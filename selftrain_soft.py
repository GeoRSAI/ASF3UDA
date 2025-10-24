import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_read.dataset_border import BuildingDataset_crdom
from models.asf3net import ASF3SEG
from utils import iou
import config

# ----- 超参设置 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 伪标签策略参数
initial_threshold = 0.9  # 初始硬阈值 tau
min_threshold = 0.5  # 最低阈值
threshold_decay = 0.05  # 每 N 轮降低多少
temperature = 1.5  # 温度系数 T
weight_gamma = 2.0  # 权重指数 γ
lambda_soft = 0.5  # 软交叉熵在总代价中的权重
lambda_dice = 0.5  # Dice Loss 在总代价中的权重
decay_interval = 5  # 每 decay_interval 轮更新一次阈值和 γ


# ----- 工具函数 -----
def temperature_scale(p, T):
    """
    对概率图 p 做温度缩放
    p: Tensor, shape (B,1,H,W), 值在 (0,1)
    T: float
    return: Tensor 同 shape 的缩放后概率
    """
    # 避免 log(0)
    eps = 1e-6
    p_clamped = p.clamp(eps, 1 - eps)
    logit = torch.log(p_clamped / (1 - p_clamped))
    scaled = torch.sigmoid(logit / T)
    return scaled


def compute_soft_pseudo_loss(pred, pseudo_prob, weight_map):
    """
    加权软交叉熵
    pred       : Tensor, 模型输出概率, shape (B,1,H,W)
    pseudo_prob: Tensor, 伪标签概率, shape (B,1,H,W)
    weight_map : Tensor, 像素权重,   shape (B,1,H,W)
    """
    # BCE with logits 如果你的pred是logits，把下面一行换成 F.binary_cross_entropy_with_logits
    loss = - (pseudo_prob * torch.log(pred + 1e-6) +
              (1 - pseudo_prob) * torch.log(1 - pred + 1e-6))
    weighted = (weight_map * loss).mean()
    return weighted


def dice_loss(pred, pseudo_prob, eps=1e-6):
    """
    Soft Dice Loss
    pred       : Tensor, 模型输出概率, shape (B,1,H,W)
    pseudo_prob: Tensor, 伪标签概率, shape (B,1,H,W)
    """
    inter = (pred * pseudo_prob).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + pseudo_prob.sum(dim=[1, 2, 3])
    dice = 1 - (2 * inter + eps) / (union + eps)
    return dice.mean()


def generate_pseudo(p, tau, gamma):
    """
    从原始概率图 p 生成伪标签概率 pseudo_prob 和权重 weight_map
    p  : Tensor, 原始模型输出概率, shape (B,1,H,W)
    tau: float, 硬阈值
    gamma: float, 权重指数
    """
    # 先做温度缩放
    p_t = temperature_scale(p, temperature)

    # 硬标签掩码
    hard_pos = (p_t >= tau).float()
    hard_neg = (p_t <= 1 - tau).float()
    soft_mask = 1 - (hard_pos + hard_neg)  # 中间区间

    # 生成最终伪标签概率：硬正例用1，硬负例用0，其余保持 p_t
    pseudo_prob = hard_pos + soft_mask * p_t

    # 根据伪标签概率生成权重
    weight_map = torch.where(
        soft_mask.bool(),
        p_t.pow(gamma),  # 中间区间用 p^γ
        torch.ones_like(p_t)  # 硬标签区间权重=1
    )
    return pseudo_prob, weight_map


# ----- 示例训练循环 -----
def train(model, trainloader, optimizer):
    model.to(device)
    bce_loss = nn.BCELoss()  # 如果输出是概率图
    iou_his1 = iou_his2 = 0
    tau = initial_threshold
    gamma = weight_gamma

    for epoch in range(config.NUM_EPOCH):
        model.train()
        iou_his1 = iou_his2
        loop = tqdm(trainloader, leave=True)
        train_loss = 0
        iou_item = 0
        iou_sum = 0
        iou_src = 0
        for i, (name, src_img, src_label, src_border, tar_img, tar_label, tar_border) in enumerate(loop):
            src_img, src_label, src_border = src_img.to(config.DEVICE), src_label.to(config.DEVICE), src_border.to(config.DEVICE)
            tar_img, tar_label, tar_border = tar_img.to(config.DEVICE), tar_label.to(config.DEVICE), tar_border.to(config.DEVICE)

            final_src, *_ = seg_model(src_img)
            final_tar, *_ = seg_model(tar_img)

            # ------ 源域监督 ------
            loss_s = bce_loss(final_src, src_label)

            # ------ 目标域伪标签损失 ------
            pseudo_prob, weight_map = generate_pseudo(final_tar.detach(), tau, gamma)
            loss_soft = compute_soft_pseudo_loss(final_tar, pseudo_prob, weight_map)
            loss_dice = dice_loss(final_tar, pseudo_prob)

            # 总损失
            loss = loss_s + lambda_soft * loss_soft + lambda_dice * loss_dice
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iou_src = iou(final_src, src_label)
            iou_item = iou(final_tar, tar_label)
            iou_sum += iou_item

            loop.set_description(f"Epoch[{epoch+1}/{config.NUM_EPOCH}]")
            loop.set_postfix(loss=train_loss, iou_src=iou_src , iou_item=iou_item, iou_avg=iou_sum/(i+1))

        # 动态更新阈值和 γ
        if epoch % decay_interval == 0:
            tau = max(tau - threshold_decay, min_threshold)
            gamma = max(gamma - 0.1, 1.0)  # 例如逐步降低 γ

        print(f'Epoch [{epoch}/{config.NUM_EPOCH}] '
              f'Loss_s:{loss_s.item():.4f} '
              f'Soft:{loss_soft.item():.4f} '
              f'Dice:{loss_dice.item():.4f} '
              f'Tau:{tau:.2f} Gamma:{gamma:.2f}')
        
        iou_his2 = iou_sum
        if iou_his2 > iou_his1:
            torch.save(seg_model, config.MODEL_SAVE_NAME)

# ----- 用法示例 -----

dataset = BuildingDataset_crdom(config.train_dic, config.val_dic)
trainloader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, pin_memory=True)
seg_model = (torch.load(config.MODEL_LOAD_NAME) if config.LOAD_MODEL
                 else ASF3SEG().to(config.DEVICE))
optimizer   = torch.optim.Adam(seg_model.parameters(), lr=config.LEARNING_RATE)

train(seg_model, trainloader, optimizer)
