from tqdm import tqdm
import torch
import config
from models.asf3net import ASF3SEG
from models.output_discriminator import OutputDiscriminator
from torch.utils.data import DataLoader
from dataset_read.dataset_domain import BuildingDataset
from utils import iou
import torch.nn.functional as F
from collections import deque

criterion_bce = torch.nn.BCELoss()
src_number=0
tar_number=1

class FeatureMemoryBank:
    def __init__(self, max_size=50):
        self.queue_s = deque(maxlen=max_size)
        self.queue_t = deque(maxlen=max_size)

    def enqueue(self, feat_s: torch.Tensor, feat_t: torch.Tensor):
        # 直接存储转置后的二维特征，不再增加维度
        if feat_s is not None and feat_t is not None:
            self.queue_s.append(feat_s.t().detach().clone())  # [N, C]
            self.queue_t.append(feat_t.t().detach().clone())  # [N, C]

    def get_memory(self):
        if len(self.queue_s) > 0 and len(self.queue_t) > 0:
            # 拼接所有样本 [sum(N), C]
            mem_s = torch.cat(list(self.queue_s), dim=0)  # [sum(N), C]
            mem_t = torch.cat(list(self.queue_t), dim=0)  # [sum(N), C]
            # 转置为 [C, sum(N)] 以符合协方差计算要求
            return mem_s.t(), mem_t.t()  # [C, sum(N)]
        else:
            return None, None

def extract_building_features(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    feat: [B, C, H, W]
    mask: [B, 1, H, W] binary
    returns: [C, N] where N = number of building pixels
             or None if N == 0
    """
    B, C, H, W = feat.shape
    feat_flat = feat.view(B, C, -1)                # [B, C, H*W]
    mask_flat = mask.view(B, 1, -1)                # [B, 1, H*W]
    mask_b = mask_flat.expand(-1, C, -1).bool()    # [B, C, H*W]
    if mask_b.sum() == 0:
        return None  # No building pixels
    selected = feat_flat[mask_b].view(C, -1)       # [C, N]
    return selected

def covariance_matrix(X: torch.Tensor) -> torch.Tensor:
    """ Compute CxC covariance from [C, N] features or return None if invalid """
    # X: [C, N]
    if X is None or X.numel() == 0 or X.shape[1] < 2:
        return None
    mu = X.mean(dim=1, keepdim=True)               # [C,1]
    X_centered = X - mu                            # [C, N]
    N = X.shape[1]
    cov = (X_centered @ X_centered.t()) / (N - 1)   # [C, C]
    return cov

def coral_loss(cov_s: torch.Tensor, cov_t: torch.Tensor) -> torch.Tensor:
    """ Frobenius norm between two covariance matrices """
    return torch.norm(cov_s - cov_t, p='fro')**2 / (4 * cov_s.shape[0]**2)

def coral(src_fea, tar_fea, src_gt, tar_gt):
    Ms = (src_gt >= 0.5).float()
    Ms = F.interpolate(Ms, size=src_fea.shape[2:], mode='nearest')
    # print(Ms.shape)
    Mt = (tar_gt >= 0.5).float()
    Mt = F.interpolate(Mt, size=tar_fea.shape[2:], mode='nearest')
    Xs = extract_building_features(src_fea, Ms)  # [C, Ns]
    Xt = extract_building_features(tar_fea, Mt)
    cov_s = covariance_matrix(Xs)
    cov_t = covariance_matrix(Xt)
    return Xs, Xt, cov_s, cov_t

def train(source_loader, target_loader, seg_model, discriminator, optimizer, optimizer_D):
    
    for epoch in range(config.NUM_EPOCH):
        loss_seg_all = 0
        loss_gene_all = 0
        loss_adv_all = 0
        loss_coral_all = 0
        loss_one_all = 0

        memory_bank1 = FeatureMemoryBank(max_size=128)
        memory_bank2 = FeatureMemoryBank(max_size=128)
        memory_bank3 = FeatureMemoryBank(max_size=128)
        
        seg_model.train()
        discriminator.train()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        num_batches = min(len(source_loader), len(target_loader))
        
        with tqdm(total=num_batches, desc="Training", unit="batch", dynamic_ncols=True) as pbar:
            for i in range(num_batches):
                srcimg_name, src_img, src_label, src_border = next(source_iter)
                tarimg_name, tar_img, tar_label, tar_border = next(target_iter)
                src_img = src_img.to(config.DEVICE)
                src_label = src_label.to(config.DEVICE)
                src_border = src_border.to(config.DEVICE)
                tar_img = tar_img.to(config.DEVICE)

                loss_gene = 0
                loss_coral = 0
                loss_coral1 = 0
                loss_coral2 = 0
                loss_coral3 = 0
                
                # for param in seg_model.parameters():
                #     param.requires_grad = True
                for param in discriminator.parameters():
                    param.requires_grad = False
                optimizer.zero_grad()
                
                final_src,edge_src,src_f = seg_model(src_img)
                final_tar,edge_tar,tar_f = seg_model(tar_img)
                loss_obj = criterion_bce(final_src, src_label)
                loss_edge = criterion_bce(edge_src, src_border)

                Xs1, Xt1, cov_s1, cov_t1 = coral(src_f[0], tar_f[0], final_src, final_tar)
                if cov_s1 is not None and cov_t1 is not None:
                    loss_coral1 = coral_loss(cov_s1, cov_t1)
                
                Xs2, Xt2, cov_s2, cov_t2 = coral(src_f[1], tar_f[1], final_src, final_tar)
                if cov_s2 is not None and cov_t2 is not None:
                    loss_coral2 = coral_loss(cov_s2, cov_t2)
                
                Xs3, Xt3, cov_s3, cov_t3 = coral(src_f[2], tar_f[2], final_src, final_tar)
                if cov_s3 is not None and cov_t3 is not None:
                    loss_coral3 = coral_loss(cov_s3, cov_t3)
                
                mem_s1, mem_t1 = memory_bank1.get_memory()
                if mem_s1 is not None and mem_t1 is not None:
                    cov_ms1 = covariance_matrix(mem_s1)
                    cov_mt1 = covariance_matrix(mem_t1)
                    if cov_ms1 is not None and cov_mt1 is not None:
                        loss_coral1 += 0.5 * coral_loss(cov_ms1, cov_mt1)
                
                mem_s2, mem_t2 = memory_bank2.get_memory()
                if mem_s2 is not None and mem_t2 is not None:
                    cov_ms2 = covariance_matrix(mem_s2)
                    cov_mt2 = covariance_matrix(mem_t2)
                    if cov_ms2 is not None and cov_mt2 is not None:
                        loss_coral2 += 0.5 * coral_loss(cov_ms2, cov_mt2)
                
                mem_s3, mem_t3 = memory_bank3.get_memory()
                if mem_s3 is not None and mem_t3 is not None:
                    cov_ms3 = covariance_matrix(mem_s3)
                    cov_mt3 = covariance_matrix(mem_t3)
                    if cov_ms3 is not None and cov_mt3 is not None:
                        loss_coral3 += 0.5 * coral_loss(cov_ms3, cov_mt3)
                
                loss_coral = loss_coral1 + loss_coral2 + loss_coral3
                loss_coral_all += loss_coral
                loss_seg = config.LAMBDA_SEG_MAIN * loss_obj + config.LAMBDA_SEG_AUX * loss_edge
                loss_seg_all += loss_seg.item()
                loss_one = config.LAMBDA_SEG_MAIN * loss_obj + config.LAMBDA_SEG_AUX * loss_edge + config.LAMBDA_SEG_COVAR * loss_coral
                loss_one_all += loss_one.item()

                iou_src_item = iou(final_src, src_label)
                iou_src_sum += iou_src_item
                
                d_out = discriminator(final_tar)
                loss_adv = criterion_bce(d_out, torch.FloatTensor(d_out.data.size()).fill_(src_number).cuda(config.DEVICE))
                loss_adv = config.LAMBDA_ADV_MAIN * loss_adv
                loss_adv_all += loss_adv.item()
                
                loss_gene = loss_one + loss_adv
                loss_gene_all += loss_gene.item()
                loss_gene.backward()
                optimizer.step()

                if Xs1 is not None and Xt1 is not None:
                    memory_bank1.enqueue(Xs1.detach(), Xt1.detach())
                if Xs2 is not None and Xt2 is not None:
                    memory_bank2.enqueue(Xs2.detach(), Xt2.detach())
                if Xs3 is not None and Xt3 is not None:
                    memory_bank3.enqueue(Xs3.detach(), Xt3.detach())
                
                # train discrimination
                for param in discriminator.parameters():
                    param.requires_grad = True
                optimizer_D.zero_grad()

                with torch.no_grad():
                    final_src_disc = final_src.clone()  # 直接使用已计算的结果
                    final_tar_disc = final_tar.clone()

                d_out = discriminator(final_src_disc)
                d_loss_1 = criterion_bce(d_out, torch.FloatTensor(d_out.data.size()).fill_(src_number).cuda(config.DEVICE))
                d_loss_1.backward()

                d_out = discriminator(final_tar_disc)
                d_loss_2 = criterion_bce(d_out, torch.FloatTensor(d_out.data.size()).fill_(tar_number).cuda(config.DEVICE))
                d_loss_2.backward()

                optimizer_D.step()

                pbar.set_postfix({
                'epoch': f"{epoch + 1}/{config.NUM_EPOCH}",
                'src': f"{iou_src_sum/(i+1):.4f}",
                'coral': f"{loss_coral_all/(i+1):.4f}",
                'adv': f"{loss_adv_all/(i+1):.4f}",
                })
                pbar.update(1)  # 更新进度

        torch.save(seg_model, config.MODEL_SAVE_NAME)

if __name__ == "__main__":
    dataset_src = BuildingDataset(config.train_dic)
    dataset_tar = BuildingDataset(config.val_dic)
    trainloader_src = DataLoader(dataset_src, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    trainloader_tar = DataLoader(dataset_tar, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    if config.LOAD_MODEL:
        seg_model = torch.load(config.MODEL_LOAD_NAME)
    else:
        seg_model = ASF3SEG().to(config.DEVICE)
    discriminator = OutputDiscriminator(num_classes=1, ndf=64).to(config.DEVICE)
    optimizer = torch.optim.SGD(seg_model.parameters(), lr=config.LEARNING_RATE)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), config.LEARNING_RATE)
    train(trainloader_src, trainloader_tar, seg_model, discriminator, optimizer, optimizer_D)
