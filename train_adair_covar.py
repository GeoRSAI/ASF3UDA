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
    
    #Comprehensive code and will be made available upon publication of the accompanying paper.

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
