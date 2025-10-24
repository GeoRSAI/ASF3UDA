from tqdm import tqdm
import torch
import config
from models.resunet import ResUNet
from models.asf3net import ASF3SEG
from models.output_discriminator import OutputDiscriminator
from torch.utils.data import DataLoader
from dataset_read.dataset_domain import BuildingDataset
from utils import iou
from matplotlib import pyplot as plt
import numpy as np
import os

criterion_bce = torch.nn.BCELoss()
src_number=0
tar_number=1

def train(source_loader, target_loader, seg_model, discriminator, optimizer, optimizer_D):

    iou_his1 = 0
    iou_his2 = 0
    for epoch in range(config.NUM_EPOCH):
        train_loss = 0
        iou_src_sum = 0
        iou_tar_sum = 0
        iou_src = 0
        loss_seg_all = 0
        
        # train segmentation module
        seg_model.train()
        discriminator.train()

        iou_his1 = iou_his2
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

                # for param in seg_model.parameters():
                #     param.requires_grad = True
                for param in discriminator.parameters():
                    param.requires_grad = False

                optimizer.zero_grad()
                final_src = seg_model(src_img)
                final_tar = seg_model(tar_img)
                # final_src, edge_src, *_ = seg_model(src_img)
                # final_tar, edge_tar, *_ = seg_model(tar_img)

                loss_obj = criterion_bce(final_src, src_label)
                # loss_edge = criterion_bce(edge_src, src_border)
                # loss_seg = config.LAMBDA_SEG_MAIN * loss_obj + config.LAMBDA_SEG_AUX * loss_edge
                loss_seg = config.LAMBDA_SEG_MAIN * loss_obj
                loss_seg.backward()
                loss_seg_all += loss_seg.item()
                
                iou_src_item = iou(final_src, src_label)
                iou_src_item += iou_src_item
                
                # final_tar = final_tar.detach()
                d_out = discriminator(final_tar)
                adv_loss = criterion_bce(d_out, torch.FloatTensor(d_out.data.size()).fill_(src_number).cuda(config.DEVICE))
                adv_loss = config.LAMBDA_ADV_MAIN * adv_loss
                adv_loss.backward()

                optimizer.step()
                
                # train discrimination
                # for param in seg_model.parameters():
                #     param.requires_grad = False
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
                'seg_loss': f"{loss_seg_all/num_batches:.4f}",
                'iou_src_avg': f"{iou_src_sum/num_batches:.4f}",
                'iou_src_item': f"{iou_src_item:.4f}",
                })
                pbar.update(1)  # 更新进度
        
        iou_his2 = iou_tar_sum
        if iou_his2 > iou_his1:
            torch.save(seg_model, config.MODEL_SAVE_NAME)

if __name__ == "__main__":
    dataset_src = BuildingDataset(config.val_dic)
    dataset_tar = BuildingDataset(config.train_dic)
    trainloader_src = DataLoader(dataset_src, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    trainloader_tar = DataLoader(dataset_tar, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    if config.LOAD_MODEL:
        seg_model = torch.load(config.MODEL_LOAD_NAME)
    else:
        seg_model = ResUNet(backbone='resnet50', pretrained=True).to(config.DEVICE)
    discriminator = OutputDiscriminator(num_classes=1, ndf=64).to(config.DEVICE)
    optimizer = torch.optim.sgd(seg_model.parameters(), lr=config.LEARNING_RATE)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), config.LEARNING_RATE)
    train(trainloader_src, trainloader_tar, seg_model, discriminator, optimizer, optimizer_D)

