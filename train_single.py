from tqdm import tqdm
import torch
import torch.nn as nn
import config
from models.asf3net import ASF3SEG
from torch.utils.data import DataLoader
from dataset_read.dataset_border import BuildingDataset_crdom
from utils import iou
import os


def train():
    dataset = BuildingDataset_crdom(config.whu_dic)
    
    if config.LOAD_MODEL:
        seg_model = torch.load(config.MODEL_LOAD_NAME)
    else:
        seg_model = ASF3SEG().to(config.DEVICE)
    
    optimizer = torch.optim.Adam(seg_model.parameters(), lr=config.LEARNING_RATE)
    trainloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    criterion_bce = nn.BCELoss()

    for epoch in range(config.NUM_EPOCH):
        loop = tqdm(trainloader, leave=True)
        train_loss = 0
        iou_sum = 0
        seg_model.train()
        for batch_idx, (img, label, border) in enumerate(loop):
            img, label, border = img.to(config.DEVICE), label.to(config.DEVICE), border.to(config.DEVICE)
            img = img.to(config.DEVICE)
            label = label.to(config.DEVICE)
            optimizer.zero_grad()

            final, edge, *_ = seg_model(img)
            loss_obj = criterion_bce(final, label)
            loss_edge = criterion_bce(edge, border)
            loss_seg = config.LAMBDA_SEG_MAIN * loss_obj + config.LAMBDA_SEG_AUX * loss_edge
            loss_seg.backward()
            optimizer.step()
            iou_item = iou(final, label)
            iou_sum += iou_item

            train_loss = train_loss + loss_seg.item()

            loop.set_description(f'Epoch[{epoch+1}/{config.NUM_EPOCH}]')
            loop.set_postfix(loss=train_loss, iou_src=iou_item, iou_avg=iou_sum/(batch_idx+1))

        # if epoch % 5 == 0:
        #     torch.save(seg_model, f"{config.MODEL_SAVE_NAME}_{epoch}")

if __name__ == "__main__":
    train()