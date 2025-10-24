import os
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class BuildingDataset_crdom(Dataset):
    def __init__(self, root_srcdic):
        self.root_srcdic = root_srcdic

        self.srcpath = os.path.join(root_srcdic, 'img256')

        self.srcimg_list = sorted(os.listdir(self.srcpath))

        self.length_data = len(self.srcimg_list)
        
        self.img_transform = transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC)
        self.label_transform = transforms.Resize(128, interpolation=transforms.InterpolationMode.NEAREST)
        
        # 标签只需要转换为Tensor
        self.tensor_transform = transforms.ToTensor()


    def __len__(self):
        return self.length_data
    
    def __getitem__(self, index):
        # 加载源域数据
        srcimg_name = os.path.splitext(self.srcimg_list[index % self.length_data])[0]
        src_img = Image.open(os.path.join(self.root_srcdic, 'img256', srcimg_name+'.jpg')).convert("RGB")
        src_label = Image.open(os.path.join(self.root_srcdic, 'label256', srcimg_name+'.png')).convert("L")
        # src_img = self.img_transform(src_img)
        # src_label = self.label_transform(src_label)

        src_label_np = np.array(src_label)
        str_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        src_border = cv2.dilate(src_label_np, str_element) - cv2.erode(src_label_np, str_element)
        
        # 应用转换
        src_img = self.tensor_transform(src_img)
        src_label = self.tensor_transform(src_label)
        src_border = self.tensor_transform(src_border)

        # 确保标签是0或1（处理可能的插值残留）
        src_label = (src_label > 0.5).float()
        
        return src_img, src_label, src_border
    
if __name__ == "__main__":
    root_srcdic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/inria/train"
    root_tardic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/WHU/AerialImage/train"
    input_size = 128
    dataset = BuildingDataset_crdom(root_srcdic, root_tardic)
    src_img, src_label, src_border, tar_img, tar_label, tar_border = dataset[115]