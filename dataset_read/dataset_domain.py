from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import cv2

class BuildingDataset(Dataset):
    def __init__(self, root_dic):
        self.root_dic = root_dic
        self.path = os.path.join(root_dic, 'img256')
        self.img_list = sorted(os.listdir(self.path))
        self.img_len = len(self.img_list)
    
        self.img_transform = transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC)
        self.label_transform = transforms.Resize(128, interpolation=transforms.InterpolationMode.NEAREST)
        self.data_tensor = transforms.ToTensor()

    def __len__(self):
        return self.img_len
    
    def __getitem__(self, index):
        
        img_name = os.path.splitext(self.img_list[index % self.img_len])[0]
        img = Image.open(os.path.join(self.root_dic, 'img256', img_name+'.jpg')).convert("RGB")
        label = Image.open(os.path.join(self.root_dic, 'label256', img_name+'.png')).convert("L")
        img = self.img_transform(img)
        label = self.label_transform(label)

        label_np = np.array(label)
        str_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        border = cv2.dilate(label_np, str_element) - cv2.erode(label_np, str_element)

        img = self.data_tensor(img)
        label = self.data_tensor(label)
        border = self.data_tensor(border)

        label = (label > 0.5).float()
        border = (border > 0.5).float()

        return img_name, img, label, border

if __name__ == "__main__":
    root_dic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/WHU/AerialImage/test"
    dataset = BuildingDataset(root_dic)
    print(dataset[1])