import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

"""
Load the Dataset: VOC2007
Author: Jiaxuan Felix Li
"""
class myDataset(Dataset):
    def __init__(self, train_data_path, label_doc_path, transform = None, target_transform = None):
        self.img_path = train_data_path
        self.label_doc_path = label_doc_path
        self.transform = transform
        self.target_transform = target_transform
        self.imgs_names = []
        print(f"Load Data from {train_data_path} ...")
        self.total_img = 0
        for xml in os.listdir(self.label_doc_path):
            if not xml.endswith('.xml'):
                print("Cannot find xml")
                break
            xml_file = os.path.join(label_doc_path, xml)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            self.imgs_names.append(root[1].text)
            self.total_img += 1
        print("Dataset Loading status: Done")
        
    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):
        filename = self.imgs_names[idx]
        img_path = os.path.join(self.img_path,filename)
        img = cv2.imread(img_path)
        # Visualization
        # cv2.imshow("img", img)
        # key = cv2.waitKey(0)
        return img, filename

img_dataset_path = "../../VOC_DATASET/VOCtrainval_06_Nov_2007/VOCdevkit/VOC2007/JPEGImages"
label_dataset_path = "../../VOC_DATASET/VOCtrainval_06_Nov_2007/VOCdevkit/VOC2007/Annotations"

dt = myDataset(img_dataset_path, label_dataset_path)
print(dt.__len__())
print(dt.__getitem__(0)[1])

