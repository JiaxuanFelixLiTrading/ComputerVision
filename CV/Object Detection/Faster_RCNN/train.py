import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.datacollection import myDataset

# # ignore warning
# import warnings
# warnings.filterwarnings("ignore")
img_dataset_path = "../VOC_DATASET/VOCtrainval_06_Nov_2007/VOCdevkit/VOC2007/JPEGImages"
label_dataset_path = "../VOC_DATASET/VOCtrainval_06_Nov_2007/VOCdevkit/VOC2007/Annotations"
training_data= myDataset(img_dataset_path, label_dataset_path,Train= True)
train_loader = DataLoader(dataset = training_data, batch_size=6, shuffle= False, num_workers=4)

Epoch = 3
for epoch in range(Epoch):
    print(f"Epoch ----------> {epoch}")
    for i, x in enumerate(training_data):
        print(i, x[0], "----------->",x[1])
        