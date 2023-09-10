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


def load_img_info(path, filtered):
    """
    load the img information from xml
    save as following data structure:
        {img1_name:[[width, height, depth], [object1 name, xmin, ymin, xmax, ymax], [object2 name, xmin, ymin, xmax, ymax], ....],
         img2_name:[[width, height, depth], [object1 name, xmin, ymin, xmax, ymax], [object2 name, xmin, ymin, xmax, ymax], ....],
         ...
        }
    """
    image_names = []
    num_of_imgs = 0
    img_dict = dict()
    for xml in os.listdir(path):
        if not xml.endswith('.xml'):
            print("Cannot find xml")
            break
        # get the tree of .xml
        else:
            if xml in filtered:
                xml_file = os.path.join(path, xml)
                tree = ET.parse(xml_file)
                root = tree.getroot()
                image_names.append(root[1].text)
                num_of_imgs += 1
                img_dict[root[1].text] = get_info_from_xml(tree)
    return img_dict, num_of_imgs, image_names


# tree for one image
def get_info_from_xml(tree):
    get_img_info = []
    for elems in tree.iter():
        size_info = []
        if elems.tag == "size":
            for e in elems:
                size_info.append(float(e.text))
            get_img_info.append(size_info)
        if elems.tag == "object":
            object = []
            for obj in elems:
                if obj.tag == "name":
                    object.append(obj.text) # append name of the object
                if obj.tag == "bndbox":
                    for e in obj:
                        object.append(int(e.text)) # append xmin, ymin, xmax, ymax
            get_img_info.append(object)
    return get_img_info


def draw_boundingbox(img, img_info):
    for i, obj in enumerate(img_info):
        if i == 0:
            continue
        else:
            object_name = obj[0]
            xmin, ymin, xmax, ymax = obj[1], obj[2], obj[3], obj[4]
            print((xmin, ymin), (xmax, ymax))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(img, object_name, (xmin-10, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img


class myDataset(Dataset):
    def __init__(self, train_data_path, label_doc_path, transform = None, target_transform = None, Train = False):
        self.img_path = train_data_path # image path
        self.label_doc_path = label_doc_path # xml path
        self.transform = transform # data augmentation
        self.target_transform = target_transform # target data augmentation
        print(f"Load Data from {train_data_path} ...")
        self.image_list = []
        if Train == True:
            modified_path = os.path.join(os.path.dirname(train_data_path), "ImageSets/Main/train.txt")
            modified_path = modified_path.replace("\\", "/")
            with open(modified_path) as file:
                self.image_list = [line.rstrip()+".xml" for line in file]
        else:
            modified_path = os.path.join(os.path.dirname(train_data_path), "ImageSets/Main/val.txt")
            modified_path = modified_path.replace("\\", "/")
            with open(modified_path) as file:
                self.image_list = [line.rstrip()+".xml" for line in file]
        print(self.image_list)

        self.imgs_info, self.total_img, self.imgs_names = load_img_info(self.label_doc_path, self.image_list)
        print("Dataset Loading status: Done")
        
    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):
        filename = self.imgs_names[idx]
        img_path = os.path.join(self.img_path,filename)
        img = cv2.imread(img_path)
        #Visualization
        #draw bounding box
        # print(filename)
        # img_copy = img.copy()
        # img_copy = draw_boundingbox(img, self.imgs_info[filename])
        # cv2.imshow("img", img_copy)
        # cv2.waitKey(0)
        return img, self.imgs_info[filename]

# img_dataset_path = "../../VOC_DATASET/VOCtrainval_06_Nov_2007/VOCdevkit/VOC2007/JPEGImages"
# label_dataset_path = "../../VOC_DATASET/VOCtrainval_06_Nov_2007/VOCdevkit/VOC2007/Annotations"

# dt = myDataset(img_dataset_path, label_dataset_path, Train= False)
# print(dt.__len__())
# print(dt.__getitem__(0)[1])

