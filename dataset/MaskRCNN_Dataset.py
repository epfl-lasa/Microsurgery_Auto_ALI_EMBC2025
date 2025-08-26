#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:38:23 2024

@author: luyin
"""

import os
import torch
from ly.constants.read_yaml import get_para
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2 as T




#para = get_para('/home/luyin/Desktop/ly_torch/ly/constants/para_MaskRCNN.yaml')
#dataset_path = para['dataset_path']


class CustomDataset_MaskRCNN(torch.utils.data.Dataset):
    def __init__(self, folder_path, transforms=None,modified_anno=None):
        self.transforms = transforms
        anno_path = os.path.join(folder_path, '_annotations.coco.json')
        if modified_anno:
            anno_path = os.path.join(folder_path,modified_anno)
        dataset = datasets.CocoDetection(folder_path, anno_path)
        self.dataset_wrap = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "masks","boxes", "labels", "masks","area","iscrowd","image_id"))

    def __getitem__(self, idx):
        
        img = tv_tensors.Image(self.dataset_wrap[idx][0])
        target = self.dataset_wrap[idx][1]

        
        if self.transforms is not None:
        #     if 'labels' not in target.keys():
        #         target['boxes'] = target['boxes'] = target['labels'] = target['area'] = target['iscrowd'] = None
            if 'labels' not in target:
                canvas_size = tuple(T.functional.get_size(img))
                
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target["boxes"] = T.functional.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                target["boxes"],
                format=tv_tensors.BoundingBoxFormat.XYWH,
                canvas_size=canvas_size,),new_format=tv_tensors.BoundingBoxFormat.XYXY,)
                
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
                target['area'] = torch.zeros((0,), dtype=torch.float32)
                target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
                
                target['masks'] = torch.zeros((0, img.shape[-2], img.shape[-1]), dtype=torch.uint8)  # Adjust based on your mask shape

                
             
            if self.transforms is not None:
                img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.dataset_wrap)
    
    



def transform_train_MaskRCNN():
    train_transform = T.Compose(
        [   T.Resize(size=(1536,2048)),
            T.RandomRotation(degrees=(0, 30)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),

            
            # T.RandomChoice([
            #     T.ColorJitter(brightness=0.5,hue=0.3),
            #     T.RandomAutocontrast(p=1.0),
            #     T.RandomAdjustSharpness(sharpness_factor=2),
            #     T.RandomAdjustSharpness(sharpness_factor=0.5),
            #     T.RandomEqualize(),
            #     T.RandomGrayscale(p=0.5),
            #     ]),
                        
            #T.TrivialAugmentWide(),
            
            T.SanitizeBoundingBoxes(),
 
            T.ToDtype(torch.float, scale=True),
            T.ToPureTensor()
            ])
    return train_transform




def transform_val_MaskRCNN():
    val_transform = T.Compose(
        [  T.Resize(size=(1536,2048)),
           T.ToDtype(torch.float, scale=True),
           T.ToPureTensor()
            ])
    return val_transform

def dataset_train_MaskRCNN(dataset_path):
    subset_path = os.path.join(dataset_path,'train')
    dataset_train = CustomDataset_MaskRCNN(subset_path, transform_train_MaskRCNN())
    return dataset_train

def dataset_val_MaskRCNN(dataset_path):
    subset_path = os.path.join(dataset_path,'val')
    dataset_val = CustomDataset_MaskRCNN(subset_path, transform_val_MaskRCNN()) 
    return dataset_val

def dataset_test_MaskRCNN(dataset_path):
    subset_path = os.path.join(dataset_path,'test')
    dataset_val = CustomDataset_MaskRCNN(subset_path, transform_val_MaskRCNN())
    print(subset_path)
    return dataset_val

def collate_fn_MaskRCNN(batch):
    images = [item[0] for item in batch]  # Assuming item[0] is a tensor of shape [C, H, W]
    targets = []
    for idx, item in enumerate(batch):
        target = {}
        img = images[idx]  # Current image tensor
        _, height, width = img.shape  # Extract height and width
        target['boxes'] = item[1]['boxes'] if 'boxes' in item[1] else torch.zeros((0, 4))
        target['labels'] = item[1]['labels'] if 'labels' in item[1] else torch.zeros((0,), dtype=torch.int64)
        target['image_id'] = item[1]['image_id']
        # Handle masks, adjusting the shape based on current image dimensions
        if 'masks' in item[1]:
            target['masks'] = item[1]['masks']
        else:
            target['masks'] = torch.zeros((0, height, width))  # Adjusted for dynamic size

        targets.append(target)

    images = torch.stack(images, 0)  # Stack images to create a batch
    return images, targets



