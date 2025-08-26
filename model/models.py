#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:38:08 2024

@author: luyin
"""

import pytorch_lightning 
from transformers import ViTForImageClassification, AdamW, get_scheduler
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR as pt_StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR,LambdaLR, CosineAnnealingLR
import os

log_dir = '/home/luyin/Desktop/TorchVideo_Project/ly/test/lightning_logs'



class ViTLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, **kwargs):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(kwargs['model_name'],
                                                              num_labels= kwargs['n_class'],
                                                              id2label = kwargs['id2label'],
                                                              label2id = kwargs['label2id'])
        self.save_hyperparameters()

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        #lr_scheduler = get_scheduler('StepLR', optimizer,scheduler_specific_kwargs= {'gamma': 0.1})
        lr_scheduler = pt_StepLR(optimizer,step_size=10)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torchmetrics
import torch
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(
            kwargs['model_name'],
            label2id=kwargs['label2id'],
            id2label=kwargs['id2label'],
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        self.num_classes = len(kwargs['label2id'])
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        
        self.train_confmat = torchmetrics.ConfusionMatrix(task = 'multiclass',num_classes=self.num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(task = 'multiclass',num_classes=self.num_classes)
    def forward(self, x):
        return self.model(x)
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self.model(batch["pixel_values"])['logits']

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        
        return loss, accuracy, labels, predictions
    
    
    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, preds = self.common_step(batch, batch_idx)     
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.train_acc(preds, labels)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, logger=True)
        
        self.train_confmat(preds, labels)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, preds = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.val_acc(preds, labels)
        self.log('validation_acc', self.val_acc, on_step=False, on_epoch=True, logger=True)
        self.val_confmat(preds, labels)
        
    def make_np_dir(self):
        if not os.listdir(log_dir):
            n_version = 0
            np_path = os.path.join(log_dir, f'version_{n_version}','np')
            os.makedirs(np_path,exist_ok=True)
        else:
            folder_names = [file for file in os.listdir(log_dir) if file.startswith('version')]
            sorted_folder_names = sorted(folder_names, key=lambda x: int(x.split('_')[-1]))
            n_version = int(sorted_folder_names[-1].split('_')[-1])
            np_path = os.path.join(log_dir, f'version_{n_version}','np')
            os.makedirs(np_path,exist_ok=True)
        return np_path
    
    def on_validation_epoch_end(self):
        # Log confusion matrix as an image (to TensorBoard, for example)
        np_path = self.make_np_dir()
        train_confmat = self.train_confmat.compute().cpu().numpy()
        val_confmat = self.val_confmat.compute().cpu().numpy()
        print(val_confmat)
        np.save(os.path.join(np_path,f'train_confmat_{self.current_epoch}.npy'),train_confmat)
        np.save(os.path.join(np_path,f'val_confmat_{self.current_epoch}.npy'),val_confmat)
        self.train_confmat.reset()
        self.val_confmat.reset()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        #lr_scheduler = get_scheduler('StepLR', optimizer,scheduler_specific_kwargs= {'gamma': 0.1})
        lr_scheduler = pt_StepLR(optimizer,step_size=50)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class MaskRCNN_Lightning_Model(pytorch_lightning.LightningModule):
    def __init__(self, ckpt_finetuned=None, mask_loss_weight = 1.0,**kwargs):
        super().__init__()
        num_classes = kwargs['num_classes']

        
        self.model = models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT",trainable_backbone_layers=kwargs['trainable_backbone_layers'])
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        if ckpt_finetuned:
            self.model.load_state_dict(torch.load(ckpt_finetuned))
        
        self.mask_loss_weight = mask_loss_weight
        
        self.valid_preds = []
        self.valid_targets = []

        
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def get_feature_map(self,images):
        return self.model.backbone(images)

    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        #print(loss_dict.keys())
        losses = sum(loss for loss in loss_dict.values()) 
        weighted_loss = loss_dict['loss_classifier'] + \
                        loss_dict['loss_box_reg'] + \
                        self.mask_loss_weight * loss_dict['loss_mask']  + \
                        loss_dict['loss_objectness'] + \
                        loss_dict['loss_rpn_box_reg'] 
                        

        
        self.log("train_loss", losses, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss_classifier", loss_dict['loss_classifier'], on_step=False, on_epoch=True, logger=True)
        self.log("train_loss_box_reg", loss_dict['loss_box_reg'], on_step=False, on_epoch=True, logger=True)
        self.log("train_loss_mask", loss_dict['loss_mask'], on_step=False, on_epoch=True, logger=True)
        self.log("train_objectness", loss_dict['loss_objectness'], on_step=False, on_epoch=True, logger=True)
        self.log("train_loss_box_rpn", loss_dict['loss_rpn_box_reg'], on_step=False, on_epoch=True, logger=True)
        
        return weighted_loss
    
        
    def validation_step(self, batch, batch_idx):
        #self.model.training = True
        images, targets = batch
        outputs = self.model(images,targets)
        import torch.nn.functional as F
        # Function to downscale masks
        def downscale_mask(mask, scale_factor=0.5):
            # Ensure mask is 4D: [num_instances, channels, height, width]
            if mask.dim() == 2:  # [height, width]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
            elif mask.dim() == 3:  # [num_instances, height, width]
                mask = mask.unsqueeze(1)  # [num_instances, 1, height, width]

            # Downscale mask
            mask = F.interpolate(mask.float(), scale_factor=scale_factor, mode='bilinear', align_corners=False)

            # Remove the channel dimension
            return mask.squeeze(1)


        # Collect predictions and ground truths
        for output, target in zip(outputs, targets):
            
            downscaled_pred_masks = [downscale_mask(mask) for mask in output['masks']]
            if downscaled_pred_masks:
                downscaled_pred_masks = torch.stack(downscaled_pred_masks)
            else:
                continue

            # Downscale target masks
            downscaled_target_masks = [downscale_mask(mask) for mask in target['masks']]
            downscaled_target_masks = torch.stack(downscaled_target_masks)
            
            self.valid_preds.append({
                'image_id': target['image_id'],
                'boxes': output['boxes'].cpu().numpy(),
                'scores': output['scores'].cpu().numpy(),
                'labels': output['labels'].cpu().numpy(),
                'masks': downscaled_pred_masks.cpu().numpy()
            })
            self.valid_targets.append({
                'image_id': target['image_id'],
                'boxes': target['boxes'].cpu().numpy(),
                'labels': target['labels'].cpu().numpy(),
                'masks': downscaled_target_masks.cpu().numpy()
            })
        
        torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self):
        box_mAP, mask_mAP = self.compute_coco_map(self.valid_preds, self.valid_targets)
        combine_mAP = (box_mAP+mask_mAP) * 0.5
        self.log("val_box_mAP", box_mAP, on_epoch=True, logger=True)
        self.log("val_mask_mAP", mask_mAP, on_epoch=True, logger=True)
        self.log("combine_mAP", combine_mAP, on_epoch=True, logger=True)
        del self.valid_preds
        del self.valid_targets
        import gc
        gc.collect()
        self.valid_preds = []
        self.valid_targets = []
        
        
    def compute_coco_map(self, preds, targets):
        coco_gt_dict, coco_dt_dict = prepare_coco_format(preds, targets)
        if not coco_dt_dict:
            print("Warning: No valid detection results.")
            return 0.0, 0.0
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        
        coco_dt = coco_gt.loadRes(coco_dt_dict)
        
        # Box mAP
        coco_eval_box = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval_box.evaluate()
        coco_eval_box.accumulate()
        coco_eval_box.summarize()
        box_mAP = coco_eval_box.stats[0]  # mAP for boxes
        #print('..............................................................')

        
        # Mask mAP
        coco_eval_mask = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval_mask.evaluate()
        coco_eval_mask.accumulate()
        coco_eval_mask.summarize()
        mask_mAP = coco_eval_mask.stats[0]  # mAP for masks
        
        return box_mAP, mask_mAP

        
    def make_np_dir(self):
        if not os.listdir(log_dir):
            n_version = 0
            np_path = os.path.join(log_dir, f'version_{n_version}','np')
            os.makedirs(np_path,exist_ok=True)
        else:
            folder_names = [file for file in os.listdir(log_dir) if file.startswith('version')]
            sorted_folder_names = sorted(folder_names, key=lambda x: int(x.split('_')[-1]))
            n_version = int(sorted_folder_names[-1].split('_')[-1])
            np_path = os.path.join(log_dir, f'version_{n_version}','np')
            os.makedirs(np_path,exist_ok=True)
        return np_path
    
        
    # def configure_optimizers(self):
    #     params = [p for p in self.model.parameters() if p.requires_grad]
        
    #     # Use AdamW optimizer
    #     optimizer = AdamW(params, lr=5e-4, weight_decay=1e-3)
        
    #     # Define the warmup schedule
    #     warmup_steps = 10
    #     def lr_lambda(current_step: int):
    #         if current_step <= warmup_steps:
    #             return float(current_step) / float(max(1, warmup_steps))


    #     # Combine the warmup with ExponentialLR
    #     warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    #     main_scheduler = ExponentialLR(optimizer, gamma=0.1 ** (1/100))

    #     # Chain the warmup and main scheduler
    #     scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.SequentialLR(
    #             optimizer,
    #             schedulers=[warmup_scheduler, main_scheduler],
    #             milestones=[warmup_steps]  # The step at which to switch from warmup to main scheduler
    #         ),
    #         'name': 'learning_rate',
    #         'interval': 'step',
    #         'frequency': 1
    #     }

    #     return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Use AdamW optimizer
        optimizer = AdamW(params, lr=5e-4, weight_decay=1e-3)
        
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        #scheduler = CosineAnnealingLR(optimizer, 5, eta_min=5e-6,last_epoch=-1, verbose='deprecated')
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=50,
                                          cycle_mult=1.0,
                                          max_lr=5e-4,
                                          min_lr=5e-5,
                                          warmup_steps=5,
                                          gamma=0.5**(1/8))

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    

    
    # def configure_optimizers(self):
    #     from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
    #     params = [p for p in self.model.parameters() if p.requires_grad]
        
    #     optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
    #     scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                       first_cycle_steps=400,
    #                                       cycle_mult=1.0,
    #                                       max_lr=0.1,
    #                                       min_lr=0.001,
    #                                       warmup_steps=50,
    #                                       gamma=0.5)

    #     return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    
    
    

def prepare_coco_format(preds, targets):
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_dt = []
    
    if not targets:
        print("Warning: No targets available.")
        return coco_gt, coco_dt
    
    annotation_id = 1
    for target in targets:
        image_id = target['image_id']
        height, width = target['masks'].shape[-2:]

        coco_gt['images'].append({'id': image_id, 'height': height, 'width': width})
        
        for i in range(len(target['boxes'])):
            coco_gt['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(target['labels'][i]),
                'bbox': target['boxes'][i].tolist(),
                'segmentation': mask_to_rle(target['masks'][i]),
                'area': float(np.sum(target['masks'][i])),
                'iscrowd': 0,
                'height': height,
                'width': width
            })
            annotation_id += 1

    for pred in preds:
        image_id = pred['image_id']
        height, width = pred['masks'].shape[-2:]

        for i in range(len(pred['boxes'])):
            coco_dt.append({
                'image_id': image_id,
                'category_id': int(pred['labels'][i]),
                'bbox': pred['boxes'][i].tolist(),
                'score': pred['scores'][i],
                'segmentation': mask_to_rle(pred['masks'][i]),
                'height': height,
                'width': width
            })
    
    # Assuming a single category for simplicity; update as needed for your case
    coco_gt['categories'] = [{'id': i} for i in range(1, max([int(t['labels'].max().item()) for t in targets]) + 1)]
    return coco_gt, coco_dt

from pycocotools import mask as mask_util
def mask_to_rle(mask):
    # Ensure mask is a binary mask (0s and 1s)
    mask = mask > 0.5
    
    # Convert tensor to numpy array if it's not already
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Ensure mask is in the correct shape
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]  # Remove the channel dimension if it exists
    
    #print(np.amax(mask.astype(np.uint8)))
    # Convert binary mask to RLE format
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))

    # Decode the counts to string for compatibility
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from efficientnet_pytorch import EfficientNet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone



class CustomMaskRCNN_Lightning_Model(MaskRCNN_Lightning_Model):
    def __init__(self, num_classes, backbone='resnet50', ckpt_finetuned=None, mask_loss_weight=1.0, **kwargs):
        super().__init__(ckpt_finetuned=ckpt_finetuned, mask_loss_weight=mask_loss_weight, num_classes=num_classes, **kwargs)
        
        # Custom initialization logic for different backbones
        if backbone == 'resnet50_coco':
            #self.model = models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT",trainable_backbone_layers=kwargs['trainable_backbone_layers'])
            self.model = models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT",trainable_backbone_layers=kwargs['trainable_backbone_layers'])
        elif backbone.startswith('res') or backbone.startswith('wide'):
            print('--------------------------------------------------------------')
            self.model = self._create_resnet_maskrcnn(num_classes,backbone)
            print(f'{backbone} initialized')
            print('--------------------------------------------------------------')
        elif backbone.startswith('efficientnet'):
            self.model = self._create_efficientnet_maskrcnn(backbone, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace the box predictor to match the number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor to match the number of classes
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # Load fine-tuned weights if provided
        if ckpt_finetuned:
            self.model.load_state_dict(torch.load(ckpt_finetuned))
        
        self.mask_loss_weight = mask_loss_weight
        self.valid_preds = []
        self.valid_targets = []

    def _create_resnet_maskrcnn(self, num_classes,backbone):
        # Create the ResNet-101 backbone with FPN
        #backbone = resnet_fpn_backbone(backbone, pretrained=True)
        if backbone == 'resnet18' or backbone == 'resnet34':
            backbone = resnet_fpn_backbone(backbone, weights="IMAGENET1K_V1",trainable_layers=5)
        elif backbone.startswith('wide_resnet101_2'):
            backbone = resnet_fpn_backbone(backbone, weights="IMAGENET1K_V2",trainable_layers=5)
        else:
            backbone = resnet_fpn_backbone(backbone, weights="IMAGENET1K_V2",trainable_layers=5)
        model = MaskRCNN(backbone, num_classes=num_classes)
        return model

    def _create_efficientnet_maskrcnn(self, backbone_name, num_classes):
        # Create EfficientNet backbone with FPN
        efficientnet_backbone = EfficientNet.from_pretrained(backbone_name) if 'efficientnet' in backbone_name else EfficientNet.from_name(backbone_name)
        backbone = BackboneWithFPN(efficientnet_backbone, in_channels_list=[40, 112, 320], out_channels=256)
        model = MaskRCNN(backbone, num_classes=num_classes)
        return model



# from detectron2.modeling import build_model
# from detectron2.config import get_cfg
# from detectron2 import model_zoo

# import torch.nn.functional as F
# from detectron2.utils.events import EventStorage
# from detectron2.structures import Instances, Boxes, BitMasks,PolygonMasks
# from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
# from detectron2.data.datasets import register_coco_instances
# from detectron2.checkpoint import DetectionCheckpointer

# class Detectron2LightningModel(MaskRCNN_Lightning_Model):
#     def __init__(self, cfg, *args, **kwargs):
#         super(Detectron2LightningModel, self).__init__(*args, **kwargs)
#         self.cfg = cfg
#         self.model = build_model(cfg)
#         self.checkpointer = DetectionCheckpointer(self.model)
#         self.checkpointer.load(cfg.MODEL.WEIGHTS)
#         self.model.train()
    
#     def forward(self, images, targets=None):
#         return self.model(images, targets)

#     def validation_step(self, batch, batch_idx):
#         images, targets = batch
#         print('-------------------------------------------------------')
#         print('val', self.model.training)
#         outputs = self.model(images)

#         # Function to downscale masks
#         def downscale_mask(mask, scale_factor=0.5):
#             if mask.dim() == 2:  # [height, width]
#                 mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
#             elif mask.dim() == 3:  # [num_instances, height, width]
#                 mask = mask.unsqueeze(1)  # [num_instances, 1, height, width]

#             mask = F.interpolate(mask.float(), scale_factor=scale_factor, mode='bilinear', align_corners=False)
#             return mask.squeeze(1)

#         # Collect predictions and ground truths
#         for output, target in zip(outputs, targets):
#             downscaled_pred_masks = [downscale_mask(mask) for mask in output['masks']]
#             if downscaled_pred_masks:
#                 downscaled_pred_masks = torch.stack(downscaled_pred_masks)
#             else:
#                 continue

#             downscaled_target_masks = [downscale_mask(mask) for mask in target['masks']]
#             downscaled_target_masks = torch.stack(downscaled_target_masks)

#             self.valid_preds.append({
#                 'image_id': target['image_id'],
#                 'boxes': output['boxes'].cpu().numpy(),
#                 'scores': output['scores'].cpu().numpy(),
#                 'labels': output['labels'].cpu().numpy(),
#                 'masks': downscaled_pred_masks.cpu().numpy()
#             })
#             self.valid_targets.append({
#                 'image_id': target['image_id'],
#                 'boxes': target['boxes'].cpu().numpy(),
#                 'labels': target['labels'].cpu().numpy(),
#                 'masks': downscaled_target_masks.cpu().numpy()
#             })

#         torch.cuda.empty_cache()

    