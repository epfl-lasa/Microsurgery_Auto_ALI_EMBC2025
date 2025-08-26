#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# Local imports
from dataset.MaskRCNN_Dataset import dataset_train_MaskRCNN, dataset_val_MaskRCNN, collate_fn_MaskRCNN
from model.models import MaskRCNN_Lightning_Model, CustomMaskRCNN_Lightning_Model
from constants.read_yaml import get_para
from train.lightning_setup import (
    callback_save_top_k_val_loss,
    callback_per_n_epoch,
    callback_lr_monitor,
    callback_save_top_k_train_loss,
    callback_save_top_k_mask_mAP,
    callback_save_top_k_box_mAP,
    callback_save_top_k_combine_mAP,
)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_fold(fold_idx: int, backbone: str, batch_size: int, max_epochs: int = 400):
    """Train one fold with a given backbone."""
    dataset_path = Path(
        f"/home/luyin/Desktop/ly_torch/ly/scripts/Image_Processing/contrast/dataset/5_fold_cross_no_val_balanced/fold_{fold_idx+1}"
    )

    # Dataset & DataLoader
    dataset_train = dataset_train_MaskRCNN(dataset_path)
    dataloader_train = DataLoader(
        dataset_train,
        collate_fn=collate_fn_MaskRCNN,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
    )

    # Define model
    model_params = {"num_classes": 3, "trainable_backbone_layers": 5, "backbone": backbone}
    model = CustomMaskRCNN_Lightning_Model(**model_params)

    # Callbacks (you can switch set here)
    callbacks = [
        callback_lr_monitor(),
        callback_save_top_k_train_loss(2),
        # Optionally: callback_save_top_k_box_mAP(2), callback_save_top_k_mask_mAP(2)
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        accumulate_grad_batches=8,  # adjust as needed
    )

    print(f"[INFO] Starting training: Fold {fold_idx+1}, Backbone={backbone}")
    trainer.fit(model, train_dataloaders=dataloader_train)

if __name__ == "__main__":
    set_seed(42)

    BATCH_SIZE = 2
    MAX_EPOCHS = 400
    BACKBONES = ["resnet50_coco"]  # extend this list if needed

    for fold_idx in range(5):
        for backbone in BACKBONES:
            train_fold(fold_idx, backbone, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS)
