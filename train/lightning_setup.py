#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:48:26 2024

@author: luyin
"""

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, Callback



def callback_early_stop():
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=True,
        mode='min'
    )
    return early_stop_callback

def callback_save_top_k(k):
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_acc",
        filename="{epoch:02d}-{validation_acc:.3f}",
        save_top_k=k,
        mode="max",
    )
    return checkpoint_callback


def callback_save_top_k_val_loss(k):
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_loss',
        filename="{epoch:02d}-{val_loss:.3f}",
        save_top_k= k,
        mode= 'min',
    )
    return checkpoint_callback


def callback_save_top_k_train_loss(k):
    checkpoint_callback = ModelCheckpoint(
        monitor = 'train_loss',
        filename="{epoch:02d}-{train_loss:.3f}",
        save_top_k= k,
        mode= 'min',
    )
    return checkpoint_callback


def callback_save_top_k_box_mAP(k):
    checkpoint_callback = ModelCheckpoint( 
        monitor = 'val_box_mAP',
        filename="{epoch:02d}-{val_box_mAP:.3f}",
        save_top_k= k,
        mode= 'max',
    )
    return checkpoint_callback

def callback_save_top_k_combine_mAP(k):
    checkpoint_callback = ModelCheckpoint(
        monitor = 'combine_mAP',
        filename="{epoch:02d}-{combine_mAP:.3f}",
        save_top_k= k,
        mode= 'max',
    )
    return checkpoint_callback


def callback_save_top_k_mask_mAP(k):
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_mask_mAP',
        filename="{epoch:02d}-{val_mask_mAP:.3f}",
        save_top_k= k,
        mode= 'max',
    )
    return checkpoint_callback

def callback_per_n_epoch(n):
    ckpt_callback_per_n_epoch = ModelCheckpoint(
        filename="every-{epoch:02d}-{train_loss:.3f}",
        every_n_epochs = n,
        save_top_k=-1,
    )
    return ckpt_callback_per_n_epoch

def callback_lr_monitor():
    return LearningRateMonitor(logging_interval='epoch')





# def trainer_setup():
#     trainer = Trainer(
#         max_epochs=300, gradient_clip_val=0.1,
#         callbacks=[checkpoint_callback, lr_monitor]
#      )
#     return trainer


















