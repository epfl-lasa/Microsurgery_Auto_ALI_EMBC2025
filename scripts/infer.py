#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import re
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.functional import to_pil_image
from pycocotools.coco import COCO
from pycocotools import mask as mask_util

# Local imports
from dataset.MaskRCNN_Dataset import transform_val_MaskRCNN
from model.models import MaskRCNN_Lightning_Model, CustomMaskRCNN_Lightning_Model
from constants.read_yaml import get_para
from util.visHelper import (
    resize_image_to_target_longest_side,
    paint_ground_truth_mask,
    compute_iou,
    compute_bbox_iou,
    compute_distance_and_size_metrics,
    calculate_precision_recall,
    extract_loss,
    check_serialization_issue,
    convert_to_serializable,
)


# CONFIGURATION
FOLD_NUMBER = 3
CKPT_VERSION = FOLD_NUMBER - 1
FOLDER_PATH = Path(
    f"/home/luyin/Desktop/ly_torch/ly/scripts/Image_Processing/contrast/dataset/5_fold_cross_no_val_balanced/fold_{FOLD_NUMBER}/test"
)
CKPT_DIR = Path(
    f"/home/luyin/Desktop/ly_torch/ly/scripts/Image_Processing/contrast/lightning_logs/version_{CKPT_VERSION}/checkpoints/"
)

THRESH_CLS = 0.0
THRESH_MASK = 0.5
BATCH_SIZE = 1
CROP_W = 1280

LABEL2NAME = {"1": "Stitch", "2": "Vessel"}
LABEL2COLOR = {"Stitch": "green", "Vessel": "yellow", "Clamp": "Orange"}
MODEL_PARA = {"num_classes": 3, "trainable_backbone_layers": 5, "backbone": "resnet50_coco"}

# HELPERS
def setup_output_folders(base: Path):
    """Create output folders and return paths as a dict."""
    save_folder = base.parent / f"{base.name}_infer"
    folders = {
        "root": save_folder,
        "np": save_folder / "np",
        "vessel_mask": save_folder / "vessel_masks",
        "suture_mask": save_folder / "suture_masks",
        "cropped": save_folder / "cropped",
        "cropped_ori": save_folder / "cropped_ori",
        "cropped_vessel": save_folder / "cropped_vessel",
        "cropped_suture": save_folder / "cropped_suture",
    }
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)
    return folders


def load_checkpoint(ckpt_dir: Path) -> str:
    ckpts = sorted(os.listdir(ckpt_dir), key=extract_loss)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return str(ckpt_dir / ckpts[0])


def load_model(ckpt_path: str, device: torch.device):
    model = CustomMaskRCNN_Lightning_Model.load_from_checkpoint(ckpt_path, **MODEL_PARA)
    model.eval()
    return model.to(device)


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to uint8 image [0,255]."""
    return (255.0 * (tensor - tensor.min()) / (tensor.max() - tensor.min())).to(torch.uint8)


def save_image(arr: np.ndarray, path: Path):
    Image.fromarray(arr).save(path)


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup paths
    folders = setup_output_folders(FOLDER_PATH)
    ckpt_path = load_checkpoint(CKPT_DIR)
    ckpt_name = Path(ckpt_path).name
    para = get_para("/home/luyin/Desktop/ly_torch/ly/constants/para_MaskRCNN.yaml")

    # model
    model = load_model(ckpt_path, device)

    # dataset
    files = sorted([f for f in os.listdir(FOLDER_PATH) if f.endswith("jpg")])
    annotations_path = FOLDER_PATH / "_annotations.coco.json"
    coco = COCO(annotations_path)
    filename_to_id = {img["file_name"]: img["id"] for img in coco.loadImgs(coco.getImgIds())}

    img_list, img_name_list = [], []
    metrics = {}

    for i, img_name in enumerate(files):
        img_path = FOLDER_PATH / img_name
        image = Image.open(img_path).convert("RGB")

        # ground truth
        image_id = filename_to_id.get(img_name, None)
        if image_id is None:
            print(f"Image ID not found for {img_name}")
            continue
        gt_anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        image_gt = paint_ground_truth_mask(image, gt_anns)

        # transform
        img = tv_tensors.Image(image)
        transform = transform_val_MaskRCNN()
        img = transform(img).to(device)

        img_list.append(img)
        img_name_list.append(img_name)

        # batch processing
        if len(img_list) == BATCH_SIZE or i == len(files) - 1:
            img_batch = torch.stack(img_list, 0)
            with torch.no_grad():
                output_batch = model(img_batch)
                feature_map_batch = model.get_feature_map(img_batch)

            for j, out in enumerate(output_batch):
                img = normalize_image(img_list[j])
                img_name = Path(img_name_list[j]).stem
                img_ori = img.clone()
                detections = out

                # prepare accumulators
                vessel_bbx, suture_bbx, vessel_mask = None, None, None
                c, x, y = img_ori.shape
                suture_mask = torch.zeros((x, y), dtype=img_ori.dtype, device=img_ori.device)
                pred_boxes, pred_labels, gt_boxes, gt_labels = [], [], [], []
                img_metrics = {"distance": [], "size": [], "precision_recall": {}}

                # ground truth boxes
                for gt in gt_anns:
                    box = [gt["bbox"][0], gt["bbox"][1], gt["bbox"][0] + gt["bbox"][2], gt["bbox"][1] + gt["bbox"][3]]
                    gt_boxes.append(box)
                    gt_labels.append(LABEL2NAME[str(gt["category_id"])])

                # predictions
                for label, score, box, mask in zip(detections["labels"], detections["scores"], detections["boxes"], detections["masks"]):
                    if score > THRESH_CLS:
                        label_name = LABEL2NAME[str(label.cpu().numpy())]
                        pred_labels.append(label_name)
                        pred_boxes.append(box.cpu().numpy())
                        mask = (mask >= THRESH_MASK).bool()

                        if label_name == "Vessel":
                            vessel_bbx, vessel_mask = box, mask.clone()
                        elif label_name == "Stitch":
                            suture_bbx = box
                            suture_mask += mask[0]

                        # draw masks and boxes
                        color = LABEL2COLOR[label_name]
                        img = draw_segmentation_masks(img, mask, alpha=0.5, colors=color)
                        img = draw_bounding_boxes(img, box.reshape((1, 4)), colors="black")

                        # mask IoU
                        for gt in gt_anns:
                            iou = compute_iou(mask[0].cpu().numpy().astype(np.uint8), mask_util.decode(coco.annToRLE(gt)))
                            if iou > 0.5:
                                img = draw_bounding_boxes(img, box.reshape((1, 4)), [f"Mask IoU: {iou:.2f}"], colors="white", width=1)

                        # bbox IoU
                        for gt_box, gt_label in zip(gt_boxes, gt_labels):
                            if gt_label == label_name:
                                iou_bbox = compute_bbox_iou(box.cpu().numpy(), gt_box)
                                if iou_bbox > 0.5:
                                    img_metrics["distance"].append(
                                        compute_distance_and_size_metrics(box.cpu().numpy(), gt_box)[0]
                                    )

                # precision/recall
                for cls in set(pred_labels + gt_labels):
                    cls_pred = [b for b, l in zip(pred_boxes, pred_labels) if l == cls]
                    cls_gt = [b for b, l in zip(gt_boxes, gt_labels) if l == cls]
                    p, r = calculate_precision_recall(cls_pred, cls_gt)
                    img_metrics["precision_recall"][cls] = {"precision": p, "recall": r}

                metrics[img_name] = img_metrics

                # save images
                bgr_image = img.permute((1, 2, 0)).cpu().numpy()
                save_image(bgr_image, folders["root"] / f"{img_name}.png")
                image_gt.save(folders["root"] / f"{img_name}_gt.png")

                # crop & save masks
                if vessel_bbx is not None:
                    x1, y1, x2, y2 = vessel_bbx.int().cpu().numpy()
                    # vessel
                    vessel_mask_np = (vessel_mask.squeeze().cpu().numpy() > THRESH_MASK).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(vessel_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        cv2.drawContours(vessel_mask_np, [cv2.boxPoints(cv2.minAreaRect(largest)).astype(int)], 0, (0, 255, 0), 6)
                    save_image(cv2.cvtColor(vessel_mask_np, cv2.COLOR_GRAY2BGR), folders["vessel_mask"] / f"{img_name}_vessel.png")

                if suture_bbx is not None and vessel_bbx is not None:
                    cropped_img = bgr_image[y1:y2, x1:x2, :]
                    save_image(resize_image_to_target_longest_side(Image.fromarray(cropped_img), CROP_W),
                               folders["cropped"] / f"crop_{img_name}.png")

            img_list, img_name_list = [], []

    # serialize metrics
    metrics_serializable = convert_to_serializable(metrics)
    check_serialization_issue(metrics)
    with open(f"{FOLD_NUMBER}_metrics.json", "w") as f:
        json.dump(metrics_serializable, f, indent=4)


if __name__ == "__main__":
    run_inference()
