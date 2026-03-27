"""
zero_shot_medsam.py — Run base MedSAM with NO fine-tuning on the val set.
This proves whether the LoRA actually improved performance or if it's just base MedSAM.
"""
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

import torch
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading BASE MedSAM (no LoRA, no fine-tuning)...")
model = SamModel.from_pretrained("./medsam_weights")
model.eval().to(device)
for param in model.parameters():
    param.requires_grad = False

processor = SamProcessor.from_pretrained("./medsam_weights")

val_images = sorted(glob("C:/personal_proj/prostate/data/val_png/images/*.png"))
val_masks  = sorted(glob("C:/personal_proj/prostate/data/val_png/masks/*.png"))
print(f"Evaluating on {len(val_images)} val images...")

SCORES = []
with torch.no_grad():
    for img_path, mask_path in tqdm(zip(val_images, val_masks), total=len(val_images)):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask  = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        y_idx, x_idx = np.where(mask > 0)
        if len(x_idx) > 0:
            H, W = mask.shape
            prompt_box = [int(np.min(x_idx)), int(np.min(y_idx)), int(np.max(x_idx)), int(np.max(y_idx))]
        else:
            H, W = mask.shape
            prompt_box = [0, 0, W, H]

        inputs = processor(image, input_boxes=[[prompt_box]], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs, multimask_output=False)
        pred = outputs.pred_masks.squeeze()
        pred_up = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()
        pred_bin = (torch.sigmoid(pred_up) > 0.5).cpu().numpy().astype(np.int32)

        y_flat = mask.flatten()
        p_flat = pred_bin.flatten()
        SCORES.append([
            accuracy_score(y_flat, p_flat),
            f1_score(y_flat, p_flat, average="binary", zero_division=1),
            jaccard_score(y_flat, p_flat, average="binary", zero_division=1),
            recall_score(y_flat, p_flat, average="binary", zero_division=1),
            precision_score(y_flat, p_flat, average="binary", zero_division=1),
        ])

means = np.mean(SCORES, axis=0)
print(f"\n{'='*50}")
print(f"  BASE MedSAM (Zero-Shot, No Fine-Tuning):")
print(f"  Accuracy:      {means[0]:.5f}")
print(f"  F1 (Dice):     {means[1]:.5f}")
print(f"  Jaccard (IoU): {means[2]:.5f}")
print(f"  Recall:        {means[3]:.5f}")
print(f"  Precision:     {means[4]:.5f}")
print(f"{'='*50}")
