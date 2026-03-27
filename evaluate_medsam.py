"""
evaluate_medsam.py — Evaluate the trained MedSAM LoRA adapter on the validation set.
Computes Dice, IoU (Jaccard), Accuracy, Precision, and Recall.
"""
import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, jaccard_score, precision_score, recall_score,
)

import torch
from transformers import SamProcessor

sys.path.append(os.path.abspath("C:/personal_proj/lora_trainer"))
from injector import inject_lora

sys.path.append(os.path.abspath("C:/personal_proj/prostate"))
from medsam_model import build_medsam_lora

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedSAM LoRA on prostate val set")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/medsam_r8_best.pth",
                        help="Path to trained LoRA checkpoint")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank used during training")
    parser.add_argument("--targets", nargs="+", default=["qkv", "proj"])
    parser.add_argument("--val-images", type=str,
                        default="C:/personal_proj/prostate/data/val_png/images")
    parser.add_argument("--val-masks", type=str,
                        default="C:/personal_proj/prostate/data/val_png/masks")
    parser.add_argument("--results-dir", type=str, default="./results/medsam_lora")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load model ---
    print("Building MedSAM + LoRA model...")
    model = build_medsam_lora(r=args.r, target_layers=args.targets)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # The checkpoint only contains trainable (LoRA + decoder) params
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"  Missing keys (expected frozen base layers): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    # --- Load processor ---
    print("Loading Processor...")
    processor = SamProcessor.from_pretrained("./medsam_weights")

    # --- Load val images ---
    image_paths = sorted(glob(os.path.join(args.val_images, "*.png")))
    mask_paths  = sorted(glob(os.path.join(args.val_masks, "*.png")))
    print(f"Found {len(image_paths)} validation images")

    if len(image_paths) == 0:
        print(f"ERROR: No images found in {args.val_images}")
        exit(1)

    os.makedirs(args.results_dir, exist_ok=True)

    SCORE = []
    with torch.no_grad():
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
            name = os.path.splitext(os.path.basename(img_path))[0]

            # Load and preprocess
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_bin = (mask > 127).astype(np.uint8)

            # Generate bounding box prompt from GT mask
            y_idx, x_idx = np.where(mask_bin > 0)
            if len(x_idx) > 0:
                H, W = mask_bin.shape
                prompt_box = [int(np.min(x_idx)), int(np.min(y_idx)),
                              int(np.max(x_idx)), int(np.max(y_idx))]
            else:
                H, W = mask_bin.shape
                prompt_box = [0, 0, W, H]

            # Run through SAM processor and model
            inputs = processor(image_rgb, input_boxes=[[prompt_box]], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, multimask_output=False)
            # pred_masks: (1, 1, 1, H, W) -> squeeze to (H, W)
            pred_mask_low = outputs.pred_masks.squeeze()
            # The model outputs low-res (256x256) masks. Upsample to original size.
            pred_mask_up = torch.nn.functional.interpolate(
                pred_mask_low.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze()
            pred_bin = (torch.sigmoid(pred_mask_up) > 0.5).cpu().numpy().astype(np.int32)

            # Save side-by-side comparison
            gt_vis = np.stack([mask_bin * 255] * 3, axis=-1)
            pred_vis = np.stack([pred_bin * 255] * 3, axis=-1)
            divider = np.ones((H, 10, 3), dtype=np.uint8) * 128
            comparison = np.concatenate([image, divider, gt_vis, divider, pred_vis], axis=1)
            cv2.imwrite(os.path.join(args.results_dir, f"{name}.png"), comparison)

            # Compute metrics
            y_flat    = mask_bin.flatten()
            pred_flat = pred_bin.flatten()

            acc  = accuracy_score(y_flat, pred_flat)
            f1   = f1_score(y_flat, pred_flat, average="binary", zero_division=1)
            jac  = jaccard_score(y_flat, pred_flat, average="binary", zero_division=1)
            rec  = recall_score(y_flat, pred_flat, average="binary", zero_division=1)
            prec = precision_score(y_flat, pred_flat, average="binary", zero_division=1)
            SCORE.append([name, acc, f1, jac, rec, prec])

    # Summary
    means = np.mean([s[1:] for s in SCORE], axis=0)
    print(f"\n{'='*50}")
    print(f"  MedSAM LoRA Final Results (r={args.r}):")
    print(f"  Accuracy:         {means[0]:.5f}")
    print(f"  F1 (Dice):        {means[1]:.5f}")
    print(f"  Jaccard (IoU):    {means[2]:.5f}")
    print(f"  Recall:           {means[3]:.5f}")
    print(f"  Precision:        {means[4]:.5f}")
    print(f"{'='*50}")

    # Save CSV
    csv_path = f"./results/score_medsam_r{args.r}.csv"
    os.makedirs("./results", exist_ok=True)
    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv(csv_path, index=False)
    print(f"\nPer-image metrics saved to: {csv_path}")
    print(f"Visual comparisons saved to: {args.results_dir}/")

if __name__ == "__main__":
    main()
