"""
precompute_embeddings.py — Run this ONCE before training.

The MedSAM vision encoder is frozen during LoRA fine-tuning, which means it
produces identical embeddings for the same image every single epoch. Running
1722 images × 50 epochs = 86,100 redundant ViT-B forward passes. 

This script runs each image through the frozen vision encoder exactly once
and saves the resulting embedding to disk. A full training run then skips the
encoder entirely, reducing per-epoch time from ~60 min → ~2-5 min.
"""
import os
import sys
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from transformers import SamModel, SamProcessor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load base model (no LoRA, just the frozen encoder) ---
    print("Loading MedSAM model for feature extraction...")
    model = SamModel.from_pretrained("./medsam_weights")
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    processor = SamProcessor.from_pretrained("./medsam_weights")

    # --- Locate training images ---
    import cv2
    train_image_dir = "C:/personal_proj/prostate/data/train_png/images"
    train_mask_dir  = "C:/personal_proj/prostate/data/train_png/masks"

    image_paths = sorted(glob(os.path.join(train_image_dir, "*.png")))
    mask_paths  = sorted(glob(os.path.join(train_mask_dir,  "*.png")))
    print(f"Found {len(image_paths)} images to precompute.")

    out_dir = "./embeddings_cache"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (img_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Precomputing")):
            # --- Load image & mask ---
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask  = (mask > 127).astype(np.uint8)

            # --- Generate bounding box prompt from ground truth mask ---
            y_idx, x_idx = np.where(mask > 0)
            if len(x_idx) > 0:
                H, W = mask.shape
                noise = 15
                x_min = max(0, int(np.min(x_idx)) - np.random.randint(0, noise))
                x_max = min(W, int(np.max(x_idx)) + np.random.randint(0, noise))
                y_min = max(0, int(np.min(y_idx)) - np.random.randint(0, noise))
                y_max = min(H, int(np.max(y_idx)) + np.random.randint(0, noise))
                prompt_box = [x_min, y_min, x_max, y_max]
            else:
                H, W = mask.shape
                prompt_box = [0, 0, W, H]

            # --- Run through SamProcessor (resizes to 1024×1024 and normalizes) ---
            inputs = processor(image, input_boxes=[[prompt_box]], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            input_boxes  = inputs["input_boxes"].to(device)

            # --- Run ONLY the vision encoder (the expensive frozen part) ---
            image_embeddings = model.get_image_embeddings(pixel_values)

            # --- Save embedding + prompt box + label ---
            torch.save({
                "embedding":   image_embeddings.squeeze(0).cpu(),   # (256, 64, 64)
                "input_boxes": input_boxes.squeeze(0).cpu(),         # (1, 4)
                "label":       torch.tensor(mask, dtype=torch.float32),
            }, os.path.join(out_dir, f"{idx:05d}.pt"))

    print(f"\nDone! {len(image_paths)} embeddings saved to {out_dir}/")
    print("You can now run: python train_medsam.py --epochs 50 --r 8 --grad-accum 4")

if __name__ == "__main__":
    main()
