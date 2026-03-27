import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SamProcessor

from dataset import MedSAMDataset, CachedMedSAMDataset
from medsam_model import build_medsam_lora

# Import the dice loss from your previous project
sys.path.append(os.path.abspath("C:/personal_proj/prostate"))
from metrics import dice_loss

# Top-level collate functions for Windows multiprocessing compatibility
# Handles both MedSAMDataset (pixel_values) and CachedMedSAMDataset (embedding)
def collate_fn(batch):
    inputs_list, labels_list = zip(*batch)
    labels = torch.stack(labels_list)
    # MedSAMDataset returns pixel_values; CachedMedSAMDataset returns embedding
    if "pixel_values" in inputs_list[0]:
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in inputs_list]),
            "input_boxes":  torch.stack([x["input_boxes"]  for x in inputs_list]),
        }, labels
    else:
        return {
            "embedding":   torch.stack([x["embedding"]   for x in inputs_list]),
            "input_boxes": torch.stack([x["input_boxes"] for x in inputs_list]),
        }, labels

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Training for MedSAM LoRA")
    parser.add_argument("--r", type=int, default=8, choices=[4, 8, 16, 32], help="LoRA Rank parameter")
    # HuggingFace SAM uses:
    #   'qkv'  -> fused Q, K, V attention projection (single Linear layer)
    #   'proj' -> output projection after attention
    #   'lin1' / 'lin2' -> MLP (feed-forward) blocks in each ViT layer
    parser.add_argument("--targets", nargs="+", default=["qkv", "proj"], help="Layers to inject LoRA into")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Actual forward-pass batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps, effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # Cap dataset size for fast ablation runs (e.g. 500 instead of 1722 images)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training images for faster runs")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- MEDSAM ABLATION RUN ---")
    print(f"Rank: {args.r}, Targets: {args.targets}")
    print(f"Batch Size: {args.batch_size}, Grad Accumulation: {args.grad_accum}")
    
    print("Loading Processor from Local Directory...")
    processor = SamProcessor.from_pretrained("./medsam_weights")
    
    train_image_dir = "C:/personal_proj/prostate/data/train_png/images"
    train_mask_dir = "C:/personal_proj/prostate/data/train_png/masks"
    
    cache_dir = "./embeddings_cache"
    if os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) > 0:
        print(f"Found embedding cache at {cache_dir} — using CachedMedSAMDataset (fast mode)")
        dataset = CachedMedSAMDataset(cache_dir)
    else:
        print("No embedding cache found. Run precompute_embeddings.py first for faster training!")
        print("Falling back to slow MedSAMDataset with raw images...")
        processor = SamProcessor.from_pretrained("./medsam_weights")
        dataset = MedSAMDataset(train_image_dir, train_mask_dir, processor)
    
    # Subsample the dataset if --max-samples is set (for fast ablation experiments)
    if args.max_samples and args.max_samples < len(dataset):
        from torch.utils.data import Subset
        indices = list(range(args.max_samples))
        dataset = Subset(dataset, indices)
        print(f"Using {args.max_samples}/{len(dataset)} samples (--max-samples flag)")
    
    # We must use a custom collate_fn because SamProcessor outputs complex tensor dicts
    # (Note: collate_fn is defined at module level for Windows multiprocessing compatibility)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,       # 0 = main process only (avoids Windows pickle errors)
        pin_memory=True,     # Faster GPU transfer
        persistent_workers=False,
    )
    
    print("Building Model...")
    model = build_medsam_lora(r=args.r, target_layers=args.targets)
    model.to(device)
    
    # Verify LoRA injection succeeded
    injected = [(name, mod) for name, mod in model.named_modules()
                if "LoRA" in type(mod).__name__]
    print(f"LoRA layers injected: {len(injected)}")
    for name, _ in injected[:5]:
        print(f"  {name}")
    if len(injected) == 0:
        raise RuntimeError("LoRA injection failed — 0 layers found! Check target layer names.")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    # Fix: Use non-deprecated GradScaler API
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # --- CALLBACKS ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    
    # Early Stopping state
    best_loss = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_name = f"checkpoints/medsam_r{args.r}_best.pth"

    print("\nStarting Training Loop...")
    model.train()
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (inputs, labels) in enumerate(progress):
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.unsqueeze(1).to(device)
            
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # FAST PATH: cached embeddings — skip the frozen ViT-B encoder entirely
                    if "embedding" in inputs:
                        # FAST PATH: pass pre-computed embedding directly to SamModel.
                        # When image_embeddings is provided, HuggingFace SamModel skips
                        # the vision encoder and handles positional encoding internally.
                        outputs = model(
                            image_embeddings=inputs["embedding"],
                            input_boxes=inputs["input_boxes"],
                            multimask_output=False,
                        )
                    else:
                        # SLOW PATH: raw images — full forward pass through vision encoder
                        outputs = model(
                            pixel_values=inputs["pixel_values"],
                            input_boxes=inputs["input_boxes"],
                            multimask_output=False,
                        )
                    pred_masks = outputs.pred_masks.squeeze(2)  # (B, 1, 256, 256)

                    labels_resized = torch.nn.functional.interpolate(labels.float(), size=(256, 256))
                    loss = dice_loss(labels_resized, torch.sigmoid(pred_masks))
                    loss = loss / args.grad_accum
                    
                scaler.scale(loss).backward()
                
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            
            epoch_loss += loss.item() * args.grad_accum
            progress.set_postfix({"loss": loss.item() * args.grad_accum})
            
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # --- ReduceLROnPlateau ---
        scheduler.step(avg_loss)
        
        # --- ModelCheckpoint (save ONLY if loss improved) ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(trainable_state_dict, checkpoint_name)
            print(f"  ✓ New best loss: {best_loss:.4f} — checkpoint saved!")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
        
        # --- EarlyStopping ---
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
            break

    print(f"\nTraining complete. Best checkpoint saved to: {checkpoint_name}")

if __name__ == "__main__":
    main()
