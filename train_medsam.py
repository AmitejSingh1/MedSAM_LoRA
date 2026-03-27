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

from dataset import MedSAMDataset
from medsam_model import build_medsam_lora

# Import the dice loss from your previous project
sys.path.append(os.path.abspath("C:/personal_proj/prostate"))
from metrics import dice_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Training for MedSAM LoRA")
    parser.add_argument("--r", type=int, default=8, choices=[4, 8, 16, 32], help="LoRA Rank parameter")
    parser.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"], help="Layers to inject LoRA into")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    # Batch size must be kept tiny (1 or 2) because MedSAM image encoder is massive
    parser.add_argument("--batch-size", type=int, default=1, help="Actual forward-pass batch size")
    # We use gradient accumulation to simulate a larger batch size (e.g. batch=1, grad_accum=8 simulates batch=8)
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- MEDSAM ABLATION RUN ---")
    print(f"Rank: {args.r}, Targets: {args.targets}")
    print(f"Batch Size: {args.batch_size}, Grad Accumulation: {args.grad_accum}")
    
    print("Loading Processor...")
    processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
    
    train_image_dir = "C:/personal_proj/prostate/data/train_png/images"
    train_mask_dir = "C:/personal_proj/prostate/data/train_png/masks"
    
    print("Initializing Dataset...")
    dataset = MedSAMDataset(train_image_dir, train_mask_dir, processor)
    
    # We must use a custom collate_fn because SamProcessor outputs complex tensor dicts
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "input_boxes": torch.stack([x["input_boxes"] for x in batch]),
        }, torch.stack([x["labels"] for x in batch])
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    print("Building Model...")
    model = build_medsam_lora(r=args.r, target_layers=args.targets)
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    print("\nStarting Training Loop...")
    model.train()
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (inputs, labels) in enumerate(progress):
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Add channel dimension so labels match prediction shape
            labels = labels.unsqueeze(1).to(device) 
            
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # multimask_output=False forces the model to output a single definitive mask
                    outputs = model(**inputs, multimask_output=False)
                    
                    # The predicted mask logits are in outputs.pred_masks
                    # Shape: (B, 1, 1, 256, 256) -> Squeeze down to (B, 1, 256, 256)
                    pred_masks = outputs.pred_masks.squeeze(1) 
                    
                    # We calculate the loss against the low-res 256x256 logits
                    # Resize ground truth to match prediction logits
                    labels_resized = torch.nn.functional.interpolate(labels.float(), size=(256, 256))
                    
                    # Dice loss expects probabilities (sigmoid), not raw logits
                    loss = dice_loss(labels_resized, torch.sigmoid(pred_masks))
                    
                    # Gradient accumulation normalization
                    loss = loss / args.grad_accum
                    
                scaler.scale(loss).backward()
                
                # Execute optimizer step only after accumulating N batches
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.grad_accum
            progress.set_postfix({"loss": loss.item() * args.grad_accum})
            
        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save a unique checkpoint for each ablation test!
        checkpoint_name = f"checkpoints/medsam_r{args.r}_{''.join(args.targets)}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        # Extract only the LoRA weights + trainable Decoder weights
        trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
        torch.save(trainable_state_dict, checkpoint_name)

if __name__ == "__main__":
    main()
