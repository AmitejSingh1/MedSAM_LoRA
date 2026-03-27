import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from transformers import SamProcessor

class MedSAMDataset(Dataset):
    """
    Adapts the raw Prostate Segmentation PNG files to be mathematically compatible
    with Hugging Face's SamModel (MedSAM).
    
    MedSAM requires inputs to be aggressively preprocessed (e.g., exactly 1024x1024 pixels, 
    specific normalization) and it fundamentally requires a prompt (like a bounding box) 
    to know where to segment.
    """
    def __init__(self, image_dir: str, mask_dir: str, processor: SamProcessor):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Image (MedSAM expects RGB)
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Mask (Grayscale, binary)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        
        # 3. Create a Bounding Box Prompt directly from the Ground Truth mask
        # SAM models require a prompt to anchor their attention. During fine-tuning,
        # it is standard practice to generate a loose bounding box around the ground truth mask.
        
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Add random perturbation (noise) to the bounding box during training
            # This forces the model to not overfit to perfect bounding boxes!
            H, W = mask.shape
            noise = 15 # pixels of noise
            x_min = max(0, x_min - np.random.randint(0, noise))
            x_max = min(W, x_max + np.random.randint(0, noise))
            y_min = max(0, y_min - np.random.randint(0, noise))
            y_max = min(H, y_max + np.random.randint(0, noise))
            
            prompt_box = [x_min, y_min, x_max, y_max]
        else:
            # Fallback if the medical scan mask is empty
            H, W = mask.shape
            prompt_box = [0, 0, W, H]
            
        # 4. Let the official SamProcessor handle all the intense resizing and tensor creation
        inputs = self.processor(
            image, 
            input_boxes=[[prompt_box]], 
            segmentation_maps=mask, 
            return_tensors="pt"
        )
        
        # 5. Remove the arbitrary batch dimension the processor adds (DataLoader will add it back)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs

if __name__ == "__main__":
    # --- Quick Check ---
    print("Loading Processor...")
    # NOTE: You MUST have 'transformers' and 'Pillow' installed
    processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
    
    # We will test against the Prostate Data from your previous project!
    train_image_dir = "C:/personal_proj/prostate/data/train_png/images"
    train_mask_dir = "C:/personal_proj/prostate/data/train_png/masks"
    
    print("Initializing MedSAM Dataset...")
    dataset = MedSAMDataset(train_image_dir, train_mask_dir, processor)
    
    # Extract the very first scan
    sample = dataset[0]
    
    print("\n--- SAMPLE TENSOR SHAPES ---")
    print(f"Pixel Values (Image):     {sample['pixel_values'].shape}")
    print(f"Input Boxes (Prompt):     {sample['input_boxes'].shape}")
    print(f"Labels (Ground Truth):    {sample['labels'].shape}")
