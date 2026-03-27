import torch
import torch.nn as nn
from transformers import SamModel
import sys
import os

# Connect to our custom LoRA framework
sys.path.append(os.path.abspath("C:/personal_proj/lora_trainer"))
from injector import inject_lora

def build_medsam_lora(r=8, target_layers=["qkv", "proj"]):
    """
    Loads MedSAM (ViT-B backbone), injects LoRA into the Vision Encoder's fused
    attention layers (qkv = Q,K,V fused; proj = output projection), and unfreezes the mask decoder.
    """
    print("Loading Base MedSAM Model (Local Offline Directory)...")
    # Load from your newly created local folder
    model = SamModel.from_pretrained("./medsam_weights")
    
    # 1. Freeze entirely first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Inject LoRA into the Vision Encoder
    # The HF SamModel uses 'q_proj', 'k_proj', 'v_proj', 'out_proj' inside the ViT blocks.
    print(f"Injecting LoRA (r={r}) into Attention layers: {target_layers}...")
    
    # inject_lora automatically prints the injection count
    model.vision_encoder = inject_lora(
        model.vision_encoder, 
        target_layer_names=target_layers, 
        r=r
    )
    
    # 3. Architecturally Correct: Unfreeze the entire Mask Decoder
    # The mask decoder is specifically designed to handle segmentation heads
    # and is relatively small (~4M parameters).
    print("Unfreezing the Mask Decoder (~4M parameters)...")
    for param in model.mask_decoder.parameters():
        param.requires_grad = True
        
    return model

if __name__ == "__main__":
    # Use correct HuggingFace SAM layer names (fused qkv, not separate q_proj/v_proj)
    model = build_medsam_lora(r=8, target_layers=["qkv", "proj"])
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print("\n--- MEDSAM-LORA PARAMETERS ---")
    print(f"Trainable (LoRA + Decoder): {trainable_params:,}")
    print(f"Frozen (ViT-B base):        {frozen_params:,}")
    print(f"% Trainable:                {100 * trainable_params / (trainable_params + frozen_params):.4f}%")
