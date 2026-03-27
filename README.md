# MedSAM LoRA Ablation Study

This project investigates the application of Parameter-Efficient Fine-Tuning (specifically Low-Rank Adaptation, LoRA) directly onto the Vision Transformer (ViT-B) backbone of **MedSAM** (`wanglab/medsam-vit-b`) for Prostate Segmentation.

## Objective
To prove mathematically and empirically that LoRA on Attention Projections (`q_proj`, `v_proj`) of a Transformer-based Medical Segmentation model is significantly more effective and parameter-efficient than on legacy CNN architectures (like VGG-UNet). 

## Methodology
1. **Base Architecture:** MedSAM Vision Encoder (Frozen) + Mask Decoder (Trainable ~$4M$ params).
2. **LoRA Injection:** Custom framework injecting low-rank matrices specifically into the multi-head attention layers of the 89M parameter Vision Encoder.
3. **Training Routine:** 8GB VRAM constraint managed via `torch.amp` mixed precision and gradient accumulation.
4. **Metrics Evaluated:** Dice Score (F1), IoU (Jaccard), Training Time, and Trainable Parameter counts.

## Ablation configurations
- LoRA Ranks ($r$): 4, 8, 16, 32
- Targets: `[q, v]` vs `[q, k, v, out]`
