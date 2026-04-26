# MedSAM LoRA — Prostate Segmentation with Parameter-Efficient Fine-Tuning

This repository implements **LoRA (Low-Rank Adaptation)** fine-tuning on **MedSAM** (Segment Anything Model pretrained on medical images) for prostate segmentation on ultrasound MRI, demonstrating that Transformer-based architectures with attention-layer LoRA injection dramatically outperform CNN baselines.
📝 **Read the full technical writeup:** [Building a LoRA Trainer from Scratch for Medical Image Segmentation](https://substack.com/@amitejsinghdatta/p-192350772)

---
---

## Results

| Model | Dice (+/-) | IoU | Trainable Params | Total Params |
| :--- | :---: | :---: | :---: | :---: |
| Base MedSAM (Zero-Shot) | 0.519 | 0.388 | **0** | 94.1M |
| VGG-UNet (CNN Baseline) | 0.832 | 0.784 | 25.8M | 25.8M |
| **MedSAM + LoRA r=8 (Ours)** | **0.957** | **0.919** | **4.5M** | 94.2M |

**Key finding:** LoRA fine-tuning lifted Dice from 0.519 → 0.957 (+43.8pp) while training **82% fewer parameters** than the CNN baseline.

---

## Architecture

```
MedSAM (ViT-B backbone)
├── Vision Encoder [FROZEN, 89M params]
│   └── LoRA adapters injected into qkv + proj layers [TRAINABLE, ~1.2M]
└── Mask Decoder [FULLY TRAINABLE, ~4M params]
```

LoRA is injected into each of the 12 ViT-B attention blocks:
- `qkv` — fused Q/K/V projection (HuggingFace SAM naming)
- `proj` — output projection after attention

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/AmitejSingh1/MedSAM_LoRA
cd MedSAM_LoRA
```

### 1. Download MedSAM Weights

The base weights for MedSAM are required. Download them from Hugging Face and place them in the `./medsam_weights/` directory:

- **MedSAM ViT-B**: [wanglab/medsam-vit-b](https://huggingface.co/wanglab/medsam-vit-b)

### 2. Prepare Embeddings

The training pipeline uses an **Embedding Cache** to achieve 20x speedup. You do not need to download these; simply run the precompute script once (see Usage).

```bash
# Install dependencies
pip install torch transformers opencv-python numpy pandas tqdm scikit-learn
```
```

---

## Usage

### Step 1: Precompute embeddings (run ONCE, ~10 minutes)
```bash
python precompute_embeddings.py
```
Runs all training images through the frozen ViT-B encoder once and saves embeddings to `embeddings_cache/`. Reduces training from ~60 min/epoch → ~2-5 min/epoch.

### Step 2: Train
```bash
python train_medsam.py --epochs 50 --r 8 --grad-accum 4
```
Includes Early Stopping (patience=15), ReduceLROnPlateau (patience=5), and best-model checkpointing.

### Step 3: Evaluate
```bash
python evaluate_medsam.py --checkpoint checkpoints/medsam_r8_best.pth
```

### Step 4: Zero-shot baseline (optional)
```bash
python zero_shot_medsam.py
```

---

## File Structure

| File | Description |
|---|---|
| `medsam_model.py` | Loads MedSAM, injects LoRA, unfreezes decoder |
| `dataset.py` | `MedSAMDataset` (raw images) + `CachedMedSAMDataset` (fast, from cache) |
| `train_medsam.py` | Training loop with AMP, grad accumulation, early stopping |
| `precompute_embeddings.py` | One-time embedding cache generator |
| `evaluate_medsam.py` | Evaluation on val set with Dice/IoU/Acc/Prec/Recall |
| `zero_shot_medsam.py` | Baseline: untuned MedSAM on same val set |

---

## Ablation

To test different LoRA ranks for the ablation study:
```bash
python train_medsam.py --r 4  --epochs 50
python train_medsam.py --r 8  --epochs 50  # Best
python train_medsam.py --r 16 --epochs 50
python train_medsam.py --r 32 --epochs 50
```
