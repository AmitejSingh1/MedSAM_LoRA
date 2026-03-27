# MedSAM LoRA — Prostate Segmentation with Parameter-Efficient Fine-Tuning

This repository implements **LoRA (Low-Rank Adaptation)** fine-tuning on **MedSAM** (Segment Anything Model pretrained on medical images) for prostate segmentation on ultrasound MRI, demonstrating that Transformer-based architectures with attention-layer LoRA injection dramatically outperform CNN baselines.

---

## Results

| Model | Dice (F1) | IoU | Accuracy | Recall | Precision |
|---|---|---|---|---|---|
| Base MedSAM (Zero-Shot) | 0.519 | 0.388 | 0.758 | 0.506 | 0.870 |
| VGG-UNet (Published CNN Baseline) | 0.832 | 0.784 | 0.953 | 0.921 | 0.854 |
| **MedSAM + LoRA r=8 (Ours)** | **0.957** | **0.920** | **0.970** | **0.937** | **0.979** |

**Key finding:** LoRA fine-tuning lifted Dice from 0.519 → 0.957 (+43.8pp) by training only ~2.3M additional adapter parameters on top of the frozen 89M ViT-B encoder.

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

# 2. Install dependencies
pip install torch transformers opencv-python numpy pandas tqdm scikit-learn

# 3. Download MedSAM weights manually from HuggingFace:
#    https://huggingface.co/wanglab/medsam-vit-b
#    Place the downloaded files into ./medsam_weights/
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
