# DINOv2 → YOLO Knowledge Distillation

Pretrain detection backbones using knowledge distillation from DINOv2 vision foundation models.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Distillation Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Image ──┬──► DINOv2 Teacher ──► Spatial Features (B, H*W, D)  │
│           │         (frozen)              │                      │
│           │                               ▼                      │
│           │                          MSE Loss                    │
│           │                               ▲                      │
│           │                               │                      │
│           └──► YOLO Student ──► Proj Head ──► Aligned Features  │
│                 (trainable)      (trainable)    (B, H*W, D)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pretraining.git
cd pretraining

# Install with pip
pip install -e .

# With YOLO support
pip install -e ".[yolo]"

# With all dependencies
pip install -e ".[all]"
```

## Quick Start

### Training with Hydra

```bash
# Default training (YOLOv8s + DINOv2 ViT-B/14)
python src/train.py data.data_dir=/path/to/images

# Override student model
python src/train.py student=yolov8n data.data_dir=/path/to/images

# Override teacher model
python src/train.py teacher=dinov2_vitl14 data.data_dir=/path/to/images

# Use experiment config
python src/train.py experiment=yolov8s_dinov2 data.data_dir=/path/to/images

# Multi-GPU training
python src/train.py trainer=ddp trainer.devices=4 data.data_dir=/path/to/images

# CPU training (for testing)
python src/train.py trainer=cpu data.data_dir=/path/to/images
```

### Export Pretrained Model

```bash
python src/export.py checkpoint=outputs/distillation/checkpoints/last.ckpt
```

### Fine-tune with YOLO

```python
from ultralytics import YOLO

# Load pretrained backbone
model = YOLO("outputs/distillation/exported_model.pt")

# Fine-tune on detection
model.train(data="coco8.yaml", epochs=50)
```

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── train.py                 # Main training script (Hydra entry)
│   ├── export.py                # Model export script
│   │
│   ├── configs/                 # Hydra configuration
│   │   ├── train.yaml           # Main config
│   │   ├── data/                # Data configs
│   │   ├── model/               # Model configs
│   │   ├── teacher/             # Teacher configs
│   │   ├── student/             # Student configs
│   │   ├── trainer/             # Trainer configs
│   │   ├── callbacks/           # Callback configs
│   │   ├── logger/              # Logger configs
│   │   └── experiment/          # Full experiment configs
│   │
│   ├── models/                  # Model implementations
│   │   ├── teachers/            # Teacher wrappers (DINOv2)
│   │   ├── students/            # Student wrappers (YOLO, etc.)
│   │   └── components/          # Shared components
│   │
│   ├── methods/                 # Training methods
│   │   ├── distillation_module.py  # Main LightningModule
│   │   └── losses.py            # Loss functions
│   │
│   ├── data/                    # Data loading
│   │   ├── image_datamodule.py  # LightningDataModule
│   │   └── transforms.py        # Augmentations
│   │
│   └── utils/                   # Utilities
│       ├── instantiators.py     # Hydra helpers
│       ├── logging_utils.py     # Logging
│       └── utils.py             # General utilities
│
├── pyproject.toml               # Package configuration
└── README.md
```

## Supported Models

### Teachers (DINOv2)
| Config | Model | Embed Dim |
|--------|-------|-----------|
| `dinov2_vits14` | ViT-S/14 | 384 |
| `dinov2_vitb14` | ViT-B/14 | 768 |
| `dinov2_vitl14` | ViT-L/14 | 1024 |
| `dinov2_vitg14` | ViT-G/14 | 1536 |

### Students (YOLO)
| Config | Model |
|--------|-------|
| `yolov8n` | YOLOv8 Nano |
| `yolov8s` | YOLOv8 Small |
| `yolov11s` | YOLO11 Small |

## Configuration

### Key Training Parameters

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| Data directory | `data.data_dir` | - | Path to images |
| Image size | `data.image_size` | 224 | Input resolution |
| Batch size | `data.batch_size` | 128 | Batch size per device |
| Epochs | `trainer.max_epochs` | 100 | Training epochs |
| Learning rate | `model.optimizer.lr` | 1e-3 | Base learning rate |
| Teacher blocks | `teacher.n_blocks` | 2 | Intermediate blocks to distill |

### Override Examples

```bash
# Change batch size and epochs
python src/train.py data.batch_size=256 trainer.max_epochs=200

# Use W&B logging
python src/train.py logger=wandb logger.wandb.project=my_project

# Disable early stopping
python src/train.py callbacks.early_stopping=null
```

## How It Works

1. **Teacher (DINOv2)**: Frozen ViT extracts spatial features from intermediate blocks → `(B, H*W, D)`

2. **Student (YOLO)**: Backbone extracts spatial features → `(B, C, H, W)`

3. **Projection Head**: MLP maps student channels to teacher dimension → `(B, D, H, W)`

4. **Spatial Alignment**: Bilinear interpolation resizes student grid to match teacher → `(B, H*W, D)`

5. **Loss**: MSE between teacher and student spatial features

Same augmentations are applied to both teacher and student inputs for consistent targets.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Hydra 1.3+
- ultralytics (for YOLO support)

## License

MIT License
