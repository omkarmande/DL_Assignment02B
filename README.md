# Fine-tuning ResNet50 on iNaturalist Dataset

This project evaluates multiple fine-tuning strategies on the iNaturalist-12K dataset using ResNet50 and logs metrics with [Weights & Biases (wandb)](https://wandb.ai).

---

## Dataset

This script assumes the dataset is already downloaded and extracted into the following structure:

```
inaturalist_12K/
├── train/   # Training images (used with stratified 80-20 split for train/val)
└── val/     # Validation/Test images (used as test set only)
```

If you haven't downloaded it yet, you can run:

```bash
wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip -O nature_12K.zip
unzip -q nature_12K.zip
rm nature_12K.zip
```

---

## Fine-tuning Strategies

The script supports four fine-tuning strategies on ResNet50:

1. `freeze_all_except_last`: Freeze all layers except the final classification layer.
2. `freeze_first_k`: Freeze the first `k` layers.
3. `freeze_last_k`: Freeze the last `k` layers (except the final FC layer).
4. `train_from_scratch`: Train the model with randomly initialized weights.

---

## Command Line Arguments

```bash
python train_resnet.py \
  --entity <your_wandb_entity> \
  --project <your_project_name> \
  --strategy freeze_first_k \
  --k 4 \
  --epochs 10
```

### Arguments:

| Argument      | Type   | Description |
|---------------|--------|-------------|
| `--entity`    | str    | Your Weights & Biases entity name |
| `--project`   | str    | Your wandb project name |
| `--strategy`  | str    | Fine-tuning strategy (`freeze_all_except_last`, `freeze_first_k`, `freeze_last_k`, `train_from_scratch`) |
| `--k`         | int    | Number of layers to freeze/unfreeze (used in `freeze_first_k` and `freeze_last_k`) |
| `--epochs`    | int    | Number of training epochs |

---

## Logging and Evaluation

The following metrics are logged to wandb:
- Training loss per epoch
- Validation accuracy and loss
- Final test accuracy and loss

Evaluation results are printed in the terminal and also logged under tags: `val_acc`, `val_loss`, `test_acc`, `test_loss`.

---

## Model and Training Details

- **Model**: ResNet50 from torchvision (optionally pretrained on ImageNet)
- **Optimizer**: NAdam with `lr=1e-4`, `weight_decay=0.005`
- **Loss Function**: CrossEntropyLoss
- **Input Size**: 224 × 224
- **Batch Size**: 64
- **Transforms**: Resize + Normalize

---

## Author

Created for **DA6401 Assignment 02B** — Comparing fine-tuning strategies on ResNet50 using the iNaturalist dataset.

---
