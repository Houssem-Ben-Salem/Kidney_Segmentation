# Kidney Tumor Segmentation using TransUNet with Attention Gate

This project implements kidney tumor segmentation using the **TransUNet** architecture enhanced with **Attention Gates** for improved segmentation performance. It utilizes the **KiTS19** dataset for training and testing. The implementation is designed for flexibility and includes features like attention visualization, checkpointing, and integration with **Weights & Biases (W&B)** for experiment tracking.

---

## Features

- **TransUNet Architecture**: Combines transformer-based global attention with UNet's local feature extraction.
- **Attention Gates**: Integrated into the architecture to focus on relevant regions in the input image, improving segmentation performance.
- **KiTS19 Dataset Support**: Processes and trains on kidney and tumor segmentation dataset with background.
- **W&B Integration**: Logs training metrics, attention maps, and segmentation outputs for better visualization and tracking.
- **Checkpoints**: Supports saving and resuming from checkpoints (`latest_checkpoint.pth` and `best_model.pth`).
- **Validation Metrics**: Validates every few iterations and logs validation loss and Dice score.

---

## Project Structure

```
.
├── data/                     # KiTS19 dataset (not included in repo)
├── lists_kits19/             # Train/val/test file lists
├── networks/
│   ├── vit_seg_modeling.py   # TransUNet with Attention Gate
│   ├── attention_gate.py     # Attention Gate implementation
│   └── ...                   # Additional network components
├── utils/
│   └── utils.py              # Dice loss and other utilities
├── datasets/
│   └── dataset_kits19_list.py# Dataset loader for KiTS19
├── TransUNet/
│   ├── train.py              # Training script
│   ├── trainer.py            # Training loop with validation
│   └── ...                   # Other components
└── README.md                 # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU with drivers
- Virtual environment (optional but recommended)

### Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies include:
- PyTorch
- torchvision
- tqdm
- numpy
- nibabel
- scikit-learn
- wandb

---

## Usage

### 1. Dataset Setup

1. Download the KiTS19 dataset from the official website: [KiTS19 Challenge](https://kits19.grand-challenge.org/).
2. Extract it to the `data/` directory.

Ensure the structure is as follows:
```
data/
├── case_00000/
│   ├── imaging.nii.gz
│   └── segmentation.nii.gz
...
```

### 2. Generate Train/Val/Test Splits

Run the script to create splits:
```bash
python generate_kits19_lists.py
```

This generates three files:
- `train.txt`
- `val.txt`
- `test.txt`

### 3. Training

Run the training script:
```bash
python train.py \
    --root_path data \
    --list_dir lists_kits19 \
    --batch_size 8 \
    --img_size 224 \
    --max_iterations 50000 \
    --checkpoint_dir checkpoints \
    --use_attention_gate 1 # 1 if you want to use the attention gare 0 otherwise
```

### 4. Resuming Training

To resume training, ensure `latest_checkpoint.pth` or `best_model.pth` exists in the `checkpoint_dir`. The script will automatically load the weights and optimizer state.

---

## Visualizations

The project uses **Weights & Biases** for tracking and visualizing:
- Training/validation loss
- Dice scores
- Attention maps
- Segmentation outputs (input image, prediction, ground truth)

### Logging

Login to W&B before running the script:
```bash
wandb login
```

Access visualizations from the W&B dashboard.

---

## Highlights of the Attention Gate

The **Attention Gate** focuses the model's attention on the most relevant regions in the input images. This helps:
- Filter irrelevant background noise.
- Improve segmentation accuracy for small regions like tumors.

Attention maps are logged to W&B for interpretability.

---

## Example Logs

During training, you’ll see outputs like:

```
Iteration 1000: Loss 0.5632
Validation: Iteration 2000, Val Loss: 0.4301, Val Dice: 0.8763
Saved latest checkpoint at checkpoints/latest_checkpoint.pth
New best model saved at checkpoints/best_model.pth with Dice 0.8763
```

---

## Results

### Quantitative Results
Dice scores are tracked during validation and can be compared for different configurations (e.g., with/without attention gates).

### Qualitative Results
Segmentation outputs and attention maps are logged to W&B for qualitative comparison.

---

## Contributions

- **Attention Gate**: Enhances the vanilla TransUNet architecture.
- **Training Features**: Robust logging, validation, and checkpointing.
- **Visualization**: Integrated attention and segmentation outputs with W&B.

---

## Future Work

- Extend to multi-class segmentation tasks.
- Evaluate on other medical imaging datasets.
- Explore lightweight models for real-time inference.

---

## Acknowledgments

- **KiTS19 Dataset**: Kidney Tumor Segmentation Challenge.
- **TransUNet**: Transformer-based UNet architecture.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
```

### Highlights:
1. **Clear Sections**:
   - Features, installation, and usage are neatly categorized.
2. **Usage Instructions**:
   - Covers dataset setup, split generation, and training.
3. **Visualization**:
   - Explains the integration with W&B for easy experiment tracking.
4. **Effectiveness of Attention Gate**:
   - Briefly explains the improvement it adds to the architecture.
