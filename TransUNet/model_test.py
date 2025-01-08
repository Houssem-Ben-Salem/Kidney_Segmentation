import argparse
import torch
import numpy as np
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_kits19_list import KiTS19DatasetList
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar
import os
from scipy.spatial.distance import directed_hausdorff

# Argument parser (reuse configurations from training)
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='kits19/data', help='Root directory for KiTS19 data')
parser.add_argument('--list_dir', type=str, default='./lists_kits19', help='Directory containing train/val/test lists')
parser.add_argument('--dataset', type=str, default='KiTS19', help='Dataset name')
parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')
parser.add_argument('--img_size', type=int, default=224, help='Input image size')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT model name')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patch size')
parser.add_argument('--use_attention', type=int, default=0, help='Use attention gates in decoder (1 for Yes, 0 for No)')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_kits19_with_attention/best_model.pth', help='Path to the trained model checkpoint')
parser.add_argument('--test_list', type=str, default='./lists_kits19/test.txt', help='Path to the test list file')
parser.add_argument('--test_percentage', type=float, default=1.0, help='Percentage of the test dataset to use (0-100)')
parser.add_argument('--debug', type=bool, default=False, help='Enable debugging visualization and logging')
args = parser.parse_args()

# Load Model Configuration
config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = 3
config_vit.use_attention = bool(args.use_attention)
if 'R50' in args.vit_name:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
model.to(device)

# Load checkpoint with resizing for position embeddings
checkpoint = torch.load(args.checkpoint_path, map_location=device)
state_dict = checkpoint

# Adjust position embeddings size mismatch
pos_emb_key = 'transformer.embeddings.position_embeddings'
if pos_emb_key in state_dict:
    pretrained_pos_emb = state_dict[pos_emb_key]
    if pretrained_pos_emb.shape != model.state_dict()[pos_emb_key].shape:        
        # Reshape position embeddings
        ntok_new = model.state_dict()[pos_emb_key].shape[1]
        ntok_old = pretrained_pos_emb.shape[1]
        if ntok_old != ntok_new:
            pretrained_pos_emb = pretrained_pos_emb[:, :ntok_new, :]
            state_dict[pos_emb_key] = pretrained_pos_emb

# Load the adjusted state_dict
model.load_state_dict(state_dict, strict=False)
model.eval()

# Load Test Dataset
test_dataset = KiTS19DatasetList(
    list_file=args.test_list,
    base_dir=args.root_path,
    slice_size=(args.img_size, args.img_size),
    augment=False  # No augmentation during testing
)

# Adjust dataset size based on the percentage specified
total_samples = len(test_dataset)
subset_size = int(total_samples * (args.test_percentage / 100.0))
subset_indices = np.random.choice(total_samples, subset_size, replace=False)
test_subset = Subset(test_dataset, subset_indices)

# Create DataLoader for the subset
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

# Function to compute Hausdorff Distance
def hausdorff_distance(pred_mask, true_mask):
    pred_coords = np.argwhere(pred_mask)
    true_coords = np.argwhere(true_mask)

    if len(pred_coords) == 0 or len(true_coords) == 0:
        return np.inf  # No overlap

    forward_hd = directed_hausdorff(pred_coords, true_coords)[0]
    backward_hd = directed_hausdorff(true_coords, pred_coords)[0]

    return max(forward_hd, backward_hd)

# Enhanced Metrics Calculation Functions
def compute_metrics(prediction, label, num_classes):
    metrics = {"IoU": [], "Dice": [], "Hausdorff Distance": []}
    eps = 1e-6  # Small epsilon to avoid division by zero

    for class_id in range(num_classes):  # Include background (class_id=0)
        pred_mask = (prediction == class_id).astype(np.uint8)
        true_mask = (label == class_id).astype(np.uint8)

        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        pred_sum = pred_mask.sum()
        true_sum = true_mask.sum()

        # IoU
        if union > 0:
            iou = intersection / (union + eps)
        else:
            iou = 1.0 if class_id == 0 else 0.0  # Perfect background match or no foreground
        metrics["IoU"].append(iou)

        # Dice
        if pred_sum + true_sum > 0:
            dice = (2 * intersection) / (pred_sum + true_sum + eps)
        else:
            dice = 1.0 if class_id == 0 else 0.0  # Perfect background match or no foreground
        metrics["Dice"].append(dice)

        # Hausdorff Distance
        if pred_sum > 0 and true_sum > 0:
            hd = hausdorff_distance(pred_mask, true_mask)
        else:
            hd = float('nan')  # Undefined for empty masks
        metrics["Hausdorff Distance"].append(hd)

    return metrics

# Enhanced metrics aggregation to ignore NaN values
def aggregate_metrics(overall_metrics):
    avg_metrics = {}
    for key in overall_metrics:
        class_metrics = np.array(overall_metrics[key])
        # Compute mean, ignoring NaN values for Hausdorff Distance
        avg_metrics[key] = np.nanmean(class_metrics, axis=0)
    return avg_metrics

# Debugging Function
def log_debug_info(image, label, prediction, metrics, idx):
    print(f"Sample Index: {idx}")
    print(f"Metrics: {metrics}")
    #visualize_predictions(image, label, prediction, idx)

# Add a progress bar to iterate over the test_loader
overall_metrics = {"IoU": [], "Dice": [], "Hausdorff Distance": []}
with torch.no_grad():
    for idx, sample in enumerate(tqdm(test_loader, desc="Processing Test Dataset", unit="sample")):
        image = sample["image"].to(device)
        label = sample["label"].squeeze().cpu().numpy()  # Ground truth

        # Forward pass
        output = model(image)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Predicted mask

        # Compute metrics for this sample
        metrics = compute_metrics(prediction, label, args.num_classes)

        # Accumulate overall metrics
        for key in metrics:
            overall_metrics[key].append(metrics[key])

        # Debugging visualization for samples with poor metrics
        if args.debug:
            log_debug_info(image.cpu().numpy(), label, prediction, metrics, idx)

# Compute average metrics across the dataset per class
avg_metrics = aggregate_metrics(overall_metrics)

# Print average metrics per class
print("\nAverage Metrics per Class:")
for class_id in range(1, args.num_classes):  # Exclude background (class_id=0)
    class_name = {1: "Kidney", 2: "Tumor"}[class_id]
    print(f"Class {class_id} ({class_name}):")
    print(f"  IoU: {avg_metrics['IoU'][class_id - 1]:.4f}")
    print(f"  Dice: {avg_metrics['Dice'][class_id - 1]:.4f}")
    print(f"  Hausdorff Distance: {avg_metrics['Hausdorff Distance'][class_id - 1]:.4f}")

# Print overall average metrics across all classes
overall_avg_metrics = {key: np.mean(avg_metrics[key]) for key in avg_metrics}
print("\nOverall Average Metrics Across All Classes:")
for key, value in overall_avg_metrics.items():
    print(f"  {key}: {value:.4f}")