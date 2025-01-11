import argparse
import torch
import numpy as np
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_kits19_list import KiTS19DatasetList
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

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
parser.add_argument('--test_percentage', type=float, default=10.0, help='Percentage of the test dataset to use (0-100)')
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

# Function to Visualize Predictions
def visualize_predictions(image, label, prediction, idx):
    plt.figure(figsize=(15, 5))

    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(image[0], cmap='gray')  # Ensure image has at least 2D shape
    plt.title("Input Image")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    # Model Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='viridis')  # 'viridis' to highlight all classes
    plt.title("Prediction")
    plt.axis("off")

    # Save visualization or show it
    plt.savefig(f'prediction_{idx}.png')
    plt.show()

# Iterate through the test dataset and display examples with all predicted classes
with torch.no_grad():
    for idx, sample in enumerate(tqdm(test_loader, desc="Processing Test Dataset", unit="sample")):
        image = sample["image"].to(device)
        label = sample["label"].squeeze().cpu().numpy()  # Ground truth

        # Forward pass
        output = model(image)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Predicted mask

        # Check if all classes are present in the prediction
        unique_classes = np.unique(prediction)
        if set(range(args.num_classes)) <= set(unique_classes):
            # Visualize the predictions if all classes are present
            input_image = image.cpu().numpy()[0]  # [B, C, H, W] -> Take the first sample in batch
            visualize_predictions(input_image, label, prediction, idx)