import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
# Set matplotlib backend to non-GUI
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a display
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_kits19_list import KiTS19DatasetList
import argparse
import logging
from datetime import datetime
import random
from torch.utils.data import Subset
import nibabel as nib

def visualize_transformer_attention(image, attention_weights, ax, layer_idx=-1, head_idx=None, title=None):
    """
    Visualize transformer attention weights overlaid on the original image.
    """
    # Select layer
    attn = attention_weights[layer_idx]  # [B, H, N, N]
    
    # Get attention for first batch item (we only process one image at a time)
    batch_attn = attn[0]  # [H, N, N]
    
    # Average across heads or select specific head
    if head_idx is not None:
        batch_attn = batch_attn[head_idx]  # [N, N]
    else:
        batch_attn = batch_attn.mean(0)  # [N, N]
    
    # Reshape to spatial dimensions [H, W, H, W]
    n = int(np.sqrt(batch_attn.shape[0]))
    batch_attn = batch_attn.reshape(n, n, n, n)
    
    # Average attention across queries to get attention per grid cell
    spatial_attn = batch_attn.mean(axis=(2, 3))
    
    # Resize to image dimensions
    h, w = image.shape
    # Use simple interpolation instead of cv2
    spatial_attn_resized = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            # Simple nearest neighbor interpolation
            ni = int(i * spatial_attn.shape[0] / h)
            nj = int(j * spatial_attn.shape[1] / w)
            spatial_attn_resized[i, j] = spatial_attn[ni, nj]
    
    # Plot original image
    ax.imshow(image, cmap='gray')
    
    # Overlay attention map
    attention_map = ax.imshow(spatial_attn_resized, alpha=0.6, cmap='jet')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Transformer Attention (Layer {layer_idx}, {'Avg Heads' if head_idx is None else f'Head {head_idx}'})", 
                    fontsize=14, fontweight='bold')
    
    ax.axis('off')
    return attention_map

def visualize_decoder_attention(image, attention_map, ax, block_idx, title=None):
    """
    Visualize decoder attention gate maps overlaid on the original image.
    """
    if attention_map is None:
        ax.imshow(image, cmap='gray')
        ax.set_title(f"No Attention Map (Block {block_idx})", fontsize=14, fontweight='bold')
        ax.axis('off')
        return None
    
    # Get attention map for first batch item (we only process one image at a time)
    attn = attention_map[0, 0].cpu().numpy()  # [H', W']
    
    # Resize to match image dimensions - simple approach without cv2
    h, w = image.shape
    attn_resized = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            # Simple nearest neighbor interpolation
            ni = int(i * attn.shape[0] / h)
            nj = int(j * attn.shape[1] / w)
            attn_resized[i, j] = attn[ni, nj]
    
    # Plot original image
    ax.imshow(image, cmap='gray')
    
    # Overlay attention map
    attention_overlay = ax.imshow(attn_resized, alpha=0.6, cmap='jet')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Decoder Attention (Block {block_idx})", fontsize=14, fontweight='bold')
    
    ax.axis('off')
    return attention_overlay

def show_segmentation(ax, image, segmentation, title=None):
    """
    Display a segmentation map overlaid on the original image.
    
    Args:
        ax: Matplotlib axis
        image: Original grayscale image
        segmentation: Segmentation map with class indices
        title: Optional title
    """
    # Display original image
    ax.imshow(image, cmap='gray')
    
    # Create a color mask for each class
    # 0: background (transparent)
    # 1: kidney (red, semi-transparent)
    # 2: tumor (blue, semi-transparent)
    mask = np.zeros((*segmentation.shape, 4), dtype=np.float32)  # RGBA
    
    # Kidney mask (red)
    kidney_mask = segmentation == 1
    mask[kidney_mask, 0] = 1.0  # R
    mask[kidney_mask, 3] = 0.5  # Alpha
    
    # Tumor mask (blue)
    tumor_mask = segmentation == 2
    mask[tumor_mask, 2] = 1.0  # B
    mask[tumor_mask, 3] = 0.5  # Alpha
    
    # Overlay segmentation
    ax.imshow(mask)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

def compute_dice(pred, target, class_idx):
    """
    Compute Dice coefficient for a specific class.
    
    Args:
        pred: Prediction tensor with class indices
        target: Target tensor with class indices
        class_idx: Class index to compute Dice for
    
    Returns:
        Dice coefficient (float)
    """
    # Create binary masks for the class
    pred_mask = (pred == class_idx).float()
    target_mask = (target == class_idx).float()
    
    # Compute Dice coefficient
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    
    # Handle empty masks
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2. * intersection / union).item()

def visualize_attention_maps_with_seg(image, gt_seg, pred_seg, transformer_attn, decoder_attn, 
                                     case_id, slice_idx, kidney_dice, tumor_dice, save_dir):
    """
    Create comprehensive visualization of attention maps from both transformer and decoder,
    along with segmentation results.
    """
    # Create a multi-row figure with increased size for better visibility
    fig = plt.figure(figsize=(22, 16))
    
    # Row 1: Image and segmentation (larger size)
    ax_orig = plt.subplot2grid((4, 4), (0, 0), colspan=1)
    ax_orig.imshow(image, cmap='gray')
    ax_orig.set_title("Original Image", fontsize=14, fontweight='bold')
    ax_orig.axis('off')
    
    ax_gt = plt.subplot2grid((4, 4), (0, 1), colspan=1)
    show_segmentation(ax_gt, image, gt_seg, title="Ground Truth")
    ax_gt.set_title("Ground Truth", fontsize=14, fontweight='bold')
    
    ax_pred = plt.subplot2grid((4, 4), (0, 2), colspan=1)
    show_segmentation(ax_pred, image, pred_seg, title="Prediction")
    ax_pred.set_title("Prediction", fontsize=14, fontweight='bold')
    
    # Add metrics information
    metrics_text = f"Kidney Dice: {kidney_dice:.4f}\nTumor Dice: {tumor_dice:.4f}"
    ax_metrics = plt.subplot2grid((4, 4), (0, 3), colspan=1)
    ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=14, fontweight='bold')
    ax_metrics.axis('off')
    
    # Row 2 & 3: Transformer attention maps (last layer, different heads)
    # Make them bigger by using 2 rows instead of 1
    last_layer_idx = len(transformer_attn) - 1
    attn = transformer_attn[last_layer_idx]  # [B, H, N, N]
    num_heads = min(6, attn.shape[1])  # Show up to 6 attention heads
    
    # First row of transformer heads (3 heads)
    for h in range(min(3, num_heads)):
        ax_trans = plt.subplot2grid((4, 4), (1, h+1))
        visualize_transformer_attention(
            image, transformer_attn, ax_trans, 
            layer_idx=last_layer_idx, head_idx=h,
            title=f"Transformer Head {h}"
        )
        ax_trans.set_title(f"Transformer Head {h}", fontsize=14, fontweight='bold')
    
    # Second row of transformer heads (remaining heads)
    for h in range(3, num_heads):
        ax_trans = plt.subplot2grid((4, 4), (2, h-2))
        visualize_transformer_attention(
            image, transformer_attn, ax_trans, 
            layer_idx=last_layer_idx, head_idx=h,
            title=f"Transformer Head {h}"
        )
        ax_trans.set_title(f"Transformer Head {h}", fontsize=14, fontweight='bold')
    
    # Add average of all heads
    ax_trans_avg = plt.subplot2grid((4, 4), (1, 0))
    visualize_transformer_attention(
        image, transformer_attn, ax_trans_avg, 
        layer_idx=last_layer_idx, head_idx=None,
        title="Transformer (All Heads)"
    )
    ax_trans_avg.set_title("Transformer (All Heads)", fontsize=14, fontweight='bold')
    
    # Add legend for attention map
    cax_trans = plt.subplot2grid((4, 4), (2, 3))
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='jet'), cax=cax_trans)
    cbar.set_label('Transformer Attention Intensity', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    
    # Row 4: Decoder attention maps
    for i, attn_map in enumerate(decoder_attn):
        if i >= 4:  # Show at most 4 decoder blocks in a row
            break
        ax_dec = plt.subplot2grid((4, 4), (3, i))
        visualize_decoder_attention(
            image, attn_map, ax_dec, i,
            title=f"Decoder Block {i}"
        )
        ax_dec.set_title(f"Decoder Block {i}", fontsize=14, fontweight='bold')
    
    plt.suptitle(f"Case {case_id}, Slice {slice_idx} - Attention Maps and Segmentation", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout with room for suptitle
    
    # Save figure as PDF with high quality
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"attention_case{case_id}_slice{slice_idx}.pdf")
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()
    
    return save_path

def load_case_directly(case_id, data_dir, img_size):
    """
    Load a specific case directly from the KiTS19 dataset.
    
    Args:
        case_id: Case ID (e.g., '00086')
        data_dir: Base directory for KiTS19 data
        img_size: Size to resize slices to
    
    Returns:
        images: List of preprocessed image slices
        labels: List of preprocessed label slices
        slice_indices: List of slice indices
    """
    case_path = os.path.join(data_dir, f"case_{case_id}")
    image_path = os.path.join(case_path, "imaging.nii.gz")
    segmentation_path = os.path.join(case_path, "segmentation.nii.gz")
    
    logging.info(f"Loading image from: {image_path}")
    logging.info(f"Loading segmentation from: {segmentation_path}")
    
    # Load the NIfTI files
    image_nii = nib.load(image_path)
    segmentation_nii = nib.load(segmentation_path)
    
    # Get the data as numpy arrays
    image_data = image_nii.get_fdata()
    segmentation_data = segmentation_nii.get_fdata()
    
    logging.info(f"Image shape: {image_data.shape}, Segmentation shape: {segmentation_data.shape}")
    
    # Normalize image data to [0, 1]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # Convert to the right orientation (assuming axial slices)
    # For KiTS19, we usually want to view axial slices (axis 2)
    num_slices = image_data.shape[2]
    
    images = []
    labels = []
    slice_indices = []
    kidney_slices = 0
    tumor_slices = 0
    
    logging.info(f"Processing {num_slices} total slices from case {case_id}")
    
    for slice_idx in range(num_slices):
        # Extract the slice
        image_slice = image_data[:, :, slice_idx]
        segmentation_slice = segmentation_data[:, :, slice_idx]
        
        # Check if the slice contains kidney or tumor
        has_kidney = np.any(segmentation_slice == 1)
        has_tumor = np.any(segmentation_slice == 2)
        
        # Skip slices without kidney or tumor
        if not (has_kidney or has_tumor):
            continue
        
        if has_kidney:
            kidney_slices += 1
        if has_tumor:
            tumor_slices += 1
        
        # Resize to the desired dimensions
        # Use scipy's resize function for better quality
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (img_size / image_slice.shape[0], img_size / image_slice.shape[1])
        
        # Resize image using interpolation
        resized_image = zoom(image_slice, zoom_factors, order=1)  # order=1 for bilinear interpolation
        
        # Resize segmentation using nearest neighbor to preserve labels
        resized_segmentation = zoom(segmentation_slice, zoom_factors, order=0)  # order=0 for nearest neighbor
        
        # Add channel dimension and convert to tensor
        image_tensor = torch.from_numpy(resized_image).float().unsqueeze(0)
        label_tensor = torch.from_numpy(resized_segmentation).long()
        
        images.append(image_tensor)
        labels.append(label_tensor)
        slice_indices.append(slice_idx)
    
    logging.info(f"Found {len(images)} slices with kidney or tumor")
    logging.info(f"Kidney present in {kidney_slices} slices, tumor present in {tumor_slices} slices")
    
    return images, labels, slice_indices

def visualize_case_attention(args, case_id="00086"):
    """
    Visualize attention maps for a specific case, focusing on slices with kidney and tumor.
    Only saves the top N slices by average Dice score.
    """
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'attention_vis_case{case_id}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(results_dir, 'attention_log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Initialize model
    logging.info("Initializing model...")
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_attention = True  # Must use attention for visualization
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                  int(args.img_size / args.vit_patches_size))
    
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # Load model weights
    logging.info(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Load the specific case data directly
    logging.info(f"Loading case {case_id} directly from {args.root_path}")
    
    # Direct loading of the specific case
    images, labels, slice_indices = load_case_directly(case_id, args.root_path, args.img_size)
    
    logging.info(f"Processing {len(images)} slices with kidney or tumor from case {case_id}")
    
    # Create directory for visualizations
    vis_dir = os.path.join(results_dir, 'attention_maps')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each slice and collect dice scores
    slice_results = []
    
    with torch.no_grad():
        for i in tqdm(range(len(images)), desc="Analyzing slices"):
            try:
                image = images[i].cuda()
                label = labels[i].cuda()
                slice_idx = slice_indices[i]
                
                # Forward pass with attention map return
                output, attention_maps = model(image.unsqueeze(0), return_attn=True)
                
                # Get prediction
                pred = torch.argmax(output, dim=1)[0]  # Remove batch dimension
                
                # Compute Dice scores
                kidney_dice = compute_dice(pred, label, class_idx=1)
                tumor_dice = compute_dice(pred, label, class_idx=2)
                
                # Calculate average dice score
                avg_dice = (kidney_dice + tumor_dice) / 2.0
                
                # Store results
                slice_results.append({
                    'index': i,
                    'slice_idx': slice_idx,
                    'image': image.cpu(),
                    'label': label.cpu(),
                    'pred': pred.cpu(),
                    'kidney_dice': kidney_dice,
                    'tumor_dice': tumor_dice,
                    'avg_dice': avg_dice,
                    'transformer_attn': attention_maps['transformer'],  # Keep as CUDA tensors for now
                    'decoder_attn': attention_maps['decoder']
                })
                
            except Exception as e:
                logging.error(f"Error processing slice {i}: {e}")
                continue
    
    # Sort slices by average Dice score (highest first)
    slice_results.sort(key=lambda x: x['avg_dice'], reverse=True)
    
    # Determine how many top slices to save
    num_to_save = min(args.top_n, len(slice_results))
    logging.info(f"Saving visualizations for the top {num_to_save} slices by average Dice score")
    
    # Save visualizations for top slices
    for i, result in enumerate(tqdm(slice_results[:num_to_save], desc="Saving visualizations")):
        logging.info(f"Visualizing top slice {i+1}/{num_to_save}: "
                    f"Slice index {result['slice_idx']}, "
                    f"Kidney Dice: {result['kidney_dice']:.4f}, "
                    f"Tumor Dice: {result['tumor_dice']:.4f}, "
                    f"Avg Dice: {result['avg_dice']:.4f}")
        
        # Visualize and save attention maps with segmentation
        save_path = visualize_attention_maps_with_seg(
            result['image'].cpu().numpy()[0],  # First channel
            result['label'].cpu().numpy(), 
            result['pred'].cpu().numpy(), 
            result['transformer_attn'], 
            result['decoder_attn'], 
            case_id, 
            result['slice_idx'], 
            result['kidney_dice'], 
            result['tumor_dice'], 
            vis_dir
        )
    
    # Write a summary of results
    with open(os.path.join(results_dir, 'dice_scores.csv'), 'w') as f:
        f.write('slice_idx,kidney_dice,tumor_dice,avg_dice\n')
        for result in slice_results:
            f.write(f"{result['slice_idx']},{result['kidney_dice']:.4f},{result['tumor_dice']:.4f},{result['avg_dice']:.4f}\n")
    
    logging.info(f"Completed attention map visualization for top {num_to_save} slices of case {case_id}.")
    logging.info(f"Results saved to {results_dir}")
    return results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize attention maps for TransUNet')
    parser.add_argument('--root_path', type=str, default='kits19/data',
                      help='root dir for KiTS19 data')
    parser.add_argument('--list_dir', type=str, default='./lists_kits19',
                      help='list dir')
    parser.add_argument('--output_dir', type=str, default='./attention_results',
                      help='output dir')
    parser.add_argument('--model_path', type=str, required=True,
                      help='path to trained model weights')
    parser.add_argument('--num_classes', type=int, default=3,
                      help='output channel of network')
    parser.add_argument('--img_size', type=int, default=224,
                      help='input patch size')
    parser.add_argument('--n_skip', type=int, default=3,
                      help='number of skip connections')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                      help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, 
                      help='ViT patch size')
    parser.add_argument('--case_id', type=str, default='00086',
                      help='case ID to visualize (default: 00086)')
    parser.add_argument('--split', type=str, default='test',
                      help='dataset split to use (train, val, test)')
    parser.add_argument('--top_n', type=int, default=100,
                      help='number of top slices to visualize based on Dice score')
    
    args = parser.parse_args()
    
    results_dir = visualize_case_attention(args, case_id=args.case_id)
    print(f"Results saved to: {results_dir}")