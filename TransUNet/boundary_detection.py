import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from skimage import measure, feature
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
from scipy.ndimage import zoom
import argparse
import logging
from datetime import datetime
import nibabel as nib
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def setup_logging(results_dir):
    """Set up logging configuration."""
    os.makedirs(results_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(results_dir, 'boundary_analysis_log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
    return logging.getLogger()

def load_case_data(case_id, data_dir, img_size):
    """Load a specific case from KiTS19 dataset."""
    case_path = os.path.join(data_dir, f"case_{case_id}")
    image_path = os.path.join(case_path, "imaging.nii.gz")
    segmentation_path = os.path.join(case_path, "segmentation.nii.gz")
    
    # Load the NIfTI files
    image_nii = nib.load(image_path)
    segmentation_nii = nib.load(segmentation_path)
    
    # Get the data as numpy arrays
    image_data = image_nii.get_fdata()
    segmentation_data = segmentation_nii.get_fdata()
    
    # Normalize image data to [0, 1]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # For axial slices (axis 2)
    num_slices = image_data.shape[2]
    
    images = []
    labels = []
    slice_indices = []
    original_shapes = []
    
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
        
        # Save original shape before resizing
        original_shapes.append(image_slice.shape)
        
        # Resize using scipy's zoom
        zoom_factors = (img_size / image_slice.shape[0], img_size / image_slice.shape[1])
        
        # Resize image using interpolation
        resized_image = zoom(image_slice, zoom_factors, order=1)
        
        # Resize segmentation using nearest neighbor
        resized_segmentation = zoom(segmentation_slice, zoom_factors, order=0)
        
        # Add channel dimension and convert to tensor
        image_tensor = torch.from_numpy(resized_image).float().unsqueeze(0)
        label_tensor = torch.from_numpy(resized_segmentation).long()
        
        images.append(image_tensor)
        labels.append(label_tensor)
        slice_indices.append(slice_idx)
    
    return images, labels, slice_indices, original_shapes

def compute_dice(pred, target, class_idx):
    """Compute Dice coefficient for a specific class."""
    pred_mask = (pred == class_idx).float()
    target_mask = (target == class_idx).float()
    
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2. * intersection / union).item()

def extract_boundaries(segmentation, class_idx):
    """Extract boundary pixels for a given class."""
    # Create binary mask for the class
    mask = (segmentation == class_idx).astype(np.uint8)
    
    # If no pixels of this class, return empty boundary
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    
    # Use erosion to find boundary
    eroded = ndimage.binary_erosion(mask)
    boundary = np.logical_and(mask, np.logical_not(eroded))
    
    return boundary

def compute_boundary_dice(pred_boundary, gt_boundary, distance_tolerance=2):
    """
    Compute Dice coefficient for boundaries with distance tolerance.
    
    Args:
        pred_boundary: Binary mask of predicted boundary
        gt_boundary: Binary mask of ground truth boundary
        distance_tolerance: Maximum distance (in pixels) to consider a boundary match
    
    Returns:
        Dice coefficient for boundaries
    """
    if not np.any(gt_boundary):
        return 1.0 if not np.any(pred_boundary) else 0.0
    
    # Dilate ground truth boundary by the tolerance distance
    gt_dilated = ndimage.binary_dilation(
        gt_boundary, 
        structure=ndimage.generate_binary_structure(2, 1),
        iterations=distance_tolerance
    )
    
    # Find matching boundary pixels (true positives)
    true_positives = np.logical_and(pred_boundary, gt_dilated).sum()
    
    # Calculate Dice
    return (2 * true_positives) / (pred_boundary.sum() + gt_boundary.sum())

def compute_hausdorff_distance(pred_mask, gt_mask):
    """
    Compute the Hausdorff distance between two binary masks.
    
    Args:
        pred_mask: Binary mask of prediction
        gt_mask: Binary mask of ground truth
    
    Returns:
        Hausdorff distance (modified with log to handle outliers)
    """
    from scipy.spatial.distance import directed_hausdorff
    
    # If either mask is empty, return a large value
    if not np.any(pred_mask) or not np.any(gt_mask):
        return 100.0  # A large value indicating failure
    
    # Get boundary coordinates
    pred_boundary = extract_boundaries(pred_mask, 1)
    gt_boundary = extract_boundaries(gt_mask, 1)
    
    # Get coordinates of boundary pixels
    pred_coords = np.array(np.where(pred_boundary)).T
    gt_coords = np.array(np.where(gt_boundary)).T
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 100.0
    
    # Calculate directed Hausdorff distances
    d1 = directed_hausdorff(pred_coords, gt_coords)[0]
    d2 = directed_hausdorff(gt_coords, pred_coords)[0]
    
    # Take the maximum (true Hausdorff)
    return max(d1, d2)

def process_attention_map(attention_maps, img_size, layer_idx=-1):
    """Process transformer attention map to spatial dimensions."""
    attn = attention_maps['transformer'][layer_idx][0]  # [H, N, N]
    
    # Average across heads
    avg_attn = attn.mean(0)  # [N, N]
    
    # Reshape to spatial dimensions
    n = int(np.sqrt(avg_attn.shape[0]))
    spatial_attn = avg_attn.reshape(n, n, n, n).mean(axis=(2, 3))
    
    # Resize to image dimensions
    attn_resized = torch.nn.functional.interpolate(
        spatial_attn.unsqueeze(0).unsqueeze(0), 
        size=(img_size, img_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().cpu().numpy()
    
    return attn_resized

def compute_attention_metrics_at_boundary(attention_map, boundary_mask, non_boundary_mask):
    """Calculate attention metrics specifically at boundary regions."""
    # Mean attention at boundary
    boundary_attention = attention_map[boundary_mask].mean() if boundary_mask.sum() > 0 else 0
    
    # Mean attention at non-boundary regions
    non_boundary_attention = attention_map[non_boundary_mask].mean() if non_boundary_mask.sum() > 0 else 0
    
    # Ratio of boundary attention to non-boundary attention
    boundary_ratio = boundary_attention / non_boundary_attention if non_boundary_attention > 0 else 0
    
    return {
        'boundary_attention': boundary_attention,
        'non_boundary_attention': non_boundary_attention,
        'boundary_ratio': boundary_ratio
    }

def find_challenging_boundaries(results, min_tumor_size=100):
    """Find slices with challenging tumor boundaries but reasonable size."""
    challenging = []
    
    for result in results:
        # Check if tumor is present and has reasonable size
        if result['has_tumor'] and result['tumor_size'] > min_tumor_size:
            # If tumor Dice is high but boundary Dice is lower, it's a challenging boundary case
            if result['tumor_dice'] > 0.7 and result['tumor_boundary_dice'] < 0.6:
                challenging.append(result)
    
    # Sort by the difference between tumor Dice and boundary Dice
    challenging.sort(key=lambda x: x['tumor_dice'] - x['tumor_boundary_dice'], reverse=True)
    
    return challenging[:min(5, len(challenging))]

def create_boundary_visualization(result, results_dir, case_id, include_decoder_attention=True):
    """Create a comprehensive visualization showing boundary detection with attention."""
    # Extract data from result
    image = result['image'].cpu().numpy()[0]
    gt_seg = result['label'].cpu().numpy()
    pred_seg = result['pred'].cpu().numpy()
    attention_map = result['attention_map']
    slice_idx = result['slice_idx']
    
    # Extract boundaries
    gt_kidney_boundary = extract_boundaries(gt_seg, 1)
    gt_tumor_boundary = extract_boundaries(gt_seg, 2)
    pred_kidney_boundary = extract_boundaries(pred_seg, 1)
    pred_tumor_boundary = extract_boundaries(pred_seg, 2)
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Original CT with ground truth boundaries
    axes[0, 0].imshow(image, cmap='gray')
    
    # Plot ground truth kidney boundary in red
    if gt_kidney_boundary.sum() > 0:
        kidney_y, kidney_x = np.where(gt_kidney_boundary)
        axes[0, 0].scatter(kidney_x, kidney_y, c='red', s=10, alpha=0.8, marker='.')
    
    # Plot ground truth tumor boundary in blue
    if gt_tumor_boundary.sum() > 0:
        tumor_y, tumor_x = np.where(gt_tumor_boundary)
        axes[0, 0].scatter(tumor_x, tumor_y, c='blue', s=10, alpha=0.8, marker='.')
    
    axes[0, 0].set_title(f'Original CT with GT Boundaries\nSlice {slice_idx}', fontsize=12)
    axes[0, 0].axis('off')
    
    # Plot 2: Segmentation comparison (GT vs Pred)
    # Create a custom colormap for segmentation
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.5), (0, 0, 1, 0.5)]  # transparent, red, blue
    segmentation_cmap = ListedColormap(colors[:len(np.unique(gt_seg))])
    
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].imshow(gt_seg, cmap=segmentation_cmap, alpha=0.6)
    
    # Add predicted boundaries
    if pred_kidney_boundary.sum() > 0:
        kidney_y, kidney_x = np.where(pred_kidney_boundary)
        axes[0, 1].scatter(kidney_x, kidney_y, c='yellow', s=5, alpha=1.0, marker='.')
    
    if pred_tumor_boundary.sum() > 0:
        tumor_y, tumor_x = np.where(pred_tumor_boundary)
        axes[0, 1].scatter(tumor_x, tumor_y, c='cyan', s=5, alpha=1.0, marker='.')
    
    axes[0, 1].set_title(f'GT Segmentation with Predicted Boundaries\nK-Dice: {result["kidney_dice"]:.3f}, T-Dice: {result["tumor_dice"]:.3f}', fontsize=12)
    axes[0, 1].axis('off')
    
    # Plot 3: Attention map
    im = axes[0, 2].imshow(attention_map, cmap='jet')
    
    # Add ground truth boundaries on attention map
    if gt_kidney_boundary.sum() > 0:
        kidney_y, kidney_x = np.where(gt_kidney_boundary)
        axes[0, 2].scatter(kidney_x, kidney_y, c='white', s=5, alpha=0.5, marker='.')
    
    if gt_tumor_boundary.sum() > 0:
        tumor_y, tumor_x = np.where(gt_tumor_boundary)
        axes[0, 2].scatter(tumor_x, tumor_y, c='yellow', s=5, alpha=0.5, marker='.')
    
    axes[0, 2].set_title(f'Attention Map with GT Boundaries\nBoundary Attention Ratio: {result["tumor_boundary_ratio"]:.2f}', fontsize=12)
    axes[0, 2].axis('off')
    
    # Add colorbar for attention map
    cbar = plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
    
    # Plot 4: Boundary-focused attention
    # Create a mask that highlights the boundary regions
    boundary_highlight = np.zeros_like(image)
    
    if gt_tumor_boundary.sum() > 0:
        # Dilate boundary to make it more visible
        dilated_boundary = ndimage.binary_dilation(
            gt_tumor_boundary, 
            structure=ndimage.generate_binary_structure(2, 1),
            iterations=3
        )
        boundary_highlight[dilated_boundary] = 1
    
    # Show image with boundary regions highlighted
    axes[1, 0].imshow(image, cmap='gray')
    
    # Use a red colormap for boundary highlight
    axes[1, 0].imshow(boundary_highlight, cmap='Reds', alpha=0.7)
    
    axes[1, 0].set_title('Tumor Boundary Region Highlighted', fontsize=12)
    axes[1, 0].axis('off')
    
    # Plot 5: Attention overlay on original image
    axes[1, 1].imshow(image, cmap='gray')
    axes[1, 1].imshow(attention_map, cmap='jet', alpha=0.6)
    
    # Add predicted boundaries
    if pred_kidney_boundary.sum() > 0:
        kidney_y, kidney_x = np.where(pred_kidney_boundary)
        axes[1, 1].scatter(kidney_x, kidney_y, c='white', s=5, alpha=0.7, marker='.')
    
    if pred_tumor_boundary.sum() > 0:
        tumor_y, tumor_x = np.where(pred_tumor_boundary)
        axes[1, 1].scatter(tumor_x, tumor_y, c='yellow', s=5, alpha=0.7, marker='.')
    
    axes[1, 1].set_title('Attention Overlay with Predicted Boundaries', fontsize=12)
    axes[1, 1].axis('off')
    
    # Plot 6: Boundary attention comparison
    if include_decoder_attention and 'decoder_attention_maps' in result:
        # Show decoder attention (from last decoder block) focused on boundary
        decoder_attention = result['decoder_attention_maps'][-1]
        
        if decoder_attention is not None:
            axes[1, 2].imshow(image, cmap='gray')
            axes[1, 2].imshow(decoder_attention, cmap='jet', alpha=0.6)
            
            # Add ground truth boundaries
            if gt_tumor_boundary.sum() > 0:
                tumor_y, tumor_x = np.where(gt_tumor_boundary)
                axes[1, 2].scatter(tumor_x, tumor_y, c='white', s=5, alpha=0.7, marker='.')
            
            axes[1, 2].set_title('Decoder Attention at Boundary', fontsize=12)
        else:
            axes[1, 2].imshow(image, cmap='gray')
            axes[1, 2].set_title('Decoder Attention Not Available', fontsize=12)
    else:
        # Show zoomed boundary region with attention
        if gt_tumor_boundary.sum() > 0:
            # Find tumor boundary centroid
            tumor_y, tumor_x = np.where(gt_tumor_boundary)
            center_y, center_x = int(np.mean(tumor_y)), int(np.mean(tumor_x))
            
            # Define zoom region (50x50 pixels around center)
            zoom_size = 50
            y_min = max(0, center_y - zoom_size//2)
            y_max = min(image.shape[0], center_y + zoom_size//2)
            x_min = max(0, center_x - zoom_size//2)
            x_max = min(image.shape[1], center_x + zoom_size//2)
            
            # Extract zoomed regions
            zoomed_image = image[y_min:y_max, x_min:x_max]
            zoomed_attention = attention_map[y_min:y_max, x_min:x_max]
            zoomed_gt_boundary = gt_tumor_boundary[y_min:y_max, x_min:x_max]
            zoomed_pred_boundary = pred_tumor_boundary[y_min:y_max, x_min:x_max]
            
            # Show zoomed image with attention overlay
            axes[1, 2].imshow(zoomed_image, cmap='gray')
            axes[1, 2].imshow(zoomed_attention, cmap='jet', alpha=0.7)
            
            # Add boundary points
            if zoomed_gt_boundary.sum() > 0:
                zy, zx = np.where(zoomed_gt_boundary)
                axes[1, 2].scatter(zx, zy, c='white', s=15, alpha=0.8, marker='.')
            
            if zoomed_pred_boundary.sum() > 0:
                zy, zx = np.where(zoomed_pred_boundary)
                axes[1, 2].scatter(zx, zy, c='yellow', s=15, alpha=0.8, marker='.')
            
            axes[1, 2].set_title('Zoomed Tumor Boundary with Attention', fontsize=12)
        else:
            axes[1, 2].imshow(image, cmap='gray')
            axes[1, 2].set_title('No Tumor Boundary to Zoom', fontsize=12)
    
    axes[1, 2].axis('off')
    
    # Create legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Kidney (GT)'),
        Patch(facecolor='blue', alpha=0.5, label='Tumor (GT)'),
        Patch(facecolor='none', edgecolor='yellow', label='Predicted Kidney Boundary'),
        Patch(facecolor='none', edgecolor='cyan', label='Predicted Tumor Boundary')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    # Add statistics as text
    stats_text = (
        f"Tumor Boundary Dice: {result['tumor_boundary_dice']:.3f}\n"
        f"Boundary Attention: {result['tumor_boundary_attention']:.3f}\n"
        f"Non-Boundary Attention: {result['tumor_non_boundary_attention']:.3f}\n"
        f"Boundary/Non-Boundary Ratio: {result['tumor_boundary_ratio']:.3f}\n"
        f"Hausdorff Distance: {result['tumor_hausdorff']:.3f}"
    )
    
    fig.text(0.05, 0.03, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.suptitle(f"Case {case_id}, Slice {slice_idx}: Attention Mechanisms for Boundary Detection", 
                fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    # Save figure
    save_path = os.path.join(results_dir, f'boundary_case_{case_id}_slice_{slice_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_boundary_comparison_figure(results, results_dir, case_id):
    """Create a figure comparing boundary detection metrics vs. attention."""
    # Convert results to DataFrame
    data = []
    
    for result in results:
        if result['has_tumor']:
            data.append({
                'slice_idx': result['slice_idx'],
                'tumor_dice': result['tumor_dice'],
                'tumor_boundary_dice': result['tumor_boundary_dice'],
                'tumor_boundary_ratio': result['tumor_boundary_ratio'],
                'tumor_size': result['tumor_size'],
                'tumor_hausdorff': result['tumor_hausdorff']
            })
    
    df = pd.DataFrame(data)
    
    # Create a figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Scatter plot of boundary attention ratio vs. boundary Dice
    sns.scatterplot(
        data=df, x='tumor_boundary_ratio', y='tumor_boundary_dice',
        size='tumor_size', hue='tumor_dice',
        palette='viridis', sizes=(20, 200), alpha=0.7,
        ax=axes[0, 0]
    )
    
    # Add regression line
    sns.regplot(
        data=df, x='tumor_boundary_ratio', y='tumor_boundary_dice',
        scatter=False, color='red', ax=axes[0, 0]
    )
    
    # Add correlation statistics
    if len(df) > 1:
        corr, p_value = np.corrcoef(df['tumor_boundary_ratio'], df['tumor_boundary_dice'])[0, 1], 0.05  # Placeholder p-value
        axes[0, 0].annotate(
            f"Correlation: {corr:.3f}\np-value: {p_value:.4f}",
            xy=(0.05, 0.05), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    axes[0, 0].set_title('Boundary Attention Ratio vs. Boundary Dice Score', fontsize=14)
    axes[0, 0].set_xlabel('Boundary/Non-Boundary Attention Ratio', fontsize=12)
    axes[0, 0].set_ylabel('Boundary Dice Score', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of Hausdorff distance vs. boundary attention ratio
    sns.scatterplot(
        data=df, x='tumor_boundary_ratio', y='tumor_hausdorff',
        size='tumor_size', hue='tumor_dice',
        palette='viridis', sizes=(20, 200), alpha=0.7,
        ax=axes[0, 1]
    )
    
    # Add regression line
    sns.regplot(
        data=df, x='tumor_boundary_ratio', y='tumor_hausdorff',
        scatter=False, color='red', ax=axes[0, 1]
    )
    
    # Add correlation statistics
    if len(df) > 1:
        corr, p_value = np.corrcoef(df['tumor_boundary_ratio'], df['tumor_hausdorff'])[0, 1], 0.05  # Placeholder p-value
        axes[0, 1].annotate(
            f"Correlation: {corr:.3f}\np-value: {p_value:.4f}",
            xy=(0.05, 0.05), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    axes[0, 1].set_title('Boundary Attention Ratio vs. Hausdorff Distance', fontsize=14)
    axes[0, 1].set_xlabel('Boundary/Non-Boundary Attention Ratio', fontsize=12)
    axes[0, 1].set_ylabel('Hausdorff Distance (lower is better)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Comparison of overall Dice vs. boundary Dice
    df['dice_difference'] = df['tumor_dice'] - df['tumor_boundary_dice']
    
    sns.scatterplot(
        data=df, x='tumor_dice', y='tumor_boundary_dice',
        size='tumor_size', hue='tumor_boundary_ratio',
        palette='viridis', sizes=(20, 200), alpha=0.7,
        ax=axes[1, 0]
    )
    
    # Add identity line (x=y)
    identity_line = np.linspace(*axes[1, 0].get_xlim())
    axes[1, 0].plot(identity_line, identity_line, '--', color='gray')
    
    axes[1, 0].set_title('Tumor Dice vs. Boundary Dice', fontsize=14)
    axes[1, 0].set_xlabel('Overall Tumor Dice Score', fontsize=12)
    axes[1, 0].set_ylabel('Tumor Boundary Dice Score', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Box plot of boundary attention ratios by tumor Dice categories
    df['tumor_perf_category'] = pd.cut(
        df['tumor_dice'], 
        bins=[0, 0.5, 0.75, 1.0], 
        labels=['Low (<0.5)', 'Medium (0.5-0.75)', 'High (>0.75)']
    )
    
    # Only plot if we have data in at least 2 categories
    if len(df['tumor_perf_category'].dropna().unique()) >= 2:
        sns.boxplot(
            data=df, x='tumor_perf_category', y='tumor_boundary_ratio',
            ax=axes[1, 1], palette='viridis'
        )
        
        sns.stripplot(
            data=df, x='tumor_perf_category', y='tumor_boundary_ratio',
            ax=axes[1, 1], color='black', alpha=0.5, size=5, jitter=True
        )
        
        axes[1, 1].set_title('Boundary Attention Ratio by Tumor Dice Category', fontsize=14)
        axes[1, 1].set_xlabel('Tumor Dice Category', fontsize=12)
        axes[1, 1].set_ylabel('Boundary/Non-Boundary Attention Ratio', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for categories', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('Boundary Attention Ratio by Category', fontsize=14)
        axes[1, 1].axis('off')
    
    plt.suptitle(f'Case {case_id}: Relationship Between Attention and Boundary Detection', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = os.path.join(results_dir, f'boundary_metrics_case_{case_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def analyze_boundary_detection(args):
    """Analyze how attention mechanisms improve boundary detection."""
    # Set up results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'boundary_analysis_{args.case_id}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(results_dir)
    
    # Load model
    logger.info("Initializing model...")
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_attention = True
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                   int(args.img_size / args.vit_patches_size))
    
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # Load model weights
    logger.info(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Load case data
    logger.info(f"Loading case {args.case_id} data...")
    images, labels, slice_indices, original_shapes = load_case_data(
        args.case_id, args.root_path, args.img_size
    )
    
    logger.info(f"Found {len(images)} slices with kidney or tumor")
    
    # Process each slice and collect metrics
    results = []
    
    logger.info("Processing slices and collecting boundary metrics...")
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
                
                # Compute overall Dice scores
                kidney_dice = compute_dice(pred, label, class_idx=1)
                tumor_dice = compute_dice(pred, label, class_idx=2)
                avg_dice = (kidney_dice + tumor_dice) / 2.0
                
                # Process attention maps
                attention_map = process_attention_map(attention_maps, args.img_size)
                
                # Get ground truth segmentation
                gt_seg = label.cpu().numpy()
                pred_seg = pred.cpu().numpy()
                
                # Extract boundaries
                gt_kidney_boundary = extract_boundaries(gt_seg, 1)
                gt_tumor_boundary = extract_boundaries(gt_seg, 2)
                pred_kidney_boundary = extract_boundaries(pred_seg, 1)
                pred_tumor_boundary = extract_boundaries(pred_seg, 2)
                
                # Compute boundary Dice scores
                kidney_boundary_dice = compute_boundary_dice(pred_kidney_boundary, gt_kidney_boundary)
                tumor_boundary_dice = compute_boundary_dice(pred_tumor_boundary, gt_tumor_boundary)
                
                # Compute Hausdorff distance for tumor
                tumor_hausdorff = compute_hausdorff_distance(
                    (pred_seg == 2).astype(np.uint8), 
                    (gt_seg == 2).astype(np.uint8)
                )
                
                # Check if kidney and tumor are present
                has_kidney = np.any(gt_seg == 1)
                has_tumor = np.any(gt_seg == 2)
                
                # Calculate region sizes
                kidney_size = (gt_seg == 1).sum().item()
                tumor_size = (gt_seg == 2).sum().item()
                
                # Create non-boundary masks
                tumor_non_boundary = np.logical_and(gt_seg == 2, ~gt_tumor_boundary)
                
                # Calculate attention metrics
                if has_tumor:
                    tumor_attention_metrics = compute_attention_metrics_at_boundary(
                        attention_map, gt_tumor_boundary, tumor_non_boundary
                    )
                else:
                    tumor_attention_metrics = {
                        'boundary_attention': 0,
                        'non_boundary_attention': 0,
                        'boundary_ratio': 0
                    }
                
                # Process decoder attention maps if available
                decoder_attention_maps = None
                if 'decoder' in attention_maps and attention_maps['decoder']:
                    decoder_attention_maps = []
                    for decoder_attn in attention_maps['decoder']:
                        if decoder_attn is None:
                            decoder_attention_maps.append(None)
                        else:
                            # Get first batch, first channel
                            attn = decoder_attn[0, 0]  # [H', W']
                            
                            # Resize to match image dimensions
                            attn_resized = torch.nn.functional.interpolate(
                                attn.unsqueeze(0).unsqueeze(0),
                                size=(args.img_size, args.img_size),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze().cpu().numpy()
                            
                            decoder_attention_maps.append(attn_resized)
                
                # Store results
                result = {
                    'slice_idx': slice_idx,
                    'image': image,
                    'label': label,
                    'pred': pred,
                    'attention_map': attention_map,
                    'decoder_attention_maps': decoder_attention_maps,
                    'kidney_dice': kidney_dice,
                    'tumor_dice': tumor_dice,
                    'avg_dice': avg_dice,
                    'kidney_boundary_dice': kidney_boundary_dice,
                    'tumor_boundary_dice': tumor_boundary_dice,
                    'tumor_hausdorff': tumor_hausdorff,
                    'has_kidney': has_kidney,
                    'has_tumor': has_tumor,
                    'kidney_size': kidney_size,
                    'tumor_size': tumor_size,
                    'tumor_boundary_attention': tumor_attention_metrics['boundary_attention'],
                    'tumor_non_boundary_attention': tumor_attention_metrics['non_boundary_attention'],
                    'tumor_boundary_ratio': tumor_attention_metrics['boundary_ratio']
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing slice {i}: {e}")
                continue
    
    # Find challenging boundary cases
    logger.info("Finding challenging boundary cases...")
    challenging_cases = find_challenging_boundaries(results)
    
    logger.info(f"Found {len(challenging_cases)} challenging boundary cases")
    
    # Create visualizations for challenging cases
    for i, case in enumerate(challenging_cases):
        logger.info(f"Creating visualization for challenging case {i+1}: Slice {case['slice_idx']}")
        save_path = create_boundary_visualization(case, results_dir, args.case_id)
        logger.info(f"Saved visualization to {save_path}")
    
    # Create summary figure
    logger.info("Creating boundary metrics comparison figure...")
    comparison_path = create_boundary_comparison_figure(results, results_dir, args.case_id)
    logger.info(f"Saved comparison figure to {comparison_path}")
    
    logger.info(f"Analysis complete. Results saved to {results_dir}")
    return results_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze how attention mechanisms improve boundary detection')
    parser.add_argument('--root_path', type=str, default='kits19/data',
                        help='root dir for KiTS19 data')
    parser.add_argument('--output_dir', type=str, default='./boundary_results',
                        help='output dir')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained model weights')
    parser.add_argument('--case_id', type=str, default='00024',
                        help='case ID to analyze (default: 00086)')
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
    
    args = parser.parse_args()
    
    results_dir = analyze_boundary_detection(args)
    print(f"Results saved to: {results_dir}")