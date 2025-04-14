import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from scipy.ndimage import zoom
from scipy import stats
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
        filename=os.path.join(results_dir, 'analysis_log.txt'),
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
    
    return images, labels, slice_indices

def compute_dice(pred, target, class_idx):
    """Compute Dice coefficient for a specific class."""
    pred_mask = (pred == class_idx).float()
    target_mask = (target == class_idx).float()
    
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2. * intersection / union).item()

def process_attention_transformer(transformer_attn, img_size, layer_idx=-1):
    """Process transformer attention map to spatial dimensions."""
    attn = transformer_attn[layer_idx][0]  # [H, N, N]
    
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

def create_attention_segmentation_comparison(high_performing, low_performing, case_id, results_dir):
    """
    Create a comprehensive figure comparing attention and segmentation between 
    high-performing and low-performing examples.
    
    Args:
        high_performing: List of dictionaries with high-performing examples
        low_performing: List of dictionaries with low-performing examples
        case_id: Case ID
        results_dir: Directory to save results
        
    Returns:
        Path to saved figure
    """
    # Number of examples to show in each category
    n_examples = min(3, len(high_performing), len(low_performing))
    
    # Create the figure
    fig, axes = plt.subplots(4, n_examples, figsize=(n_examples*5, 16))
    
    # Row titles
    row_titles = ["Original CT", "Ground Truth & Prediction", "Transformer Attention", "Attention Overlay"]
    
    # Color maps for visualization
    seg_colors = [
        [0, 0, 0, 0],      # Background (transparent)
        [1, 0, 0, 0.5],    # Kidney (red, semi-transparent)
        [0, 0, 1, 0.5]     # Tumor (blue, semi-transparent)
    ]
    
    # Process high-performing examples
    for i in range(n_examples):
        example = high_performing[i]
        
        # Row 1: Original CT image
        axes[0, i].imshow(example['image'].cpu().numpy()[0], cmap='gray')
        axes[0, i].set_title(f"Slice {example['slice_idx']}\nDice: {example['avg_dice']:.3f}")
        axes[0, i].axis('off')
        
        # Row 2: Ground Truth & Prediction
        # Create a color mask for segmentation
        gt_seg = example['label'].cpu().numpy()
        pred_seg = example['pred'].cpu().numpy()
        
        # Show the original image
        axes[1, i].imshow(example['image'].cpu().numpy()[0], cmap='gray')
        
        # Create segmentation overlay
        overlay = np.zeros((*gt_seg.shape, 4))
        for class_idx in range(1, 3):  # Kidney and tumor
            gt_mask = (gt_seg == class_idx)
            pred_mask = (pred_seg == class_idx)
            
            # Areas where GT and prediction agree
            correct_mask = gt_mask & pred_mask
            overlay[correct_mask] = seg_colors[class_idx]
            
            # Areas where they disagree (outline only)
            if class_idx == 1:  # Kidney
                missed = gt_mask & ~pred_mask
                overlay[missed] = [1, 0.7, 0.7, 0.3]  # Light red for missed kidney
                
                false_pos = ~gt_mask & pred_mask
                overlay[false_pos] = [1, 0.5, 0.5, 0.3]  # Different red for false positives
            else:  # Tumor
                missed = gt_mask & ~pred_mask
                overlay[missed] = [0.7, 0.7, 1, 0.3]  # Light blue for missed tumor
                
                false_pos = ~gt_mask & pred_mask
                overlay[false_pos] = [0.5, 0.5, 1, 0.3]  # Different blue for false positives
        
        # Show the segmentation overlay
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"GT & Pred (K:{example['kidney_dice']:.2f}, T:{example['tumor_dice']:.2f})")
        axes[1, i].axis('off')
        
        # Row 3: Attention map
        attn_map = example['attention_map']
        im = axes[2, i].imshow(attn_map, cmap='jet')
        axes[2, i].set_title(f"Attention Map\nFocus Ratio: {example['attention_ratio']:.2f}")
        axes[2, i].axis('off')
        
        # Row 4: Attention overlay on image
        axes[3, i].imshow(example['image'].cpu().numpy()[0], cmap='gray')
        axes[3, i].imshow(attn_map, alpha=0.6, cmap='jet')
        axes[3, i].set_title("Attention Overlay")
        axes[3, i].axis('off')
    
    # Add a vertical separator
    for ax in axes[:, n_examples-1]:
        ax.axvline(x=ax.get_xlim()[1] + 5, color='k', linestyle='-', lw=2)
    
    # Process low-performing examples
    for i in range(n_examples):
        example = low_performing[i]
        
        # Row 1: Original CT image
        axes[0, i].imshow(example['image'].cpu().numpy()[0], cmap='gray')
        axes[0, i].set_title(f"Slice {example['slice_idx']}\nDice: {example['avg_dice']:.3f}")
        axes[0, i].axis('off')
        
        # Row 2: Ground Truth & Prediction
        # Create a color mask for segmentation
        gt_seg = example['label'].cpu().numpy()
        pred_seg = example['pred'].cpu().numpy()
        
        # Show the original image
        axes[1, i].imshow(example['image'].cpu().numpy()[0], cmap='gray')
        
        # Create segmentation overlay
        overlay = np.zeros((*gt_seg.shape, 4))
        for class_idx in range(1, 3):  # Kidney and tumor
            gt_mask = (gt_seg == class_idx)
            pred_mask = (pred_seg == class_idx)
            
            # Areas where GT and prediction agree
            correct_mask = gt_mask & pred_mask
            overlay[correct_mask] = seg_colors[class_idx]
            
            # Areas where they disagree (outline only)
            if class_idx == 1:  # Kidney
                missed = gt_mask & ~pred_mask
                overlay[missed] = [1, 0.7, 0.7, 0.3]  # Light red for missed kidney
                
                false_pos = ~gt_mask & pred_mask
                overlay[false_pos] = [1, 0.5, 0.5, 0.3]  # Different red for false positives
            else:  # Tumor
                missed = gt_mask & ~pred_mask
                overlay[missed] = [0.7, 0.7, 1, 0.3]  # Light blue for missed tumor
                
                false_pos = ~gt_mask & pred_mask
                overlay[false_pos] = [0.5, 0.5, 1, 0.3]  # Different blue for false positives
        
        # Show the segmentation overlay
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"GT & Pred (K:{example['kidney_dice']:.2f}, T:{example['tumor_dice']:.2f})")
        axes[1, i].axis('off')
        
        # Row 3: Attention map
        attn_map = example['attention_map']
        im = axes[2, i].imshow(attn_map, cmap='jet')
        axes[2, i].set_title(f"Attention Map\nFocus Ratio: {example['attention_ratio']:.2f}")
        axes[2, i].axis('off')
        
        # Row 4: Attention overlay on image
        axes[3, i].imshow(example['image'].cpu().numpy()[0], cmap='gray')
        axes[3, i].imshow(attn_map, alpha=0.6, cmap='jet')
        axes[3, i].set_title("Attention Overlay")
        axes[3, i].axis('off')
    
    # Add row labels
    for i, title in enumerate(row_titles):
        fig.text(0.01, 0.75 - i*0.18, title, va='center', ha='left', 
                fontsize=14, rotation=90, fontweight='bold')
    
    # Add main titles for both sections
    fig.text(0.3, 0.97, "High-Performing Segmentation", fontsize=16, fontweight='bold', ha='center')
    fig.text(0.7, 0.97, "Low-Performing Segmentation", fontsize=16, fontweight='bold', ha='center')
    
    # Add a colorbar for the attention maps
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=20)
    
    # Add a legend for segmentation colors
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array(seg_colors[1]), markersize=15, label='Kidney (Correct)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array([1, 0.7, 0.7, 0.3]), markersize=15, label='Kidney (Missed)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array(seg_colors[2]), markersize=15, label='Tumor (Correct)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array([0.7, 0.7, 1, 0.3]), markersize=15, label='Tumor (Missed)'),
    ]
    
    legend_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_elements, loc='center', ncol=4)
    
    plt.suptitle(f"Case {case_id}: Attention Focus vs. Segmentation Performance", fontsize=18, y=0.995)
    plt.tight_layout(rect=[0.02, 0.03, 0.9, 0.95])
    
    save_path = os.path.join(results_dir, f'attention_segmentation_comparison_{case_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_correlation_figure(results, results_dir, case_id):
    """
    Create a comprehensive correlation figure showing the relationship between
    attention focus and segmentation accuracy.
    
    Args:
        results: List of result dictionaries
        results_dir: Directory to save results
        case_id: Case ID
        
    Returns:
        Path to saved figure
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create a figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Scatter plot of attention ratio vs. tumor dice score
    sns.scatterplot(
        data=df, x='attention_ratio', y='tumor_dice', 
        hue='has_tumor', size='tumor_size',
        palette=['gray', 'blue'], sizes=(20, 200),
        ax=axes[0, 0], alpha=0.7
    )
    
    # Add regression line for tumor slices only
    tumor_df = df[df['has_tumor']]
    if len(tumor_df) > 1:
        sns.regplot(
            data=tumor_df, x='attention_ratio', y='tumor_dice',
            scatter=False, color='blue', ax=axes[0, 0]
        )
        
        # Add correlation statistics
        corr, p_value = stats.pearsonr(tumor_df['attention_ratio'], tumor_df['tumor_dice'])
        axes[0, 0].annotate(
            f"Correlation: {corr:.3f}\np-value: {p_value:.4f}",
            xy=(0.05, 0.05), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    axes[0, 0].set_title('Attention Focus Ratio vs. Tumor Dice Score', fontsize=14)
    axes[0, 0].set_xlabel('Attention Ratio (tumor region / background)', fontsize=12)
    axes[0, 0].set_ylabel('Tumor Dice Score', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Attention heatmap for high-performing tumor segmentation
    # Sort by tumor dice score
    df_sorted = df[df['has_tumor']].sort_values('tumor_dice', ascending=False)
    
    # Get top 25% of slices
    top_tumor = df_sorted.iloc[:max(1, len(df_sorted)//4)]
    bottom_tumor = df_sorted.iloc[-max(1, len(df_sorted)//4):]
    
    # Calculate average attention map for top slices
    if len(top_tumor) > 0:
        top_attn_maps = np.array([r['attention_map'] for r in top_tumor.to_dict('records')])
        avg_top_attn = np.mean(top_attn_maps, axis=0)
        
        im = axes[0, 1].imshow(avg_top_attn, cmap='jet')
        axes[0, 1].set_title(f'Average Attention Map\nHigh Tumor Dice (>{top_tumor["tumor_dice"].min():.2f})', fontsize=14)
        axes[0, 1].axis('off')
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=axes[0, 1])
        cbar.set_label('Average Attention', rotation=270, labelpad=15)
    
    # Plot 3: Box plot comparing attention ratio across dice score categories
    # Create categories
    df['tumor_perf_category'] = pd.cut(
        df['tumor_dice'], 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low (0-0.3)', 'Medium (0.3-0.6)', 'High (0.6-1.0)']
    )
    
    # Only use slices with tumor for the boxplot
    tumor_df = df[df['has_tumor']]
    
    sns.boxplot(
        data=tumor_df, x='tumor_perf_category', y='attention_ratio',
        ax=axes[1, 0], palette='Blues'
    )
    
    sns.stripplot(
        data=tumor_df, x='tumor_perf_category', y='attention_ratio',
        ax=axes[1, 0], color='black', alpha=0.5, size=5, jitter=True
    )
    
    axes[1, 0].set_title('Attention Ratio by Tumor Dice Score Category', fontsize=14)
    axes[1, 0].set_xlabel('Tumor Dice Score Category', fontsize=12)
    axes[1, 0].set_ylabel('Attention Ratio (tumor/background)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Attention heatmap for low-performing tumor segmentation
    if len(bottom_tumor) > 0:
        bottom_attn_maps = np.array([r['attention_map'] for r in bottom_tumor.to_dict('records')])
        avg_bottom_attn = np.mean(bottom_attn_maps, axis=0)
        
        im = axes[1, 1].imshow(avg_bottom_attn, cmap='jet')
        axes[1, 1].set_title(f'Average Attention Map\nLow Tumor Dice (<{bottom_tumor["tumor_dice"].max():.2f})', fontsize=14)
        axes[1, 1].axis('off')
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Average Attention', rotation=270, labelpad=15)
    
    plt.suptitle(f'Case {case_id}: Relationship Between Attention Focus and Tumor Segmentation Accuracy', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = os.path.join(results_dir, f'attention_tumor_correlation_{case_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def calculate_attention_metrics(attention_map, segmentation):
    """Calculate metrics for attention concentration in different regions."""
    # Ensure all inputs are numpy arrays
    if torch.is_tensor(segmentation):
        segmentation = segmentation.cpu().numpy()
    
    # Create masks for different regions
    kidney_mask = (segmentation == 1)
    tumor_mask = (segmentation == 2)
    background_mask = (segmentation == 0)
    
    # Calculate average attention in each region
    attention_kidney = attention_map[kidney_mask].mean() if kidney_mask.sum() > 0 else 0
    attention_tumor = attention_map[tumor_mask].mean() if tumor_mask.sum() > 0 else 0
    attention_background = attention_map[background_mask].mean() if background_mask.sum() > 0 else 0
    
    # Calculate attention concentration ratios
    attention_ratio_kidney = (attention_kidney / attention_background) if attention_background > 0 else 0
    attention_ratio_tumor = (attention_tumor / attention_background) if attention_background > 0 else 0
    
    # Calculate region sizes 
    kidney_size = kidney_mask.sum()
    tumor_size = tumor_mask.sum()
    
    # Check if regions exist
    has_kidney = kidney_size > 0
    has_tumor = tumor_size > 0
    
    return {
        'attention_kidney': attention_kidney,
        'attention_tumor': attention_tumor,
        'attention_background': attention_background,
        'attention_ratio_kidney': attention_ratio_kidney,
        'attention_ratio_tumor': attention_ratio_tumor,
        'has_kidney': has_kidney,
        'has_tumor': has_tumor,
        'kidney_size': kidney_size,
        'tumor_size': tumor_size
    }

def generate_key_figures(args):
    """Generate key informative figures about attention-segmentation correlation."""
    # Set up results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'key_figures_{args.case_id}_{timestamp}')
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
    images, labels, slice_indices = load_case_data(
        args.case_id, args.root_path, args.img_size
    )
    
    logger.info(f"Found {len(images)} slices with kidney or tumor")
    
    # Process each slice and collect metrics
    results = []
    
    logger.info("Processing slices and collecting metrics...")
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
                avg_dice = (kidney_dice + tumor_dice) / 2.0
                
                # Process transformer attention (average across heads of last layer)
                attention_map = process_attention_transformer(
                    attention_maps['transformer'], args.img_size
                )
                
                # Calculate attention metrics
                attention_metrics = calculate_attention_metrics(
                    attention_map, label.cpu().numpy()
                )
                
                # For slices with tumor, use the tumor attention ratio
                # For slices with only kidney, use the kidney attention ratio
                if attention_metrics['has_tumor']:
                    attention_ratio = attention_metrics['attention_ratio_tumor']
                else:
                    attention_ratio = attention_metrics['attention_ratio_kidney']
                
                # Store results
                results.append({
                    'slice_idx': slice_idx,
                    'image': image,
                    'label': label,
                    'pred': pred,
                    'kidney_dice': kidney_dice,
                    'tumor_dice': tumor_dice,
                    'avg_dice': avg_dice,
                    'attention_map': attention_map,
                    'attention_ratio': attention_ratio,
                    **attention_metrics
                })
                
            except Exception as e:
                logger.error(f"Error processing slice {i}: {e}")
                continue
    
    # Sort results by average Dice score
    results.sort(key=lambda x: x['avg_dice'])
    
    # Select examples for the comparison figure
    low_performing = results[:min(5, len(results)//4)]  # Bottom quarter
    high_performing = results[-min(5, len(results)//4):]  # Top quarter
    
    logger.info("Creating attention-segmentation comparison figure...")
    comparison_path = create_attention_segmentation_comparison(
        high_performing, low_performing, args.case_id, results_dir
    )
    
    logger.info("Creating correlation analysis figure...")
    correlation_path = create_correlation_figure(results, results_dir, args.case_id)
    
    logger.info(f"Key figures generated and saved to: {results_dir}")
    logger.info(f"Attention-Segmentation Comparison: {comparison_path}")
    logger.info(f"Correlation Analysis: {correlation_path}")
    
    return results_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate key figures showing attention-segmentation correlation')
    parser.add_argument('--root_path', type=str, default='kits19/data',
                        help='root dir for KiTS19 data')
    parser.add_argument('--output_dir', type=str, default='./key_figures',
                        help='output dir')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained model weights')
    parser.add_argument('--case_id', type=str, default='00086',
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
    
    results_dir = generate_key_figures(args)
    print(f"Results saved to: {results_dir}")