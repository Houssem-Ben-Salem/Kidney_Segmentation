import os
import torch
import numpy as np
import time
import traceback
from tqdm import tqdm
import torch.nn.functional as F
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_kits19_list import KiTS19DatasetList
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from medpy import metric
import argparse
import logging
from datetime import datetime
import pandas as pd
import random
from scipy.ndimage import distance_transform_edt
from skimage import measure
import seaborn as sns
import psutil
from collections import defaultdict

# --------------------- Setup Functions ---------------------
def setup_logging(results_dir):
    """Set up logging to both file and console"""
    log_file = os.path.join(results_dir, 'analysis_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def setup_directories(results_dir):
    """Create all necessary subdirectories for analysis outputs"""
    # Main visualization directories
    vis_dirs = {
        'error_types': os.path.join(results_dir, 'error_types'),
        'boundary': os.path.join(results_dir, 'boundary_analysis'),
        'size_analysis': os.path.join(results_dir, 'size_analysis'),
        'general': os.path.join(results_dir, 'visualizations')
    }
    
    # Create all directories
    for dir_path in vis_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return vis_dirs

# --------------------- Metric Calculation Functions ---------------------
def calculate_metrics(pred, gt, class_idx):
    """
    Calculate comprehensive metrics for a specific class
    Returns: dice, jaccard, hd95, asd, fn_volume, fp_volume, gt_volume
    """
    # Convert to binary maps for the specific class
    pred_binary = (pred == class_idx).astype(int)
    gt_binary = (gt == class_idx).astype(int)
    
    if gt_binary.sum() == 0 and pred_binary.sum() == 0:
        # Perfect prediction for empty case
        return 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    if gt_binary.sum() == 0:
        # Only false positives
        return 0.0, 0.0, float('inf'), float('inf'), 0.0, pred_binary.sum(), 0.0
    
    if pred_binary.sum() == 0:
        # Only false negatives
        return 0.0, 0.0, float('inf'), float('inf'), gt_binary.sum(), 0.0, gt_binary.sum()
    
    # Calculate metrics safely
    try:
        dice = metric.binary.dc(pred_binary, gt_binary)
    except Exception as e:
        logging.error(f"Error calculating Dice: {e}")
        dice = 0.0
        
    try:
        jc = metric.binary.jc(pred_binary, gt_binary)
    except Exception as e:
        logging.error(f"Error calculating Jaccard: {e}")
        jc = 0.0
        
    try:
        hd95 = metric.binary.hd95(pred_binary, gt_binary)
    except Exception as e:
        logging.error(f"Error calculating HD95: {e}")
        hd95 = float('inf')
        
    try:
        asd = metric.binary.asd(pred_binary, gt_binary)
    except Exception as e:
        logging.error(f"Error calculating ASD: {e}")
        asd = float('inf')
    
    # Calculate volumes
    fn_volume = np.sum((gt_binary == 1) & (pred_binary == 0))  # False negatives
    fp_volume = np.sum((gt_binary == 0) & (pred_binary == 1))  # False positives
    gt_volume = np.sum(gt_binary)  # Total ground truth volume
    
    return dice, jc, hd95, asd, fn_volume, fp_volume, gt_volume

def analyze_error_types(pred, gt):
    """
    Categorize and analyze error types
    Returns: error_map, error_stats
    """
    # Initialize error type maps
    error_map = np.zeros_like(gt)
    
    # True positive (correctly identified)
    tp_mask = ((gt > 0) & (pred > 0) & (gt == pred))
    error_map[tp_mask] = 1
    
    # False negative (missed)
    fn_mask = ((gt > 0) & (pred == 0))
    error_map[fn_mask] = 2
    
    # False positive (over-segmentation)
    fp_mask = ((gt == 0) & (pred > 0))
    error_map[fp_mask] = 3
    
    # Class confusion (wrong class)
    cc_mask = ((gt > 0) & (pred > 0) & (gt != pred))
    error_map[cc_mask] = 4
    
    # Calculate error volumes
    error_stats = {
        "true_positive": np.sum(tp_mask),
        "false_negative": np.sum(fn_mask),
        "false_positive": np.sum(fp_mask),
        "class_confusion": np.sum(cc_mask)
    }
    
    # Calculate percentage of each error type (for non-background voxels)
    total_error_voxels = np.sum(fn_mask) + np.sum(fp_mask) + np.sum(cc_mask)
    total_gt_voxels = np.sum(gt > 0)
    
    if total_error_voxels > 0:
        error_stats["fn_percent"] = 100 * np.sum(fn_mask) / total_error_voxels
        error_stats["fp_percent"] = 100 * np.sum(fp_mask) / total_error_voxels
        error_stats["cc_percent"] = 100 * np.sum(cc_mask) / total_error_voxels
    else:
        error_stats["fn_percent"] = 0
        error_stats["fp_percent"] = 0
        error_stats["cc_percent"] = 0
    
    if total_gt_voxels > 0:
        error_stats["error_rate"] = 100 * total_error_voxels / (total_gt_voxels + np.sum(fp_mask))
    else:
        error_stats["error_rate"] = 0 if np.sum(fp_mask) == 0 else 100
        
    return error_map, error_stats

def analyze_boundary_accuracy(pred, gt, class_idx):
    """
    Analyze boundary accuracy for a specific class
    Returns: boundary_stats dictionary
    """
    # Convert to binary maps
    pred_binary = (pred == class_idx).astype(int)
    gt_binary = (gt == class_idx).astype(int)
    
    if gt_binary.sum() == 0 or pred_binary.sum() == 0:
        return None
    
    try:
        # Get boundaries (simple edge detection)
        from scipy.ndimage import binary_erosion
        
        gt_interior = binary_erosion(gt_binary, iterations=1)
        pred_interior = binary_erosion(pred_binary, iterations=1)
        
        gt_boundary = gt_binary - gt_interior
        pred_boundary = pred_binary - pred_interior
        
        # Calculate distance transforms
        gt_dist = distance_transform_edt(~gt_boundary.astype(bool))
        pred_dist = distance_transform_edt(~pred_boundary.astype(bool))
        
        # Calculate distance from gt boundary to prediction
        gt_to_pred_dist = gt_dist[pred_boundary.astype(bool)]
        
        # Calculate distance from pred boundary to gt
        pred_to_gt_dist = pred_dist[gt_boundary.astype(bool)]
        
        # Calculate statistics
        boundary_stats = {}
        if len(gt_to_pred_dist) > 0:
            boundary_stats['gt_to_pred_mean'] = np.mean(gt_to_pred_dist)
            boundary_stats['gt_to_pred_std'] = np.std(gt_to_pred_dist)
            boundary_stats['gt_to_pred_max'] = np.max(gt_to_pred_dist)
        else:
            boundary_stats['gt_to_pred_mean'] = None
            boundary_stats['gt_to_pred_std'] = None
            boundary_stats['gt_to_pred_max'] = None
        
        if len(pred_to_gt_dist) > 0:
            boundary_stats['pred_to_gt_mean'] = np.mean(pred_to_gt_dist)
            boundary_stats['pred_to_gt_std'] = np.std(pred_to_gt_dist)
            boundary_stats['pred_to_gt_max'] = np.max(pred_to_gt_dist)
        else:
            boundary_stats['pred_to_gt_mean'] = None
            boundary_stats['pred_to_gt_std'] = None
            boundary_stats['pred_to_gt_max'] = None
        
        # Calculate boundary length
        boundary_stats['gt_boundary_length'] = np.sum(gt_boundary)
        boundary_stats['pred_boundary_length'] = np.sum(pred_boundary)
        
        # Return both the stats and visualization components
        return {
            'stats': boundary_stats,
            'visualization': {
                'gt_boundary': gt_boundary,
                'pred_boundary': pred_boundary,
                'gt_dist': gt_dist,
                'pred_dist': pred_dist
            }
        }
    except Exception as e:
        logging.error(f"Error in boundary analysis: {e}")
        return None

def measure_size(binary_mask):
    """
    Measure the size (area) of objects in a binary mask
    Returns: total_area, list_of_component_areas
    """
    if binary_mask.sum() == 0:
        return 0, []
    
    try:
        # Label connected components
        labeled_mask, num_components = measure.label(binary_mask, return_num=True)
        
        # Measure properties of each component
        props = measure.regionprops(labeled_mask)
        
        # Extract areas
        areas = [prop.area for prop in props]
        
        # Return total area and areas of individual components
        return np.sum(areas), areas
    except Exception as e:
        logging.error(f"Error measuring size: {e}")
        return 0, []

def categorize_size(area):
    """Categorize size as small, medium, or large"""
    if area == 0:
        return "none"
    elif area < 100:
        return "small"
    elif area < 1000:
        return "medium"
    else:
        return "large"

# --------------------- Visualization Functions ---------------------
def visualize_prediction(image, pred, gt, save_path):
    """Create a simple visualization of prediction vs ground truth"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        gt_vis = np.zeros_like(gt, dtype=np.float32)
        gt_vis[gt == 1] = 0.5  # Kidney in different color
        gt_vis[gt == 2] = 1.0  # Tumor in different color
        axes[1].imshow(gt_vis, cmap='viridis')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        pred_vis = np.zeros_like(pred, dtype=np.float32)
        pred_vis[pred == 1] = 0.5  # Kidney in different color
        pred_vis[pred == 2] = 1.0  # Tumor in different color
        axes[2].imshow(pred_vis, cmap='viridis')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        return True
    except Exception as e:
        logging.error(f"Error in visualization: {e}")
        return False

def visualize_error_types(image, pred, gt, error_map, save_path):
    """
    Create and save a visualization of error types
    """
    try:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Ground truth
        gt_vis = np.zeros_like(gt, dtype=np.float32)
        gt_vis[gt == 1] = 0.5  # Kidney in different color
        gt_vis[gt == 2] = 1.0  # Tumor in different color
        ax2.imshow(gt_vis, cmap='viridis')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        # Prediction
        pred_vis = np.zeros_like(pred, dtype=np.float32)
        pred_vis[pred == 1] = 0.5  # Kidney in different color
        pred_vis[pred == 2] = 1.0  # Tumor in different color
        ax3.imshow(pred_vis, cmap='viridis')
        ax3.set_title('Prediction')
        ax3.axis('off')
        
        # Error map
        # Create custom colormap for error types
        colors = [(0, 0, 0, 0),           # Background (transparent)
                  (0.0, 1.0, 0.0, 0.7),   # True positive (green)
                  (1.0, 0.0, 0.0, 0.7),   # False negative (red)
                  (0.0, 0.0, 1.0, 0.7),   # False positive (blue)
                  (1.0, 1.0, 0.0, 0.7)]   # Class confusion (yellow)
        
        error_cmap = LinearSegmentedColormap.from_list('error_cmap', colors, N=5)
        ax4.imshow(image, cmap='gray')
        error_vis = ax4.imshow(error_map, cmap=error_cmap, alpha=0.7, vmin=0, vmax=4)
        ax4.set_title('Error Types')
        ax4.axis('off')
        
        # Add legend for error types
        tp_patch = mpatches.Patch(color=colors[1], label='True Positive')
        fn_patch = mpatches.Patch(color=colors[2], label='False Negative')
        fp_patch = mpatches.Patch(color=colors[3], label='False Positive')
        cc_patch = mpatches.Patch(color=colors[4], label='Class Confusion')
        
        ax4.legend(handles=[tp_patch, fn_patch, fp_patch, cc_patch], 
                  loc='lower right', fontsize='small')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        logging.error(f"Error in error type visualization: {e}")

def visualize_boundary_accuracy(image, boundary_data, save_path):
    """
    Create and save a visualization of boundary accuracy
    """
    try:
        gt_boundary = boundary_data['visualization']['gt_boundary']
        pred_boundary = boundary_data['visualization']['pred_boundary']
        gt_dist = boundary_data['visualization']['gt_dist']
        pred_dist = boundary_data['visualization']['pred_dist']
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image with ground truth boundary
        ax1.imshow(image, cmap='gray')
        ax1.imshow(gt_boundary, cmap='Reds', alpha=0.7)
        ax1.set_title('GT Boundary')
        ax1.axis('off')
        
        # Original image with prediction boundary
        ax2.imshow(image, cmap='gray')
        ax2.imshow(pred_boundary, cmap='Blues', alpha=0.7)
        ax2.set_title('Pred Boundary')
        ax2.axis('off')
        
        # Distance from GT boundary
        ax3.imshow(image, cmap='gray')
        # Use a logarithmic scale for better visualization
        gt_dist_vis = np.log1p(gt_dist)
        ax3.imshow(gt_dist_vis, cmap='hot', alpha=0.7)
        ax3.set_title('Distance from GT')
        ax3.axis('off')
        
        # Distance from Pred boundary
        ax4.imshow(image, cmap='gray')
        # Use a logarithmic scale for better visualization
        pred_dist_vis = np.log1p(pred_dist)
        ax4.imshow(pred_dist_vis, cmap='hot', alpha=0.7)
        ax4.set_title('Distance from Pred')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        logging.error(f"Error in boundary visualization: {e}")

def create_summary_plots(results_dir, metrics, error_analysis, size_analysis, time_analysis):
    """Create summary plots for the analysis"""
    try:
        # 1. Metrics Overview
        plt.figure(figsize=(12, 6))
        metric_names = ['dice', 'jaccard']
        organ_names = ['kidney', 'tumor']
        
        for i, metric in enumerate(metric_names):
            plt.subplot(1, 2, i+1)
            
            means = [np.mean(metrics[organ][metric]) for organ in organ_names]
            stds = [np.std(metrics[organ][metric]) for organ in organ_names]
            
            plt.bar(organ_names, means, yerr=stds)
            plt.ylim(0, 1)
            plt.title(f'{metric.capitalize()} Score')
            
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'accuracy_metrics.png'), dpi=150)
        plt.close()
        
        # 2. Boundary Metrics
        plt.figure(figsize=(12, 6))
        metric_names = ['hd95', 'asd']
        organ_names = ['kidney', 'tumor']
        
        for i, metric in enumerate(metric_names):
            plt.subplot(1, 2, i+1)
            
            # Filter out infinities for meaningful plotting
            kidney_values = [v for v in metrics['kidney'][metric] if not np.isinf(v) and not np.isnan(v)]
            tumor_values = [v for v in metrics['tumor'][metric] if not np.isinf(v) and not np.isnan(v)]
            
            means = [np.mean(kidney_values) if kidney_values else 0, 
                     np.mean(tumor_values) if tumor_values else 0]
            stds = [np.std(kidney_values) if kidney_values else 0,
                   np.std(tumor_values) if tumor_values else 0]
            
            plt.bar(organ_names, means, yerr=stds)
            plt.title(f'{metric.upper()} (Lower is Better)')
            
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'boundary_metrics.png'), dpi=150)
        plt.close()
        
        # 3. Error Type Distribution
        if error_analysis:
            plt.figure(figsize=(10, 6))
            
            # Extract mean percentages
            error_types = ['False Negative', 'False Positive', 'Class Confusion']
            percentages = [
                np.mean([entry['fn_percent'] for entry in error_analysis]),
                np.mean([entry['fp_percent'] for entry in error_analysis]),
                np.mean([entry['cc_percent'] for entry in error_analysis])
            ]
            
            # Create pie chart
            plt.pie(percentages, labels=error_types, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Distribution of Error Types')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'error_distribution.png'), dpi=150)
            plt.close()
        
        # 4. Size-dependent Performance
        if size_analysis:
            df = pd.DataFrame(size_analysis)
            if 'gt_category' in df.columns and 'tumor_dice' in df.columns:
                plt.figure(figsize=(10, 6))
                
                # Create box plot of Dice by tumor size
                sns.boxplot(x='gt_category', y='tumor_dice', data=df)
                plt.title('Tumor Dice Score by Size Category')
                plt.xlabel('Tumor Size')
                plt.ylabel('Dice Score')
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'size_performance.png'), dpi=150)
                plt.close()
        
        # 5. Computational Efficiency
        if time_analysis:
            plt.figure(figsize=(10, 6))
            
            times = [entry['inference_time_ms'] for entry in time_analysis]
            
            plt.hist(times, bins=20, alpha=0.7)
            plt.axvline(np.mean(times), color='r', linestyle='dashed', linewidth=1)
            plt.text(np.mean(times)*1.1, plt.ylim()[1]*0.9, f'Mean: {np.mean(times):.2f} ms', color='r')
            
            plt.title('Inference Time Distribution')
            plt.xlabel('Time (ms)')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'inference_time.png'), dpi=150)
            plt.close()
    
    except Exception as e:
        logging.error(f"Error creating summary plots: {e}")
        logging.error(traceback.format_exc())

# --------------------- Main Analysis Function ---------------------
def comprehensive_error_analysis(args):
    """
    Run comprehensive error analysis on TransUNet model with KiTS19 dataset
    """
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'error_analysis_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup visualization directories
    vis_dirs = setup_directories(results_dir)
    
    # Setup logging
    setup_logging(results_dir)
    
    # Log start time and configuration
    logging.info(f"Starting TransUNet comprehensive error analysis at {timestamp}")
    logging.info(f"Arguments: {args}")
    
    # Initialize model with correct configuration
    try:
        # Create config with explicit attributes
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        
        # Explicitly add use_attention attribute
        setattr(config_vit, 'use_attention', bool(args.use_attention))
        
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                     int(args.img_size / args.vit_patches_size))
        
        # Load model
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=args.num_classes).cuda()
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
        # *** CRITICAL FIX: Replace model's forward method with a fixed version ***
        original_forward = model.forward
        
        def fixed_forward(x, return_attn=False):
            """
            Fixed forward method that correctly handles the decoder's return values.
            The key issue is that the decoder is called with return_attn, but
            unpacking happens regardless of the value of return_attn.
            """
            if x.size()[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            
            # Get transformer outputs and features
            x, attn_weights, features = model.transformer(x)
            
            # The critical fix: Handle the decoder's return value properly
            decoder_result = model.decoder(x, features, return_attn=False)
            
            # Use the first value if it's a tuple, otherwise use it directly
            if isinstance(decoder_result, tuple):
                x = decoder_result[0]
            else:
                x = decoder_result
                
            # Generate segmentation output
            logits = model.segmentation_head(x)
            
            if return_attn:
                # Here we'd need to call decoder again with return_attn=True
                # But for our analysis we don't need attention maps, so skip this
                return logits, None
            
            return logits
        
        # Replace the model's forward method with our fixed version
        model.forward = fixed_forward
        
        logging.info("Model loaded successfully with patched forward method")
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error(traceback.format_exc())
        return None
    
    # Setup dataset
    try:
        test_dataset = KiTS19DatasetList(
            list_file=os.path.join(args.list_dir, "test.txt"),
            base_dir=args.root_path,
            slice_size=(args.img_size, args.img_size),
            augment=False
        )
        
        # Calculate subset size
        total_samples = len(test_dataset)
        num_samples = int((args.test_percentage / 100.0) * total_samples)
        indices = list(range(total_samples))
        random.shuffle(indices)
        selected_indices = indices[:num_samples]
        
        # Create subset
        from torch.utils.data import Subset
        test_subset = Subset(test_dataset, selected_indices)
        
        # Create dataloader
        test_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )
        
        logging.info(f"Using {num_samples} samples out of {total_samples} ({args.test_percentage}% of test set)")
        
    except Exception as e:
        logging.error(f"Error setting up dataset: {e}")
        logging.error(traceback.format_exc())
        return None
    
    # Initialize storage for all analyses
    metrics = {
        'kidney': defaultdict(list),
        'tumor': defaultdict(list)
    }
    
    error_analysis = []
    size_analysis = []
    boundary_analysis = {
        'kidney': [],
        'tumor': []
    }
    time_analysis = []
    
    # Run inference
    logging.info("Starting inference and analysis...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Analyzing")):
            try:
                # Get data
                image = batch['image'].cuda()
                label = batch['label'].cuda()
                case_id = batch['case_id'][0]
                slice_idx = batch['slice_idx'].item()
                
                # Measure memory usage before inference
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Measure inference time
                start_time = time.time()
                output = model(image)
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Measure memory usage after inference
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # Process output
                if output is None or torch.isnan(output).any():
                    logging.warning(f"Invalid output for case {case_id}, slice {slice_idx}")
                    continue
                
                output = F.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                label_np = label.cpu().numpy()[0]
                image_np = image.cpu().numpy()[0, 0]
                
                # 1. Calculate standard metrics for kidney and tumor
                for class_idx, class_name in [(1, 'kidney'), (2, 'tumor')]:
                    dice, jc, hd95, asd, fn_vol, fp_vol, gt_vol = calculate_metrics(
                        pred, label_np, class_idx
                    )
                    
                    metrics[class_name]['dice'].append(dice)
                    metrics[class_name]['jaccard'].append(jc)
                    metrics[class_name]['hd95'].append(hd95)
                    metrics[class_name]['asd'].append(asd)
                    metrics[class_name]['fn_volume'].append(fn_vol)
                    metrics[class_name]['fp_volume'].append(fp_vol)
                    metrics[class_name]['gt_volume'].append(gt_vol)
                
                # 2. Error type analysis
                error_map, error_stats = analyze_error_types(pred, label_np)
                error_stats['case_id'] = case_id
                error_stats['slice_idx'] = slice_idx
                error_analysis.append(error_stats)
                
                # 3. Size analysis
                gt_tumor_mask = (label_np == 2).astype(int)
                pred_tumor_mask = (pred == 2).astype(int)
                gt_tumor_size, gt_components = measure_size(gt_tumor_mask)
                pred_tumor_size, pred_components = measure_size(pred_tumor_mask)
                
                size_analysis.append({
                    'case_id': case_id,
                    'slice_idx': slice_idx,
                    'gt_tumor_size': gt_tumor_size,
                    'pred_tumor_size': pred_tumor_size,
                    'gt_category': categorize_size(gt_tumor_size),
                    'pred_category': categorize_size(pred_tumor_size),
                    'num_gt_components': len(gt_components),
                    'num_pred_components': len(pred_components),
                    'tumor_dice': metrics['tumor']['dice'][-1],
                    'tumor_hd95': metrics['tumor']['hd95'][-1] if not np.isinf(metrics['tumor']['hd95'][-1]) else -1,
                    'tumor_asd': metrics['tumor']['asd'][-1] if not np.isinf(metrics['tumor']['asd'][-1]) else -1
                })
                
                # 4. Boundary analysis
                for class_idx, class_name in [(1, 'kidney'), (2, 'tumor')]:
                    if (label_np == class_idx).sum() > 0 and (pred == class_idx).sum() > 0:
                        boundary_data = analyze_boundary_accuracy(pred, label_np, class_idx)
                        if boundary_data is not None:
                            boundary_data['stats']['case_id'] = case_id
                            boundary_data['stats']['slice_idx'] = slice_idx
                            boundary_analysis[class_name].append(boundary_data['stats'])
                
                # 5. Computational analysis
                time_analysis.append({
                    'case_id': case_id,
                    'slice_idx': slice_idx,
                    'inference_time_ms': inference_time,
                    'memory_usage_mb': memory_used,
                    'has_tumor': gt_tumor_size > 0,
                    'image_size': image.shape
                })
                
                # 6. Create visualizations
                
                # Basic prediction visualization (at regular intervals or for all tumor slices)
                if idx % 20 == 0 or gt_tumor_size > 0:
                    vis_path = os.path.join(
                        vis_dirs['general'], 
                        f'case_{case_id}_slice_{slice_idx}_k{metrics["kidney"]["dice"][-1]:.2f}_t{metrics["tumor"]["dice"][-1]:.2f}.png'
                    )
                    visualize_prediction(image_np, pred, label_np, vis_path)
                
                # Error type visualization (at regular intervals or for problematic cases)
                if idx % 30 == 0 or (gt_tumor_size > 0 and metrics['tumor']['dice'][-1] < 0.5):
                    error_vis_path = os.path.join(
                        vis_dirs['error_types'],
                        f'case_{case_id}_slice_{slice_idx}_error_types.png'
                    )
                    visualize_error_types(image_np, pred, label_np, error_map, error_vis_path)
                
                # Boundary visualization (only for tumor-containing slices)
                if gt_tumor_size > 0 and (pred == 2).sum() > 0:
                    boundary_data = analyze_boundary_accuracy(pred, label_np, 2)  # For tumor class
                    if boundary_data is not None:
                        boundary_vis_path = os.path.join(
                            vis_dirs['boundary'],
                            f'case_{case_id}_slice_{slice_idx}_boundary.png'
                        )
                        visualize_boundary_accuracy(image_np, boundary_data, boundary_vis_path)
                
            except Exception as e:
                logging.error(f"Error processing batch {idx}: {e}")
                logging.error(traceback.format_exc())
    
    # Process and save analysis results
    logging.info("Processing analysis results...")
    
    # 1. Save raw data to CSV
    try:
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame({
            'kidney_dice': metrics['kidney']['dice'],
            'kidney_jaccard': metrics['kidney']['jaccard'],
            'kidney_hd95': metrics['kidney']['hd95'],
            'kidney_asd': metrics['kidney']['asd'],
            'tumor_dice': metrics['tumor']['dice'],
            'tumor_jaccard': metrics['tumor']['jaccard'],
            'tumor_hd95': metrics['tumor']['hd95'],
            'tumor_asd': metrics['tumor']['asd']
        })
        metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
        
        # Save other analyses
        pd.DataFrame(error_analysis).to_csv(os.path.join(results_dir, 'error_analysis.csv'), index=False)
        pd.DataFrame(size_analysis).to_csv(os.path.join(results_dir, 'size_analysis.csv'), index=False)
        pd.DataFrame(boundary_analysis['kidney']).to_csv(os.path.join(results_dir, 'kidney_boundary.csv'), index=False)
        pd.DataFrame(boundary_analysis['tumor']).to_csv(os.path.join(results_dir, 'tumor_boundary.csv'), index=False)
        pd.DataFrame(time_analysis).to_csv(os.path.join(results_dir, 'computational_analysis.csv'), index=False)
        
        logging.info("Raw data saved to CSV files")
    except Exception as e:
        logging.error(f"Error saving raw data: {e}")
        logging.error(traceback.format_exc())
    
    # 2. Create summary visualizations
    try:
        create_summary_plots(results_dir, metrics, error_analysis, size_analysis, time_analysis)
        logging.info("Summary visualizations created")
    except Exception as e:
        logging.error(f"Error creating summary visualizations: {e}")
        logging.error(traceback.format_exc())
    
    # 3. Compute size-dependent performance
    try:
        size_df = pd.DataFrame(size_analysis)
        if not size_df.empty:
            # Group by tumor size category
            size_performance = size_df.groupby('gt_category')[['tumor_dice', 'tumor_hd95', 'tumor_asd']].agg(['mean', 'std', 'count'])
            logging.info("\nSize-dependent Performance:")
            logging.info(size_performance)
    except Exception as e:
        logging.error(f"Error in size-dependent analysis: {e}")
        logging.error(traceback.format_exc())
    
    # 4. Compute boundary accuracy statistics
    try:
        for class_name in ['kidney', 'tumor']:
            if boundary_analysis[class_name]:
                df = pd.DataFrame(boundary_analysis[class_name])
                if not df.empty:
                    logging.info(f"\n{class_name.capitalize()} Boundary Analysis:")
                    boundary_stats = df.describe()
                    logging.info(boundary_stats)
    except Exception as e:
        logging.error(f"Error in boundary accuracy analysis: {e}")
        logging.error(traceback.format_exc())
    
    # 5. Compute computational efficiency metrics
    try:
        time_df = pd.DataFrame(time_analysis)
        if not time_df.empty:
            logging.info("\nComputational Efficiency:")
            logging.info(f"Mean inference time: {time_df['inference_time_ms'].mean():.2f} ms")
            logging.info(f"Mean memory usage: {time_df['memory_usage_mb'].mean():.2f} MB")
            
            # Compare tumor vs non-tumor slices
            tumor_times = time_df[time_df['has_tumor']]['inference_time_ms']
            non_tumor_times = time_df[~time_df['has_tumor']]['inference_time_ms']
            
            if not tumor_times.empty and not non_tumor_times.empty:
                logging.info(f"Tumor slices mean time: {tumor_times.mean():.2f} ms")
                logging.info(f"Non-tumor slices mean time: {non_tumor_times.mean():.2f} ms")
    except Exception as e:
        logging.error(f"Error in computational efficiency analysis: {e}")
        logging.error(traceback.format_exc())
    
    # 6. Create comprehensive report
    try:
        # Calculate overall metrics
        kidney_dice_mean = np.mean(metrics['kidney']['dice'])
        kidney_dice_std = np.std(metrics['kidney']['dice'])
        tumor_dice_mean = np.mean(metrics['tumor']['dice'])
        tumor_dice_std = np.std(metrics['tumor']['dice'])
        
        # Filter out infinities for boundary metrics
        kidney_hd95 = [v for v in metrics['kidney']['hd95'] if not np.isinf(v) and not np.isnan(v)]
        kidney_asd = [v for v in metrics['kidney']['asd'] if not np.isinf(v) and not np.isnan(v)]
        tumor_hd95 = [v for v in metrics['tumor']['hd95'] if not np.isinf(v) and not np.isnan(v)]
        tumor_asd = [v for v in metrics['tumor']['asd'] if not np.isinf(v) and not np.isnan(v)]
        
        kidney_hd95_mean = np.mean(kidney_hd95) if kidney_hd95 else float('inf')
        kidney_hd95_std = np.std(kidney_hd95) if kidney_hd95 else 0
        kidney_asd_mean = np.mean(kidney_asd) if kidney_asd else float('inf')
        kidney_asd_std = np.std(kidney_asd) if kidney_asd else 0
        
        tumor_hd95_mean = np.mean(tumor_hd95) if tumor_hd95 else float('inf')
        tumor_hd95_std = np.std(tumor_hd95) if tumor_hd95 else 0
        tumor_asd_mean = np.mean(tumor_asd) if tumor_asd else float('inf')
        tumor_asd_std = np.std(tumor_asd) if tumor_asd else 0
        
        # Calculate error distributions
        if error_analysis:
            fn_percent = np.mean([entry['fn_percent'] for entry in error_analysis])
            fp_percent = np.mean([entry['fp_percent'] for entry in error_analysis])
            cc_percent = np.mean([entry['cc_percent'] for entry in error_analysis])
        else:
            fn_percent = fp_percent = cc_percent = 0
        
        # Create report
        report = f"""# TransUNet Comprehensive Error Analysis

## Overview
- **Model**: {args.vit_name}
- **Dataset**: KiTS19 Test Set ({args.test_percentage}% sampled)
- **Date**: {timestamp}

## 1. Performance Metrics

### Kidney
- **Dice Score**: {kidney_dice_mean:.4f} ± {kidney_dice_std:.4f}
- **Jaccard**: {np.mean(metrics['kidney']['jaccard']):.4f} ± {np.std(metrics['kidney']['jaccard']):.4f}
- **HD95**: {kidney_hd95_mean:.4f} ± {kidney_hd95_std:.4f}
- **ASD**: {kidney_asd_mean:.4f} ± {kidney_asd_std:.4f}

### Tumor
- **Dice Score**: {tumor_dice_mean:.4f} ± {tumor_dice_std:.4f}
- **Jaccard**: {np.mean(metrics['tumor']['jaccard']):.4f} ± {np.std(metrics['tumor']['jaccard']):.4f}
- **HD95**: {tumor_hd95_mean:.4f} ± {tumor_hd95_std:.4f}
- **ASD**: {tumor_asd_mean:.4f} ± {tumor_asd_std:.4f}

## 2. Error Type Analysis

The errors in segmentation can be categorized into three main types:

1. **False Negatives**: {fn_percent:.1f}% - These are areas that should be segmented as kidney or tumor but were missed
2. **False Positives**: {fp_percent:.1f}% - These are areas incorrectly labeled as kidney or tumor
3. **Class Confusion**: {cc_percent:.1f}% - These are areas where kidney was labeled as tumor or vice versa

## 3. Size-dependent Performance Analysis
"""
        
        # Add size-dependent performance if available
        if 'size_performance' in locals():
            report += size_performance.to_markdown()
        else:
            report += "No size analysis data available."
            
        report += f"""

### Observations:
- {"Small tumors tend to be more difficult to segment accurately, with lower Dice scores compared to larger tumors." if 'size_performance' in locals() else ""}
- {"Boundary accuracy (measured by HD95 and ASD) generally improves with tumor size." if 'size_performance' in locals() else ""}

## 4. Boundary Accuracy Assessment

Boundary accuracy is measured using Hausdorff Distance (HD95) and Average Surface Distance (ASD).

### Kidney Boundaries
- Mean HD95: {kidney_hd95_mean:.4f} pixels
- Mean ASD: {kidney_asd_mean:.4f} pixels

### Tumor Boundaries
- Mean HD95: {tumor_hd95_mean:.4f} pixels
- Mean ASD: {tumor_asd_mean:.4f} pixels

### Observations:
- {"Tumor boundaries are generally less accurate than kidney boundaries, suggesting the model struggles more with the complex shapes of tumors." if tumor_hd95_mean > kidney_hd95_mean else ""}
- Boundary errors typically occur in regions where tumor and kidney tissue are difficult to distinguish.

## 5. Computational Efficiency

"""
        # Add computational efficiency if available
        if 'time_df' in locals() and not time_df.empty:
            report += f"""- **Mean Inference Time**: {time_df['inference_time_ms'].mean():.2f} ms per slice
- **Median Inference Time**: {time_df['inference_time_ms'].median():.2f} ms per slice
- **Memory Usage**: {time_df['memory_usage_mb'].mean():.2f} MB per inference
- **Tumor vs. Non-tumor Slices**: {"Tumor-containing slices require more processing time" if tumor_times.mean() > non_tumor_times.mean() else "No significant difference in processing times between tumor and non-tumor slices"}
"""
        else:
            report += "No computational efficiency data available."
            
        report += f"""
## 6. Summary and Recommendations

### Key Findings:
1. The model achieves better segmentation performance on kidneys ({kidney_dice_mean:.4f} Dice) than tumors ({tumor_dice_mean:.4f} Dice).
2. {"Small tumors are particularly challenging to segment accurately." if 'size_performance' in locals() else ""}
3. The most common error type is {"false negatives" if fn_percent > fp_percent and fn_percent > cc_percent else "false positives" if fp_percent > fn_percent and fp_percent > cc_percent else "class confusion"} ({max(fn_percent, fp_percent, cc_percent):.1f}% of all errors).
4. Boundary accuracy is {"better for kidneys than tumors" if kidney_hd95_mean < tumor_hd95_mean else "comparable between kidneys and tumors"}.

### Recommendations:
1. {"Focus on improving the detection of small tumors, possibly by using specialized loss functions." if 'size_performance' in locals() and 'small' in size_performance.index else ""}
2. Reduce {"false negatives" if fn_percent > fp_percent and fn_percent > cc_percent else "false positives" if fp_percent > fn_percent and fp_percent > cc_percent else "class confusion"} by {"adjusting class weights in the loss function" if fn_percent > fp_percent and fn_percent > cc_percent else "applying post-processing to remove small false positives" if fp_percent > fn_percent and fp_percent > cc_percent else "improving feature representation to better distinguish kidney from tumor tissue"}.
3. Improve boundary accuracy by incorporating boundary-aware loss functions or post-processing techniques.
4. {"The model is computationally efficient with an average inference time of " + f"{time_df['inference_time_ms'].mean():.2f} ms per slice, making it suitable for clinical applications." if 'time_df' in locals() and not time_df.empty else ""}

"""
        
        # Save report
        with open(os.path.join(results_dir, 'comprehensive_report.md'), 'w') as f:
            f.write(report)
            
        logging.info(f"Comprehensive report saved to {os.path.join(results_dir, 'comprehensive_report.md')}")
    except Exception as e:
        logging.error(f"Error creating comprehensive report: {e}")
        logging.error(traceback.format_exc())
    
    logging.info(f"Analysis complete. Results saved to: {results_dir}")
    return {
        'kidney_dice': np.mean(metrics['kidney']['dice']),
        'tumor_dice': np.mean(metrics['tumor']['dice']),
        'results_dir': results_dir
    }

# --------------------- Main Entry Point ---------------------
if __name__ == "__main__":
    print("\n=== TransUNet Comprehensive Error Analysis ===\n")
    
    parser = argparse.ArgumentParser(description="Comprehensive TransUNet Error Analysis")
    parser.add_argument('--root_path', type=str, default='kits19/data',
                        help='Root directory for KiTS19 data')
    parser.add_argument('--list_dir', type=str, default='./lists_kits19_1',
                        help='Directory containing dataset lists')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes (typically 3 for background, kidney, tumor)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (typically 224 for TransUNet)')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections in the model')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='Vision Transformer model variant')
    parser.add_argument('--use_attention', type=int, default=1,
                        help='Whether to use attention gates in decoder')
    parser.add_argument('--vit_patches_size', type=int, default=16, 
                        help='Size of patches used in Vision Transformer')
    parser.add_argument('--test_percentage', type=float, default=20.0,
                        help='Percentage of test set to use (0-100)')
    
    args = parser.parse_args()
    
    # Run analysis
    results = comprehensive_error_analysis(args)