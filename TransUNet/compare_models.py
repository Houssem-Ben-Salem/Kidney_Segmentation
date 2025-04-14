import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_kits19_list import KiTS19DatasetList
import matplotlib.pyplot as plt
from medpy import metric
import argparse
import logging
from datetime import datetime
import seaborn as sns
import pandas as pd
import random

def calculate_metrics(pred, gt, class_idx):
    """
    Calculate metrics for a specific class
    pred and gt are binary masks (0 or 1)
    """
    # Convert to binary maps for the specific class
    pred_binary = (pred == class_idx).astype(int)
    gt_binary = (gt == class_idx).astype(int)
    
    if gt_binary.sum() == 0 and pred_binary.sum() == 0:
        return 1.0, 1.0, 1.0, 1.0  # Perfect prediction for empty case
    
    if gt_binary.sum() == 0 or pred_binary.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0  # Avoid division by zero
    
    # Calculate metrics
    dice = metric.binary.dc(pred_binary, gt_binary)
    jc = metric.binary.jc(pred_binary, gt_binary)  # Jaccard coefficient
    hd95 = metric.binary.hd95(pred_binary, gt_binary)
    asd = metric.binary.asd(pred_binary, gt_binary)  # Average surface distance
    
    return dice, jc, hd95, asd

def visualize_comparison(image, pred1, pred2, gt, save_path, metrics1, metrics2):
    """
    Create and save a visualization of two model predictions vs ground truth
    """
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
    
    # Model 1 Prediction
    pred1_vis = np.zeros_like(pred1, dtype=np.float32)
    pred1_vis[pred1 == 1] = 0.5  # Kidney in different color
    pred1_vis[pred1 == 2] = 1.0  # Tumor in different color
    ax3.imshow(pred1_vis, cmap='viridis')
    title1 = f'Model 1 Prediction\nKidney Dice: {metrics1["kidney"]["dice"]:.2f}\nTumor Dice: {metrics1["tumor"]["dice"]:.2f}'
    ax3.set_title(title1)
    ax3.axis('off')
    
    # Model 2 Prediction
    pred2_vis = np.zeros_like(pred2, dtype=np.float32)
    pred2_vis[pred2 == 1] = 0.5  # Kidney in different color
    pred2_vis[pred2 == 2] = 1.0  # Tumor in different color
    ax4.imshow(pred2_vis, cmap='viridis')
    title2 = f'Model 2 Prediction\nKidney Dice: {metrics2["kidney"]["dice"]:.2f}\nTumor Dice: {metrics2["tumor"]["dice"]:.2f}'
    ax4.set_title(title2)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def compare_models(args):
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'model_comparison_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(results_dir, 'comparison_log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Initialize models with their specific configurations
    
    # Model 1 config - with attention gates
    config_vit1 = CONFIGS_ViT_seg[args.vit_name]
    config_vit1.n_classes = args.num_classes
    config_vit1.n_skip = args.n_skip
    config_vit1.use_attention = True  # First model uses attention gates
    
    if args.vit_name.find('R50') != -1:
        config_vit1.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                 int(args.img_size / args.vit_patches_size))
    
    # Model 2 config - standard model
    config_vit2 = CONFIGS_ViT_seg[args.vit_name]
    config_vit2.n_classes = args.num_classes
    config_vit2.n_skip = args.n_skip
    config_vit2.use_attention = False  # Second model doesn't use attention gates
    
    if args.vit_name.find('R50') != -1:
        config_vit2.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                 int(args.img_size / args.vit_patches_size))
    
    # Model 1 - with attention gates
    model1 = ViT_seg(config_vit1, img_size=args.img_size, num_classes=config_vit1.n_classes).cuda()
    
    # Load model weights
    try:
        model1.load_state_dict(torch.load(args.model1_path))
    except RuntimeError as e:
        logging.error(f"Error loading model1: {e}")
        logging.info("Attempting to load with strict=False")
        model1.load_state_dict(torch.load(args.model1_path), strict=False)
    
    model1.eval()
    
    # Model 2 - standard model without attention gates
    model2 = ViT_seg(config_vit2, img_size=args.img_size, num_classes=config_vit2.n_classes).cuda()
    
    # Load model weights
    try:
        model2.load_state_dict(torch.load(args.model2_path))
    except RuntimeError as e:
        logging.error(f"Error loading model2: {e}")
        logging.info("Attempting to load with strict=False")
        model2.load_state_dict(torch.load(args.model2_path), strict=False)
    
    model2.eval()
    
    # Initialize test dataset
    test_dataset = KiTS19DatasetList(
        list_file=os.path.join(args.list_dir, "test.txt"),
        base_dir=args.root_path,
        slice_size=(args.img_size, args.img_size),
        augment=False
    )
    
    # Calculate number of samples to use based on percentage
    total_samples = len(test_dataset)
    num_samples = int((args.test_percentage / 100.0) * total_samples)
    indices = list(range(total_samples))
    random.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    # Create a subset of the dataset
    from torch.utils.data import Subset
    test_dataset = Subset(test_dataset, selected_indices)
    
    logging.info(f"Using {num_samples} samples out of {total_samples} "
                f"({args.test_percentage}% of test set)")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one slice at a time
        shuffle=False,
        num_workers=4
    )
    
    # Storage for model predictions and metrics
    slice_metrics = []
    
    # Testing loop - First pass to calculate metrics for both models
    with torch.no_grad():
        logging.info("First pass: calculating metrics for both models...")
        for idx, batch in enumerate(tqdm(test_loader, desc='Calculating metrics')):
            image, label = batch['image'].cuda(), batch['label'].cuda()
            case_id = batch['case_id'][0]
            slice_idx = batch['slice_idx'].item()
            
            # Forward pass for both models
            output1 = model1(image)
            output2 = model2(image)
            
            # Get predictions
            output1 = F.softmax(output1, dim=1)
            output2 = F.softmax(output2, dim=1)
            
            pred1 = torch.argmax(output1, dim=1).cpu().numpy()[0]
            pred2 = torch.argmax(output2, dim=1).cpu().numpy()[0]
            label = label.cpu().numpy()[0]
            
            # Calculate metrics for both models
            metrics1 = {"kidney": {}, "tumor": {}}
            metrics2 = {"kidney": {}, "tumor": {}}
            
            for class_idx, class_name in [(1, 'kidney'), (2, 'tumor')]:
                # Metrics for model 1
                dice1, jc1, hd95_1, asd1 = calculate_metrics(pred1, label, class_idx)
                metrics1[class_name]["dice"] = dice1
                metrics1[class_name]["jaccard"] = jc1
                metrics1[class_name]["hd95"] = hd95_1
                metrics1[class_name]["asd"] = asd1
                
                # Metrics for model 2
                dice2, jc2, hd95_2, asd2 = calculate_metrics(pred2, label, class_idx)
                metrics2[class_name]["dice"] = dice2
                metrics2[class_name]["jaccard"] = jc2
                metrics2[class_name]["hd95"] = hd95_2
                metrics2[class_name]["asd"] = asd2
            
            # Calculate performance difference (model1 - model2)
            kidney_dice_diff = metrics1["kidney"]["dice"] - metrics2["kidney"]["dice"]
            tumor_dice_diff = metrics1["tumor"]["dice"] - metrics2["tumor"]["dice"]
            
            # Store slice info with metrics
            slice_metrics.append({
                "case_id": case_id,
                "slice_idx": slice_idx,
                "image": image.cpu().numpy()[0, 0],
                "label": label,
                "pred1": pred1,
                "pred2": pred2,
                "metrics1": metrics1,
                "metrics2": metrics2,
                "kidney_dice_diff": kidney_dice_diff,
                "tumor_dice_diff": tumor_dice_diff,
                # Overall performance difference
                "overall_diff": kidney_dice_diff + tumor_dice_diff
            })
    
    # Filter slices where the difference is close to the target gap
    target_gap = args.target_difference
    gap_tolerance = args.gap_tolerance
    
    # Filter slices where either kidney or tumor dice has a difference close to target_gap
    filtered_slices = []
    for s in slice_metrics:
        # Check if the difference for either class is close to our target
        if (abs(s["kidney_dice_diff"] - target_gap) <= gap_tolerance or 
            abs(s["tumor_dice_diff"] - target_gap) <= gap_tolerance):
            filtered_slices.append(s)
    
    # Sort the filtered slices by how close they are to our target gap
    filtered_slices.sort(key=lambda x: min(
        abs(x["kidney_dice_diff"] - target_gap),
        abs(x["tumor_dice_diff"] - target_gap)
    ))
    
    # If we didn't find any slices with the target gap, fallback to the original approach
    if not filtered_slices:
        logging.warning(f"No slices found with a difference close to {target_gap}. "
                       f"Falling back to slices with the largest differences.")
        # Sort slices by overall performance difference
        slice_metrics.sort(key=lambda x: x["overall_diff"], reverse=True)
        filtered_slices = slice_metrics
    
    # Visualize slices with the target gap
    top_n = min(args.num_visualizations, len(filtered_slices))
    logging.info(f"Visualizing {top_n} slices with performance difference closest to {target_gap}...")
    
    for i in range(top_n):
        slice_data = filtered_slices[i]
        if slice_data["overall_diff"] <= 0:
            logging.info("No more slices where Model 1 outperforms Model 2.")
            continue
            
        # Create the visualization
        save_path = os.path.join(
            vis_dir, 
            f'case_{slice_data["case_id"]}_slice_{slice_data["slice_idx"]}_'
            f'diff_{slice_data["overall_diff"]:.2f}.png'
        )
        
        visualize_comparison(
            slice_data["image"], 
            slice_data["pred1"], 
            slice_data["pred2"], 
            slice_data["label"], 
            save_path,
            slice_data["metrics1"],
            slice_data["metrics2"]
        )
        
        # Log details
        logging.info(f"Visualization {i+1}/{top_n}: Case {slice_data['case_id']}, "
                    f"Slice {slice_data['slice_idx']}")
        logging.info(f"  Model 1 - Kidney Dice: {slice_data['metrics1']['kidney']['dice']:.4f}, "
                    f"Tumor Dice: {slice_data['metrics1']['tumor']['dice']:.4f}")
        logging.info(f"  Model 2 - Kidney Dice: {slice_data['metrics2']['kidney']['dice']:.4f}, "
                    f"Tumor Dice: {slice_data['metrics2']['tumor']['dice']:.4f}")
        logging.info(f"  Difference - Kidney: {slice_data['kidney_dice_diff']:.4f}, "
                    f"Tumor: {slice_data['tumor_dice_diff']:.4f}")
    
    # Create summary dataframe and save to CSV
    summary_data = []
    for slice_data in slice_metrics:
        summary_data.append({
            "case_id": slice_data["case_id"],
            "slice_idx": slice_data["slice_idx"],
            "model1_kidney_dice": slice_data["metrics1"]["kidney"]["dice"],
            "model1_tumor_dice": slice_data["metrics1"]["tumor"]["dice"],
            "model2_kidney_dice": slice_data["metrics2"]["kidney"]["dice"],
            "model2_tumor_dice": slice_data["metrics2"]["tumor"]["dice"],
            "kidney_dice_diff": slice_data["kidney_dice_diff"],
            "tumor_dice_diff": slice_data["tumor_dice_diff"],
            "overall_diff": slice_data["overall_diff"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'model_comparison_summary.csv'), index=False)
    
    # Create and save comparison box plots
    plt.figure(figsize=(12, 8))
    data = {
        'Model 1 Kidney Dice': [s["metrics1"]["kidney"]["dice"] for s in slice_metrics],
        'Model 2 Kidney Dice': [s["metrics2"]["kidney"]["dice"] for s in slice_metrics],
        'Model 1 Tumor Dice': [s["metrics1"]["tumor"]["dice"] for s in slice_metrics],
        'Model 2 Tumor Dice': [s["metrics2"]["tumor"]["dice"] for s in slice_metrics]
    }
    
    df = pd.DataFrame(data)
    df_melted = pd.melt(df)
    
    sns.boxplot(data=df_melted, x='variable', y='value')
    plt.title('Distribution of Dice Scores - Model 1 vs Model 2')
    plt.xlabel('Metric')
    plt.ylabel('Dice Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dice_comparison_boxplot.png'))
    plt.close()
    
    logging.info(f"All results saved to {results_dir}")
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='kits19/data',
                      help='root dir for KiTS19 data')
    parser.add_argument('--list_dir', type=str, default='./lists_kits19',
                      help='list dir')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                      help='output dir')
    parser.add_argument('--model1_path', type=str, required=True,
                      help='path to first model weights')
    parser.add_argument('--model2_path', type=str, required=True,
                      help='path to second model weights')
    parser.add_argument('--num_classes', type=int, default=3,
                      help='output channel of network')
    parser.add_argument('--img_size', type=int, default=224,
                      help='input patch size')
    parser.add_argument('--n_skip', type=int, default=3,
                      help='number of skip connections')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                      help='select one vit model')
    # Removed the use_attention argument since it's now hardcoded for each model
    parser.add_argument('--vit_patches_size', type=int,
                      default=16, help='ViT patch size')
    parser.add_argument('--test_percentage', type=float,
                      default=100.0, help='Percentage of test set to use (0-100)')
    parser.add_argument('--num_visualizations', type=int,
                      default=10, help='Number of best predictions to visualize')
    parser.add_argument('--target_difference', type=float,
                      default=0.5, help='Target Dice score difference to look for')
    parser.add_argument('--gap_tolerance', type=float,
                      default=0.1, help='Tolerance around target difference')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_percentage <= 0 or args.test_percentage > 100:
        raise ValueError("test_percentage must be between 0 and 100")
    if args.num_visualizations <= 0:
        raise ValueError("num_visualizations must be positive")
    if args.target_difference < 0 or args.target_difference > 1:
        raise ValueError("target_difference must be between 0 and 1")
    if args.gap_tolerance < 0 or args.gap_tolerance > 1:
        raise ValueError("gap_tolerance must be between 0 and 1")
        
    compare_models(args)