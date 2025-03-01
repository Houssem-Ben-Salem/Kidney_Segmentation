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

def visualize_prediction(image, pred, gt, save_path):
    """
    Create and save a visualization of the prediction vs ground truth
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
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
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def test_transunet(args):
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'test_results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(results_dir, 'test_log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Initialize model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.use_attention = bool(args.use_attention)
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                 int(args.img_size / args.vit_patches_size))
    
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
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
    
    # Metrics storage
    metrics_dict = {
        'kidney': {'dice': [], 'jaccard': [], 'hd95': [], 'asd': []},
        'tumor': {'dice': [], 'jaccard': [], 'hd95': [], 'asd': []}
    }
    case_metrics = {}  # Store metrics per case
    
    # Testing loop
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            image, label = batch['image'].cuda(), batch['label'].cuda()
            # Get case and slice information directly from batch
            case_id = batch['case_id'][0]  # Now directly available in batch
            slice_idx = batch['slice_idx'].item()  # Now directly available in batch
            
            # Forward pass
            output = model(image)
            
            # Get predictions
            output = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            label = label.cpu().numpy()[0]
            image = image.cpu().numpy()[0, 0]  # Get first channel of first image
            
            # Calculate metrics for kidney (class 1) and tumor (class 2)
            for class_idx, class_name in [(1, 'kidney'), (2, 'tumor')]:
                dice, jc, hd95, asd = calculate_metrics(pred, label, class_idx)
                metrics_dict[class_name]['dice'].append(dice)
                metrics_dict[class_name]['jaccard'].append(jc)
                metrics_dict[class_name]['hd95'].append(hd95)
                metrics_dict[class_name]['asd'].append(asd)
            
            # Store case-wise metrics
            if case_id not in case_metrics:
                case_metrics[case_id] = {
                    'kidney': {'dice': [], 'jaccard': [], 'hd95': [], 'asd': []},
                    'tumor': {'dice': [], 'jaccard': [], 'hd95': [], 'asd': []}
                }
            
            for class_name in ['kidney', 'tumor']:
                for metric_name in ['dice', 'jaccard', 'hd95', 'asd']:
                    case_metrics[case_id][class_name][metric_name].append(
                        metrics_dict[class_name][metric_name][-1]
                    )
            
            # Save visualization if prediction is good (dice > 0.85) or contains tumor
            kidney_dice = metrics_dict['kidney']['dice'][-1]
            has_tumor = 2 in label
            if kidney_dice > 0.85 or has_tumor:
                save_path = os.path.join(
                    vis_dir, 
                    f'case_{case_id}_slice_{slice_idx}_k{kidney_dice:.2f}.png'
                )
                visualize_prediction(image, pred, label, save_path)
    
    # Calculate and log overall metrics
    results = {}
    for class_name in ['kidney', 'tumor']:
        results[class_name] = {}
        for metric_name in ['dice', 'jaccard', 'hd95', 'asd']:
            values = metrics_dict[class_name][metric_name]
            results[class_name][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            }
    
    # Calculate case-wise averages
    case_results = {}
    for case_id in case_metrics:
        case_results[case_id] = {}
        for class_name in ['kidney', 'tumor']:
            case_results[case_id][class_name] = {}
            for metric_name in ['dice', 'jaccard', 'hd95', 'asd']:
                values = case_metrics[case_id][class_name][metric_name]
                if values:  # Only calculate if we have values
                    case_results[case_id][class_name][metric_name] = np.mean(values)
    
    # Log results
    logging.info("\nOverall Results:")
    for class_name in ['kidney', 'tumor']:
        logging.info(f"\n{class_name.upper()} Metrics:")
        for metric_name in ['dice', 'jaccard', 'hd95', 'asd']:
            stats = results[class_name][metric_name]
            logging.info(f"{metric_name.upper()}: "
                        f"Mean = {stats['mean']:.4f} ± {stats['std']:.4f}, "
                        f"Median = {stats['median']:.4f}")
    
    # Create and save box plots
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    for class_name in ['kidney', 'tumor']:
        for metric_name in ['dice', 'jaccard']:
            data.append(metrics_dict[class_name][metric_name])
            labels.extend([f'{class_name}_{metric_name}'] * len(metrics_dict[class_name][metric_name]))
    
    sns.boxplot(data=pd.DataFrame({'value': np.concatenate(data), 'metric': labels}), 
                x='metric', y='value')
    plt.title('Distribution of Dice and Jaccard Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metric_distribution.png'))
    plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(case_results).transpose()
    results_df.to_csv(os.path.join(results_dir, 'case_wise_results.csv'))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='kits19/data',
                        help='root dir for KiTS19 data')
    parser.add_argument('--list_dir', type=str, default='./lists_kits19',
                        help='list dir')
    parser.add_argument('--output_dir', type=str, default='./test_results',
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
    parser.add_argument('--use_attention', type=int, default=0,
                        help='use attention gates in decoder')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='ViT patch size')
    parser.add_argument('--test_percentage', type=float,
                        default=100.0, help='Percentage of test set to use (0-100)')
    parser.add_argument('--num_visualizations', type=int,
                        default=10, help='Number of best predictions to visualize')
    parser.add_argument('--roi_only', type=int, default=0,
                        help='Test only on ROI slices (1) or all slices (0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_percentage <= 0 or args.test_percentage > 100:
        raise ValueError("test_percentage must be between 0 and 100")
    if args.num_visualizations < 0:
        raise ValueError("num_visualizations must be positive")
    if args.roi_only not in [0, 1]:
        raise ValueError("roi_only must be 0 or 1")
        
    results = test_transunet(args)