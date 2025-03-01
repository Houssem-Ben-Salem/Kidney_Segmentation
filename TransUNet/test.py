import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import nibabel as nib
from collections import defaultdict
import seaborn as sns
from datasets.dataset_kits19_list import KiTS19DatasetList
from torch.utils.data import DataLoader

def analyze_dataset(list_file, base_dir):
    """
    Analyze a dataset's distribution characteristics
    """
    # Create dataset
    dataset = KiTS19DatasetList(
        list_file=list_file,
        base_dir=base_dir,
        augment=False
    )
    
    # Initialize statistics collectors
    stats_dict = {
        'class_pixels': defaultdict(int),
        'intensities': [],
        'kidney_sizes': [],
        'tumor_sizes': [],
        'slice_positions': []
    }
    
    # Analyze each sample
    for idx in tqdm(range(len(dataset)), desc=f"Analyzing {list_file}"):
        sample = dataset[idx]
        image, label = sample['image'], sample['label']
        slice_idx = sample['slice_idx']
        
        # Count class pixels
        for class_idx in [0, 1, 2]:  # background, kidney, tumor
            stats_dict['class_pixels'][class_idx] += (label == class_idx).sum().item()
        
        # Collect intensity statistics (from non-zero regions)
        mask = image.numpy() != 0
        if mask.sum() > 0:
            stats_dict['intensities'].extend(image.numpy()[mask].flatten())
        
        # Collect organ sizes
        kidney_pixels = (label == 1).sum().item()
        tumor_pixels = (label == 2).sum().item()
        if kidney_pixels > 0:
            stats_dict['kidney_sizes'].append(kidney_pixels)
        if tumor_pixels > 0:
            stats_dict['tumor_sizes'].append(tumor_pixels)
            
        # Record slice position
        stats_dict['slice_positions'].append(slice_idx)
    
    return stats_dict

def plot_distributions(val_stats, test_stats, output_dir):
    """
    Create comparative plots of distributions
    """
    plt.style.use('seaborn')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(10, 6))
    classes = ['Background', 'Kidney', 'Tumor']
    val_dist = [val_stats['class_pixels'][i] for i in range(3)]
    test_dist = [test_stats['class_pixels'][i] for i in range(3)]
    
    val_total = sum(val_dist)
    test_total = sum(test_dist)
    val_dist = [x/val_total*100 for x in val_dist]
    test_dist = [x/test_total*100 for x in test_dist]
    
    x = np.arange(len(classes))
    width = 0.35
    plt.bar(x - width/2, val_dist, width, label='Validation')
    plt.bar(x + width/2, test_dist, width, label='Test')
    plt.xlabel('Classes')
    plt.ylabel('Percentage')
    plt.title('Class Distribution Comparison')
    plt.xticks(x, classes)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Intensity Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(val_stats['intensities'], bins=50, alpha=0.5, density=True, label='Validation')
    plt.hist(test_stats['intensities'], bins=50, alpha=0.5, density=True, label='Test')
    plt.xlabel('Intensity Values')
    plt.ylabel('Density')
    plt.title('Intensity Distribution Comparison')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'intensity_distribution.png'))
    plt.close()
    
    # 3. Organ Size Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.boxplot(data=[val_stats['kidney_sizes'], test_stats['kidney_sizes']], 
                ax=ax1)
    ax1.set_xticklabels(['Validation', 'Test'])
    ax1.set_title('Kidney Size Distribution')
    ax1.set_ylabel('Number of Pixels')
    
    sns.boxplot(data=[val_stats['tumor_sizes'], test_stats['tumor_sizes']], 
                ax=ax2)
    ax2.set_xticklabels(['Validation', 'Test'])
    ax2.set_title('Tumor Size Distribution')
    ax2.set_ylabel('Number of Pixels')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_distribution.png'))
    plt.close()
    
    # 4. Slice Position Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(val_stats['slice_positions'], bins=30, alpha=0.5, density=True, label='Validation')
    plt.hist(test_stats['slice_positions'], bins=30, alpha=0.5, density=True, label='Test')
    plt.xlabel('Slice Position')
    plt.ylabel('Density')
    plt.title('Slice Position Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'slice_position_distribution.png'))
    plt.close()

def run_statistical_tests(val_stats, test_stats):
    """
    Run statistical tests to compare distributions
    """
    results = {}
    
    # 1. Class Distribution Chi-Square Test
    val_dist = [val_stats['class_pixels'][i] for i in range(3)]
    test_dist = [test_stats['class_pixels'][i] for i in range(3)]
    chi2, p_value = stats.chi2_contingency([val_dist, test_dist])[:2]
    results['class_distribution'] = {
        'test': 'Chi-Square',
        'statistic': chi2,
        'p_value': p_value
    }
    
    # 2. Intensity Distribution KS Test
    ks_stat, p_value = stats.ks_2samp(
        np.random.choice(val_stats['intensities'], 10000),  # Sample for efficiency
        np.random.choice(test_stats['intensities'], 10000)
    )
    results['intensity_distribution'] = {
        'test': 'Kolmogorov-Smirnov',
        'statistic': ks_stat,
        'p_value': p_value
    }
    
    # 3. Organ Size Mann-Whitney U Test
    for organ in ['kidney', 'tumor']:
        stat, p_value = stats.mannwhitneyu(
            val_stats[f'{organ}_sizes'],
            test_stats[f'{organ}_sizes'],
            alternative='two-sided'
        )
        results[f'{organ}_size'] = {
            'test': 'Mann-Whitney U',
            'statistic': stat,
            'p_value': p_value
        }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', type=str, required=True,
                      help='Path to validation list file')
    parser.add_argument('--test_list', type=str, required=True,
                      help='Path to test list file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Root directory containing the data')
    parser.add_argument('--output_dir', type=str, default='./distribution_analysis',
                      help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Analyze datasets
    print("Analyzing validation set...")
    val_stats = analyze_dataset(args.val_list, args.data_dir)
    print("Analyzing test set...")
    test_stats = analyze_dataset(args.test_list, args.data_dir)
    
    # Create visualizations
    print("Creating distribution plots...")
    plot_distributions(val_stats, test_stats, args.output_dir)
    
    # Run statistical tests
    print("Running statistical tests...")
    test_results = run_statistical_tests(val_stats, test_stats)
    
    # Print results
    print("\nStatistical Test Results:")
    print("-" * 50)
    for dist_type, results in test_results.items():
        print(f"\n{dist_type.replace('_', ' ').title()}:")
        print(f"Test: {results['test']}")
        print(f"Statistic: {results['statistic']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
        if results['p_value'] < 0.05:
            print("Conclusion: Distributions are significantly different")
        else:
            print("Conclusion: No significant difference detected")

if __name__ == "__main__":
    main()