import os
import random
import numpy as np
import nibabel as nib
from collections import defaultdict
import matplotlib.pyplot as plt

def get_case_list(data_path, sample_percentage):
    """
    Get a random sample of cases based on percentage
    Args:
        data_path: Path to kits19/data directory
        sample_percentage: Percentage of cases to analyze (0-100)
    """
    # Get all valid cases
    all_cases = [case for case in os.listdir(data_path) 
                 if case.startswith('case_') and 
                 os.path.exists(os.path.join(data_path, case, 'segmentation.nii.gz'))]
    
    # Calculate number of cases to sample
    num_cases = max(1, int(len(all_cases) * sample_percentage / 100))
    
    # Randomly sample cases
    sampled_cases = random.sample(all_cases, num_cases)
    
    print(f"\nAnalyzing {num_cases} cases ({sample_percentage}% of total {len(all_cases)} cases)")
    return sorted(sampled_cases)  # Sort for consistent output

def analyze_dataset(data_path, sample_percentage=100, seed=42):
    """
    Analyze KiTS19 dataset class distribution
    Args:
        data_path: Path to kits19/data directory
        sample_percentage: Percentage of cases to analyze (0-100)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    stats = {
        'total_slices': 0,
        'slices_with_kidney': 0,
        'slices_with_tumor': 0,
        'slices_with_roi': 0,  # Slices with either kidney or tumor
        'slices_background_only': 0,
        'pixel_counts': defaultdict(int),
        'cases_with_tumor': 0,
        'analyzed_cases': []
    }
    
    # Get sampled cases
    cases = get_case_list(data_path, sample_percentage)
    stats['analyzed_cases'] = cases
    
    # Analyze each case
    for case in cases:
        print(f"Processing {case}...")
        seg_path = os.path.join(data_path, case, 'segmentation.nii.gz')
            
        # Load segmentation mask
        seg_data = nib.load(seg_path).get_fdata()
        
        # Update pixel counts
        unique, counts = np.unique(seg_data, return_counts=True)
        for val, count in zip(unique, counts):
            stats['pixel_counts'][int(val)] += count
            
        # Count slices with each class
        stats['total_slices'] += seg_data.shape[2]
        
        for slice_idx in range(seg_data.shape[2]):
            slice_data = seg_data[:,:,slice_idx]
            has_kidney = 1 in slice_data
            has_tumor = 2 in slice_data
            
            if has_kidney:
                stats['slices_with_kidney'] += 1
            if has_tumor:
                stats['slices_with_tumor'] += 1
            if has_kidney or has_tumor:
                stats['slices_with_roi'] += 1
            else:
                stats['slices_background_only'] += 1
                
        # Check if case has tumor
        if 2 in seg_data:
            stats['cases_with_tumor'] += 1
            
    return stats

def visualize_stats(stats):
    """
    Create visualizations of dataset statistics
    """
    # Pixel distribution
    total_pixels = sum(stats['pixel_counts'].values())
    class_names = ['Background', 'Kidney', 'Tumor']
    percentages = [
        (stats['pixel_counts'].get(i, 0) / total_pixels) * 100 
        for i in range(3)
    ]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    bars = plt.bar(class_names, percentages)
    plt.title('Pixel-wise Class Distribution')
    plt.ylabel('Percentage')
    
    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    # Slice statistics
    plt.subplot(1, 2, 2)
    slice_stats = [
        stats['total_slices'],
        stats['slices_with_kidney'],
        stats['slices_with_tumor']
    ]
    bars = plt.bar(['Total', 'With Kidney', 'With Tumor'], slice_stats)
    plt.title('Slice Distribution')
    plt.ylabel('Number of Slices')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_path = 'kits19/data'  # Update this path
    sample_percentage = 20  # Analyze 20% of the data
    
    # Run analysis
    stats = analyze_dataset(data_path, sample_percentage=sample_percentage)
    
    # Print summary statistics
    print("\nDataset Statistics:")
    print(f"Analyzed cases: {', '.join(stats['analyzed_cases'])}")
    print(f"Total number of slices: {stats['total_slices']}")
    print(f"Background-only slices: {stats['slices_background_only']} ({stats['slices_background_only']/stats['total_slices']*100:.1f}%)")
    print(f"Slices with ROI (kidney/tumor): {stats['slices_with_roi']} ({stats['slices_with_roi']/stats['total_slices']*100:.1f}%)")
    
    # Calculate percentages based on ROI slices instead of total slices
    roi_slices = stats['slices_with_roi']
    print(f"\nAmong ROI slices:")
    print(f"Slices containing kidney: {stats['slices_with_kidney']} ({stats['slices_with_kidney']/roi_slices*100:.1f}%)")
    print(f"Slices containing tumor: {stats['slices_with_tumor']} ({stats['slices_with_tumor']/roi_slices*100:.1f}%)")
    print(f"Cases with tumor: {stats['cases_with_tumor']} ({stats['cases_with_tumor']/len(stats['analyzed_cases'])*100:.1f}%)")
    
    # Print class distribution
    total_pixels = sum(stats['pixel_counts'].values())
    print("\nPixel-wise class distribution:")
    for class_id in range(3):  # Ensure we print all classes even if some are missing
        count = stats['pixel_counts'].get(class_id, 0)
        percentage = (count / total_pixels) * 100
        class_name = ['Background', 'Kidney', 'Tumor'][class_id]
        print(f"{class_name}: {percentage:.2f}%")
    
    # Create visualizations
    visualize_stats(stats)