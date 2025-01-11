import os
import random
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict

class DatasetAnalyzer:
    def __init__(self):
        # Regular stats for validation and test, and pre-filtering train stats
        self.stats = defaultdict(lambda: {
            'total_slices': 0,
            'background_only': 0,
            'with_roi': 0,
            'with_kidney': 0,
            'with_tumor': 0,
            'pixel_counts': defaultdict(int)
        })
        
        # Post-filtering stats for training
        self.train_filtered_stats = {
            'total_slices': 0,
            'with_roi': 0,
            'with_kidney': 0,
            'with_tumor': 0,
            'pixel_counts': defaultdict(int)
        }
    
    def analyze_case(self, case_path, split_name, is_filtered=False):
        """Analyze a single case and update statistics."""
        seg_path = os.path.join(case_path, "segmentation.nii.gz")
        segmentation = nib.load(seg_path).get_fdata()
        
        if is_filtered:
            stats = self.train_filtered_stats
        else:
            stats = self.stats[split_name]
            stats['total_slices'] += segmentation.shape[2]
        
        for slice_idx in range(segmentation.shape[2]):
            slice_data = segmentation[:,:,slice_idx]
            has_kidney = 1 in slice_data
            has_tumor = 2 in slice_data
            
            if has_kidney or has_tumor:
                if not is_filtered:
                    stats['with_roi'] += 1
                if has_kidney:
                    stats['with_kidney'] += 1
                if has_tumor:
                    stats['with_tumor'] += 1
                
                if is_filtered:
                    stats['total_slices'] += 1
                    stats['with_roi'] += 1
                    # Update pixel counts for ROI slices only
                    unique, counts = np.unique(slice_data, return_counts=True)
                    for val, count in zip(unique, counts):
                        stats['pixel_counts'][int(val)] += count
            else:
                if not is_filtered:
                    stats['background_only'] += 1
            
            if not is_filtered:
                # Update pixel counts for all slices
                unique, counts = np.unique(slice_data, return_counts=True)
                for val, count in zip(unique, counts):
                    stats['pixel_counts'][int(val)] += count
    
    def plot_distributions(self, split_name, is_filtered=False):
        """Create distribution plots for a specific split."""
        if is_filtered and split_name == 'train':
            stats = self.train_filtered_stats
            title_suffix = " (ROI only)"
            filename_suffix = "_roi_only"
        else:
            stats = self.stats[split_name]
            title_suffix = ""
            filename_suffix = ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Pixel-wise distribution
        total_pixels = sum(stats['pixel_counts'].values())
        class_names = ['Background', 'Kidney', 'Tumor']
        percentages = [
            (stats['pixel_counts'].get(i, 0) / total_pixels) * 100 
            for i in range(3)
        ]
        
        bars = ax1.bar(class_names, percentages)
        ax1.set_title(f'Pixel-wise Class Distribution - {split_name}{title_suffix}')
        ax1.set_ylabel('Percentage')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        # Slice distribution
        slice_counts = [
            stats['total_slices'],
            stats['with_kidney'],
            stats['with_tumor']
        ]
        bars = ax2.bar(['Total', 'With Kidney', 'With Tumor'], slice_counts)
        ax2.set_title(f'Slice Distribution - {split_name}{title_suffix}')
        ax2.set_ylabel('Number of Slices')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'distribution_{split_name}{filename_suffix}.png')
        plt.close()
    
    def print_stats(self, split_name, is_filtered=False):
        """Print statistics for a specific split."""
        if is_filtered and split_name == 'train':
            stats = self.train_filtered_stats
            title_suffix = " (ROI only)"
        else:
            stats = self.stats[split_name]
            title_suffix = ""
        
        print(f"\n{split_name} Statistics{title_suffix}:")
        print(f"Total number of slices: {stats['total_slices']}")
        if not is_filtered:
            print(f"Background-only slices: {stats['background_only']} "
                  f"({stats['background_only']/stats['total_slices']*100:.1f}%)")
        print(f"Slices with ROI: {stats['with_roi']} "
              f"({stats['with_roi']/stats['total_slices']*100:.1f}%)")
        
        print(f"\nAmong ROI slices:")
        print(f"Slices with kidney: {stats['with_kidney']} "
              f"({stats['with_kidney']/stats['with_roi']*100:.1f}%)")
        print(f"Slices with tumor: {stats['with_tumor']} "
              f"({stats['with_tumor']/stats['with_roi']*100:.1f}%)")
        
        total_pixels = sum(stats['pixel_counts'].values())
        print(f"\nPixel-wise class distribution:")
        for class_id in range(3):
            count = stats['pixel_counts'].get(class_id, 0)
            percentage = (count / total_pixels) * 100
            class_name = ['Background', 'Kidney', 'Tumor'][class_id]
            print(f"{class_name}: {percentage:.2f}%")

def generate_slices_list(cases, filename, data_dir, analyzer, split_name, roi_only=False):
    """Generate a list of slices for the given cases."""
    slices_list = []
    
    # First pass: collect regular statistics
    for case in cases:
        case_path = os.path.join(data_dir, case)
        analyzer.analyze_case(case_path, split_name, is_filtered=False)
    
    # Second pass for training: collect filtered statistics and generate list
    for case in cases:
        case_path = os.path.join(data_dir, case)
        seg_path = os.path.join(case_path, "segmentation.nii.gz")
        segmentation = nib.load(seg_path).get_fdata()
        
        for slice_idx in range(segmentation.shape[2]):
            if roi_only:
                slice_data = segmentation[:,:,slice_idx]
                if 1 in slice_data or 2 in slice_data:
                    slices_list.append(f"{case} {slice_idx}")
                    if split_name == 'train':
                        analyzer.analyze_case(case_path, split_name, is_filtered=True)
            else:
                slices_list.append(f"{case} {slice_idx}")
    
    random.shuffle(slices_list)
    
    with open(filename, "w") as f:
        f.writelines([slice_item + "\n" for slice_item in slices_list])
    
    return len(slices_list)

def main():
    random.seed(42)
    
    data_dir = "kits19/data"
    lists_dir = "./lists_kits19"
    os.makedirs(lists_dir, exist_ok=True)
    
    # Get and shuffle cases
    cases = sorted([c for c in os.listdir(data_dir) if c.startswith('case_')])
    random.shuffle(cases)
    
    # First split: Separate test cases (7.5%)
    test_size = int(len(cases) * 0.075)
    test_cases = cases[:test_size]
    trainval_cases = cases[test_size:]  # 92.5% for train/val
    
    analyzer = DatasetAnalyzer()
    
    # For train/val cases, we'll collect all ROI slices first
    all_roi_slices = []
    for case in trainval_cases:
        case_path = os.path.join(data_dir, case)
        seg_path = os.path.join(case_path, "segmentation.nii.gz")
        segmentation = nib.load(seg_path).get_fdata()
        
        for slice_idx in range(segmentation.shape[2]):
            slice_data = segmentation[:,:,slice_idx]
            if 1 in slice_data or 2 in slice_data:  # ROI slice
                all_roi_slices.append(f"{case} {slice_idx}")
    
    # Shuffle and split ROI slices between train and val
    random.shuffle(all_roi_slices)
    val_size = int(len(all_roi_slices) * 0.15)  # 15% of ROI slices for validation
    val_slices = all_roi_slices[:val_size]
    train_slices = all_roi_slices[val_size:]
    
    # Write train and val slices to files
    with open(os.path.join(lists_dir, "train.txt"), "w") as f:
        f.writelines([slice_item + "\n" for slice_item in train_slices])
    
    with open(os.path.join(lists_dir, "val.txt"), "w") as f:
        f.writelines([slice_item + "\n" for slice_item in val_slices])
    
    # Generate test list (all slices from test cases)
    test_slices = generate_slices_list(
        test_cases,
        os.path.join(lists_dir, "test.txt"),
        data_dir,
        analyzer,
        'test',
        roi_only=False
    )
    
    # Print summary
    print("\nDataset Split Summary:")
    print(f"Train/Val Cases: {len(trainval_cases)} cases ({len(trainval_cases)/len(cases)*100:.1f}%)")
    print(f"- Training: {len(train_slices)} ROI slices")
    print(f"- Validation: {len(val_slices)} ROI slices")
    print(f"Test Cases: {len(test_cases)} cases ({len(test_cases)/len(cases)*100:.1f}%), "
          f"{test_slices} slices (all)")
    
    # Print statistics and create plots
    # First analyze training/validation cases
    for case in trainval_cases:
        case_path = os.path.join(data_dir, case)
        analyzer.analyze_case(case_path, 'trainval', is_filtered=False)
    
    # Then analyze test cases
    for case in test_cases:
        case_path = os.path.join(data_dir, case)
        analyzer.analyze_case(case_path, 'test', is_filtered=False)
        
    # Print and plot distributions
    analyzer.print_stats('trainval', is_filtered=False)
    analyzer.plot_distributions('trainval', is_filtered=False)
    analyzer.print_stats('test', is_filtered=False)
    analyzer.plot_distributions('test', is_filtered=False)

if __name__ == "__main__":
    main()