import os
import nibabel as nib
from tqdm import tqdm
import shutil
from datetime import datetime

def verify_test_data(list_file, base_dir):
    """
    Verify the integrity of test data files.
    Returns lists of valid and problematic files.
    """
    print(f"\nVerifying data from list file: {list_file}")
    
    # Read the list file
    with open(list_file, "r") as f:
        slices = [line.strip().split() for line in f.readlines()]
    
    valid_slices = []
    problematic_files = []
    
    # Check each case
    unique_cases = set(slice_info[0] for slice_info in slices)
    for case in tqdm(unique_cases, desc="Checking cases"):
        case_path = os.path.join(base_dir, case)
        img_path = os.path.join(case_path, "imaging.nii.gz")
        seg_path = os.path.join(case_path, "segmentation.nii.gz")
        
        try:
            # Check if files exist
            if not os.path.exists(img_path):
                problematic_files.append((case, "imaging.nii.gz", "File not found"))
                continue
            if not os.path.exists(seg_path):
                problematic_files.append((case, "segmentation.nii.gz", "File not found"))
                continue
            
            # Try to load the files
            try:
                img_nifti = nib.load(img_path)
                img_data = img_nifti.get_fdata()
            except Exception as e:
                problematic_files.append((case, "imaging.nii.gz", str(e)))
                continue
                
            try:
                seg_nifti = nib.load(seg_path)
                seg_data = seg_nifti.get_fdata()
            except Exception as e:
                problematic_files.append((case, "segmentation.nii.gz", str(e)))
                continue
            
            # Check shapes match
            if img_data.shape != seg_data.shape:
                problematic_files.append(
                    (case, "both", f"Shape mismatch: img {img_data.shape} vs seg {seg_data.shape}")
                )
                continue
            
            # Check slice indices
            case_slices = [int(s[1]) for s in slices if s[0] == case]
            max_slice = img_data.shape[2] - 1
            
            invalid_slices = [s for s in case_slices if s > max_slice]
            if invalid_slices:
                problematic_files.append(
                    (case, "slices", f"Invalid slice indices: {invalid_slices}, max allowed: {max_slice}")
                )
                continue
            
            # If all checks pass, add to valid slices
            valid_slices.extend([s for s in slices if s[0] == case])
            
        except Exception as e:
            problematic_files.append((case, "unknown", str(e)))
    
    return valid_slices, problematic_files

def create_cleaned_test_list(list_file, valid_slices, output_dir):
    """Create a new test list excluding problematic cases"""
    # Create backup of original file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(output_dir, f"test_backup_{timestamp}.txt")
    shutil.copy2(list_file, backup_path)
    print(f"\nOriginal test list backed up to: {backup_path}")
    
    # Create cleaned test list
    cleaned_list_path = list_file  # Overwrite original file
    with open(cleaned_list_path, 'w') as f:
        for case, slice_idx in valid_slices:
            f.write(f"{case} {slice_idx}\n")
    
    print(f"Cleaned test list saved to: {cleaned_list_path}")
    print(f"Removed {len(set([p[0] for p in problematic_files]))} problematic cases")

def save_verification_results(valid_slices, problematic_files, output_dir="verification_results"):
    """Save verification results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save valid slices
    with open(os.path.join(output_dir, "valid_test_slices.txt"), "w") as f:
        for case, slice_idx in valid_slices:
            f.write(f"{case} {slice_idx}\n")
    
    # Save problematic files with details
    with open(os.path.join(output_dir, "problematic_files.txt"), "w") as f:
        f.write("Case | File | Error\n")
        f.write("-" * 50 + "\n")
        for case, file_type, error in problematic_files:
            f.write(f"{case} | {file_type} | {error}\n")
    
    # Print summary
    print("\nVerification Summary:")
    print(f"Total valid slices: {len(valid_slices)}")
    print(f"Total problematic cases: {len(set([p[0] for p in problematic_files]))}")
    print(f"Total problematic files: {len(problematic_files)}")
    print(f"\nResults saved in: {output_dir}")
    print("Check 'problematic_files.txt' for detailed error information")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_dir', type=str, default='./lists_kits19',
                      help='directory containing test list')
    parser.add_argument('--root_path', type=str, default='kits19/data',
                      help='root directory for data')
    parser.add_argument('--output_dir', type=str, default='verification_results',
                      help='directory to save verification results')
    parser.add_argument('--clean_list', action='store_true',
                      help='create cleaned test list removing problematic cases')
    
    args = parser.parse_args()
    
    # Verify test data
    test_list = os.path.join(args.list_dir, "test.txt")
    valid_slices, problematic_files = verify_test_data(test_list, args.root_path)
    
    # Save results
    save_verification_results(valid_slices, problematic_files, args.output_dir)
    
    # Create cleaned test list if requested
    if args.clean_list:
        create_cleaned_test_list(test_list, valid_slices, args.output_dir)