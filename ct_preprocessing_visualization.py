import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import nibabel as nib  # For reading NIfTI files

def load_ct_slice(case_id, slice_num, data_dir):
    """
    Load a specific CT slice from NIfTI files (.nii.gz)
    
    Parameters:
    case_id (str): Case identifier (e.g., 'case_00122')
    slice_num (int): Slice number to load
    data_dir (str): Directory containing CT scan data
    
    Returns:
    numpy.ndarray: CT slice as a 2D array in Hounsfield Units
    """
    # Construct path to the NIfTI file based on KiTS19 dataset structure
    nifti_path = os.path.join(data_dir, case_id, "imaging.nii.gz")
    
    try:
        # Load NIfTI file using nibabel
        nifti_img = nib.load(nifti_path)
        
        # Get the 3D volume data (already in Hounsfield Units for KiTS19)
        volume_data = nifti_img.get_fdata()
        
        # Extract the specific slice (check axis orientation)
        # KiTS19 typically has slices along the z-axis (axis 2)
        if slice_num < volume_data.shape[2]:
            image = volume_data[:, :, slice_num]
        else:
            raise IndexError(f"Slice index {slice_num} out of bounds for volume with {volume_data.shape[2]} slices")
            
    except (FileNotFoundError, nib.filebasedimages.ImageFileError, IndexError) as e:
        # Fallback for testing: create synthetic CT data if file not found
        print(f"Warning: Could not load slice {slice_num} from {nifti_path}. Error: {e}")
        print("Creating synthetic data.")
        
        # Create synthetic CT data with kidney-like structures
        image = np.random.normal(-100, 50, (512, 512))  # Background tissue
        
        # Add a kidney-like structure
        center_x, center_y = 256, 256
        radius = 100
        y, x = np.ogrid[:512, :512]
        kidney_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image[kidney_mask] = np.random.normal(50, 20, np.sum(kidney_mask))  # Kidney tissue ~50 HU
        
        # Add a tumor-like structure
        tumor_center_x, tumor_center_y = 300, 256
        tumor_radius = 30
        tumor_mask = (x - tumor_center_x)**2 + (y - tumor_center_y)**2 <= tumor_radius**2
        image[tumor_mask] = np.random.normal(20, 10, np.sum(tumor_mask))  # Tumor tissue ~20 HU
        
        # Add some contrast and structures
        image = np.clip(image, -1024, 3071)  # Clip to standard HU range
    
    return image

def standard_normalization(image):
    """
    Apply standard min-max normalization to scale image to [0, 1]
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Normalized image
    """
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def non_zero_mean_std_normalization(image):
    """
    Apply non-zero mean-std normalization as described in the report
    
    Parameters:
    image (numpy.ndarray): Input image
    
    Returns:
    numpy.ndarray: Normalized image
    """
    # Create a mask for non-zero pixels (typically not air)
    # In CT, air is typically around -1000 HU, so we can use a threshold
    non_zero_mask = image > -950
    
    if non_zero_mask.sum() > 0:
        # Calculate mean and std of non-zero (non-air) voxels
        non_zero_mean = np.mean(image[non_zero_mask])
        non_zero_std = np.std(image[non_zero_mask])
        
        # Normalize using non-zero statistics
        normalized = (image - non_zero_mean) / (non_zero_std if non_zero_std > 0 else 1)
    else:
        # Fallback if no non-zero pixels
        normalized = image
    
    return normalized

def hu_windowing(image, center=50, width=350):
    """
    Apply HU windowing to focus on a specific range of Hounsfield Units
    
    Parameters:
    image (numpy.ndarray): Input image in Hounsfield Units
    center (float): Center of the window
    width (float): Width of the window
    
    Returns:
    numpy.ndarray: Windowed image scaled to [0, 255]
    """
    # Calculate window boundaries
    window_min = center - (width / 2)
    window_max = center + (width / 2)
    
    # Apply windowing formula as described in the report
    windowed = (image - window_min) / width
    
    # Scale to [0, 255] and clip
    windowed = np.clip(windowed * 255, 0, 255)
    
    return windowed

def visualize_preprocessing(ct_slice):
    """
    Create a comparison figure of different preprocessing techniques
    
    Parameters:
    ct_slice (numpy.ndarray): Input CT slice in Hounsfield Units
    
    Returns:
    matplotlib.figure.Figure: Figure with comparison subplots
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # (a) Original CT scan with full HU range
    # For visualization, use a standard abdominal window
    orig_display = hu_windowing(ct_slice, center=50, width=400)
    axes[0].imshow(orig_display, cmap='gray')
    axes[0].set_title('(a) Original CT scan\nwith full HU range')
    axes[0].axis('off')
    
    # (b) Standard normalization
    std_norm = standard_normalization(ct_slice)
    axes[1].imshow(std_norm, cmap='gray')
    axes[1].set_title('(b) Standard normalization')
    axes[1].axis('off')
    
    # (c) CT-specific windowing approach
    windowed = hu_windowing(ct_slice, center=50, width=350)
    axes[2].imshow(windowed, cmap='gray')
    axes[2].set_title('(c) CT-specific windowing\n(center=50, width=350)')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def process_case_file(file_path, data_dir, output_dir):
    """
    Process a file containing case IDs and slice numbers
    
    Parameters:
    file_path (str): Path to the text file with case information
    data_dir (str): Directory containing CT scan data
    output_dir (str): Directory to save output figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            case_id = parts[0]
            slice_num = int(parts[1])
            
            print(f"Processing {case_id}, slice {slice_num}...")
            
            # Load the CT slice
            ct_slice = load_ct_slice(case_id, slice_num, data_dir)
            
            # Create visualization
            fig = visualize_preprocessing(ct_slice)
            
            # Save figure
            output_path = os.path.join(output_dir, f"{case_id}_slice{slice_num}_comparison.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved comparison to {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='CT Scan Preprocessing Visualization')
    parser.add_argument('--input', type=str, required=True, help='Path to text file with case IDs and slice numbers')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing KiTS19 data (e.g., path to kits19/data/)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output figures')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the case file
    process_case_file(args.input, args.data_dir, args.output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()