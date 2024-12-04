import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def load_nifti_file(filepath):
    """Load a NIfTI file and return the data array."""
    nifti_img = nib.load(filepath)
    return nifti_img.get_fdata()

def plot_slices(image_data, segmentation_data, slice_idx):
    """Plot a slice of the image and its corresponding segmentation with contours."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Image slice
    axes[0].imshow(image_data[:, :, slice_idx], cmap="gray")
    axes[0].set_title(f"Image (Slice {slice_idx})")
    axes[0].axis("off")

    # Overlay segmentation contours
    axes[1].imshow(image_data[:, :, slice_idx], cmap="gray")
    axes[1].set_title(f"Segmentation Contours (Slice {slice_idx})")
    axes[1].axis("off")
    contours = measure.find_contours(segmentation_data[:, :, slice_idx], level=0.5)
    for contour in contours:
        axes[1].plot(contour[:, 1], contour[:, 0], linewidth=1, color="red")

    plt.tight_layout()
    plt.show()

def count_classes(segmentation_data):
    """Count the number of unique classes in the segmentation."""
    unique_classes = np.unique(segmentation_data)
    print(f"Unique classes in segmentation: {unique_classes}")
    return len(unique_classes)

def check_segmentation_blank(segmentation_data):
    """Check if the segmentation data is blank or contains only one class."""
    unique_classes = np.unique(segmentation_data)
    if len(unique_classes) == 1:
        print("The segmentation data appears to be blank or contains only one class.")
    else:
        print("The segmentation data contains multiple classes.")

def segmentation_statistics(segmentation_data):
    """Calculate and display statistics about the segmentation data."""
    total_voxels = segmentation_data.size
    non_zero_voxels = np.count_nonzero(segmentation_data)
    zero_voxels = total_voxels - non_zero_voxels
    print(f"Total voxels: {total_voxels}")
    print(f"Non-zero voxels: {non_zero_voxels} ({(non_zero_voxels / total_voxels) * 100:.2f}%)")
    print(f"Zero voxels: {zero_voxels} ({(zero_voxels / total_voxels) * 100:.2f}%)")

def visualize_multiple_slices(image_data, segmentation_data, num_slices=5):
    """Visualize multiple slices from the image and segmentation data."""
    total_slices = image_data.shape[2]
    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

    for slice_idx in slice_indices:
        plot_slices(image_data, segmentation_data, slice_idx)

def visualize_class(segmentation_data, class_value, slice_idx):
    """Visualize a specific class in the segmentation data."""
    class_mask = (segmentation_data[:, :, slice_idx] == class_value)
    plt.figure(figsize=(6, 6))
    plt.imshow(class_mask, cmap="jet")
    plt.title(f"Class {class_value} Visualization (Slice {slice_idx})")
    plt.axis("off")
    plt.show()

def main():
    # File paths
    image_path = "/home/hous/Desktop/Kidney_Segmentation/kits19/data/case_00000/imaging.nii.gz"
    segmentation_path = "/home/hous/Desktop/Kidney_Segmentation/kits19/data/case_00000/segmentation.nii.gz"

    # Load image and segmentation data
    image_data = load_nifti_file(image_path)
    segmentation_data = load_nifti_file(segmentation_path)

    # Display some information about the data
    print(f"Image shape: {image_data.shape}")
    print(f"Segmentation shape: {segmentation_data.shape}")

    # Check if segmentation data is blank
    check_segmentation_blank(segmentation_data)

    # Count the unique classes in the segmentation
    num_classes = count_classes(segmentation_data)
    print(f"Number of classes in the segmentation: {num_classes}")

    # Display segmentation statistics
    segmentation_statistics(segmentation_data)

    # Visualize multiple slices
    visualize_multiple_slices(image_data, segmentation_data, num_slices=5)

    # Optional: Visualize specific classes
    visualize_class(segmentation_data, class_value=1, slice_idx=image_data.shape[2] // 2)
    visualize_class(segmentation_data, class_value=2, slice_idx=image_data.shape[2] // 2)

if __name__ == "__main__":
    main()