import os
import random
import nibabel as nib
from sklearn.model_selection import train_test_split

# Path to the dataset
data_dir = "kits19/data"

# Get all cases
cases = sorted(os.listdir(data_dir))

# Shuffle the cases for randomness
random.seed(42)  # Set seed for reproducibility
random.shuffle(cases)

# Split cases into train, validation, and test (15% test, rest split into train and val)
test_split = int(len(cases) * 0.15)  # 15% for testing
test_cases = cases[:test_split]
remaining_cases = cases[test_split:]
train_cases, val_cases = train_test_split(remaining_cases, test_size=0.2, random_state=42)  # 20% of remaining for validation

# Create lists directory
lists_dir = "./lists_kits19"
os.makedirs(lists_dir, exist_ok=True)

def generate_slices_list(cases, filename):
    """Generate a list of all slices for the given cases."""
    slices_list = []
    for case in cases:
        case_path = os.path.join(data_dir, case)
        seg_path = os.path.join(case_path, "segmentation.nii.gz")
        segmentation = nib.load(seg_path).get_fdata()
        num_slices = segmentation.shape[2]
        for slice_idx in range(num_slices):
            slices_list.append(f"{case} {slice_idx}")
    
    # Shuffle the slices for randomness
    random.shuffle(slices_list)
    
    # Write to the specified file
    with open(os.path.join(lists_dir, filename), "w") as f:
        f.writelines([slice_item + "\n" for slice_item in slices_list])

# Generate train, validation, and test slice lists
generate_slices_list(train_cases, "train.txt")
generate_slices_list(val_cases, "val.txt")
generate_slices_list(test_cases, "test.txt")

# Print summary
print(f"Generated slice lists:")
print(f"- Train cases: {len(train_cases)} ({len(train_cases) / len(cases) * 100:.2f}%)")
print(f"- Validation cases: {len(val_cases)} ({len(val_cases) / len(cases) * 100:.2f}%)")
print(f"- Test cases: {len(test_cases)} ({len(test_cases) / len(cases) * 100:.2f}%)")