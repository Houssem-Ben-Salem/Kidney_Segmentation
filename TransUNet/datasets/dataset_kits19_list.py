import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import rotate, zoom
import random

class KiTS19DatasetList(Dataset):
    def __init__(self, list_file, base_dir, slice_size=(224, 224), augment=True, num_classes=3):
        self.base_dir = base_dir
        self.slice_size = slice_size
        self.augment = augment
        self.num_classes = num_classes

        with open(list_file, "r") as f:
            self.slices = [line.strip().split() for line in f.readlines()]

    def _normalize(self, volume):
        """Normalize using mean and std of non-zero values"""
        mask = volume != 0
        mean = volume[mask].mean()
        std = volume[mask].std()
        normalized = (volume - mean) / (std + 1e-8)
        return normalized

    def _apply_window(self, image, window_center=50, window_width=350):
        """Apply CT windowing for better tissue contrast"""
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        image = np.clip(image, min_value, max_value)
        image = (image - min_value) / (max_value - min_value)
        return image

    def _augment(self, image, label):
        """Apply moderate augmentations with careful probability"""
        # Store original label values to ensure preservation
        tumor_pixels = (label == 2)
        kidney_pixels = (label == 1)

        # 1. Rotation with moderate angles (15° instead of 20°)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = rotate(image, angle, reshape=False, order=3, mode='reflect')
            label = rotate(label, angle, reshape=False, order=0, mode='constant')

        # 2. Flip (kept as is, since it's a safe augmentation)
        if random.random() > 0.5:
            axis = random.choice([0, 1])
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        # 3. Random intensity variation (mild)
        if random.random() > 0.7:  # Reduced probability
            image = image * random.uniform(0.9, 1.1)  # Mild intensity scaling
            image = np.clip(image, 0, 1)

        # Ensure label integrity
        label = label.round().astype(np.int64)
        label[tumor_pixels] = 2  # Restore tumor labels
        label[kidney_pixels & (label != 2)] = 1  # Restore kidney labels where there's no tumor

        return image, label

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        case, slice_idx = self.slices[idx]
        case_path = os.path.join(self.base_dir, case)

        # Load volume and segmentation
        img_path = os.path.join(case_path, "imaging.nii.gz")
        seg_path = os.path.join(case_path, "segmentation.nii.gz")
        volume = nib.load(img_path).get_fdata()
        segmentation = nib.load(seg_path).get_fdata()

        # Extract the slice
        slice_idx = int(slice_idx)
        image = volume[:, :, slice_idx]
        label = segmentation[:, :, slice_idx]

        # Apply CT windowing first
        image = self._apply_window(image)
        
        # Then normalize
        image = self._normalize(image)

        # Resize slices if needed
        if image.shape != self.slice_size:
            resize_factors = (self.slice_size[0] / image.shape[0], 
                            self.slice_size[1] / image.shape[1])
            image = zoom(image, resize_factors, order=3)
            label = zoom(label, resize_factors, order=0)

        # Apply augmentation
        if self.augment:
            image, label = self._augment(image, label)

        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return {"image": image, "label": label}