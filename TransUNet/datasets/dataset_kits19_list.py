import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import rotate, zoom
import random

class KiTS19DatasetList(Dataset):
    def __init__(self, list_file, base_dir, slice_size=(224, 224), augment=True):
        """
        Args:
            list_file (str): Path to the slice list file.
            base_dir (str): Path to the KiTS19 dataset.
            slice_size (tuple): Target size of slices (height, width).
            augment (bool): Apply augmentation if True.
        """
        self.base_dir = base_dir
        self.slice_size = slice_size
        self.augment = augment

        # Read list file
        with open(list_file, "r") as f:
            self.slices = [line.strip().split() for line in f.readlines()]

    def _normalize(self, volume):
        """Normalize a 3D volume to [0, 1]."""
        return (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    def _augment(self, image, label):
        """Apply random augmentations (rotation, flip)."""
        if random.random() > 0.5:  # Random rotation
            angle = random.uniform(-20, 20)
            image = rotate(image, angle, reshape=False, order=3)
            label = rotate(label, angle, reshape=False, order=0)
        if random.random() > 0.5:  # Random flip
            axis = random.choice([0, 1])
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
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

        # Normalize the entire volume
        volume = self._normalize(volume)

        # Extract the slice
        slice_idx = int(slice_idx)
        image = volume[:, :, slice_idx]
        label = segmentation[:, :, slice_idx]

        # Resize slices
        resize_factors = (self.slice_size[0] / image.shape[0], self.slice_size[1] / image.shape[1])
        image = zoom(image, resize_factors, order=3)
        label = zoom(label, resize_factors, order=0)

        # Apply augmentation
        if self.augment:
            image, label = self._augment(image, label)

        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).long()

        return {"image": image, "label": label}