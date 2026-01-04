import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LGGSegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, resize=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.resize = resize  # (W,H) or None

        self.image_paths = sorted(glob(os.path.join(images_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {images_dir}. "
                f"If your files are .tif, change '*.png' to '*.tif'."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, fname)

        if not os.path.exists(mask_path):
            raise RuntimeError(f"Mask not found for {fname}. Expected: {mask_path}")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read: {img_path} or {mask_path}")

        if self.resize is not None:
            w, h = self.resize
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)  # [1,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0)    # [1,H,W]
        return image, mask, fname


class SubsetWithFilter(Dataset):
    """
    Subset wrapper with optional filtering of empty masks.
    Filtering happens only once at init => efficient during training.
    """
    def __init__(self, base_dataset: Dataset, indices, filter_empty: bool = False):
        self.base = base_dataset
        self.indices = list(indices)

        if filter_empty:
            kept = []
            for i in self.indices:
                _, mask, _ = self.base[i]
                if mask.sum().item() > 0:
                    kept.append(i)
            self.indices = kept
            print(f"[Subset] filter_empty=True, kept {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]
