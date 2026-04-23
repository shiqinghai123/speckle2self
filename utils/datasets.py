import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from utils.image_ops import resize_image, linear_normalization


class BaseDenoisingDataset(Dataset):
    """Base class for denoising datasets with shared utilities."""
    def __init__(self, interp='linear', per_channel_norm=False):
        self.interp = interp
        self.per_channel_norm = per_channel_norm
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.3),
        ], additional_targets={
            'image0': 'image',
            'image1': 'image',
            'mask': 'image'
        })

    def preprocess_image(self, image):
        image_low = resize_image(image, 0.25, interpol=self.interp)
        image_mid = resize_image(image, 0.5, interpol=self.interp)
        image_high = image

        image_low = linear_normalization(image_low, per_channel=self.per_channel_norm)
        image_mid = linear_normalization(image_mid, per_channel=self.per_channel_norm)
        image_high = linear_normalization(image_high, per_channel=self.per_channel_norm)

        return image_low, image_mid, image_high

    def to_tensor(self, *images):
        tensors = []
        for img in images:
            if img.ndim == 2:
                tensors.append(torch.from_numpy(np.expand_dims(img, 0)).float())
            elif img.ndim == 3:
                tensors.append(torch.from_numpy(np.transpose(img, (2, 0, 1))).float())
            else:
                raise ValueError(f"Unsupported image ndim: {img.ndim}")
        return tensors


class DenoisingDatasetCCA(BaseDenoisingDataset):
    """Dataset for unsupervised denoising with noisy-only train_data.npy."""
    def __init__(self, image_dir, interp='linear', channels=1):
        super().__init__(interp, per_channel_norm=(channels > 1))
        self.images = np.load(os.path.join(image_dir, "train_data.npy"))

        if channels == 1 and self.images.ndim != 3:
            raise ValueError(f"Expected (N,H,W) for channels=1, got {self.images.shape}")
        if channels > 1 and not (self.images.ndim == 4 and self.images.shape[-1] == channels):
            raise ValueError(f"Expected (N,H,W,{channels}) for RGB mode, got {self.images.shape}")

        self.num_images = self.images.shape[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_raw = self.images[idx]
        image_low, image_mid, image_high = self.preprocess_image(image_raw)

        transformed = self.transform(
            image=image_low, image0=image_high, image1=image_mid
        )

        image_low, image_high, image_mid = self.to_tensor(
            transformed['image'], transformed['image0'], transformed['image1']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
            'image_mid': image_mid
        }


class DenoisingDatasetSimulator(BaseDenoisingDataset):
    """Dataset for simulated denoising with paired noisy and clean images."""
    def __init__(self, path, interp='linear', channels=1):
        super().__init__(interp, per_channel_norm=(channels > 1))
        data = np.load(os.path.join(path, "train_data.npy"))

        if channels == 1:
            if not (data.ndim == 4 and data.shape[1] == 2):
                raise ValueError(f"Expected (N,2,H,W) for channels=1, got {data.shape}")
            self.noisy_imgs = data[:, 0]
            self.clean_imgs = data[:, 1]
        else:
            if not (data.ndim == 5 and data.shape[1] == 2 and data.shape[-1] == channels):
                raise ValueError(f"Expected (N,2,H,W,{channels}) for channels={channels}, got {data.shape}")
            self.noisy_imgs = data[:, 0]
            self.clean_imgs = data[:, 1]

        self.num_images = self.noisy_imgs.shape[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        noisy = self.noisy_imgs[idx]
        clean = linear_normalization(self.clean_imgs[idx], per_channel=self.per_channel_norm)

        image_low, image_mid, image_high = self.preprocess_image(noisy)

        transformed = self.transform(
            image=image_low,
            image0=image_high,
            image1=image_mid,
            mask=clean
        )

        image_low, image_high, image_mid, image_clean = self.to_tensor(
            transformed['image'],
            transformed['image0'],
            transformed['image1'],
            transformed['mask']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
            'image_mid': image_mid,
            'image_clean': image_clean
        }
