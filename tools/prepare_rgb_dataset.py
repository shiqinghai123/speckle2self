import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from utils.image_ops import flat_field_correction, remove_stripe_noise


VALID_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_rgb_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {img.shape} for {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def center_crop_square(img, crop_size):
    h, w = img.shape[:2]
    if crop_size <= 0 or crop_size > min(h, w):
        return img

    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in VALID_EXT])
    if not paths:
        raise FileNotFoundError(f"No images found in: {in_dir}")

    images = []
    for path in paths:
        img = load_rgb_image(path)
        img = center_crop_square(img, args.crop_size)

        if args.flat_field:
            img = flat_field_correction(img, blur_ksize=args.flat_field_ksize)
        if args.destripe:
            img = remove_stripe_noise(img, sigma_x=args.stripe_sigma_x, sigma_y=args.stripe_sigma_y)

        images.append(img.astype(np.float32))

    arr = np.stack(images, axis=0)
    out_path = out_dir / 'train_data.npy'
    np.save(out_path, arr)
    print(f"[✓] Saved dataset: {out_path}, shape={arr.shape}, dtype={arr.dtype}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare RGB noisy dataset for Speckle2Self training.')
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with PNG/JPG/TIFF noisy images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder containing train_data.npy')
    parser.add_argument('--crop_size', type=int, default=512, help='Center crop size. <=0 disables crop')

    parser.add_argument('--flat_field', action='store_true', help='Enable flat-field correction')
    parser.add_argument('--flat_field_ksize', type=int, default=101, help='Gaussian kernel size for flat-field')

    parser.add_argument('--destripe', action='store_true', help='Enable directional stripe suppression')
    parser.add_argument('--stripe_sigma_x', type=float, default=9.0, help='Sigma X for destriping blur')
    parser.add_argument('--stripe_sigma_y', type=float, default=1.0, help='Sigma Y for destriping blur')

    main(parser.parse_args())
