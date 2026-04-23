import cv2
import numpy as np


def linear_normalization(image, per_channel=False):
    image = image.astype(np.float32)

    if per_channel and image.ndim == 3:
        min_value = np.min(image, axis=(0, 1), keepdims=True)
        max_value = np.max(image, axis=(0, 1), keepdims=True)
    else:
        min_value = np.min(image)
        max_value = np.max(image)

    denom = max_value - min_value
    if np.any(denom == 0):
        return np.zeros_like(image, dtype=np.float32)

    normalized_image = (image - min_value) / denom
    return normalized_image.astype(np.float32)


def resize_image(image, scale, interpol='linear'):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    if interpol == 'linear':
        interpol_method = cv2.INTER_LINEAR
    elif interpol == 'cubic':
        interpol_method = cv2.INTER_CUBIC
    elif interpol == 'area':
        interpol_method = cv2.INTER_AREA
    elif interpol == 'nearest':
        interpol_method = cv2.INTER_NEAREST
    else:
        raise ValueError(f"Unsupported interpolation method: {interpol}")

    resized = cv2.resize(image, dim, interpolation=interpol_method)
    return cv2.resize(resized, (image.shape[1], image.shape[0]), interpolation=interpol_method)


def remove_stripe_noise(image, sigma_x=9, sigma_y=1):
    """Suppress directional stripe noise by subtracting directional blur residual."""
    image_f = image.astype(np.float32)
    directional = cv2.GaussianBlur(image_f, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    corrected = image_f - (directional - image_f.mean(axis=(0, 1), keepdims=True))
    return np.clip(corrected, 0, None)


def flat_field_correction(image, blur_ksize=101):
    """Estimate smooth illumination field and divide to flatten background."""
    image_f = image.astype(np.float32) + 1e-6
    background = cv2.GaussianBlur(image_f, (blur_ksize, blur_ksize), 0)
    corrected = image_f / (background + 1e-6)
    corrected *= np.mean(background, axis=(0, 1), keepdims=True)
    return corrected.astype(np.float32)
