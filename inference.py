import os
import cv2
import torch
import numpy as np
import argparse
from networks.srn.net import SpeckleReductionNet
from utils.image_ops import linear_normalization


def load_model(model_path, device, channels):
    model = SpeckleReductionNet(channels=channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def to_chw_tensor(image):
    if image.ndim == 2:
        return torch.tensor(image).unsqueeze(0)
    if image.ndim == 3:
        return torch.tensor(np.transpose(image, (2, 0, 1)))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def run_inference(model, image_array, device, visualize=False):
    del visualize  # visualization for grayscale ultrasound only
    output_list = []

    with torch.no_grad():
        for img_input in image_array:
            per_channel = (img_input.ndim == 3)
            norm_input = linear_normalization(img_input, per_channel=per_channel)
            tensor_input = to_chw_tensor(norm_input).unsqueeze(0).float().to(device)

            output_tensor, _, _ = model(tensor_input, tensor_input, tensor_input)
            output_tensor = torch.clamp(output_tensor, 0, 1)

            output = output_tensor.squeeze(0).cpu().numpy()
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))

            output_list.append(output)

    return np.stack(output_list, axis=0)


def save_results(output_array, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, output_array)
    print(f"[✓] Output saved to: {save_path}")


def save_tiff16_stack(output_array, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, img in enumerate(output_array):
        img_uint16 = (np.clip(img, 0, 1) * 65535.0).astype(np.uint16)
        out_path = os.path.join(out_dir, f"denoised_{idx:04d}.tiff")
        if img_uint16.ndim == 3 and img_uint16.shape[2] == 3:
            img_uint16 = cv2.cvtColor(img_uint16, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img_uint16)
    print(f"[✓] 16-bit TIFF files saved in: {out_dir}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    images = np.load(args.data_path)
    if images.ndim == 3:
        channels = 1
        noisy_images = images
    elif images.ndim == 4 and images.shape[-1] in (1, 3):
        channels = images.shape[-1]
        noisy_images = images
    elif images.ndim == 4 and images.shape[1] == 2:
        channels = 1
        noisy_images = images[:, 0]
    elif images.ndim == 5 and images.shape[1] == 2 and images.shape[-1] in (1, 3):
        channels = images.shape[-1]
        noisy_images = images[:, 0]
    else:
        raise ValueError(f"Unsupported npy shape: {images.shape}")

    model = load_model(args.model_path, device, channels=channels)
    outputs = run_inference(model, noisy_images, device, visualize=args.visualize)

    save_results(outputs, args.output_path)
    if args.tiff16_dir:
        save_tiff16_stack(outputs, args.tiff16_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SRN model inference.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to .npy input image file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to model .pth file")
    parser.add_argument('--output_path', type=str, required=True, help="Where to save the output .npy file")
    parser.add_argument('--tiff16_dir', type=str, default='', help="Optional directory to export 16-bit TIFF outputs")
    parser.add_argument('--visualize', action='store_true', help="Deprecated grayscale visualization")
    args = parser.parse_args()

    main(args)
