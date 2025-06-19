import os
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

# Define color to label mapping
COLOR_TO_LABEL = {
    (0, 0, 0): 0,         # background
    (255, 255, 255): 1    # crack
}

def convert_rgb_mask_to_label(mask_rgb_path, save_path):
    img = Image.open(mask_rgb_path).convert('RGB')
    img_np = np.array(img)

    # Initialize output with ignore label 255
    label_mask = np.full(img_np.shape[:2], 255, dtype=np.uint8)

    for color, label in COLOR_TO_LABEL.items():
        matches = np.all(img_np == color, axis=-1)
        label_mask[matches] = label

    label_img = Image.fromarray(label_mask, mode='L')
    label_img.save(save_path)

def process_directory(root_dir):
    mask_rgb_dir = Path(root_dir) / "mask_rgb"
    mask_label_dir = Path(root_dir) / "mask"
    mask_label_dir.mkdir(exist_ok=True)

    for mask_file in sorted(mask_rgb_dir.glob("*.png")):
        output_file = mask_label_dir / mask_file.name
        convert_rgb_mask_to_label(mask_file, output_file)
        print(f"Converted: {mask_file} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert RGB masks to label masks.")
    parser.add_argument("dataset_root", nargs='?', default=".", help="Path to dataset root (default: current directory)")
    args = parser.parse_args()

    base_path = Path(args.dataset_root).resolve()

    for split in ['train', 'valid', 'test']:
        split_dir = base_path / split
        if (split_dir / "mask_rgb").exists():
            print(f"Processing: {split}/mask_rgb")
            process_directory(split_dir)

if __name__ == "__main__":
    main()

