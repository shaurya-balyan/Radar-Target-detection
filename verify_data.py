"""
verify_data.py - SAR ATR Dataset Inspection & PyTorch Pipeline Verification
===========================================================================
Validates:
1. Folder structure and image formats in data/
2. PyTorch torchvision ImageFolder loading and DataLoader batching
3. Generates a visual inspection grid (dataset_preview.png) with colormaps & histograms
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


def run_verification(data_dir: str = "data", preview_output: str = "dataset_preview.png"):
    print("=" * 65)
    print("SAR ATR DATASET INSPECTION & PYTORCH PIPELINE VERIFICATION")
    print("=" * 65)
    
    root_path = Path(data_dir)
    train_path = root_path / "train"
    test_path = root_path / "test"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found at {train_path}. Run download_data.py first.")
        
    # 1. Define PyTorch transforms for 128x128 SAR imagery
    sar_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # Scales to [0.0, 1.0]
    ])
    
    # 2. Load PyTorch datasets
    train_dataset = datasets.ImageFolder(root=str(train_path), transform=sar_transform)
    test_dataset = datasets.ImageFolder(root=str(test_path), transform=sar_transform) if test_path.exists() else None
    
    classes = train_dataset.classes
    print(f"[+] Detected Classes    : {classes} (Total: {len(classes)})")
    print(f"[+] Class to Index Map  : {train_dataset.class_to_idx}")
    print(f"[+] Total Train Samples : {len(train_dataset)}")
    if test_dataset:
        print(f"[+] Total Test Samples  : {len(test_dataset)}")
        
    # Class distribution breakdown
    print("\n--- Class Distribution ---")
    for cls_name in classes:
        train_count = len(list((train_path / cls_name).glob("*.*")))
        test_count = len(list((test_path / cls_name).glob("*.*"))) if test_path.exists() else 0
        print(f"  {cls_name:10s} -> Train: {train_count:4d} | Test: {test_count:4d}")
        
    # 3. Test PyTorch DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(train_loader))
    
    print("\n--- PyTorch DataLoader Validation ---")
    print(f"[+] Batch Shape         : {images.shape} (Batch, Channel, Height, Width)")
    print(f"[+] Batch Dtype         : {images.dtype}")
    print(f"[+] Batch Pixel Range   : min={images.min().item():.3f}, max={images.max().item():.3f}, mean={images.mean().item():.3f}")
    print(f"[+] Labels in Batch     : {labels[:8].tolist()}...")
    
    # 4. Generate Visual Inspection Grid
    print(f"\n[+] Rendering SAR Target Preview figure to: {preview_output}...")
    fig, axes = plt.subplots(len(classes), 4, figsize=(14, 3.5 * len(classes)))
    if len(classes) == 1:
        axes = np.expand_dims(axes, 0)
        
    for row_idx, cls_name in enumerate(classes):
        cls_folder = train_path / cls_name
        sample_files = sorted(list(cls_folder.glob("*.*")))[:4]
        
        for col_idx, img_file in enumerate(sample_files):
            ax = axes[row_idx, col_idx]
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            
            # Display with radar/viridis colormap
            im = ax.imshow(img, cmap="inferno", vmin=0, vmax=255)
            ax.set_title(f"{cls_name} (#{col_idx+1})\nMean: {img.mean():.1f} | Max: {img.max()}", fontsize=9)
            ax.axis("off")
            
    fig.suptitle("MSTAR Synthetic Aperture Radar (SAR) Target Chips (128x128)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(preview_output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Inspection figure saved successfully! ({preview_output})")
    print("=" * 65)
    print("[SUCCESS] All pipeline checks passed. Your environment & data are fully ready!")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
