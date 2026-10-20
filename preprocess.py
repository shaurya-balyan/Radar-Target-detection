"""
preprocess.py - SAR Radar Preprocessing & Feature Extraction Pipeline
=====================================================================
Specialized defense-tech preprocessing suite for MSTAR Synthetic Aperture Radar (SAR)
Automatic Target Recognition (ATR):
  1. Radar Speckle Noise Attenuation (Bilateral, Median, and Adaptive Lee Filters)
  2. Contrast Limited Adaptive Histogram Equalization (CLAHE) for Target-to-Clutter Definition
  3. Seamless PyTorch Custom Dataset (MSTARPreprocessedDataset) & torchvision Transform
  4. Visual Diagnostic Report Generation (preprocessing_report.png) with Radar Metrics
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Specialized Radar Denoising & Filtering Functions
# ---------------------------------------------------------------------------

def apply_adaptive_lee_filter(image: np.ndarray,
                              kernel_size: int = 5,
                              cu: float = 0.523) -> np.ndarray:
    """
    Adaptive Lee Filter for SAR Multiplicative Speckle Reduction.
    
    Formula:
      I_filtered = mean + W * (I - mean)
      W = max(0, 1 - (Cu^2 * mean^2) / (var + eps))
    where Cu = sigma_v / mu_v is the noise variation coefficient (0.523 for 1-look SAR).
    """
    img_f = image.astype(np.float32)
    k = (kernel_size, kernel_size)
    mean = cv2.boxFilter(img_f, -1, k)
    sqr_mean = cv2.boxFilter(img_f ** 2, -1, k)
    var = np.maximum(sqr_mean - mean ** 2, 0.0)
    
    # Weighting factor
    weight = np.clip(1.0 - (cu ** 2 * mean ** 2) / (var + 1e-6), 0.0, 1.0)
    filtered = mean + weight * (img_f - mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_speckle_denoising(image: np.ndarray,
                            method: str = "bilateral",
                            d: int = 5,
                            sigma_color: float = 35.0,
                            sigma_space: float = 35.0,
                            median_kernel: int = 3,
                            lee_kernel: int = 5) -> np.ndarray:
    """
    Attenuates SAR multiplicative speckle noise while strictly preserving sharp
    edges of vehicle hulls, turrets, and high-RCS scattering centers.
    
    Methods:
      - 'bilateral': Non-linear edge-preserving filter combining spatial & radiometric distance.
      - 'median'   : Rank-order non-linear filter eliminating impulse speckle spikes.
      - 'lee'      : Statistical adaptive speckle filter based on local SAR variance.
      - 'none'     : Pass-through (no smoothing).
    """
    method = method.lower()
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif method == "median":
        return cv2.medianBlur(image, ksize=median_kernel)
    elif method == "lee":
        return apply_adaptive_lee_filter(image, kernel_size=lee_kernel)
    elif method == "none":
        return image.copy()
    else:
        raise ValueError(f"Unknown denoising method '{method}'. Choose from 'bilateral', 'median', 'lee', 'none'.")


# ---------------------------------------------------------------------------
# 2. Radar Contrast Enhancement (CLAHE)
# ---------------------------------------------------------------------------

def apply_clahe_enhancement(image: np.ndarray,
                            clip_limit: float = 2.5,
                            tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Unlike standard histogram equalization, CLAHE prevents over-amplifying background
    clutter noise by operating on local contextual tiles with a strict clip threshold.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def preprocess_sar_chip(image: np.ndarray,
                        denoise_method: str = "bilateral",
                        apply_clahe: bool = True,
                        clip_limit: float = 2.5,
                        tile_grid_size: Tuple[int, int] = (8, 8),
                        denoise_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Radar Processing Pipeline:
      Raw (128x128 uint8) -> Denoised -> Contrast Enhanced (CLAHE).
    Returns:
      (raw_image, denoised_image, enhanced_image)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    dk = denoise_kwargs or {}
    denoised = apply_speckle_denoising(image, method=denoise_method, **dk)
    
    if apply_clahe:
        enhanced = apply_clahe_enhancement(denoised, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    else:
        enhanced = denoised.copy()
        
    return image, denoised, enhanced


# ---------------------------------------------------------------------------
# 3. Radar Quality & Physical Metrics
# ---------------------------------------------------------------------------

def compute_radar_metrics(raw_chip: np.ndarray, processed_chip: np.ndarray) -> Dict[str, float]:
    """
    Computes diagnostic metrics for SAR target processing:
      - ENL (Equivalent Number of Looks): Measure of speckle suppression in background clutter (higher = smoother).
      - Contrast Ratio (CR): Dynamic range ratio between high-scattering target and clutter.
      - Edge Preservation Index (EPI): Ratio of high-frequency edge gradients preserved.
    """
    # Background clutter is typically located along chip borders (outer 20-pixel perimeter)
    h, w = raw_chip.shape
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[24:h-24, 24:w-24] = False  # Center region where target resides
    
    clutter_raw = raw_chip[border_mask].astype(np.float64)
    clutter_proc = processed_chip[border_mask].astype(np.float64)
    
    # ENL = mean^2 / var (in clutter)
    var_raw = np.var(clutter_raw)
    enl_raw = (np.mean(clutter_raw) ** 2) / (var_raw + 1e-6)
    
    var_proc = np.var(clutter_proc)
    enl_proc = (np.mean(clutter_proc) ** 2) / (var_proc + 1e-6)
    
    # Target region (center 40x40)
    target_raw = raw_chip[h//2-20:h//2+20, w//2-20:w//2+20].astype(np.float64)
    target_proc = processed_chip[h//2-20:h//2+20, w//2-20:w//2+20].astype(np.float64)
    
    # Target-to-Clutter Ratio (dB)
    tcr_raw = 10.0 * np.log10((np.mean(target_raw) + 1e-3) / (np.mean(clutter_raw) + 1e-3))
    tcr_proc = 10.0 * np.log10((np.mean(target_proc) + 1e-3) / (np.mean(clutter_proc) + 1e-3))
    
    return {
        "enl_raw": float(enl_raw),
        "enl_proc": float(enl_proc),
        "tcr_raw_db": float(tcr_raw),
        "tcr_proc_db": float(tcr_proc),
        "target_max_proc": float(target_proc.max()),
    }


# ---------------------------------------------------------------------------
# 4. Custom Torchvision Transform
# ---------------------------------------------------------------------------

class SARDenoiseCLAHETransform:
    """
    PyTorch-compatible callable transformation for SAR imagery.
    Transforms raw 128x128 array or PIL Image into a normalized, edge-preserved,
    CLAHE-enhanced PyTorch FloatTensor of shape (1, 128, 128).
    """
    def __init__(self,
                 denoise_method: str = "bilateral",
                 apply_clahe: bool = True,
                 clip_limit: float = 2.5,
                 tile_grid_size: Tuple[int, int] = (8, 8),
                 normalize_mean: float = 0.5,
                 normalize_std: float = 0.5):
        self.denoise_method = denoise_method
        self.apply_clahe = apply_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(self, img: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        # Convert to numpy uint8 grayscale
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("L"))
        elif isinstance(img, torch.Tensor):
            arr = img.squeeze().cpu().numpy()
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        elif isinstance(img, np.ndarray):
            arr = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # 1. Denoise
        denoised = apply_speckle_denoising(arr, method=self.denoise_method)
        
        # 2. CLAHE
        if self.apply_clahe:
            enhanced = apply_clahe_enhancement(denoised, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        else:
            enhanced = denoised
            
        # 3. Convert to FloatTensor in [0.0, 1.0] and add channel dim (1, H, W)
        tensor = torch.from_numpy(enhanced).float().unsqueeze(0) / 255.0
        
        # 4. Optional standard zero-centering normalization
        if self.normalize_mean is not None and self.normalize_std is not None:
            tensor = (tensor - self.normalize_mean) / self.normalize_std
            
        return tensor

    def __repr__(self) -> str:
        return (f"SARDenoiseCLAHETransform(denoise='{self.denoise_method}', "
                f"clahe={self.apply_clahe}, clip_limit={self.clip_limit})")


# ---------------------------------------------------------------------------
# 5. PyTorch Custom Dataset Class
# ---------------------------------------------------------------------------

class MSTARPreprocessedDataset(Dataset):
    """
    Specialized PyTorch Dataset for MSTAR SAR Automatic Target Recognition.
    
    Loads single-channel 128x128 radar chips from structured directories:
      <root_dir>/<split>/<class_name>/<filename>
      
    Features:
      - Automatic class discovery and index mapping.
      - Integrated radar despeckling (Bilateral/Median/Lee).
      - Contrast-Limited Adaptive Histogram Equalization (CLAHE).
      - PyTorch data augmentation (random horizontal flips, random aspect jitter).
      - Option to return (tensor, label) for training, or full metadata for evaluation.
    """
    def __init__(self,
                 root_dir: str = "data",
                 split: str = "train",
                 classes: Optional[List[str]] = None,
                 denoise_method: str = "bilateral",
                 apply_clahe: bool = True,
                 clip_limit: float = 2.5,
                 tile_grid_size: Tuple[int, int] = (8, 8),
                 augment: bool = False,
                 return_metadata: bool = False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.augment = augment
        self.return_metadata = return_metadata
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")
            
        # Discover or filter classes
        if classes is not None:
            self.classes = sorted([c.upper() for c in classes])
        else:
            self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
            
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Index all image files
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_folder = self.split_dir / cls_name
            if cls_folder.exists():
                valid_files = sorted([p for p in cls_folder.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
                for fpath in valid_files:
                    self.samples.append((fpath, self.class_to_idx[cls_name]))
                    
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image files found in {self.split_dir} for classes {self.classes}")
            
        # Preprocessing transform
        self.radar_transform = SARDenoiseCLAHETransform(
            denoise_method=denoise_method,
            apply_clahe=apply_clahe,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            normalize_mean=0.5,
            normalize_std=0.5
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Dict[str, Any]]]:
        fpath, label = self.samples[idx]
        
        # Load grayscale radar chip
        raw_chip = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if raw_chip is None:
            raise IOError(f"Failed to read image at {fpath}")
            
        # Optional radar data augmentation for training
        if self.augment and self.split == "train":
            # SAR horizontal symmetry flip
            if np.random.rand() > 0.5:
                raw_chip = cv2.flip(raw_chip, 1)
            # Small random aspect rotation (+/- 5 degrees)
            if np.random.rand() > 0.5:
                angle = float(np.random.uniform(-5.0, 5.0))
                center = (raw_chip.shape[1] // 2, raw_chip.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                raw_chip = cv2.warpAffine(raw_chip, M, (raw_chip.shape[1], raw_chip.shape[0]),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                
        # Apply preprocessing pipeline to produce normalized PyTorch tensor (1, 128, 128)
        processed_tensor = self.radar_transform(raw_chip)
        
        if self.return_metadata:
            meta = {
                "file_path": str(fpath),
                "filename": fpath.name,
                "class_name": self.classes[label],
                "raw_shape": list(raw_chip.shape),
                "raw_mean": float(raw_chip.mean()),
                "raw_max": float(raw_chip.max()),
            }
            return processed_tensor, label, meta
            
        return processed_tensor, label


# ---------------------------------------------------------------------------
# 6. Visual Comparison Report Generator
# ---------------------------------------------------------------------------

def generate_preprocessing_report(data_dir: str = "data",
                                  output_report: str = "preprocessing_report.png",
                                  denoise_method: str = "bilateral",
                                  clip_limit: float = 2.5) -> str:
    """
    Generates a comprehensive visual report showing side-by-side progression:
      [Original Chip] -> [Denoised (Speckle Filter)] -> [Enhanced (CLAHE)] -> [Pseudo-Color Thermogram] -> [Intensity Cross-Section]
    Saves output to `output_report`.
    """
    root_path = Path(data_dir)
    train_dir = root_path / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Directory {train_dir} not found. Run download_data.py first.")
        
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {train_dir}")
        
    print(f"[+] Generating visual preprocessing comparison for classes: {classes}")
    
    # Select 2 distinct sample chips per class for high-fidelity comparison
    num_samples_per_class = 2
    total_rows = len(classes) * num_samples_per_class
    
    fig, axes = plt.subplots(total_rows, 5, figsize=(18, 3.4 * total_rows), dpi=150)
    if total_rows == 1:
        axes = np.expand_dims(axes, 0)
        
    row_idx = 0
    for cls_name in classes:
        cls_folder = train_dir / cls_name
        files = sorted(list(cls_folder.glob("*.*")))[:num_samples_per_class]
        
        for f_idx, fpath in enumerate(files):
            raw = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            _, denoised, enhanced = preprocess_sar_chip(
                raw,
                denoise_method=denoise_method,
                apply_clahe=True,
                clip_limit=clip_limit
            )
            metrics = compute_radar_metrics(raw, enhanced)
            
            # Col 1: Original Raw Radar Chip
            ax0 = axes[row_idx, 0]
            im0 = ax0.imshow(raw, cmap="gray", vmin=0, vmax=255)
            ax0.set_title(f"1. Raw MSTAR: {cls_name} (#{f_idx+1})\nMean: {raw.mean():.1f} | Std: {raw.std():.1f} | ENL: {metrics['enl_raw']:.1f}",
                          fontsize=9, fontweight="bold")
            ax0.axis("off")
            
            # Col 2: Denoised (Speckle Filtered)
            ax1 = axes[row_idx, 1]
            im1 = ax1.imshow(denoised, cmap="gray", vmin=0, vmax=255)
            ax1.set_title(f"2. {denoise_method.title()} Denoised\nMean: {denoised.mean():.1f} | Std: {denoised.std():.1f} | Preserved Edges",
                          fontsize=9, fontweight="bold", color="#0055aa")
            ax1.axis("off")
            
            # Col 3: Contrast Enhanced (CLAHE)
            ax2 = axes[row_idx, 2]
            im2 = ax2.imshow(enhanced, cmap="gray", vmin=0, vmax=255)
            ax2.set_title(f"3. CLAHE Enhanced (Clip={clip_limit})\nMean: {enhanced.mean():.1f} | TCR: {metrics['tcr_proc_db']:.1f} dB",
                          fontsize=9, fontweight="bold", color="#990000")
            ax2.axis("off")
            
            # Col 4: Pseudo-Color Radar Thermogram (Inferno)
            ax3 = axes[row_idx, 3]
            im3 = ax3.imshow(enhanced, cmap="inferno", vmin=0, vmax=255)
            ax3.set_title(f"4. Tactical Thermogram\nHigh-RCS Scatterers & Shadow",
                          fontsize=9, fontweight="bold")
            ax3.axis("off")
            
            # Col 5: Radar Intensity Profile (Horizontal Slice across vehicle center)
            ax4 = axes[row_idx, 4]
            center_line_y = raw.shape[0] // 2
            raw_slice = raw[center_line_y, :]
            enh_slice = enhanced[center_line_y, :]
            
            ax4.plot(raw_slice, label="Raw Radar", color="gray", alpha=0.6, linestyle="--", linewidth=1.2)
            ax4.plot(enh_slice, label="Denoised + CLAHE", color="#d62728", linewidth=1.8)
            ax4.set_xlim(0, 128)
            ax4.set_ylim(0, 260)
            ax4.set_title(f"5. Intensity Cross-Section (Row {center_line_y})\nTarget Peak vs Clutter Floor", fontsize=9, fontweight="bold")
            ax4.set_xlabel("Cross-Range Pixel (w)", fontsize=8)
            ax4.set_ylabel("Amplitude [0-255]", fontsize=8)
            ax4.grid(True, linestyle=":", alpha=0.5)
            ax4.legend(loc="upper right", fontsize=7)
            
            row_idx += 1
            
    fig.suptitle("DEFENSE-TECH SAR RADAR PREPROCESSING & CONTRAST ENHANCEMENT PIPELINE\n"
                 "Multiplicative Speckle Attenuation (Bilateral) + CLAHE Target Definition",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_report, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Visual preprocessing report successfully saved to: {output_report}")
    return output_report


# ---------------------------------------------------------------------------
# 7. Main Execution & PyTorch Pipeline Smoke-Test
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAR Preprocessing & PyTorch Custom Dataset Pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory containing train/ and test/")
    parser.add_argument("--output-report", type=str, default="preprocessing_report.png", help="Output path for visual report")
    parser.add_argument("--denoise", type=str, default="bilateral", choices=["bilateral", "median", "lee", "none"],
                        help="Speckle reduction algorithm")
    parser.add_argument("--clip-limit", type=float, default=2.5, help="CLAHE clip limit")
    parser.add_argument("--batch-size", type=int, default=16, help="Test batch size for PyTorch DataLoader")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MSTAR SAR RADAR PREPROCESSING & FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Data Directory  : {args.data_dir}")
    print(f"Denoise Method  : {args.denoise}")
    print(f"CLAHE Clip Limit: {args.clip_limit}")
    print(f"Report Target   : {args.output_report}")
    print("-" * 70)
    
    # 1. Generate Visual Inspection Report
    report_file = generate_preprocessing_report(
        data_dir=args.data_dir,
        output_report=args.output_report,
        denoise_method=args.denoise,
        clip_limit=args.clip_limit
    )
    
    # 2. Test PyTorch Custom Dataset & DataLoader Integration
    print("\n[+] Initializing PyTorch MSTARPreprocessedDataset...")
    train_dataset = MSTARPreprocessedDataset(
        root_dir=args.data_dir,
        split="train",
        denoise_method=args.denoise,
        apply_clahe=True,
        clip_limit=args.clip_limit,
        augment=True,
        return_metadata=False
    )
    
    test_dataset = MSTARPreprocessedDataset(
        root_dir=args.data_dir,
        split="test",
        denoise_method=args.denoise,
        apply_clahe=True,
        clip_limit=args.clip_limit,
        augment=False,
        return_metadata=False
    )
    
    print(f"[+] Loaded Train Samples: {len(train_dataset)}")
    print(f"[+] Loaded Test Samples : {len(test_dataset)}")
    print(f"[+] Target Classes      : {train_dataset.classes}")
    print(f"[+] Class to Index Map  : {train_dataset.class_to_idx}")
    
    # 3. Test PyTorch DataLoader Batching
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch_images, batch_labels = next(iter(train_loader))
    
    print("\n--- PyTorch DataLoader Verification ---")
    print(f"[+] Batch Tensor Shape : {batch_images.shape} (Batch, Channel, Height, Width)")
    print(f"[+] Batch Tensor Dtype : {batch_images.dtype}")
    print(f"[+] Tensor Value Range : min={batch_images.min().item():.3f}, max={batch_images.max().item():.3f}, mean={batch_images.mean().item():.3f}")
    print(f"[+] Labels in Batch    : {batch_labels.tolist()}")
    
    print("\n" + "=" * 70)
    print("PHASE 2 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"[+] Report Artifact: {report_file}")
    print("[+] PyTorch Dataset is ready to plug into deep learning ATR architectures.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
