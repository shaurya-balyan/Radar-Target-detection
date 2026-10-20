"""
synthetic_radar.py - High-Fidelity Synthetic SAR (Synthetic Aperture Radar) Generator
======================================================================================
Simulates realistic 128x128 single-channel SAR chips modeling the DARPA/AFRL MSTAR
characteristics for military vehicle target recognition:
  - Multiplicative coherent speckle noise (Rayleigh / Gamma distribution)
  - Vehicle target geometric silhouettes and class-specific structures (T72, BMP2, BTR70, etc.)
  - Realistic aspect angle rotation (0 to 360 degrees)
  - Radar shadow occlusion regions along slant-range illumination
  - Specular point scattering centers (trihedrals/dihedrals with radar point spread function)
"""

import os
import math
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


# Standard vehicle physical footprint parameters (in pixels for 128x128 chips at 0.3m resolution)
VEHICLE_SPECS: Dict[str, Dict[str, Any]] = {
    "T72": {
        "hull_length": 46,
        "hull_width": 26,
        "turret_radius": 10,
        "turret_offset": (-2, 0),
        "barrel_length": 32,
        "barrel_width": 2.5,
        "has_tracks": True,
        "num_scatterers": 16,
        "shadow_length": 18,
        "type": "tank",
    },
    "BMP2": {
        "hull_length": 42,
        "hull_width": 22,
        "turret_radius": 8,
        "turret_offset": (-4, 0),
        "barrel_length": 22,
        "barrel_width": 2.0,
        "has_tracks": True,
        "num_scatterers": 13,
        "shadow_length": 14,
        "type": "ifv",
    },
    "BTR70": {
        "hull_length": 48,
        "hull_width": 20,
        "turret_radius": 6,
        "turret_offset": (-3, 0),
        "barrel_length": 14,
        "barrel_width": 1.5,
        "has_tracks": False,  # 8-wheel APC
        "num_scatterers": 12,
        "shadow_length": 12,
        "type": "apc",
    },
    "2S1": {
        "hull_length": 48,
        "hull_width": 24,
        "turret_radius": 11,
        "turret_offset": (4, 0),
        "barrel_length": 36,
        "barrel_width": 3.0,
        "has_tracks": True,
        "num_scatterers": 15,
        "shadow_length": 20,
        "type": "howitzer",
    },
    "BRDM2": {
        "hull_length": 36,
        "hull_width": 18,
        "turret_radius": 6,
        "turret_offset": (0, 0),
        "barrel_length": 12,
        "barrel_width": 1.5,
        "has_tracks": False,
        "num_scatterers": 10,
        "shadow_length": 10,
        "type": "scout",
    },
    "BTR60": {
        "hull_length": 46,
        "hull_width": 20,
        "turret_radius": 6,
        "turret_offset": (-2, 0),
        "barrel_length": 12,
        "barrel_width": 1.5,
        "has_tracks": False,
        "num_scatterers": 11,
        "shadow_length": 12,
        "type": "apc",
    },
    "D7": {
        "hull_length": 38,
        "hull_width": 26,
        "turret_radius": 0,
        "turret_offset": (0, 0),
        "barrel_length": 0,
        "barrel_width": 0,
        "has_tracks": True,
        "num_scatterers": 18,
        "shadow_length": 16,
        "type": "bulldozer",
    },
    "T62": {
        "hull_length": 45,
        "hull_width": 25,
        "turret_radius": 9.5,
        "turret_offset": (-1, 0),
        "barrel_length": 30,
        "barrel_width": 2.5,
        "has_tracks": True,
        "num_scatterers": 15,
        "shadow_length": 17,
        "type": "tank",
    },
    "ZIL131": {
        "hull_length": 46,
        "hull_width": 18,
        "turret_radius": 0,
        "turret_offset": (0, 0),
        "barrel_length": 0,
        "barrel_width": 0,
        "has_tracks": False,
        "num_scatterers": 14,
        "shadow_length": 14,
        "type": "truck",
    },
    "ZSU23_4": {
        "hull_length": 44,
        "hull_width": 24,
        "turret_radius": 12,
        "turret_offset": (-1, 0),
        "barrel_length": 20,
        "barrel_width": 4.0,  # Quad autocannon cluster
        "has_tracks": True,
        "num_scatterers": 20,
        "shadow_length": 18,
        "type": "spaag",
    },
}


def generate_speckle_noise(shape: Tuple[int, int] = (128, 128),
                           scale: float = 14.0,
                           num_looks: int = 1) -> np.ndarray:
    """
    Simulates coherent radar clutter using multi-look Rayleigh distribution.
    In SAR, speckle amplitude sqrt(I^2 + Q^2) follows a Rayleigh distribution:
      mean = scale * sqrt(pi / 2) ≈ scale * 1.253.
    For scale=14.5, mean amplitude is ~18.2, matching real MSTAR clutter.
    """
    real_part = np.random.normal(0, scale / np.sqrt(num_looks), size=(num_looks, *shape))
    imag_part = np.random.normal(0, scale / np.sqrt(num_looks), size=(num_looks, *shape))
    amplitude = np.sqrt(np.mean(real_part**2 + imag_part**2, axis=0))
    return amplitude.astype(np.float32)


def generate_target_mask(target_class: str,
                         aspect_deg: float,
                         shape: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates target body, shadow, and scattering center coordinates rotated by aspect angle.
    Returns:
        body_mask: float array in [0, 1] for target body
        shadow_mask: float array in [0, 1] for radar shadow
        scatterer_map: float array of point scattering peaks
    """
    h, w = shape
    center_y, center_x = h // 2, w // 2
    
    spec = VEHICLE_SPECS.get(target_class.upper(), VEHICLE_SPECS["T72"])
    
    # Local coordinate canvas with extra margins for rotation
    canvas_size = max(h, w) * 2
    local_body = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    local_scatterers = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    
    lc_y, lc_x = canvas_size // 2, canvas_size // 2
    
    # 1. Base hull polygon
    hl = spec["hull_length"]
    hw = spec["hull_width"]
    pts = np.array([
        [lc_x - hl // 2, lc_y - hw // 2],
        [lc_x + hl // 2 - 4, lc_y - hw // 2],
        [lc_x + hl // 2, lc_y - hw // 4],
        [lc_x + hl // 2, lc_y + hw // 4],
        [lc_x + hl // 2 - 4, lc_y + hw // 2],
        [lc_x - hl // 2, lc_y + hw // 2],
    ], dtype=np.int32)
    cv2.fillPoly(local_body, [pts], 255)
    
    # Turret / superstructure
    tr = spec["turret_radius"]
    if tr > 0:
        to_x, to_y = spec["turret_offset"]
        t_center = (lc_x + to_x, lc_y + to_y)
        cv2.circle(local_body, t_center, int(tr), 255, -1)
        
        # Gun barrel
        bl = spec["barrel_length"]
        bw = spec["barrel_width"]
        if bl > 0:
            b_end = (int(t_center[0] + bl), int(t_center[1]))
            cv2.line(local_body, t_center, b_end, 255, max(1, int(bw)))
    
    # 2. Strong specular scattering centers (corner reflectors, gun mantlet, chassis edges)
    num_scat = spec["num_scatterers"]
    np.random.seed(int(abs(aspect_deg * 100)) % (2**31 - 1))
    
    if spec["has_tracks"]:
        # Periodic track and road-wheel specular scatterers
        for x_pos in np.linspace(lc_x - hl // 2 + 4, lc_x + hl // 2 - 4, 7):
            local_scatterers[int(lc_y - hw // 2), int(x_pos)] = np.random.uniform(180, 255)
            local_scatterers[int(lc_y + hw // 2), int(x_pos)] = np.random.uniform(180, 255)
    else:
        # Wheeled APC distinct wheel scatterers
        for x_pos in np.linspace(lc_x - hl // 2 + 5, lc_x + hl // 2 - 5, 4):
            local_scatterers[int(lc_y - hw // 2), int(x_pos)] = np.random.uniform(170, 240)
            local_scatterers[int(lc_y + hw // 2), int(x_pos)] = np.random.uniform(170, 240)
            
    # Specular returns from turret ring and gun mantlet
    if tr > 0:
        to_x, to_y = spec["turret_offset"]
        local_scatterers[lc_y + to_y, lc_x + to_x] = np.random.uniform(220, 255)
        local_scatterers[lc_y + to_y - 2, lc_x + to_x + int(tr // 2)] = np.random.uniform(210, 255)
        
    # Highlights inside the hull
    body_y, body_x = np.where(local_body > 0)
    if len(body_y) > 0:
        chosen_idx = np.random.choice(len(body_y), size=min(num_scat, len(body_y)), replace=False)
        for idx in chosen_idx:
            local_scatterers[body_y[idx], body_x[idx]] = np.random.uniform(180, 255)
            
    # 3. Radar Shadow: extends behind the vehicle in slant-range direction (+y)
    shadow_len = spec["shadow_length"]
    kernel_shadow = np.ones((shadow_len, 1), dtype=np.uint8)
    dilated_for_shadow = cv2.dilate(local_body, kernel_shadow, iterations=1)
    local_shadow = cv2.subtract(dilated_for_shadow, local_body)
    
    # 4. Rotate all layers by aspect angle
    M = cv2.getRotationMatrix2D((lc_x, lc_y), -aspect_deg, 1.0)
    rot_body = cv2.warpAffine(local_body, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
    rot_shadow = cv2.warpAffine(local_shadow, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
    rot_scatterers = cv2.warpAffine(local_scatterers, M, (canvas_size, canvas_size), flags=cv2.INTER_NEAREST)
    
    # 5. Crop back to centered (h, w)
    y1 = lc_y - h // 2
    y2 = y1 + h
    x1 = lc_x - w // 2
    x2 = x1 + w
    
    crop_body = (rot_body[y1:y2, x1:x2].astype(np.float32) / 255.0)
    crop_shadow = (rot_shadow[y1:y2, x1:x2].astype(np.float32) / 255.0)
    crop_scatterers = rot_scatterers[y1:y2, x1:x2]
    
    # Radar Point Spread Function (PSF): 2D sinc-like Gaussian beam profile
    psf_scatterers = cv2.GaussianBlur(crop_scatterers, (5, 5), sigmaX=1.1, sigmaY=1.1)
    
    return crop_body, crop_shadow, psf_scatterers


def generate_sar_chip(target_class: str = "T72",
                      aspect_deg: Optional[float] = None,
                      shape: Tuple[int, int] = (128, 128),
                      clutter_scale: float = 14.5,
                      target_scale: float = 55.0) -> np.ndarray:
    """
    Generates a single 128x128 SAR chip adhering to MSTAR physical characteristics:
    - Target body backscatter + speckle
    - Target shadow region (suppressed backscatter)
    - Strong point scattering centers with radar PSF
    - Background speckle clutter
    """
    if aspect_deg is None:
        aspect_deg = float(np.random.uniform(0.0, 360.0))
        
    body_mask, shadow_mask, scatterers = generate_target_mask(target_class, aspect_deg, shape)
    
    # Coherent background clutter (mean ~18, std ~10)
    clutter = generate_speckle_noise(shape, scale=clutter_scale, num_looks=1)
    
    # Radar shadow suppresses background clutter backscatter to near zero (< 3)
    shadow_factor = 1.0 - (shadow_mask * 0.92)
    base_image = clutter * shadow_factor
    
    # Target diffuse backscatter
    target_speckle = generate_speckle_noise(shape, scale=target_scale, num_looks=1)
    target_component = target_speckle * body_mask
    
    # Composite: replace clutter in target region with target response
    composite = base_image * (1.0 - body_mask) + target_component
    
    # Add specular point scatterers (peaks up to 255)
    composite = np.maximum(composite, scatterers)
    
    # Receiver thermal noise floor
    thermal_noise = np.random.normal(1.5, 0.8, size=shape)
    composite = np.maximum(composite + thermal_noise, 0.0)
    
    # Clip to 8-bit dynamic range [0, 255]
    chip_uint8 = np.clip(composite, 0, 255).astype(np.uint8)
    return chip_uint8


def generate_synthetic_sar_dataset(output_dir: str = "data",
                                   classes: Optional[list] = None,
                                   samples_per_class: int = 50,
                                   train_ratio: float = 0.8,
                                   shape: Tuple[int, int] = (128, 128),
                                   save_npy: bool = False) -> Dict[str, int]:
    """
    Generates a structured SAR training/testing dataset in `output_dir`:
      data/
        train/
          BMP2/
          T72/
          BTR70/
        test/
          BMP2/
          T72/
          BTR70/
    """
    if classes is None:
        classes = ["BMP2", "T72", "BTR70"]
        
    out_path = Path(output_dir)
    stats = {"train": 0, "test": 0}
    
    num_train = int(samples_per_class * train_ratio)
    num_test = samples_per_class - num_train
    
    for cls in classes:
        train_dir = out_path / "train" / cls
        test_dir = out_path / "test" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Training samples with uniform aspect angle coverage
        train_aspects = np.linspace(0, 360, num_train, endpoint=False) + np.random.uniform(-3, 3, num_train)
        for i, asp in enumerate(train_aspects):
            chip = generate_sar_chip(cls, aspect_deg=asp, shape=shape)
            filename = f"synth_{cls}_train_{i:04d}_deg{int(asp)%360:03d}.png"
            cv2.imwrite(str(train_dir / filename), chip)
            if save_npy:
                np.save(str(train_dir / filename.replace(".png", ".npy")), chip)
            stats["train"] += 1
            
        # Testing samples with interleaved aspect angles
        test_aspects = np.linspace(0, 360, num_test, endpoint=False) + np.random.uniform(5, 12, num_test)
        for i, asp in enumerate(test_aspects):
            chip = generate_sar_chip(cls, aspect_deg=asp, shape=shape)
            filename = f"synth_{cls}_test_{i:04d}_deg{int(asp)%360:03d}.png"
            cv2.imwrite(str(test_dir / filename), chip)
            if save_npy:
                np.save(str(test_dir / filename.replace(".png", ".npy")), chip)
            stats["test"] += 1
            
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic SAR radar chips")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--classes", nargs="+", default=["BMP2", "T72", "BTR70"], help="Vehicle classes")
    parser.add_argument("--samples", type=int, default=50, help="Samples per class")
    parser.add_argument("--save-npy", action="store_true", help="Also save .npy raw matrices")
    args = parser.parse_args()
    
    print(f"Generating synthetic SAR dataset for classes: {args.classes}")
    counts = generate_synthetic_sar_dataset(
        output_dir=args.output_dir,
        classes=args.classes,
        samples_per_class=args.samples,
        save_npy=args.save_npy
    )
    print(f"Dataset generated successfully! Counts: {counts}")
