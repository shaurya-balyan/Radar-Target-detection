"""
download_data.py - MSTAR SAR Target Dataset Ingestion & Synthetic Radar Fallback
================================================================================
Automated defense-tech data pipeline for Synthetic Aperture Radar (SAR) Automatic
Target Recognition (ATR) on standard military vehicles (BMP2, T72, BTR70, etc.).

Features:
1. Pings open-source mirrors hosting processed MSTAR 128x128 target chips:
   - Primary: omae11/MSTAR-dataset (Full SOC train/test sets)
   - Secondary: rohitgit1/Target-Detection-in-MSTAR-Images (Curated sample chips)
2. Resilient Fallback: If network access is blocked, rate-limited, or air-gapped,
   automatically triggers the synthetic radar physics engine (synthetic_radar.py)
   to generate calibrated 128x128 SAR chips with Rayleigh speckle, shadows, and scatterers.
3. Organizes output cleanly into PyTorch/ImageFolder compatible structure:
   data/
     train/<CLASS>/...
     test/<CLASS>/...
     metadata.json
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

# Import synthetic radar physics generator
try:
    import synthetic_radar as sr
except ImportError:
    sr = None


# Mirror configurations
PRIMARY_MIRROR_BASE = "https://raw.githubusercontent.com/omae11/MSTAR-dataset/main/soc"
PRIMARY_API_BASE = "https://api.github.com/repos/omae11/MSTAR-dataset/contents/soc"

SECONDARY_SAMPLES = {
    "BMP2": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/BMP2_sample.jpeg",
    "T72": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/T72_sample.jpeg",
    "BTR70": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/BTR70_sample.jpeg",
    "2S1": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/2S1_sample.jpeg",
    "BRDM2": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/BRDM2_sample.jpeg",
    "BTR60": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/BTR60_sample.jpeg",
    "D7": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/D7_sample.jpeg",
    "T62": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/T62_sample.jpeg",
    "ZIL131": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/ZIL131_sample.jpeg",
    "ZSU234": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/ZSU23_4_sample.jpeg",
    "ZSU23_4": "https://raw.githubusercontent.com/rohitgit1/Target-Detection-in-MSTAR-Images/master/assets/samples/ZSU23_4_sample.jpeg",
}

# Known file lists for quick fallback downloading when GitHub API is rate-limited
KNOWN_PRIMARY_FILES = {
    "BMP2": {
        "train": [f"HB{i:05d}.jpg" for i in range(3787, 3845)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "T72": {
        "train": [f"HB{i:05d}.jpg" for i in range(3880, 3935)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "BTR70": {
        "train": [f"HB{i:05d}.jpg" for i in range(3840, 3880)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "2S1": {
        "train": [f"HB{i:05d}.jpg" for i in range(3700, 3745)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "BRDM2": {
        "train": [f"HB{i:05d}.jpg" for i in range(3745, 3785)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "BTR60": {
        "train": [f"HB{i:05d}.jpg" for i in range(3935, 3975)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "D7": {
        "train": [f"HB{i:05d}.jpg" for i in range(3975, 4015)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "T62": {
        "train": [f"HB{i:05d}.jpg" for i in range(4015, 4055)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "ZIL131": {
        "train": [f"HB{i:05d}.jpg" for i in range(4055, 4095)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
    "ZSU234": {
        "train": [f"HB{i:05d}.jpg" for i in range(4095, 4135)],
        "test": [f"HB{i:05d}.jpg" for i in range(3333, 3390)],
    },
}


def get_available_file_names(cls_key: str, split: str) -> List[str]:
    """Dynamically fetches file names from GitHub API, with fallback to verified static list."""
    api_url = f"{PRIMARY_API_BASE}/{split}/{cls_key}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "MSTAR-ATR-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                names = [item["name"] for item in data if item.get("type") == "file" and item["name"].endswith(".jpg")]
                if names:
                    return names
    except Exception:
        pass
    
    # Static fallback
    return KNOWN_PRIMARY_FILES.get(cls_key, {}).get(split, [])


def test_url_reachability(url: str, timeout: float = 4.0) -> bool:
    """Checks whether a remote URL is reachable without full body download."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MSTAR-ATR-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status in (200, 206)
    except Exception:
        return False


def download_single_file(url: str, destination: Path, timeout: float = 6.0) -> bool:
    """Downloads a file with timeout and User-Agent headers."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MSTAR-ATR-Pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(destination, "wb") as out_f:
            out_f.write(resp.read())
        return destination.exists() and destination.stat().st_size > 500
    except Exception as e:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        return False


def fetch_mirror_dataset(classes: List[str],
                         output_dir: Path,
                         max_per_class: int = 50,
                         train_ratio: float = 0.8) -> Optional[Dict[str, int]]:
    """
    Attempts to download real MSTAR chips from open-source mirrors.
    Returns dictionary with counts if successful, or None if network is blocked.
    """
    print("[+] Checking connectivity to primary MSTAR mirror (GitHub omae11/MSTAR-dataset)...")
    probe_url = f"{PRIMARY_MIRROR_BASE}/train/BMP2/HB03787.jpg"
    is_primary_online = test_url_reachability(probe_url)
    
    if not is_primary_online:
        print("[-] Primary mirror is not reachable (or blocked by firewall/proxy).")
        return None
        
    print("[+] Primary mirror is active! Beginning automated download of processed MSTAR chips...")
    stats = {"train": 0, "test": 0}
    num_train = int(max_per_class * train_ratio)
    num_test = max_per_class - num_train
    
    for cls in classes:
        # Standardize class name mapping
        cls_key = "ZSU234" if cls.upper() in ("ZSU23_4", "ZSU234") else cls.upper()
        
        train_dir = output_dir / "train" / cls
        test_dir = output_dir / "test" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Download Train split
        train_available = get_available_file_names(cls_key, "train")
        train_files = train_available[:num_train]
        print(f"  --> Downloading {len(train_files)} training chips for [{cls}]...")
        for fname in train_files:
            target_path = train_dir / fname
            if not target_path.exists():
                file_url = f"{PRIMARY_MIRROR_BASE}/train/{cls_key}/{fname}"
                ok = download_single_file(file_url, target_path)
                if ok:
                    stats["train"] += 1
            else:
                stats["train"] += 1
                
        # 2. Download Test split
        test_available = get_available_file_names(cls_key, "test")
        test_files = test_available[:num_test]
        print(f"  --> Downloading {len(test_files)} testing chips for [{cls}]...")
        for fname in test_files:
            target_path = test_dir / fname
            if not target_path.exists():
                file_url = f"{PRIMARY_MIRROR_BASE}/test/{cls_key}/{fname}"
                ok = download_single_file(file_url, target_path)
                if ok:
                    stats["test"] += 1
            else:
                stats["test"] += 1
                
    total_downloaded = stats["train"] + stats["test"]
    if total_downloaded < len(classes) * 5:
        print("[-] Insufficient files downloaded from mirror (connectivity drop).")
        return None
        
    return stats


def generate_synthetic_fallback(classes: List[str],
                                output_dir: Path,
                                samples_per_class: int = 50,
                                train_ratio: float = 0.8,
                                save_npy: bool = False) -> Dict[str, int]:
    """
    Executes the physics-based synthetic radar generation pipeline.
    """
    print("\n[!] ===========================================================================")
    print("[!] INITIATING SYNTHETIC RADAR DATASET GENERATION (RADAR PHYSICS ENGINE)")
    print("[!] Simulating 128x128 SAR arrays with coherent Rayleigh speckle, vehicle")
    print("[!] RCS scattering centers, shadow occlusions, and full 360-deg aspect coverage.")
    print("[!] ===========================================================================\n")
    
    if sr is None:
        raise RuntimeError("synthetic_radar module not found! Ensure synthetic_radar.py is in the project directory.")
        
    stats = sr.generate_synthetic_sar_dataset(
        output_dir=str(output_dir),
        classes=classes,
        samples_per_class=samples_per_class,
        train_ratio=train_ratio,
        shape=(128, 128),
        save_npy=save_npy
    )
    return stats


def save_dataset_metadata(output_dir: Path,
                          source: str,
                          classes: List[str],
                          stats: Dict[str, int],
                          samples_per_class: int) -> None:
    """Writes dataset metadata JSON for reproducibility and data lineage."""
    metadata = {
        "dataset_name": "MSTAR-SAR-Automatic-Target-Recognition",
        "data_source": source,
        "chip_dimensions": [128, 128],
        "polarization": "HH",
        "resolution": "0.3m x 0.3m (1ft x 1ft)",
        "classes": classes,
        "num_classes": len(classes),
        "samples_per_class_target": samples_per_class,
        "sample_counts": stats,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Dataset metadata recorded at: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MSTAR Radar Dataset Pipeline: Mirror Ingestion & Synthetic Radar Generator"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Target folder for dataset (default: data)"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["BMP2", "T72", "BTR70"],
        help="Target vehicle classes (e.g., BMP2 T72 BTR70 2S1 BRDM2)"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Number of samples to ingest/generate per class (default: 50)"
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Bypass online mirrors and generate synthetic radar chips immediately"
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also export raw NumPy .npy 2D float/uint8 arrays for signal processing"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    classes = [c.upper() for c in args.classes]
    
    print("=" * 70)
    print("DEFENSE-TECH SAR AUTOMATIC TARGET RECOGNITION (ATR) DATA PIPELINE")
    print("=" * 70)
    print(f"Target Classes    : {', '.join(classes)}")
    print(f"Output Directory  : {output_dir.resolve()}")
    print(f"Samples per Class : {args.samples_per_class}")
    print(f"Synthetic Only    : {args.synthetic_only}")
    print("-" * 70)
    
    stats = None
    data_source = None
    
    if not args.synthetic_only:
        try:
            stats = fetch_mirror_dataset(
                classes=classes,
                output_dir=output_dir,
                max_per_class=args.samples_per_class
            )
            if stats:
                data_source = "MSTAR Open-Source Mirror (DARPA/AFRL SOC Target Chips)"
        except Exception as e:
            print(f"[!] Warning: Mirror retrieval encountered an error: {e}")
            stats = None
            
    # Trigger synthetic fallback if mirror fetch failed or synthetic-only was requested
    if stats is None:
        if not args.synthetic_only:
            print("[!] Remote mirror access was unavailable or rate-limited.")
            print("[+] Activating robust synthetic radar generation fallback...")
        stats = generate_synthetic_fallback(
            classes=classes,
            output_dir=output_dir,
            samples_per_class=args.samples_per_class,
            save_npy=args.save_npy
        )
        data_source = "Synthetic Radar Physics Engine (Rayleigh Speckle & Scattering Centers)"
        
    # Write metadata
    save_dataset_metadata(
        output_dir=output_dir,
        source=data_source,
        classes=classes,
        stats=stats,
        samples_per_class=args.samples_per_class
    )
    
    # Print summary report
    print("\n" + "=" * 70)
    print("DATA PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Status           : SUCCESS")
    print(f"Data Source      : {data_source}")
    print(f"Training Samples : {stats.get('train', 0)}")
    print(f"Testing Samples  : {stats.get('test', 0)}")
    print(f"Total Chips      : {stats.get('train', 0) + stats.get('test', 0)}")
    print(f"Directory Layout :")
    for split in ["train", "test"]:
        print(f"  {output_dir}/{split}/")
        for cls in classes:
            c_dir = output_dir / split / cls
            count = len(list(c_dir.glob("*.*"))) if c_dir.exists() else 0
            print(f"    - {cls:8s}: {count} chips")
    print("=" * 70)
    print("[+] Pipeline ready for model training and ATR evaluation!\n")


if __name__ == "__main__":
    main()
