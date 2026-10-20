# Radar Target Detection & Automatic Target Recognition (ATR)
### Defense-Tech Deep Learning Suite for Synthetic Aperture Radar (SAR) - MSTAR Dataset

---

## 1. Project Overview
This repository contains a defense-grade machine learning foundation for **Automatic Target Recognition (ATR)** operating on **Synthetic Aperture Radar (SAR)** imagery, centered on the benchmark DARPA/AFRL **MSTAR** (Moving and Stationary Target Acquisition and Recognition) dataset.

Unlike optical computer vision, SAR images are formed by coherent microwave backscatter (X-band, 1ft $\times$ 1ft resolution, HH polarization). SAR images present unique physical phenomena:
* **Multiplicative Speckle Noise**: High-frequency interference modeling coherent Rayleigh/Gamma distributed clutter.
* **Specular Scattering Centers**: Dihedral and trihedral reflectors (turrets, gun barrels, wheel bogies) producing high-amplitude returns with radar Point Spread Functions (PSF).
* **Radar Shadow Regions**: Low-intensity occlusions cast along the radar slant-range illumination vector.

---

## 2. Directory Structure

```text
Radar Target detection/
├── .venv/                      # Isolated Python 3.13 virtual environment
├── data/                       # Ingested & organized SAR dataset
│   ├── train/                  # Training split (128x128 single-channel chips)
│   │   ├── BMP2/               # Infantry Fighting Vehicle
│   │   ├── BTR70/              # Armored Personnel Carrier (8x8)
│   │   └── T72/                # Main Battle Tank
│   ├── test/                   # Testing split
│   │   ├── BMP2/
│   │   ├── BTR70/
│   │   └── T72/
│   └── metadata.json           # Data lineage, sensor params & sample counts
├── download_data.py            # Automated multi-mirror ingestion & fallback pipeline
├── synthetic_radar.py          # SAR physics simulation engine (Rayleigh speckle & target modeling)
├── verify_data.py              # PyTorch ImageFolder/DataLoader validation & preview generator
├── dataset_preview.png         # Visualized 128x128 radar chips across classes
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git exclusions (venv, models, caches)
└── README.md                   # System documentation & quickstart
```

---

## 3. Environment Setup & Activation

The virtual environment has been created and populated with all required dependencies:
* `torch` & `torchvision` (Deep Learning & Neural Network backbones)
* `opencv-python` & `scipy` (Radar signal & image processing)
* `numpy` & `pandas` (Array math & metadata handling)
* `scikit-learn` (Baseline ML models & performance metrics)
* `matplotlib` (Radar signature visualization & plotting)
* `streamlit` (Tactical radar web dashboard)

### Activating the Virtual Environment (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If prompted by PowerShell execution policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

---

## 4. Data Pipeline Execution (`download_data.py`)

The data pipeline script automatically resolves open-source mirrors and includes an autonomous offline fallback:

### Standard Ingestion (BMP2, T72, BTR70)
```powershell
.\.venv\Scripts\python.exe download_data.py --classes BMP2 T72 BTR70 --samples-per-class 50
```

### Full 10-Class MSTAR Ingestion
```powershell
.\.venv\Scripts\python.exe download_data.py --classes BMP2 T72 BTR70 2S1 BRDM2 BTR60 D7 T62 ZIL131 ZSU234 --samples-per-class 50
```

### Direct Synthetic Radar Generation (Offline / Air-Gapped)
If network access is blocked or you want to generate purely simulated radar chips for unit testing:
```powershell
.\.venv\Scripts\python.exe download_data.py --synthetic-only --classes BMP2 T72 BTR70 --samples-per-class 60 --save-npy
```

---

## 5. Synthetic Radar Physics Engine (`synthetic_radar.py`)

If remote downloads are blocked by authentication, firewall, or air-gapping, `synthetic_radar.py` synthesizes calibrated 128x128 radar arrays:
1. **Coherent Rayleigh Clutter**: Models background ground return with multi-look speckle:
   $$A = \sqrt{\text{Re}^2 + \text{Im}^2}, \quad \text{Re}, \text{Im} \sim \mathcal{N}(0, \sigma^2)$$
2. **Geometric Silhouettes**: Accurate aspect rotation ($\theta \in [0^\circ, 360^\circ]$) for tanks (T72), IFVs (BMP2), and wheeled APCs (BTR70).
3. **Corner Scatterers**: Metallic point scatterers convolved with 2D Gaussian radar point spread functions (PSF).
4. **Radar Shadows**: Slant-range occlusion where radar returns drop toward the receiver thermal noise floor.

---

---

## 6. Verifying the Pipeline (`verify_data.py`)

Run the verification script to test basic PyTorch `torchvision.datasets.ImageFolder` and `DataLoader` integration:
```powershell
.\.venv\Scripts\python.exe verify_data.py
```
This produces `dataset_preview.png` displaying samples from each target class rendered with radar colormaps.

---

## 7. Phase 2: Radar Preprocessing & Feature Extraction (`preprocess.py`)

SAR images exhibit multiplicative coherent speckle and low dynamic range contrast. `preprocess.py` provides a specialized defense-tech preprocessing suite:

### Core Preprocessing Operations
1. **Edge-Preserving Speckle Attenuation**:
   * **Bilateral Filter**: Non-linear filtering combining spatial proximity with radiometric range similarity, smoothing clutter while preserving the sharp metallic boundaries of vehicle hulls.
   * **Median Filter**: Non-linear rank filter suppressing impulse spikes without boundary smudging.
   * **Adaptive Lee Filter**: Exact SAR statistical filter utilizing local variance and 1-look coherent noise ratio ($C_u = 0.523$).
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**:
   * Localized tile-based histogram equalization ($8 \times 8$ grid, clip limit $2.5$) that stretches target returns without over-amplifying low-RCS background clutter.
3. **Seamless PyTorch Integration**:
   * `SARDenoiseCLAHETransform`: Torchvision-compatible callable transform producing normalized $(1, 128, 128)$ tensors.
   * `MSTARPreprocessedDataset`: Subclasses `torch.utils.data.Dataset`, supporting automatic class mapping, data augmentation (aspect rotation, flips), and zero-worker multi-process loading.

### Generating the Visual Preprocessing Report
```powershell
.\.venv\Scripts\python.exe preprocess.py --data-dir data --output-report preprocessing_report.png --denoise bilateral --clip-limit 2.5
```
This generates `preprocessing_report.png` containing a 5-column diagnostic view:
* **Raw MSTAR Radar Chip**
* **Denoised Chip** (Speckle suppressed)
* **CLAHE Enhanced Chip** (Target highlighted)
* **Tactical Thermogram** (Inferno colormap showing scattering centers & radar shadow)
* **Intensity Cross-Section** (Row slice of target peak vs clutter floor)

---

---

## 8. Phase 3: Deep Convolutional Neural Network Training (`train.py`)

A specialized 4-stage convolutional neural network (`RadarTargetCNN`) is implemented to classify single-channel 128x128 SAR targets (**BMP-2**, **BTR-70**, **T-72**).

### Architecture Highlights
* **Conv Blocks 1–4**: Feature channels $1 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 128$ with `BatchNorm2d`, `ReLU`, `MaxPool2d(2, 2)`, and spatial `Dropout2d` (0.10 to 0.25) to prevent overfitting on background clutter.
* **Global Pooling**: `AdaptiveAvgPool2d((4, 4))` producing 2,048 robust target features.
* **Regularized Classifier**: `Linear(2048, 256) -> BatchNorm1d -> ReLU -> Dropout(0.40) -> Linear(256, 3)`.
* **Optimization**: Adam (`lr=1e-3`, `weight_decay=1e-4`), StepLR scheduling (decay by 0.5 every 5 epochs), and `CrossEntropyLoss`.

### Running the Training Pipeline
```powershell
.\.venv\Scripts\python.exe train.py --epochs 10 --batch-size 32 --lr 0.001
```

### Generated Artifacts
* `best_model.pth`: Top validation checkpoint containing model weights, class indices, and training metadata.
* `training_performance.png`: Multi-panel convergence visualization showing:
  1. **Loss vs. Epochs** (Train Loss vs. Validation Loss with minimum marker)
  2. **Accuracy vs. Epochs** (Train Accuracy vs. Validation Accuracy)
  3. **Test Confusion Matrix** evaluating class discrimination on held-out test chips.

---

## 9. Phase 4: Tactical Radar Web Console (`app.py`)

A production-grade tactical combat radar console tailored with the signature **Indian Army Tactical Identity**:
* **Indian Army Tactical Aesthetic**: Midnight Black (`#0B0F12`), Indian Army Olive Green (`#3B5323` / `#4B6F44`), and Tactical Gold/Sand (`#C2B280` / `#D4AF37`) with tactical corner targeting brackets (`◤` / `◢`) and monospaced HUD typography (`'Share Tech Mono'`).
* **Header & Telemetry**:
  * Header: `⚔️ INDIAN ARMY TACTICAL RADAR SERVICE ⚔️ // AUTOMATIC TARGET RECOGNITION (ATR) CONSOLE`
  * Status Bar: `[NODE: SEC-SECTOR-ALPHA-9] | [STATUS: OPERATIONAL // LIVE FEED] | [BAND: X-BAND 9.60 GHz (HH)] | [RES: 0.3m x 0.3m] | [SECURITY: TOP SECRET // DEFENCE ONLY]`
* **Flashing Tactical Threat Alert Card**: Large, high-visibility pulsing alert card (`[ ! ] AI THREAT DETECTION: POSITIVE TARGET ISOLATED`) dynamically matching the identified platform threat rating (`CRITICAL THREAT // TIER-1 HEAVY ARMOR` for T-72, `HIGH THREAT` for BMP-2, etc.).
* **Browser-Native Base64 Audio Engine**: Synthesizes a mechanical rifle bolt-locking/chamber click WAV audio embedded directly as an inline Base64 HTML5 audio string (`data:audio/wav;base64,...`) triggering upon pulse transmission or target file ingestion with zero external asset dependencies.
* **Dynamic Index Synchronization Core**: Automatically extracts `class_to_idx` from `best_model.pth` and computes `idx_to_class[predicted_index]` to guarantee zero indexing mismatch between PyTorch logits and front-end intelligence dossiers.
* **Automated Feature Isolation Reticle**: OpenCV contour clustering drawing a steel-blue bounding box `[X, Y, W, H]` (`#4682B4`) and precise gold crosshair reticle ticks `[ + ]` (`#D4AF37`) centered on the vehicle centroid.
* **Live Radar Metrics Grid**: Computes real-time telemetry tracking raw vs. filtered signal mean, variance, Net TCR Gain ($+\text{dB}$), and Equivalent Number of Looks (ENL) clutter smoothing coefficient tracking live against the clutter scale.
* **Secure MILSIGREP Intel Export**: One-click download button (`IA-ATR-INTEL-REPORT.TXT`) packaging all radar telemetry, classification probability vectors, and tactical combat dossiers into a structured, authenticated military intelligence format.

### Launching the Tactical Console
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser to: **`http://localhost:8501`**






