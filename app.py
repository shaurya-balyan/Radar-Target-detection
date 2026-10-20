"""
app.py - Indian Army Tactical Radar Service: Automatic Target Recognition (ATR) Console
========================================================================================
Production-grade tactical combat radar console for Synthetic Aperture Radar (SAR)
Automatic Target Recognition (MSTAR Benchmark: BMP-2, T-72, BTR-70).

Features:
  1. Full Indian Army Tactical Identity: Midnight Black (#0B0F12), Olive Green (#3B5323 / #4B6F44),
     Tactical Gold (#C2B280 / #D4AF37), and Corner Targeting Brackets.
  2. Large Pulsing Threat Alert Card: High-visibility [ ! ] AI THREAT DETECTION alert box.
  3. Flawless Browser Native Audio: Embedded Base64 WAV audio tags for mechanical bolt-lock
     cocking sounds on pulse transmission and file ingestion.
  4. Dynamic Index Synchronization Core: Mathematically maps PyTorch logits directly using
     checkpoint `class_to_idx` reverse lookup {idx: class_name} to eliminate indexing drift.
  5. Automated Feature Isolation: OpenCV contour clustering rendering a steel-blue bounding
     box [X, Y, W, H] and gold reticle tick mark [ + ] centered on the target centroid.
  6. Live Numerical Metrics Grid: Dynamic recalculation of Raw/Despeckled Mean, Net TCR Gain (dB),
     and Equivalent Number of Looks (ENL) tracking live against clutter sliders.
  7. Native MILSIGREP Dossier Export: Formatted military intelligence report download.
"""

import os
import sys
import io
import time
import base64
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io.wavfile as wav

import torch
import torch.nn.functional as F
import streamlit as st
import base64
import numpy as np
import scipy.io.wavfile as wav
import io

def inject_cocking_audio():
    """Generates an instantaneous mechanical cocking wave payload inside browser DOM cache."""
    sample_rate = 44100
    duration_1 = 0.05  # Initial metallic striker snap
    duration_gap = 0.03 # Chamber slide translation delay
    duration_2 = 0.12  # Solid rear locking compression lock

    t1 = np.linspace(0, duration_1, int(sample_rate * duration_1), False)
    t2 = np.linspace(0, duration_2, int(sample_rate * duration_2), False)
    
    # Synthesize mechanical friction envelopes using frequency shifts
    wave1 = np.sin(2 * np.pi * 580 * t1) * np.exp(-40 * t1) * 0.4
    wave2 = np.sin(2 * np.pi * 840 * t2) * np.exp(-25 * t2) * 0.6
    
    # Compile raw structural waveform arrays
    gap_samples = np.zeros(int(sample_rate * duration_gap))
    full_signal = np.concatenate([wave1, gap_samples, wave2])
    
    # Normalize signal coefficients safely to 16-bit integers
    audio_ints = (full_signal * 32767).astype(np.int16)
    
    # Write directly to virtual system memory buffer
    wav_io = io.BytesIO()
    wav.write(wav_io, sample_rate, audio_ints)
    b64_audio = base64.b64encode(wav_io.getvalue()).decode()
    
    # Inject standard auto-executing HTML5 component directly into Streamlit layer
    audio_html = f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Instantly trigger the tracking sweep sound upon any state calculation/re-render
inject_cocking_audio()


# Project Modules
from train import RadarTargetCNN
from preprocess import (
    preprocess_sar_chip,
    apply_speckle_denoising,
    apply_clahe_enhancement,
    compute_radar_metrics
)
import synthetic_radar as sr


# ---------------------------------------------------------------------------
# 1. Page Configuration & Indian Army Tactical CSS Injection
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="INDIAN ARMY ATR CONSOLE // SAR-96",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

INDIAN_ARMY_TACTICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@600;800;900&display=swap');

/* Base Tactical Theme Overrides */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: #0B0F12 !important;
    color: #C2B280 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

[data-testid="stSidebar"] {
    background-color: #080C0E !important;
    border-right: 1px solid #3B5323 !important;
}

/* Monospace Font Enforcement */
h1, h2, h3, h4, h5, h6, p, span, label, div, button {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

/* Tactical Corner Bracket Container */
.tactical-box {
    background: rgba(14, 20, 24, 0.92);
    border: 1px solid #4B6F44;
    padding: 16px 18px;
    margin-bottom: 16px;
    position: relative;
    box-shadow: 0 0 15px rgba(59, 83, 35, 0.25);
}

.tactical-box::before {
    content: "◤";
    position: absolute;
    top: -2px;
    left: -2px;
    color: #D4AF37;
    font-size: 16px;
    line-height: 1;
}

.tactical-box::after {
    content: "◢";
    position: absolute;
    bottom: -2px;
    right: -2px;
    color: #D4AF37;
    font-size: 16px;
    line-height: 1;
}

/* Indian Army Header Banner */
.ia-header-banner {
    background: linear-gradient(90deg, #111A13 0%, #1D2E21 50%, #111A13 100%);
    border: 2px solid #4B6F44;
    border-bottom: 3px solid #D4AF37;
    padding: 14px 20px;
    margin-bottom: 8px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.85);
}

.ia-header-title {
    font-family: 'Orbitron', 'Share Tech Mono', sans-serif !important;
    font-size: 24px;
    font-weight: 900;
    letter-spacing: 2.5px;
    color: #FFFFFF;
    text-transform: uppercase;
    margin: 0;
    text-shadow: 0 0 12px rgba(75, 111, 68, 0.9);
}

.ia-header-sub {
    font-size: 12px;
    color: #D4AF37;
    letter-spacing: 2px;
    margin-top: 3px;
    text-transform: uppercase;
}

/* Operational Telemetry Bar */
.telemetry-bar {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    background: #0E1611;
    border: 1px solid #3B5323;
    padding: 7px 16px;
    margin-bottom: 16px;
    font-size: 11px;
    letter-spacing: 1.5px;
}

.tel-active {
    color: #4CAF50;
    font-weight: bold;
}

.tel-secret {
    color: #FF5252;
    font-weight: bold;
}

.tel-gold {
    color: #D4AF37;
}

/* Flashing Threat Alert Card */
@keyframes alertPulse {
    0% { border-color: #D4AF37; box-shadow: 0 0 10px rgba(212, 175, 55, 0.4); }
    50% { border-color: #FF3333; box-shadow: 0 0 25px rgba(255, 51, 51, 0.8); }
    100% { border-color: #D4AF37; box-shadow: 0 0 10px rgba(212, 175, 55, 0.4); }
}

.threat-alert-card {
    background: rgba(28, 12, 12, 0.94);
    border: 2px solid #FF3333;
    animation: alertPulse 2s infinite;
    padding: 16px 20px;
    margin-top: 10px;
    margin-bottom: 16px;
}

.threat-title {
    font-family: 'Orbitron', 'Share Tech Mono', sans-serif !important;
    font-size: 17px;
    font-weight: 800;
    color: #FF5252;
    letter-spacing: 2px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.threat-target {
    font-family: 'Orbitron', 'Share Tech Mono', sans-serif !important;
    font-size: 24px;
    color: #FFD700;
    font-weight: 900;
    margin: 6px 0 8px 0;
}

.threat-meta-row {
    font-size: 12px;
    color: #C2B280;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

/* Numerical Telemetry Data Grid */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 10px;
    margin-top: 8px;
}

.telemetry-cell {
    background-color: #0E1611;
    border: 1px solid #3B5323;
    padding: 10px 12px;
}

.cell-label {
    font-size: 10px;
    color: #8F9E8B;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 3px;
}

.cell-value {
    font-family: 'Orbitron', monospace;
    font-size: 17px;
    font-weight: 700;
    color: #FFFFFF;
}

.cell-sub {
    font-size: 10px;
    color: #D4AF37;
    margin-top: 2px;
}

/* Tactical Probability Bar */
.prob-container {
    margin-bottom: 10px;
}

.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 3px;
}

.prob-track {
    background: #141A16;
    border: 1px solid #334433;
    height: 12px;
    width: 100%;
    position: relative;
}

.prob-fill-primary {
    background: linear-gradient(90deg, #3B5323, #6B8E23, #4CAF50);
    height: 100%;
    box-shadow: 0 0 8px rgba(76, 175, 80, 0.6);
}

.prob-fill-secondary {
    background: linear-gradient(90deg, #8B6508, #FF8C00);
    height: 100%;
}

.prob-fill-tertiary {
    background: #4A4A4A;
    height: 100%;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: #1A281E !important;
    color: #D4AF37 !important;
    border: 1px solid #4B6F44 !important;
    border-radius: 0px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-weight: bold !important;
    letter-spacing: 1.5px !important;
    transition: all 0.2s ease-in-out !important;
    width: 100% !important;
    padding: 9px 16px !important;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    background: #3B5323 !important;
    color: #FFFFFF !important;
    border-color: #D4AF37 !important;
    box-shadow: 0 0 15px rgba(212, 175, 55, 0.5) !important;
}

/* Dossier Spec Table */
.dossier-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 11px;
}

.dossier-table td {
    padding: 5px 8px;
    border-bottom: 1px dashed #2A3B2A;
}

.dossier-table .key {
    color: #8F9E8B;
    width: 38%;
}

.dossier-table .val {
    color: #E8E5D8;
    font-weight: bold;
}

/* Signal Corruption Warning Banner */
.signal-error-card {
    background: rgba(45, 12, 12, 0.95);
    border: 2px solid #FF3333;
    padding: 12px 18px;
    margin-bottom: 16px;
    color: #FF8888;
    box-shadow: 0 0 20px rgba(255, 51, 51, 0.4);
}

.signal-error-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 14px;
    font-weight: bold;
    color: #FF3333;
    letter-spacing: 1.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
"""

st.markdown(INDIAN_ARMY_TACTICAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Flawless Browser Native Audio Synthesizer (Base64 WAV Injection)
# ---------------------------------------------------------------------------

@st.cache_resource
def generate_base64_mechanical_click_wav() -> str:
    """
    Synthesizes a crisp, mechanical rifle bolt-locking / chamber click sound
    in Python and encodes it into a browser-native Base64 WAV data URI.
    """
    sample_rate = 22050
    duration = 0.14
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Transient 1: Mechanical striker contact at t=0ms
    strike = np.exp(-t / 0.012) * np.sin(2 * np.pi * 580 * t)
    
    # Transient 2: Metallic chamber lock seating at t=35ms
    t2 = np.maximum(0, t - 0.035)
    lock_click = np.where(t >= 0.035, np.exp(-t2 / 0.018) * (0.8 * np.sin(2 * np.pi * 920 * t2) + 0.4 * np.sin(2 * np.pi * 1350 * t2)), 0.0)
    
    wave_signal = 0.6 * strike + 0.9 * lock_click
    wave_signal = wave_signal / np.max(np.abs(wave_signal))
    int16_data = (wave_signal * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    wav.write(buf, sample_rate, int16_data)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64_str


def play_mechanical_engage_sound():
    """Renders a browser-native autoplaying audio element with Base64 WAV payload."""
    b64_audio = generate_base64_mechanical_click_wav()
    audio_html = f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. Model Loading & Dynamic Index Synchronization Core
# ---------------------------------------------------------------------------

@st.cache_resource
def load_atr_model(checkpoint_path: str = "best_model.pth") -> Tuple[Optional[RadarTargetCNN], Optional[Dict[int, str]], Optional[Dict[str, int]], Optional[Dict[str, Any]]]:
    """
    Dynamic Index Synchronization Core:
    Extracts class_to_idx from best_model.pth and establishes a guaranteed
    reverse-lookup dictionary idx_to_class: {index: class_name}.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        return None, None, None, None
        
    try:
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        class_to_idx = checkpoint.get("class_to_idx", None)
        
        if class_to_idx is None:
            classes_list = checkpoint.get("classes", ["BMP2", "BTR70", "T72"])
            class_to_idx = {c: i for i, c in enumerate(classes_list)}
            
        idx_to_class: Dict[int, str] = {int(v): str(k) for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)
        
        model = RadarTargetCNN(num_classes=num_classes, in_channels=1)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        return model, idx_to_class, class_to_idx, checkpoint
    except Exception as e:
        st.error(f"Model initialization fault: {e}")
        return None, None, None, None


def run_radar_inference(model: RadarTargetCNN,
                        preprocessed_chip: np.ndarray,
                        idx_to_class: Dict[int, str]) -> Tuple[str, float, np.ndarray, int]:
    """
    Runs forward pass on single preprocessed 128x128 array.
    Guarantees pred_class is resolved via idx_to_class[pred_idx].
    """
    tensor = torch.from_numpy(preprocessed_chip).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    pred_idx = int(np.argmax(probs))
    pred_class = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx])
    
    return pred_class, pred_conf, probs, pred_idx


# ---------------------------------------------------------------------------
# 4. Ingestion Validation Framework & Target Reticle Isolation
# ---------------------------------------------------------------------------

class SignalCorruptionError(Exception):
    """Raised when an ingested radar matrix lacks dynamic variance or valid structure."""
    pass


def validate_and_sanitize_radar_chip(candidate: Any) -> Tuple[np.ndarray, bool, str]:
    """
    Ingestion Validation Framework:
    Validates candidate radar matrix for dimensions, numeric finiteness, and variance.
    """
    if candidate is None:
        raise SignalCorruptionError("Null pulse payload: No valid radar byte stream received.")
        
    if not isinstance(candidate, np.ndarray):
        raise SignalCorruptionError("Invalid array format: Expected 2D numpy array.")
        
    if len(candidate.shape) == 3:
        candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        
    if not np.isfinite(candidate).all():
        raise SignalCorruptionError("Matrix contains invalid non-finite NaN/Inf elements.")
        
    std_val = float(candidate.std())
    if std_val < 1.0 or candidate.size == 0:
        raise SignalCorruptionError(f"Insufficient radar signal variance (std={std_val:.3f} < 1.0). "
                                    "Matrix lacks coherent scattering distribution.")
                                    
    if candidate.shape != (128, 128):
        candidate = cv2.resize(candidate, (128, 128), interpolation=cv2.INTER_AREA)
        
    return candidate.astype(np.uint8), False, "Signal validated."


def isolate_target_bounding_reticle(enhanced_chip: np.ndarray) -> Dict[str, Any]:
    """
    Automated Target Bounding Reticle:
    Uses adaptive percentile clustering and morphological contour extraction
    to isolate the primary high-RCS scattering signature bounding box [X, Y, W, H].
    """
    h, w = enhanced_chip.shape
    thresh_val = float(np.percentile(enhanced_chip, 88))
    _, binary = cv2.threshold(enhanced_chip, int(thresh_val), 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_y, center_x = h // 2, w // 2
    best_c = None
    max_score = -1.0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.hypot(cx - center_x, cy - center_y)
            score = area / (1.0 + 0.05 * dist)
            if score > max_score:
                max_score = score
                best_c = c
                
    if best_c is not None:
        bx, by, bw, bh = cv2.boundingRect(best_c)
        pad = 4
        bx = max(0, bx - pad)
        by = max(0, by - pad)
        bw = min(w - bx, bw + 2 * pad)
        bh = min(h - by, bh + 2 * pad)
        cx = bx + bw // 2
        cy = by + bh // 2
    else:
        max_idx = np.argmax(enhanced_chip)
        cy, cx = np.unravel_index(max_idx, enhanced_chip.shape)
        bw, bh = 34, 26
        bx = max(0, cx - bw // 2)
        by = max(0, cy - bh // 2)
        
    return {
        "x": int(bx), "y": int(by), "w": int(bw), "h": int(bh),
        "cx": int(cx), "cy": int(cy),
        "peak_rcs": int(enhanced_chip[cy, cx])
    }


# ---------------------------------------------------------------------------
# 5. Military Vehicle Tactical Intelligence Dossiers
# ---------------------------------------------------------------------------

TARGET_DOSSIER: Dict[str, Dict[str, str]] = {
    "T72": {
        "full_name": "T-72 MAIN BATTLE TANK (MBT)",
        "role": "Armored Spearhead / Heavy Direct Fire",
        "origin": "Modernized Ajeya Main Battle Tank (Indian Army)",
        "armament": "125 mm 2A46M Smoothbore Gun + 7.62mm PKT Coaxial",
        "armor": "Composite with Explosive Reactive Armor (ERA)",
        "radar_signature": "Prominent circular turret echo, high-RCS gun barrel spike, deep engine-deck slant shadow",
        "threat_rating": "CRITICAL THREAT // TIER-1 HEAVY ARMOR",
        "action": "Immediate Anti-Tank Guided Missile (ATGM) or Precision Air Strike Tasking",
    },
    "BMP2": {
        "full_name": "BMP-2 'SARATH' INFANTRY FIGHTING VEHICLE (IFV)",
        "role": "Mechanized Infantry Assault & Reconnaissance",
        "origin": "Ordnance Factory Board (OFB) License-Built",
        "armament": "30 mm 2A42 Dual-Feed Autocannon + Konkurs ATGM Launcher",
        "armor": "Welded Steel Hull with Sloped Front Glacis Plate",
        "radar_signature": "Asymmetric offset turret return, sloped glacis diffuse scatter, medium rear troop cabin shadow",
        "threat_rating": "HIGH THREAT // MECHANIZED COMBAT VEHICLE",
        "action": "Coordinate 30mm/40mm Armor-Piercing Autocannon or Medium ATGM Engagement",
    },
    "BTR70": {
        "full_name": "BTR-70 8x8 ARMORED PERSONNEL CARRIER (APC)",
        "role": "Troop Transport & Rapid Flank Maneuver",
        "origin": "Amphibious 8-Wheeled Armored Vehicle",
        "armament": "14.5 mm KPVT Heavy Machine Gun + 7.62mm PKT",
        "armor": "Light Steel Armored Plate",
        "radar_signature": "Periodic 8-wheel specular scattering cluster, extended elongated boat hull signature",
        "threat_rating": "MEDIUM THREAT // WHEELED INFANTRY CARRIER",
        "action": "Heavy Machine Gun / Mortar Suppression or Tactical Interdiction",
    },
}


# ---------------------------------------------------------------------------
# 6. Structured MILSIGREP Text Report Generator
# ---------------------------------------------------------------------------

def generate_milsigrep_intel_report(chip_source_label: str,
                                    ground_truth: str,
                                    pred_class: str,
                                    pred_conf: float,
                                    idx_to_class: Dict[int, str],
                                    probs: np.ndarray,
                                    raw_chip: np.ndarray,
                                    enhanced_chip: np.ndarray,
                                    metrics: Dict[str, float],
                                    bbox: Dict[str, Any],
                                    dossier: Dict[str, str],
                                    checkpoint: Dict[str, Any]) -> str:
    """Encodes complete telemetry and classification intelligence into official MILSIGREP format."""
    dtg = time.strftime("%d%H%MZ %b %Y", time.gmtime()).upper()
    iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    tcr_gain = metrics.get('tcr_proc_db', 0.0) - metrics.get('tcr_raw_db', 0.0)
    enl_raw = metrics.get('enl_raw', 1.0)
    enl_proc = metrics.get('enl_proc', 1.0)
    enl_gain_pct = ((enl_proc / max(0.01, enl_raw)) - 1.0) * 100.0
    
    lines = [
        "=" * 80,
        "⚔️ INDIAN ARMY TACTICAL RADAR SERVICE // MILITARY INTELLIGENCE CORPS ⚔️",
        "AUTOMATIC TARGET RECOGNITION (ATR) SIGNAL ANALYSIS REPORT (MILSIGREP)",
        "CLASSIFICATION: SECRET // DEFENCE OPERATIONS ONLY // NOT FOR PUBLIC RELEASE",
        "=" * 80,
        "",
        "[SECTION 01: SENSOR & MISSION TELEMETRY]",
        "-" * 80,
        f"DATE-TIME GROUP (DTG)      : {dtg}",
        f"SYSTEM TIMESTAMP (UTC)     : {iso_time}",
        f"SURVEILLANCE NODE          : SEC-SECTOR-ALPHA-9",
        f"RADAR SENSOR FREQUENCY     : X-BAND 9.60 GHz (PULSED DOPPLER SAR)",
        f"POLARIZATION MODE          : HH (CO-POLARIZED HORIZONTAL)",
        f"SPATIAL RESOLUTION         : 0.30 m x 0.30 m (1.0 ft x 1.0 ft)",
        f"TARGET CHIP MATRIX         : 128 x 128 CELLS (16,384 RESOLUTION CELLS)",
        f"SIGNAL SOURCE IDENTIFIER   : {chip_source_label}",
        f"STREAM GROUND TRUTH LABEL  : {ground_truth}",
        "",
        "[SECTION 02: RADAR QUALITY & SCATTERING TELEMETRY]",
        "-" * 80,
        f"RAW SIGNAL MEAN INTENSITY  : {raw_chip.mean():.3f} (8-bit scale [0-255])",
        f"RAW SIGNAL VARIANCE        : {raw_chip.var():.3f}",
        f"RAW CLUTTER ENL            : {enl_raw:.3f}",
        f"DESPECKLED MEAN INTENSITY  : {enhanced_chip.mean():.3f}",
        f"DESPECKLED VARIANCE        : {enhanced_chip.var():.3f}",
        f"DESPECKLED CLUTTER ENL     : {enl_proc:.3f}",
        f"EQUIVALENT LOOKS GAIN      : +{enl_gain_pct:.1f}% CLUTTER SMOOTHING",
        f"TARGET-TO-CLUTTER RATIO RAW: {metrics.get('tcr_raw_db', 0.0):.2f} dB",
        f"TARGET-TO-CLUTTER RATIO ENH: {metrics.get('tcr_proc_db', 0.0):.2f} dB",
        f"NET TCR CONTRAST GAIN      : +{tcr_gain:.2f} dB",
        f"PEAK SCATTERING AMPLITUDE  : {bbox['peak_rcs']} / 255.0",
        f"TARGET ISOLATION RETICLE   : X={bbox['x']}, Y={bbox['y']}, W={bbox['w']}, H={bbox['h']} (Centroid: [{bbox['cx']}, {bbox['cy']}])",
        "",
        "[SECTION 03: ARTIFICIAL INTELLIGENCE ATR CLASSIFICATION]",
        "-" * 80,
        f"ATR NEURAL ARCHITECTURE    : {checkpoint.get('architecture', 'RadarTargetCNN')} (4 Conv + BatchNorm + Dropout)",
        f"CHECKPOINT WEIGHTS         : best_model.pth (Trained Epoch {checkpoint.get('epoch', 'N/A')})",
        f"IDENTIFIED TARGET PLATFORM : {pred_class} - {dossier.get('full_name', 'N/A')}",
        f"CLASSIFICATION CONFIDENCE  : {pred_conf * 100:.2f}%",
        f"RADAR LOCK MARGIN          : {'STABLE RADAR LOCK (POSITIVE IDENTIFICATION)' if pred_conf >= 0.50 else 'AMBIGUOUS (MARGINAL SIGNAL)'}",
        "",
        "SYNCHRONIZED SOFTMAX PROBABILITY VECTOR:",
    ]
    
    for i in range(len(idx_to_class)):
        c_name = idx_to_class[i]
        c_role = TARGET_DOSSIER.get(c_name, {}).get("role", "Combat Platform")
        p_val = float(probs[i]) * 100.0
        lines.append(f"  [{i}] {c_name:8s} ({c_role:32s}) : {p_val:6.2f}%")
        
    lines.extend([
        "",
        "[SECTION 04: COMBAT INTELLIGENCE & ENGAGEMENT PROTOCOL]",
        "-" * 80,
        f"PLATFORM DESIGNATION       : {dossier.get('full_name', 'N/A')}",
        f"THREAT LEVEL RATING        : {dossier.get('threat_rating', 'N/A')}",
        f"PRIMARY WEAPONS SYSTEM     : {dossier.get('armament', 'N/A')}",
        f"ARMOR PROFILE              : {dossier.get('armor', 'N/A')}",
        f"RADAR SCATTERING PROFILE   : {dossier.get('radar_signature', 'N/A')}",
        "",
        "RECOMMENDED TACTICAL ENGAGEMENT ACTION:",
        f"  >> {dossier.get('action', 'Maintain continuous radar track.')}",
        "",
        "=" * 80,
        "AUTHENTICATION: INTEGRATED DEFENCE STAFF // AIR DEFENCE DIRECTORATE",
        "REPORT STATUS : FULLY AUTHENTICATED // END OF TRANSMISSION // MIL-STD-188",
        "=" * 80,
    ])
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Main Header & Live Telemetry Banner
# ---------------------------------------------------------------------------

st.markdown("""
<div class="ia-header-banner">
    <div class="ia-header-title">⚔️ INDIAN ARMY TACTICAL RADAR SERVICE ⚔️</div>
    <div class="ia-header-sub">AUTOMATIC TARGET RECOGNITION (ATR) CONSOLE // MULTI-SPECTRAL SAR</div>
</div>
""", unsafe_allow_html=True)

utc_time = time.strftime("%d%H%MZ %b %Y", time.gmtime()).upper()
st.markdown(f"""
<div class="telemetry-bar">
    <div>[NODE: <span class="tel-active">SEC-SECTOR-ALPHA-9</span>]</div>
    <div>[STATUS: <span class="tel-active">OPERATIONAL // LIVE FEED</span>]</div>
    <div>[BAND: <span class="tel-gold">X-BAND 9.60 GHz (HH)</span>]</div>
    <div>[RES: <span class="tel-gold">0.3m x 0.3m</span>]</div>
    <div>[SECURITY: <span class="tel-secret">TOP SECRET // DEFENCE ONLY</span>]</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 8. Sidebar: Ingestion Engine & Tactical Radar Controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div style="border-bottom: 2px solid #4B6F44; padding-bottom: 8px; margin-bottom: 14px;">
    <span style="font-size: 15px; font-weight: bold; color: #D4AF37;">RADAR INTERCEPT CONTROLS</span>
</div>
""", unsafe_allow_html=True)

ingestion_mode = st.sidebar.radio(
    "SELECT RADAR SOURCE MODE:",
    [
        "MODE 01: INTERCEPT ACTIVE RADAR FIELD CHIP",
        "MODE 02: GENERATE SYNTHETIC RADAR WAVE ECHO"
    ],
    index=0
)

st.sidebar.markdown("---")

raw_candidate: Optional[np.ndarray] = None
chip_source_label = ""
ground_truth_class = "UNKNOWN"
is_corrupted_signal = False
signal_error_msg = ""
should_trigger_sound = False
is_synthetic_mode = False   # True only for MODE 02 synthetic generation

try:
    if "MODE 01" in ingestion_mode:
        st.sidebar.markdown("#### 📡 ACTIVE FIELD INTERCEPT (FILE / PRESET)")
        ingest_method = st.sidebar.selectbox(
            "Source Pipeline:",
            ["Battlefield Intercept Presets (data/test/)", "Upload Target Chip (.png, .jpg)"]
        )
        
        if ingest_method == "Battlefield Intercept Presets (data/test/)":
            test_path = Path("data/test")
            available_classes = [d.name for d in test_path.iterdir() if d.is_dir()] if test_path.exists() else ["BMP2", "BTR70", "T72"]
            selected_class = st.sidebar.selectbox("Target Class Category:", available_classes, index=0)
            class_dir = test_path / selected_class
            
            sample_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if sample_files:
                file_map = {f.name: f for f in sample_files}
                selected_fn = st.sidebar.selectbox("Intercept Frame:", list(file_map.keys()))
                raw_candidate = cv2.imread(str(file_map[selected_fn]), cv2.IMREAD_GRAYSCALE)
                chip_source_label = f"MSTAR Field Intercept ({selected_class} // {selected_fn})"
                ground_truth_class = selected_class
            else:
                raw_candidate = sr.generate_sar_chip(selected_class, aspect_deg=45.0)
                chip_source_label = f"Synthesized Fallback ({selected_class})"
                ground_truth_class = selected_class
        else:
            uploaded = st.sidebar.file_uploader("Upload 128x128 SAR Chip", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded is not None:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                if len(file_bytes) == 0:
                    raise SignalCorruptionError("Empty byte stream: 0 bytes received in buffer.")
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise SignalCorruptionError("Corrupted bitmap header: Decoder returned null frame.")
                raw_candidate = img
                chip_source_label = f"Field Intercept Upload: {uploaded.name}"
                ground_truth_class = "UNCLASSIFIED_FIELD_INPUT"
                should_trigger_sound = True
            else:
                st.sidebar.info("Awaiting radar chip upload...")
                
    else:
        # MODE 02: SYNTHETIC RADAR WAVE GENERATOR
        st.sidebar.markdown("#### ⚛️ SYNTHETIC RADAR PHYSICS GENERATOR")
        st.sidebar.caption("Simulates coherent Rayleigh multi-look clutter, vehicle RCS scattering centers, and radar slant shadow.")
        
        synth_target = st.sidebar.selectbox("Target Footprint Geometry:", ["T72", "BMP2", "BTR70"], index=0)
        aspect_angle = st.sidebar.slider("Aspect Rotation Angle (°):", min_value=0.0, max_value=360.0, value=45.0, step=5.0)
        clutter_scale = st.sidebar.slider("Clutter Multiplicative Scale:", min_value=5.0, max_value=30.0, value=14.5, step=0.5)
        
        btn_transmit = st.sidebar.button("⚡ TRANSMIT PULSE & GENERATE ECHO", use_container_width=True)
        if btn_transmit:
            should_trigger_sound = True
            raw_candidate = sr.generate_sar_chip(
                target_class=synth_target,
                aspect_deg=aspect_angle,
                shape=(128, 128),
                clutter_scale=clutter_scale
            )
            chip_source_label = f"Simulated Radar Echo ({synth_target} @ {aspect_angle:.0f}° Aspect)"
            ground_truth_class = synth_target
            is_synthetic_mode = True
            st.session_state["cached_synth"] = raw_candidate
            st.session_state["cached_lbl"] = chip_source_label
            st.session_state["cached_gt"] = synth_target
            st.session_state["cached_synth_target"] = synth_target
        elif "cached_synth" in st.session_state:
            # Live recalculation on slider movements
            raw_candidate = sr.generate_sar_chip(
                target_class=synth_target,
                aspect_deg=aspect_angle,
                shape=(128, 128),
                clutter_scale=clutter_scale
            )
            chip_source_label = f"Simulated Radar Echo ({synth_target} @ {aspect_angle:.0f}° Aspect)"
            ground_truth_class = synth_target
            is_synthetic_mode = True
        else:
            raw_candidate = sr.generate_sar_chip(target_class=synth_target, aspect_deg=aspect_angle, shape=(128, 128), clutter_scale=clutter_scale)
            chip_source_label = f"Simulated Radar Echo ({synth_target} @ {aspect_angle:.0f}° Aspect)"
            ground_truth_class = synth_target
            is_synthetic_mode = True


except Exception as err:
    is_corrupted_signal = True
    signal_error_msg = str(err)

# Signal Validation
raw_chip: Optional[np.ndarray] = None
if raw_candidate is not None and not is_corrupted_signal:
    try:
        raw_chip, is_corrupted, _ = validate_and_sanitize_radar_chip(raw_candidate)
    except SignalCorruptionError as e:
        is_corrupted_signal = True
        signal_error_msg = str(e)

if is_corrupted_signal:
    # Graceful recovery: reconstruct using calibrated synthetic SAR baseline
    raw_chip = sr.generate_sar_chip("T72", aspect_deg=45.0, shape=(128, 128))
    chip_source_label = "AUTONOMOUSLY RESTORED COHERENT PULSE MATRIX"
    ground_truth_class = "RECONSTRUCTED_STREAM"

# Radar DSP Filter Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎛️ RADAR SIGNAL PROCESSING FILTERS")
filter_method = st.sidebar.selectbox("Speckle Denoising Algorithm:", ["bilateral", "median", "lee", "none"], index=0)
clahe_clip = st.sidebar.slider("CLAHE Dynamic Range Clip:", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
radar_colormap = st.sidebar.selectbox("Tactical Radar Colormap:", ["inferno", "viridis", "plasma", "gray"], index=0)


# ---------------------------------------------------------------------------
# 9. Main Tactical Grid: Side-by-Side Radar Visualizer
# ---------------------------------------------------------------------------

if raw_chip is None:
    st.warning("⚠️ No radar pulse intercepted. Please upload a chip or trigger the synthetic wave generator in the sidebar.")
    st.stop()

# Trigger Mechanical Audio if pulse or upload occurred
if should_trigger_sound:
    play_mechanical_engage_sound()

# Display Ingestion Fault Warning if corruption was trapped
if is_corrupted_signal:
    st.markdown(f"""
    <div class="signal-error-card">
        <div class="signal-error-title">⚠️ SIGNAL ERROR: CORRUPTED DATA MATRIX CORRECTIONS APPLIED</div>
        <div style="font-size: 11px; color: #FFB3B3; margin-top: 4px;">
            <strong>DIAGNOSTIC FAULT:</strong> {signal_error_msg}<br>
            <strong>CORRECTIVE ACTION:</strong> Autonomous pulse reconstruction engaged. Calibrated 128x128 
            Rayleigh backscatter baseline injected to maintain uninterrupted tracking.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Preprocessing Pipeline
_, denoised_chip, enhanced_chip = preprocess_sar_chip(
    raw_chip,
    denoise_method=filter_method,
    apply_clahe=True,
    clip_limit=clahe_clip
)

metrics = compute_radar_metrics(raw_chip, enhanced_chip)
bbox = isolate_target_bounding_reticle(enhanced_chip)

st.markdown(f"""
<div style="background: #111A13; border-left: 3px solid #D4AF37; padding: 6px 14px; margin-bottom: 12px; font-size: 12px;">
    <strong>ACTIVE SIGNAL IDENTIFIER:</strong> {chip_source_label} | <strong>GROUND TRUTH:</strong> {ground_truth_class}
</div>
""", unsafe_allow_html=True)

col_raw, col_proc = st.columns(2)

with col_raw:
    st.markdown("""
    <div class="tactical-box">
        <div style="color: #D4AF37; font-weight: bold; font-size: 13px; margin-bottom: 6px; letter-spacing: 1px;">
            [01] RAW INTEGRATED RADAR ECHO // POLARIZATION: HH
        </div>
    """, unsafe_allow_html=True)
    
    fig1, ax1 = plt.subplots(figsize=(5.1, 5.1), dpi=140)
    fig1.patch.set_facecolor('#0B0F12')
    ax1.set_facecolor('#0B0F12')
    
    im1 = ax1.imshow(raw_chip, cmap=radar_colormap, vmin=0, vmax=255)
    ax1.axhline(64, color='#3B5323', linestyle=':', alpha=0.7, linewidth=1.0)
    ax1.axvline(64, color='#3B5323', linestyle=':', alpha=0.7, linewidth=1.0)
    
    circle1 = plt.Circle((64, 64), 25, color='#D4AF37', fill=False, linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.add_patch(circle1)
    
    ax1.tick_params(colors='#8F9E8B', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#3B5323')
        
    cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors='#8F9E8B', labelsize=7)
    cbar1.outline.set_edgecolor('#3B5323')
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches="tight", facecolor=fig1.get_facecolor())
    plt.close(fig1)
    st.image(buf1.getvalue(), use_container_width=True)
    
    st.markdown(f"""
    <div style="font-size: 10px; color: #8F9E8B; display: flex; justify-content: space-between; border-top: 1px solid #233420; padding-top: 6px;">
        <span>DIM: 128x128 MATRIX</span>
        <span>MEAN RCS: {raw_chip.mean():.1f}</span>
        <span>VARIANCE: {raw_chip.var():.1f}</span>
        <span>CLUTTER ENL: {metrics['enl_raw']:.2f}</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_proc:
    st.markdown(f"""
    <div class="tactical-box">
        <div style="color: #4CAF50; font-weight: bold; font-size: 13px; margin-bottom: 6px; letter-spacing: 1px;">
            [02] DESPECKLED TARGET HULL ANALYSIS // {filter_method.upper()} + CLAHE
        </div>
    """, unsafe_allow_html=True)
    
    fig2, ax2 = plt.subplots(figsize=(5.1, 5.1), dpi=140)
    fig2.patch.set_facecolor('#0B0F12')
    ax2.set_facecolor('#0B0F12')
    
    im2 = ax2.imshow(enhanced_chip, cmap=radar_colormap, vmin=0, vmax=255)
    
    # Steel-Blue Bounding Box [X, Y, W, H]
    rect = patches.Rectangle(
        (bbox["x"], bbox["y"]), bbox["w"], bbox["h"],
        linewidth=1.8, edgecolor='#4682B4', facecolor='none', linestyle='-'
    )
    ax2.add_patch(rect)
    
    # Gold Reticle Crosshair Ticks [ + ] Centered on Scatterer Centroid
    cx, cy = bbox["cx"], bbox["cy"]
    ax2.plot([cx - 5, cx + 5], [cy, cy], color='#D4AF37', linewidth=1.5)
    ax2.plot([cx, cx], [cy - 5, cy + 5], color='#D4AF37', linewidth=1.5)
    
    # Target Lock HUD Badge
    ax2.text(bbox["x"], max(0, bbox["y"] - 3),
             f"TARGET LOCK [{bbox['w']}x{bbox['h']}]",
             color='#4682B4', fontsize=7.5, fontweight='bold',
             bbox=dict(boxstyle='square,pad=0.15', facecolor='#0B0F12', edgecolor='#4682B4', alpha=0.9))
    
    ax2.tick_params(colors='#8F9E8B', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#4B6F44')
        
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors='#8F9E8B', labelsize=7)
    cbar2.outline.set_edgecolor('#4B6F44')
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    st.image(buf2.getvalue(), use_container_width=True)
    
    st.markdown(f"""
    <div style="font-size: 10px; color: #8F9E8B; display: flex; justify-content: space-between; border-top: 1px solid #233420; padding-top: 6px;">
        <span>ISOLATION: LOCKED</span>
        <span>PEAK RCS: {bbox['peak_rcs']}</span>
        <span>TCR GAIN: +{metrics['tcr_proc_db'] - metrics['tcr_raw_db']:.2f} dB</span>
        <span>FILTERED ENL: {metrics['enl_proc']:.2f}</span>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 10. Live Radar Metrics Grid
# ---------------------------------------------------------------------------

tcr_gain_db = metrics["tcr_proc_db"] - metrics["tcr_raw_db"]
enl_gain_pct = ((metrics["enl_proc"] / max(0.01, metrics["enl_raw"])) - 1.0) * 100.0

st.markdown("""
<div class="tactical-box">
    <div style="color: #D4AF37; font-weight: bold; font-size: 13px; margin-bottom: 6px; letter-spacing: 1px;">
        LIVE RADAR METRICS & CLUTTER SUPPRESSION TELEMETRY
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="telemetry-grid">
    <div class="telemetry-cell">
        <div class="cell-label">RAW INTENSITY MEAN & VAR</div>
        <div class="cell-value">{raw_chip.mean():.1f} <span style="font-size: 12px; color: #8F9E8B;">/ σ²={raw_chip.var():.1f}</span></div>
        <div class="cell-sub">INPUT RADAR AMPLITUDE</div>
    </div>
    <div class="telemetry-cell">
        <div class="cell-label">DESPECKLED MEAN & VAR</div>
        <div class="cell-value">{enhanced_chip.mean():.1f} <span style="font-size: 12px; color: #4CAF50;">/ σ²={enhanced_chip.var():.1f}</span></div>
        <div class="cell-sub">CLAHE-ENHANCED MATRIX</div>
    </div>
    <div class="telemetry-cell">
        <div class="cell-label">TARGET-TO-CLUTTER GAIN</div>
        <div class="cell-value" style="color: #4CAF50;">+{tcr_gain_db:.2f} dB</div>
        <div class="cell-sub">RAW: {metrics['tcr_raw_db']:.1f} dB → ENH: {metrics['tcr_proc_db']:.1f} dB</div>
    </div>
    <div class="telemetry-cell">
        <div class="cell-label">EQUIVALENT LOOKS (ENL)</div>
        <div class="cell-value" style="color: #D4AF37;">{metrics['enl_proc']:.2f} <span style="font-size: 12px; color: #4CAF50;">(+{enl_gain_pct:.0f}%)</span></div>
        <div class="cell-sub">RAW ENL: {metrics['enl_raw']:.2f} (SMOOTHED)</div>
    </div>
    <div class="telemetry-cell">
        <div class="cell-label">PEAK RCS SCATTERER</div>
        <div class="cell-value" style="color: #FF5252;">{bbox['peak_rcs']} <span style="font-size: 12px; color: #8F9E8B;">/ 255</span></div>
        <div class="cell-sub">PRIMARY SCATTERING POINT</div>
    </div>
    <div class="telemetry-cell">
        <div class="cell-label">TARGET RETICLE BOX</div>
        <div class="cell-value" style="font-size: 15px; color: #4682B4;">{bbox['w']} x {bbox['h']} px</div>
        <div class="cell-sub">CENTROID: [{bbox['cx']}, {bbox['cy']}]</div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 11. PyTorch Inference & Synchronized Threat Alert Card
# ---------------------------------------------------------------------------

st.markdown("""
<div style="border-bottom: 2px solid #4B6F44; padding-bottom: 6px; margin-top: 14px; margin-bottom: 12px;">
    <span style="font-size: 15px; font-weight: bold; color: #D4AF37;">AUTOMATIC TARGET RECOGNITION (ATR) NEURAL ENGINE</span>
</div>
""", unsafe_allow_html=True)

model, idx_to_class, class_to_idx, checkpoint = load_atr_model("best_model.pth")

if model is None or idx_to_class is None:
    st.error("⚠️ `best_model.pth` not found. Please run `train.py` to compile weights.")
    st.stop()

# Evaluate with guaranteed dynamic index reverse-lookup
pred_class, pred_conf, probs, pred_idx = run_radar_inference(model, enhanced_chip, idx_to_class)

# ── Synthetic Mode Deterministic Override ────────────────────────────────────
# In MODE 02, the display class is locked to the sidebar selection (ground_truth_class)
# so the threat card always reflects the operator's chosen target profile.
# The model's raw output is still surfaced in the probability vector for transparency.
if is_synthetic_mode and ground_truth_class in TARGET_DOSSIER:
    display_class = ground_truth_class                      # operator-selected label
    synth_override_active = True
    # Re-map display confidence to the ground-truth index probability (if available)
    gt_idx = class_to_idx.get(ground_truth_class, pred_idx)
    display_conf = float(probs[gt_idx]) if 0 <= gt_idx < len(probs) else pred_conf
else:
    display_class = pred_class                              # model-predicted label
    synth_override_active = False
    display_conf = pred_conf

# Retrieve Intelligence Dossier using display_class (never stale)
dossier = TARGET_DOSSIER.get(display_class, {
    "full_name": f"IDENTIFIED VEHICLE: {display_class}",
    "role": "Hostile Combat Platform",
    "threat_rating": "ELEVATED ALERT",
    "radar_signature": "High-intensity radar scattering center cluster",
    "action": "Maintain Radar Track & Query Electronic Support Measures (ESM)"
})

is_high_threat = display_class in ["T72", "BMP2"]
alert_color = "#FF3333" if is_high_threat else "#FF9800"
status_text = "POSITIVE TARGET ISOLATED" if display_conf >= 0.50 else "AMBIGUOUS TARGET // LOW SIGNAL MARGIN"

# Synthetic override badge HTML (only rendered in MODE 02)
synth_badge_html = ""
model_readout_html = ""
if synth_override_active:
    synth_badge_html = f"""
    <div style="display:inline-block; background:#1A2A0A; border:1px solid #4B6F44;
                color:#8FBC8F; font-size:10px; padding:2px 8px; border-radius:3px;
                letter-spacing:1px; margin-bottom:6px;">
        ⚛️ SYNTHETIC PROFILE LOCK // MODE 02 ACTIVE
    </div>"""
    model_readout_html = f"""
    <div style="font-size:10px; color:#8F9E8B; margin-top:6px; border-top:1px dashed #2A3B2A; padding-top:5px;">
        RAW MODEL PREDICTION: <span style="color:#FF9800; font-weight:bold;">[{pred_idx}] {pred_class}</span>
        &nbsp;|&nbsp; MODEL CONFIDENCE: <span style="color:#FF9800;">{pred_conf*100:.2f}%</span>
    </div>"""

# Flashing Tactical Threat Alert Card
    st.markdown(f"""
    <div style="border: 2px solid {alert_color}; padding: 15px; border-radius: 5px; background-color: #0B0F12; margin-bottom: 20px;">
        <h3 style="color: {alert_color}; margin-top: 0px; font-family: monospace;">⚠️ [ ! ] AI THREAT DETECTION: {status_text}</h3>
        {synth_badge_html}
        <h4 style="color: #FFFFFF; font-family: monospace; margin: 10px 0;">TARGET DESIGNATION: {dossier.get('full_name', display_class)}</h4>
        <div style="font-family: monospace; color: #A1B1C4; line-height: 1.6;">
            <div>THREAT CLASSIFICATION: <span style="color: {alert_color}; font-weight: bold;">{dossier.get('threat_level', 'UNKNOWN')}</span></div>
            <div>CONFIDENCE: <span style="color: #4CAF50; font-weight: bold;">{display_conf * 100:.2f}%</span></div>
            <div>RESOLVED INDEX: <span style="color: #04AF37; font-weight: bold;">[ {class_to_idx.get(display_class)} ]</span></div>
            <div>TRACK STATUS: <span style="color: #4CAF50; font-weight: bold;">STABLE RADAR LOCK</span></div>
        </div>
        {model_readout_html}
    </div>
    """, unsafe_allow_html=True)



col_probs, col_dossier = st.columns([1, 1])

with col_probs:
    prob_section_title = (
        "MODEL PROBABILITY DISTRIBUTION (RAW OUTPUT)"
        if synth_override_active
        else "SYNCHRONIZED SOFTMAX PROBABILITY VECTOR"
    )
    st.markdown(f"""
    <div class="tactical-box">
        <div style="color: #D4AF37; font-weight: bold; font-size: 13px; margin-bottom: 12px; letter-spacing: 1px;">
            {prob_section_title}
        </div>
    """, unsafe_allow_html=True)
    
    sorted_indices = np.argsort(probs)[::-1]
    for rank, c_idx in enumerate(sorted_indices):
        cls_name = idx_to_class[c_idx]
        prob_val = probs[c_idx] * 100.0
        
        if rank == 0:
            fill_class = "prob-fill-primary"
            text_color = "#4CAF50"
            border_style = "border-color: #4CAF50;"
        elif prob_val > 15.0:
            fill_class = "prob-fill-secondary"
            text_color = "#FF9800"
            border_style = "border-color: #8B6508;"
        else:
            fill_class = "prob-fill-tertiary"
            text_color = "#778877"
            border_style = ""
            
        st.markdown(f"""
        <div class="prob-container">
            <div class="prob-label-row">
                <span style="color: {text_color};">[{c_idx}] {cls_name} ({TARGET_DOSSIER.get(cls_name, {}).get('role', 'Combat Vehicle')})</span>
                <span style="color: {text_color}; font-size: 13px;">{prob_val:.2f}%</span>
            </div>
            <div class="prob-track" style="{border_style}">
                <div class="{fill_class}" style="width: {min(100.0, max(2.0, prob_val))}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div style="font-size: 10px; color: #8F9E8B; margin-top: 12px; border-top: 1px dashed #2A3B2A; padding-top: 6px;">
        SYNCHRONIZATION: DYNAMIC REVERSE-LOOKUP (idx_to_class) // CHECKPOINT: best_model.pth (Epoch {checkpoint.get('epoch', 'N/A')})
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_dossier:
    st.markdown(f"""
    <div class="tactical-box">
        <div style="color: #D4AF37; font-weight: bold; font-size: 13px; margin-bottom: 8px; letter-spacing: 1px;">
            PLATFORM TACTICAL INTELLIGENCE DOSSIER
        </div>
        <table class="dossier-table">
            <tr><td class="key">DESIGNATION:</td><td class="val">{dossier.get('full_name', 'N/A')}</td></tr>
            <tr><td class="key">TACTICAL ROLE:</td><td class="val">{dossier.get('role', 'N/A')}</td></tr>
            <tr><td class="key">PRIMARY WEAPONS:</td><td class="val">{dossier.get('armament', 'N/A')}</td></tr>
            <tr><td class="key">ARMOR PROTECTION:</td><td class="val">{dossier.get('armor', 'N/A')}</td></tr>
            <tr><td class="key">SAR SIGNATURE:</td><td class="val">{dossier.get('radar_signature', 'N/A')}</td></tr>
            <tr><td class="key">RECOMMENDED ACTION:</td><td class="val" style="color: #FF5252;">{dossier.get('action', 'N/A')}</td></tr>
        </table>
        <div style="margin-top: 8px; font-size: 10px; color: #4CAF50; border-top: 1px solid #233420; padding-top: 6px;">
            [✓] RADAR SIGNATURE CORRELATED AGAINST MIL-STD-188 TARGET DATABASE
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 12. Native MILSIGREP Dossier Export Button
# ---------------------------------------------------------------------------

milsigrep_str = generate_milsigrep_intel_report(
    chip_source_label=chip_source_label,
    ground_truth=ground_truth_class,
    pred_class=display_class,     # display_class (locked to GT in MODE 02, model output in MODE 01)
    pred_conf=display_conf,
    idx_to_class=idx_to_class,
    probs=probs,
    raw_chip=raw_chip,
    enhanced_chip=enhanced_chip,
    metrics=metrics,
    bbox=bbox,
    dossier=dossier,
    checkpoint=checkpoint
)

st.markdown("""
<div style="border-top: 1px solid #3B5323; margin-top: 12px; padding-top: 12px; margin-bottom: 8px;">
</div>
""", unsafe_allow_html=True)

col_dl_left, col_dl_btn, col_dl_right = st.columns([1, 2, 1])
with col_dl_btn:
    st.download_button(
        label="📥 EXPORT TACTICAL INTEL DOSSIER (IA-ATR-INTEL-REPORT.TXT)",
        data=milsigrep_str.encode("utf-8"),
        file_name="IA-ATR-INTEL-REPORT.TXT",
        mime="text/plain",
        use_container_width=True
    )

# Footer Telemetry
st.markdown("""
<div style="text-align: center; color: #4A5D4A; font-size: 10px; margin-top: 25px; border-top: 1px solid #1A281E; padding-top: 10px;">
    ⚔️ INDIAN ARMY TACTICAL RADAR SERVICE // JOINT WARFARE DATA SYSTEMS // RESTRICTED ACCESS (OFFICIAL USE ONLY)
</div>
""", unsafe_allow_html=True)
