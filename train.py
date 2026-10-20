"""
train.py - Deep Learning Neural Network Training for SAR ATR
============================================================
Trains a specialized Deep Convolutional Neural Network (RadarTargetCNN)
for single-channel 128x128 SAR target recognition (MSTAR dataset).

Features:
  1. 4-Stage Deep CNN with Batch Normalization, Max Pooling, and Spatial Dropout
  2. Adam optimizer with StepLR learning rate scheduling & CrossEntropyLoss
  3. Validation tracking after each epoch (Loss & Accuracy)
  4. Best model checkpoint saving to `best_model.pth` (with metadata & class map)
  5. Convergence visualization saved to `training_performance.png`
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# Import Phase 2 preprocessed dataset
from preprocess import MSTARPreprocessedDataset


# ---------------------------------------------------------------------------
# 1. Deep Convolutional Neural Network Architecture
# ---------------------------------------------------------------------------

class RadarTargetCNN(nn.Module):
    """
    Deep Convolutional Neural Network tailored for 128x128 Single-Channel SAR Imagery.
    
    Architecture:
      Conv Block 1: (1, 128, 128) -> (32, 64, 64)  + BatchNorm + ReLU + MaxPool + Dropout2d(0.10)
      Conv Block 2: (32, 64, 64)  -> (64, 32, 32)  + BatchNorm + ReLU + MaxPool + Dropout2d(0.15)
      Conv Block 3: (64, 32, 32)  -> (128, 16, 16) + BatchNorm + ReLU + MaxPool + Dropout2d(0.20)
      Conv Block 4: (128, 16, 16) -> (128, 8, 8)   + BatchNorm + ReLU + MaxPool + Dropout2d(0.25)
      AdaptiveAvgPool2d((4, 4))   -> (128, 4, 4) = 2048 features
      Dense Classifier:
        Linear(2048, 256) -> BatchNorm1d -> ReLU -> Dropout(0.40) -> Linear(256, num_classes)
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 1):
        super(RadarTargetCNN, self).__init__()
        
        # Conv Block 1: Low-level edge & scattering center detection
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.10)
        )
        
        # Conv Block 2: Vehicle boundary & component localization
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15)
        )
        
        # Conv Block 3: Structural vehicle features (turret, gun barrel, chassis)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20)
        )
        
        # Conv Block 4: High-level target radar signatures & shadow patterns
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25)
        )
        
        # Global Pooling & Feature Extraction
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dense Classifier with clutter-regularizing dropout
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.40),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# 2. Training & Evaluation Routines
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Runs a single training epoch and returns (mean_loss, accuracy_percentage)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    """Evaluates the model on validation/test set and returns (loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# 3. Performance Visualization Generator
# ---------------------------------------------------------------------------

def plot_training_performance(history: Dict[str, List[float]],
                              classes: List[str],
                              cm: np.ndarray,
                              output_path: str = "training_performance.png"):
    """
    Plots training vs validation convergence curves and the final confusion matrix.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig = plt.figure(figsize=(15, 6), dpi=150)
    
    # 1. Loss Subplot
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, history["train_loss"], 'o-', color='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 's--', color='#d62728', linewidth=2, label='Val Loss')
    best_loss_idx = int(np.argmin(history["val_loss"]))
    ax1.scatter([best_loss_idx + 1], [history["val_loss"][best_loss_idx]], color='#2ca02c', s=120, zorder=5,
                label=f'Min Val Loss: {history["val_loss"][best_loss_idx]:.3f}')
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)
    
    # 2. Accuracy Subplot
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, history["train_acc"], 'o-', color='#1f77b4', linewidth=2, label='Train Acc')
    ax2.plot(epochs, history["val_acc"], 's--', color='#d62728', linewidth=2, label='Val Acc')
    best_acc_idx = int(np.argmax(history["val_acc"]))
    ax2.scatter([best_acc_idx + 1], [history["val_acc"][best_acc_idx]], color='#2ca02c', s=120, zorder=5,
                label=f'Max Val Acc: {history["val_acc"][best_acc_idx]:.1f}%')
    ax2.set_title("Classification Accuracy (%) vs. Epochs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)
    
    # 3. Confusion Matrix Subplot
    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title("Test Confusion Matrix (Best Model)", fontsize=11, fontweight="bold")
    
    tick_marks = np.arange(len(classes))
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(classes, rotation=25)
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(classes)
    
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax3.text(j, i, f"{val}", horizontalalignment="center",
                     color="white" if val > thresh else "black", fontweight="bold")
                     
    ax3.set_ylabel("True Target Class", fontsize=10)
    ax3.set_xlabel("Predicted Target Class", fontsize=10)
    
    plt.suptitle("MSTAR SAR ATR: DEEP CONVOLUTIONAL NEURAL NETWORK TRAINING REPORT\n"
                 "Architecture: RadarTargetCNN | Optimizer: Adam (lr=1e-3) | Input: (1, 128, 128)",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Training performance curve and confusion matrix saved to: {output_path}")


# ---------------------------------------------------------------------------
# 4. Main Training Pipeline Execution
# ---------------------------------------------------------------------------

def run_training(data_dir: str = "data",
                 epochs: int = 10,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 denoise_method: str = "bilateral",
                 output_model: str = "best_model.pth",
                 output_plot: str = "training_performance.png",
                 seed: int = 42):
    """Executes the full training loop with checkpointing and validation tracking."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("PHASE 3: DEEP CONVOLUTIONAL NEURAL NETWORK TRAINING (SAR ATR)")
    print("=" * 70)
    print(f"Compute Device      : {device}")
    print(f"Data Directory      : {data_dir}")
    print(f"Epochs              : {epochs}")
    print(f"Batch Size          : {batch_size}")
    print(f"Learning Rate       : {lr}")
    print(f"Denoising Filter    : {denoise_method}")
    print(f"Model Checkpoint    : {output_model}")
    print(f"Performance Graph   : {output_plot}")
    print("-" * 70)
    
    # 1. Instantiate Preprocessed Datasets
    print("[+] Loading MSTAR Preprocessed Training Dataset (with data augmentation)...")
    train_dataset = MSTARPreprocessedDataset(
        root_dir=data_dir,
        split="train",
        denoise_method=denoise_method,
        apply_clahe=True,
        clip_limit=2.5,
        augment=True
    )
    
    print("[+] Loading MSTAR Preprocessed Validation/Test Dataset...")
    val_dataset = MSTARPreprocessedDataset(
        root_dir=data_dir,
        split="test",
        denoise_method=denoise_method,
        apply_clahe=True,
        clip_limit=2.5,
        augment=False
    )
    
    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    print(f"[+] Classes ({len(classes)})        : {classes}")
    print(f"[+] Train Samples       : {len(train_dataset)}")
    print(f"[+] Validation Samples  : {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 2. Build Model
    model = RadarTargetCNN(num_classes=len(classes), in_channels=1).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] RadarTargetCNN Total Trainable Parameters: {total_params:,}")
    
    # 3. Loss, Optimizer, and LR Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 4. Training Loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    best_preds = []
    best_labels = []
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Val Loss':^12} | {'Val Acc (%)':^14} | {'Status'}")
    print("-" * 70)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Check if this is the best model so far
        is_best = val_acc > best_val_acc or (val_acc == best_val_acc and epoch == 1)
        status_msg = ""
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_preds = val_preds
            best_labels = val_labels
            status_msg = f"[BEST SAVED @ {val_acc:.1f}%]"
            
            # Save model checkpoint with rich metadata for downstream inference
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": val_loss,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "architecture": "RadarTargetCNN",
                "in_channels": 1,
                "num_classes": len(classes),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            torch.save(checkpoint, output_model)
            
        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc:^14.2f} | {val_loss:^12.4f} | {val_acc:^14.2f} | {status_msg}")
        
    total_elapsed = time.time() - start_time
    print("-" * 70)
    print(f"[+] Training completed in {total_elapsed:.2f} seconds ({total_elapsed/epochs:.2f}s/epoch)")
    print(f"[+] Optimal Model reached at Epoch {best_epoch} with Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[+] Best Model weights saved to: {Path(output_model).resolve()}")
    
    # 5. Compute Final Confusion Matrix & Classification Metrics
    cm = confusion_matrix(best_labels, best_preds, labels=range(len(classes)))
    print("\n--- Final Validation Classification Report ---")
    print(classification_report(best_labels, best_preds, target_names=classes, digits=3, zero_division=0))
    
    # 6. Generate Training Performance Curves
    plot_training_performance(history, classes, cm, output_path=output_plot)
    
    print("\n" + "=" * 70)
    print("PHASE 3 NEURAL NETWORK TRAINING COMPLETED SUCCESSFULLY!")
    print(f"[+] Best Model Checkpoint : {output_model}")
    print(f"[+] Performance Graph     : {output_plot}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train Deep CNN for SAR Radar ATR")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory containing train/ and test/")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--denoise", type=str, default="bilateral", choices=["bilateral", "median", "lee", "none"],
                        help="Denoising filter for input preprocessing")
    parser.add_argument("--output-model", type=str, default="best_model.pth", help="Target checkpoint filename")
    parser.add_argument("--output-plot", type=str, default="training_performance.png", help="Performance curve plot filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        denoise_method=args.denoise,
        output_model=args.output_model,
        output_plot=args.output_plot,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
