#!/usr/bin/env python3
"""
SnapBrick Project - YOLOv11 Model Training & DataLoader Pipeline
Trains YOLOv11 on synthetic LEGO dataset with GPU acceleration (CUDA),
mixed-precision (FP16), real-time metric tracking, and weights export.
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def check_hardware_and_environment() -> Dict[str, Any]:
    """Inspect and report GPU acceleration, CUDA capability, and PyTorch environment."""
    print("\n=======================================================")
    print(" 🧠 SnapBrick ML-Core: Hardware & Environment Audit")
    print("=======================================================")

    env_info = {
        "torch_available": HAS_TORCH,
        "ultralytics_available": HAS_ULTRALYTICS,
        "cuda_available": False,
        "device_name": "CPU",
        "device_count": 0,
        "vram_gb": 0.0,
    }

    if not HAS_TORCH:
        print("[ERROR] PyTorch is not installed in the current Python environment!", file=sys.stderr)
        print("Install with: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124", file=sys.stderr)
        return env_info

    print(f" • PyTorch Version:     {torch.__version__}")

    if not HAS_ULTRALYTICS:
        print("[ERROR] Ultralytics is not installed!", file=sys.stderr)
        print("Install with: uv pip install ultralytics", file=sys.stderr)
        return env_info

    import ultralytics
    print(f" • Ultralytics Version: {ultralytics.__version__}")

    if torch.cuda.is_available():
        env_info["cuda_available"] = True
        env_info["device_count"] = torch.cuda.device_count()
        env_info["device_name"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        env_info["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)

        print(f" • CUDA Acceleration:   ⚡ ENABLED (CUDA {torch.version.cuda})")
        print(f" • Primary GPU:         {env_info['device_name']}")
        print(f" • Total VRAM:          {env_info['vram_gb']} GB")
        print(f" • cuDNN Version:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    else:
        print(" • CUDA Acceleration:   ⚠️ DISABLED (Running on CPU fallback)")

    print("=======================================================\n")
    return env_info


def train_yolo_model(
    data_yaml: Path,
    model_name: str = "yolo11m.pt",
    epochs: int = 150,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project_dir: Path = Path("ml-core/runs"),
    exp_name: str = "snapbrick_yolo11m_poc",
    lr0: float = 0.01,
    patience: int = 25,
    workers: int = 2,
    cache: str = "disk"
) -> Optional[Path]:
    """Execute YOLOv11 fine-tuning on the synthetic dataset."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"[ERROR] Dataset configuration file not found at: {data_yaml}")

    env_info = check_hardware_and_environment()
    if not env_info["torch_available"] or not env_info["ultralytics_available"]:
        sys.exit(1)

    # Determine execution device (CUDA 0 or CPU fallback)
    target_device = device if env_info["cuda_available"] else "cpu"

    print("=======================================================")
    print(" 🚀 Starting YOLOv11 Training Pipeline")
    print("=======================================================")
    print(f" • Base Model Architecture: {model_name}")
    print(f" • Dataset Configuration:   {data_yaml}")
    print(f" • Target Device:           {target_device} ({env_info['device_name']})")
    print(f" • Image Resolution:        {imgsz}x{imgsz}")
    print(f" • Batch Size:              {batch_size}")
    print(f" • Training Epochs:         {epochs} (Early Stopping Patience: {patience})")
    print(f" • DataLoader Workers:      {workers}")
    print(f" • Dataset Cache Mode:      {cache} (Zero-RAM SSD Caching)")
    print(f" • Project Output:          {project_dir / exp_name}")
    print("=======================================================\n")

    # Load pre-trained weights
    model = YOLO(model_name)

    # PyTorch WSL2 stability: set multiprocessing sharing strategy to file_system to avoid /dev/shm crashes
    try:
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
    except Exception:
        pass

    # Normalize cache argument for Ultralytics (False, 'disk', or 'ram')
    cache_arg: Any = False
    if cache.lower() in ("disk", "npy"):
        cache_arg = "disk"
    elif cache.lower() in ("ram", "true"):
        cache_arg = "ram"
    else:
        cache_arg = False

    # Launch Training
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=target_device,
        project=str(project_dir.resolve()),
        name=exp_name,
        patience=patience,
        workers=workers,
        cache=cache_arg,      # 'disk' caches preprocessed .npy on SSD (instant loading, zero RAM spike)
        lr0=lr0,
        cos_lr=True,          # Cosine Annealing Learning Rate Scheduler
        amp=True,             # Automatic Mixed Precision (FP16) for Tensor Cores
        mosaic=1.0,           # Mosaic Data Augmentation
        mixup=0.15,           # MixUp Augmentation
        degrees=15.0,         # Random rotation (+/- 15 deg)
        translate=0.1,        # Random translation (+/- 10%)
        scale=0.5,            # Random scaling (+/- 50%)
        fliplr=0.5,           # Random horizontal flip
        flipud=0.0,           # Vertical flip disabled (gravity reference)
        hsv_h=0.015,          # Subtle HSV Hue jitter
        hsv_s=0.7,            # HSV Saturation jitter
        hsv_v=0.4,            # HSV Value/Brightness jitter
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # Validate and report final metrics
    print("\n=======================================================")
    print(" 📊 Validating Final Model Performance on Test Split")
    print("=======================================================")
    val_metrics = model.val(data=str(data_yaml.resolve()), split="test")

    map50 = val_metrics.box.map50
    map50_95 = val_metrics.box.map
    precision = val_metrics.box.mp
    recall = val_metrics.box.mr

    print(f"\n🎯 [FINAL METRICS SUMMARY]")
    print(f" • mAP@50:      {map50 * 100:.2f}%")
    print(f" • mAP@50-95:   {map50_95 * 100:.2f}%")
    print(f" • Precision:   {precision * 100:.2f}%")
    print(f" • Recall:      {recall * 100:.2f}%")

    # Export best weights to standardized ml-core/weights/ directory
    weights_dir = project_dir.parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_pt_src = project_dir / exp_name / "weights" / "best.pt"
    best_pt_dst = weights_dir / "best.pt"

    if best_pt_src.exists():
        shutil.copy(best_pt_src, best_pt_dst)
        print(f"\n✅ Best model weights saved to: {best_pt_dst}")
    
    print("=======================================================\n")
    return best_pt_dst if best_pt_dst.exists() else None


def main():
    parser = argparse.ArgumentParser(description="SnapBrick YOLOv11 Training Script")
    parser.add_argument("--data", type=str, default="", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Pretrained model weights (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (16 is ideal for yolo11m on 8GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image square resolution (640)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g., '0') or 'cpu'")
    parser.add_argument("--project", type=str, default="", help="Output directory for runs and checkpoints")
    parser.add_argument("--name", type=str, default="snapbrick_yolo11m_poc", help="Experiment name")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience epochs")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader background worker threads (2 is optimal for WSL2)")
    parser.add_argument("--cache", type=str, default="disk", choices=["disk", "ram", "none"], help="Dataset cache mode ('disk' is ultra-fast on SSD with 0 RAM overhead; 'ram' may freeze WSL; 'none' to disable)")
    parser.add_argument("--check_only", action="store_true", help="Audit hardware and environment only without training")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ml_core_dir = script_dir.parent
    project_root = ml_core_dir.parent

    if args.check_only:
        check_hardware_and_environment()
        return

    # Resolve paths
    if args.data:
        data_yaml = Path(args.data).resolve()
    else:
        data_yaml = ml_core_dir / "dataset" / "dataset.yaml"

    if args.project:
        project_dir = Path(args.project).resolve()
    else:
        project_dir = ml_core_dir / "runs"

    train_yolo_model(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project_dir=project_dir,
        exp_name=args.name,
        lr0=args.lr0,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache
    )


if __name__ == "__main__":
    main()
