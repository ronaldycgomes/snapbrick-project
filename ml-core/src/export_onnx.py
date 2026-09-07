#!/usr/bin/env python3
"""
SnapBrick Project - Model Optimization & ONNX Exporter
Exports trained YOLOv11 PyTorch weights (best.pt) to ONNX format with FP16 / dynamic shapes
for high-speed production inference and backend/mobile integration.
"""

import sys
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def export_to_onnx(weights_path: Path, imgsz: int = 640, half: bool = True, dynamic: bool = False) -> Path:
    """Export YOLOv11 model to ONNX runtime format."""
    if not HAS_ULTRALYTICS:
        print("[ERROR] Ultralytics is required to export models.", file=sys.stderr)
        sys.exit(1)

    if not weights_path.exists():
        raise FileNotFoundError(f"[ERROR] Weights not found at: {weights_path}")

    print("\n=======================================================")
    print(" ⚡ SnapBrick MLOps: Exporting Model to ONNX")
    print("=======================================================")
    print(f" • Input Weights:  {weights_path}")
    print(f" • Resolution:     {imgsz}x{imgsz}")
    print(f" • FP16 Half-Prec: {'Enabled' if half else 'Disabled'}")
    print(f" • Dynamic Shapes: {'Enabled' if dynamic else 'Disabled'}")
    print("=======================================================\n")

    model = YOLO(str(weights_path))

    # Export to ONNX
    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=True,
        opset=17
    )

    exported_path = Path(exported_path_str)
    print(f"\n✅ ONNX Model successfully exported to: {exported_path}")
    print(f" • File Size: {exported_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("=======================================================\n")
    return exported_path


def main():
    parser = argparse.ArgumentParser(description="SnapBrick YOLOv11 ONNX Exporter")
    parser.add_argument("--weights", type=str, default="", help="Path to best.pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Square image size (640)")
    parser.add_argument("--no_half", action="store_true", help="Disable FP16 half precision")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch/input shapes")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ml_core_dir = script_dir.parent

    if args.weights:
        weights_path = Path(args.weights).resolve()
    else:
        weights_path = ml_core_dir / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = ml_core_dir / "runs" / "snapbrick_poc" / "weights" / "best.pt"

    export_to_onnx(
        weights_path=weights_path,
        imgsz=args.imgsz,
        half=not args.no_half,
        dynamic=args.dynamic
    )


if __name__ == "__main__":
    main()
