#!/usr/bin/env python3
"""
SnapBrick Project - Inference & Color Classifier Pipeline
Detects LEGO parts from photos using YOLOv11 (best.pt), extracts ROIs,
identifies color in CIE L*a*b* space, and exports structured inventory JSON.
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


# ==============================================================================
# LEGO Reference Palette in CIE L*a*b* Space
# ==============================================================================

# Standard sRGB values for official LEGO colors
LEGO_SRGB_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "Red": (201, 26, 9),
    "Blue": (0, 85, 191),
    "Yellow": (242, 205, 55),
    "Green": (0, 133, 43),
    "Black": (27, 42, 52),
    "White": (244, 244, 244),
    "Orange": (254, 138, 24),
    "Light Gray": (138, 146, 141),
    "Dark Gray": (84, 89, 85),
    "Pearl Gold": (204, 156, 43),
    "Metallic Silver": (192, 192, 192),
    "Dark Red": (114, 14, 15),
    "Dark Blue": (10, 52, 99),
    "Lime": (165, 202, 24),
    "Tan": (222, 198, 156),
    "Brown": (88, 57, 39),
    "Trans-Orange": (235, 120, 20),
    "Trans-Red": (200, 30, 30),
    "Trans-Yellow": (240, 210, 40),
    "Trans-Blue": (40, 120, 220),
}


def rgb_to_cielab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert sRGB (0-255) to CIE L*a*b* color space."""
    # 1. sRGB to linear RGB
    r_lin = r / 255.0
    g_lin = g / 255.0
    b_lin = b / 255.0

    r_lin = ((r_lin + 0.055) / 1.055) ** 2.4 if r_lin > 0.04045 else r_lin / 12.92
    g_lin = ((g_lin + 0.055) / 1.055) ** 2.4 if g_lin > 0.04045 else g_lin / 12.92
    b_lin = ((b_lin + 0.055) / 1.055) ** 2.4 if b_lin > 0.04045 else b_lin / 12.92

    # 2. Linear RGB to CIE XYZ (D65 illuminant)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # Normalize for D65 white point
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # 3. XYZ to CIE L*a*b*
    fx = x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    fy = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    fz = z ** (1 / 3) if z > 0.008856 else (7.787 * z) + (16 / 116)

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)

    return (L, a, b_val)


# Precompute reference LAB palette
LEGO_LAB_PALETTE: Dict[str, Tuple[float, float, float]] = {
    name: rgb_to_cielab(r, g, b) for name, (r, g, b) in LEGO_SRGB_PALETTE.items()
}


def classify_roi_color(roi_bgr: np.ndarray) -> str:
    """Classify the dominant color of a cropped piece ROI using CIE Delta E distance."""
    if roi_bgr is None or roi_bgr.size == 0:
        return "Unknown"

    h, w = roi_bgr.shape[:2]
    if h < 3 or w < 3:
        return "Unknown"

    # Crop the central 50% core of the ROI to avoid background table pixels
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    core = roi_bgr[y1:y2, x1:x2]
    if core.size == 0:
        core = roi_bgr

    # Compute median color in core (more robust to highlights/shadows than mean)
    median_b = float(np.median(core[:, :, 0]))
    median_g = float(np.median(core[:, :, 1]))
    median_r = float(np.median(core[:, :, 2]))

    sample_lab = rgb_to_cielab(int(median_r), int(median_g), int(median_b))

    # Delta E Euclidean distance matching
    best_color = "Unknown"
    min_delta_e = float("inf")

    for color_name, ref_lab in LEGO_LAB_PALETTE.items():
        # Weighted Delta E: lower weight on L* (brightness) to be invariant to lighting
        dL = (sample_lab[0] - ref_lab[0]) * 0.6
        da = sample_lab[1] - ref_lab[1]
        db = sample_lab[2] - ref_lab[2]
        delta_e = math.sqrt(dL * dL + da * da + db * db)

        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_color = color_name

    return best_color


# ==============================================================================
# Model Loading & Inference Engine
# ==============================================================================

def load_yaml_names(yaml_path: Path) -> Dict[int, str]:
    """Parse class names mapping from dataset.yaml."""
    names = {}
    if not yaml_path.exists():
        return names
    with open(yaml_path, "r", encoding="utf-8") as f:
        in_names = False
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    try:
                        c_id = int(parts[0].strip())
                        c_name = parts[1].strip().strip('"').strip("'")
                        names[c_id] = c_name
                    except ValueError:
                        pass
                elif not stripped:
                    continue
                else:
                    break
    return names


def run_detection(
    image_path: Path,
    weights_path: Path,
    dataset_yaml: Path,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    output_dir: Path = Path("ml-core/runs/inference")
) -> Dict[str, Any]:
    """Run full inference pipeline: detection + color classification + JSON export."""
    if not HAS_ULTRALYTICS or not HAS_CV2:
        print("[ERROR] OpenCV and Ultralytics are required.", file=sys.stderr)
        sys.exit(1)

    if not image_path.exists():
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"[ERROR] Weights not found: {weights_path}")

    class_names = load_yaml_names(dataset_yaml)

# ==============================================================================
# Sliced High-Resolution Inference (SAHI / Quadrant-based Detection)
# ==============================================================================

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def apply_global_nms(candidates: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """Filter duplicate overlapping detections across multiple slices preserving highest score."""
    if not candidates:
        return []

    # Sort by confidence descending
    sorted_candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_candidates:
        should_keep = True
        for k in kept:
            # Check IoU
            iou = compute_iou(cand["bbox"], k["bbox"])
            if iou > iou_thresh:
                should_keep = False
                break
        if should_keep:
            kept.append(cand)

    return kept


def run_detection(
    image_path: Path,
    weights_path: Path,
    dataset_yaml: Path,
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.45,
    use_sliced: bool = True,
    output_dir: Path = Path("ml-core/runs/inference")
) -> Dict[str, Any]:
    """Run full inference pipeline: High-Res Sliced Detection + color classification + JSON export."""
    if not HAS_ULTRALYTICS or not HAS_CV2:
        print("[ERROR] OpenCV and Ultralytics are required.", file=sys.stderr)
        sys.exit(1)

    if not image_path.exists():
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"[ERROR] Weights not found: {weights_path}")

    class_names = load_yaml_names(dataset_yaml)

    print("\n=======================================================")
    print(" 🔍 SnapBrick High-Resolution Vision & Recognition Engine")
    print("=======================================================")
    print(f" • Input Image:     {image_path.name}")
    print(f" • Model Weights:   {weights_path}")
    print(f" • Confidence Cut:  {conf_thresh * 100:.0f}%")
    print(f" • High-Res Slices: {'Enabled (Multi-Quadrant SAHI)' if use_sliced else 'Disabled (Single Full Pass)'}")
    print("=======================================================\n")

    # Load model
    model = YOLO(str(weights_path))

    # Robust image loading (handles smartphone EXIF orientation)
    img_bgr = cv2.imread(str(image_path.resolve()))
    if img_bgr is None:
        try:
            from PIL import Image, ImageOps
            pil_img = Image.open(image_path.resolve())
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load image at {image_path}: {e}")

    h_orig, w_orig = img_bgr.shape[:2]
    raw_candidates: List[Dict[str, Any]] = []

    # 1. Global Full-Image Inference (catches large pieces like Plate 1x8, Wings)
    full_results = model.predict(source=img_bgr, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
    for r in full_results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            raw_candidates.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [max(0, x1), max(0, y1), min(w_orig - 1, x2), min(h_orig - 1, y2)]
            })

    # 2. Sliced Multi-Quadrant Inference for Tiny Pieces (Studs, Pins, Slopes)
    if use_sliced and (w_orig > 1000 or h_orig > 1000):
        # 3x3 overlapping grid with 25% overlap
        grid_x = 3
        grid_y = 3
        tile_w = int(w_orig / (grid_x - 0.25 * (grid_x - 1)))
        tile_h = int(h_orig / (grid_y - 0.25 * (grid_y - 1)))
        step_x = int(tile_w * 0.75)
        step_y = int(tile_h * 0.75)

        slices = []
        for gy in range(grid_y):
            for gx in range(grid_x):
                sx1 = min(gx * step_x, max(0, w_orig - tile_w))
                sy1 = min(gy * step_y, max(0, h_orig - tile_h))
                sx2 = min(sx1 + tile_w, w_orig)
                sy2 = min(sy1 + tile_h, h_orig)
                slices.append((sx1, sy1, sx2, sy2))

        for sx1, sy1, sx2, sy2 in slices:
            tile_crop = img_bgr[sy1:sy2, sx1:sx2]
            if tile_crop.size == 0:
                continue
            slice_results = model.predict(source=tile_crop, conf=conf_thresh, iou=iou_thresh, imgsz=640, verbose=False)
            for r in slice_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    lx1, ly1, lx2, ly2 = [int(v) for v in box.xyxy[0].tolist()]
                    # Translate slice coords to global coords
                    gx1 = max(0, min(w_orig - 1, lx1 + sx1))
                    gy1 = max(0, min(h_orig - 1, ly1 + sy1))
                    gx2 = max(0, min(w_orig - 1, lx2 + sx1))
                    gy2 = max(0, min(h_orig - 1, ly2 + sy1))

                    raw_candidates.append({
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": [gx1, gy1, gx2, gy2]
                    })

    # 3. Apply Global NMS to merge overlapping slice detections
    filtered_detections = apply_global_nms(raw_candidates, iou_thresh=iou_thresh)

    detected_items: List[Dict[str, Any]] = []
    summary_counts: Dict[str, int] = {}
    annotated_img = img_bgr.copy()

    BOX_COLORS = [
        (40, 40, 255), (40, 255, 40), (255, 140, 40), (40, 220, 255),
        (255, 40, 255), (0, 255, 255), (0, 165, 255), (200, 50, 200)
    ]

    for det in filtered_detections:
        cls_id = det["class_id"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        full_label = class_names.get(cls_id, f"Class_{cls_id}")
        part_id = ""
        if "(" in full_label and ")" in full_label:
            part_id = full_label[full_label.rfind("(") + 1:full_label.rfind(")")]
        else:
            part_id = str(cls_id)

        # High-res ROI extraction and color classification
        roi = img_bgr[y1:y2, x1:x2]
        color_name = classify_roi_color(roi)

        item = {
            "class_id": cls_id,
            "part_id": part_id,
            "name": full_label,
            "color": color_name,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2]
        }
        detected_items.append(item)

        key = f"{part_id}_{color_name.replace(' ', '_')}"
        summary_counts[key] = summary_counts.get(key, 0) + 1

        color_bgr = BOX_COLORS[cls_id % len(BOX_COLORS)]
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_bgr, 3)

        display_text = f"{full_label.split('(')[0].strip()} [{color_name}] {conf:.2f}"
        t_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        c2 = x1 + t_size[0] + 8, max(0, y1 - t_size[1] - 10)
        cv2.rectangle(annotated_img, (x1, y1), c2, color_bgr, -1)
        cv2.putText(annotated_img, display_text, (x1 + 4, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / f"detected_highres_{image_path.name}"
    out_json_path = output_dir / f"inventory_{image_path.stem}.json"

    cv2.imwrite(str(out_img_path), annotated_img)

    inventory_payload = {
        "image": image_path.name,
        "total_pieces_detected": len(detected_items),
        "inventory": detected_items,
        "summary_counts": summary_counts
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(inventory_payload, f, indent=2, ensure_ascii=False)

    print(f"[RESULT] ✅ High-Res Inference: Detected {len(detected_items)} LEGO pieces in {image_path.name}!")
    print(f" • Annotated Visual:  {out_img_path}")
    print(f" • Inventory JSON:    {out_json_path}")
    print("\n--- Summary of Detected Inventory ---")
    for k, v in summary_counts.items():
        print(f" • {k}: {v}x")
    print("=======================================================\n")

    return inventory_payload


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SnapBrick YOLOv11 Detection & Color Classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file (e.g. data-pipeline/output/IMG_0032.jpg)")
    parser.add_argument("--weights", type=str, default="", help="Path to trained best.pt weights")
    parser.add_argument("--yaml", type=str, default="", help="Path to dataset.yaml")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--no_sliced", action="store_true", help="Disable multi-quadrant SAHI sliced inference")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for visual and JSON results")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ml_core_dir = script_dir.parent
    project_root = ml_core_dir.parent

    # Path resolution
    img_path = Path(args.image).resolve()

    if args.weights:
        weights_path = Path(args.weights).resolve()
    else:
        weights_path = ml_core_dir / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = ml_core_dir / "runs" / "snapbrick_yolo11m_poc" / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = ml_core_dir / "runs" / "snapbrick_poc" / "weights" / "best.pt"

    if args.yaml:
        yaml_path = Path(args.yaml).resolve()
    else:
        yaml_path = ml_core_dir / "dataset" / "dataset.yaml"

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = ml_core_dir / "runs" / "inference"

    run_detection(
        image_path=img_path,
        weights_path=weights_path,
        dataset_yaml=yaml_path,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        use_sliced=not args.no_sliced,
        output_dir=out_dir
    )


if __name__ == "__main__":
    main()
