#!/usr/bin/env python3
"""
SnapBrick Project - Unified Two-Stage Inference Pipeline (detect_pipeline.py)
Industrial Two-Stage Vision Pipeline:
  Stage 1: High-Recall Class-Agnostic Detector (lego_piece) with Multi-Quadrant SAHI Slicing
  Stage 2: High-Resolution 224x224 Contextual Crop Geometry Classifier
  Stage 3: Background-Subtracted CIE L*a*b* Color Segmenter
Validates real physical photos (IMG_0033.jpg / IMG_0032.jpg) end-to-end.
"""

import sys
import os
import math
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from ultralytics import YOLO
except ImportError as e:
    print(f"[ERROR] Required libraries missing: {e}", file=sys.stderr)
    sys.exit(1)


# ==============================================================================
# LEGO Color Palette & CIE Lab Matching
# ==============================================================================

OFFICIAL_LEGO_LAB_PALETTE = {
    "Red": (53.2, 80.1, 67.2),
    "Blue": (32.3, 9.2, -56.3),
    "Yellow": (89.5, -9.1, 95.8),
    "Green": (46.2, -51.7, 28.3),
    "Black": (12.0, 0.5, -1.2),
    "White": (96.0, -0.5, 1.2),
    "Light Gray": (75.0, -1.0, -0.5),
    "Dark Gray": (45.0, -0.8, -1.0),
    "Orange": (66.5, 52.4, 78.6),
    "Lime": (78.0, -38.2, 70.1),
    "Tan": (82.0, 2.5, 24.5),
    "Brown": (34.0, 22.0, 26.0),
    "Dark Blue": (22.0, 4.0, -30.0),
}


def rgb_to_cielab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert sRGB (0-255) to CIE L*a*b* (D65 standard illuminant)."""
    def pivot(v):
        v = v / 255.0
        return ((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92

    r_lin = pivot(r)
    g_lin = pivot(g)
    b_lin = pivot(b)

    x = (r_lin * 0.4124 + g_lin * 0.3576 + b_lin * 0.1805) / 0.95047
    y = (r_lin * 0.2126 + g_lin * 0.7152 + b_lin * 0.0722) / 1.00000
    z = (r_lin * 0.0193 + g_lin * 0.1192 + b_lin * 0.9505) / 1.08883

    def f(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    L = max(0.0, min(100.0, (116.0 * f(y)) - 16.0))
    a = 500.0 * (f(x) - f(y))
    b_val = 200.0 * (f(y) - f(z))
    return (L, a, b_val)


def classify_roi_color(roi_bgr: np.ndarray, local_table_bgr: np.ndarray, piece_id: int = 0) -> str:
    """
    Robust, physics-based LEGO color classification via:
      1. Local table background subtraction with 2D color distance
      2. Morphological erosion to strip table-edge bleed and contact shadows
      3. Specular glare filtering (rejects white highlights on shiny ABS plastic)
      4. Calibrated HSV quorum voting + CIE L*a*b* Delta-E fallback.
    """
    if roi_bgr.size == 0:
        return "Unknown"

    h_img, w_img = roi_bgr.shape[:2]
    table_vec = local_table_bgr.astype(np.float32).ravel()[:3]

    # --- Step 1: 2D Euclidean distance map from table color ---
    diff_bgr = roi_bgr.astype(np.float32) - table_vec
    dists_2d = np.linalg.norm(diff_bgr, axis=2)

    # Adaptive threshold: Plastic pixels must differ significantly in color from surrounding table
    TABLE_DIST_THRESH = 24.0
    raw_plastic_mask = (dists_2d > TABLE_DIST_THRESH).astype(np.uint8)

    # Morphological erosion: strips 2px border where table wood grain blends into plastic edge
    kernel_size = 3 if min(h_img, w_img) > 25 else 1
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded_mask = cv2.erode(raw_plastic_mask, kernel, iterations=1)
    else:
        eroded_mask = raw_plastic_mask

    # If erosion preserved sufficient plastic pixels, use eroded; otherwise fallback gracefully
    if np.sum(eroded_mask) >= max(20, int(h_img * w_img * 0.05)):
        final_mask = eroded_mask
    elif np.sum(raw_plastic_mask) >= max(10, int(h_img * w_img * 0.03)):
        final_mask = raw_plastic_mask
    else:
        # Fallback to center core region (inner 35%)
        final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.ellipse(
            final_mask,
            (w_img // 2, h_img // 2),
            (max(1, int(w_img * 0.35)), max(1, int(h_img * 0.35))),
            0, 0, 360, 1, -1
        )

    plastic_pixels = roi_bgr[final_mask == 1].reshape(-1, 3)
    if len(plastic_pixels) < 5:
        return "Unknown"

    # --- Step 2: Glare & Specular Reflection Filtering ---
    px_bgr = plastic_pixels.reshape(-1, 1, 3).astype(np.uint8)
    px_hsv = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_arr = px_hsv[:, 0].astype(np.int16)
    s_arr = px_hsv[:, 1].astype(np.int16)
    v_arr = px_hsv[:, 2].astype(np.int16)

    # Filter out pure white specular glares (V >= 250, S <= 15) UNLESS most of the piece is white
    is_mostly_white = np.mean(v_arr) > 210 and np.mean(s_arr) < 35
    if not is_mostly_white:
        valid_px = ~((v_arr >= 250) & (s_arr <= 15))
        if np.sum(valid_px) >= 10:
            h_arr = h_arr[valid_px]
            s_arr = s_arr[valid_px]
            v_arr = v_arr[valid_px]
            plastic_pixels = plastic_pixels[valid_px]

    # --- Step 3: Canonical LEGO ABS Polymer Signatures ---
    votes = {
        "Black":      int(np.sum(v_arr <= 65)),
        "White":      int(np.sum((s_arr <= 32) & (v_arr >= 215))),
        "Light Gray": int(np.sum((s_arr <= 48) & (v_arr >= 85) & (v_arr < 215))),
        "Dark Gray":  int(np.sum((s_arr <= 35) & (v_arr >= 65) & (v_arr < 85))),
        "Red":        int(np.sum(((h_arr <= 10) | (h_arr >= 165)) & (s_arr >= 70) & (v_arr >= 50))),
        "Blue":       int(np.sum((h_arr >= 90) & (h_arr <= 135) & (s_arr >= 55) & (v_arr >= 45))),
        "Green":      int(np.sum((h_arr >= 35) & (h_arr <= 88) & (s_arr >= 45) & (v_arr >= 45))),
        "Yellow":     int(np.sum((h_arr >= 22) & (h_arr <= 34) & (s_arr >= 90) & (v_arr >= 115))),
        "Orange":     int(np.sum((h_arr >= 10) & (h_arr <= 20) & (s_arr >= 160) & (v_arr >= 135))),
        "Lime":       int(np.sum((h_arr >= 25) & (h_arr <= 50) & (s_arr >= 95) & (v_arr >= 85))),
    }

    best_color, max_votes = max(votes.items(), key=lambda item: item[1])

    # Diagnostic print for debugging
    if piece_id > 0:
        top3 = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join([f"{k}:{v}" for k, v in top3 if v > 0])
        print(f"   [Color #{piece_id:02d}] Plastic={len(plastic_pixels)}px | Winner={best_color} ({max_votes}px) | Top: {top3_str}")

    # Winner quorum: winner must have at least 15 pixels
    if max_votes >= 15:
        return best_color

    # Fallback to CIE Lab Delta-E if votes are inconclusive
    med_b = float(np.median(plastic_pixels[:, 0]))
    med_g = float(np.median(plastic_pixels[:, 1]))
    med_r = float(np.median(plastic_pixels[:, 2]))
    L, a, b_val = rgb_to_cielab(int(med_r), int(med_g), int(med_b))

    min_de = float("inf")
    fallback_color = "Unknown"
    for name, ref in OFFICIAL_LEGO_LAB_PALETTE.items():
        de = math.sqrt((L - ref[0]) ** 2 + (a - ref[1]) ** 2 + (b_val - ref[2]) ** 2)
        if de < min_de:
            min_de = de
            fallback_color = name

    return fallback_color


# ==============================================================================
# SAHI & NMS Utilities
# ==============================================================================

def compute_iou(b1: List[int], b2: List[int]) -> float:
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = a1 + a2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def apply_global_nms(candidates: List[Dict[str, Any]], iou_thresh: float = 0.40, containment_thresh: float = 0.50) -> List[Dict[str, Any]]:
    """Global NMS with bi-directional containment suppression to eliminate sub-slice duplicate boxes."""
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in sorted_cands:
        should_keep = True
        to_remove = []
        c_box = cand["bbox"]
        c_area = (c_box[2] - c_box[0]) * (c_box[3] - c_box[1])

        for k in kept:
            k_box = k["bbox"]
            k_area = (k_box[2] - k_box[0]) * (k_box[3] - k_box[1])
            iou = compute_iou(c_box, k_box)

            xi1 = max(c_box[0], k_box[0])
            yi1 = max(c_box[1], k_box[1])
            xi2 = min(c_box[2], k_box[2])
            yi2 = min(c_box[3], k_box[3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            min_area = min(c_area, k_area)
            containment = inter_area / min_area if min_area > 0 else 0.0

            if iou > iou_thresh:
                should_keep = False
                break

            if containment > containment_thresh:
                # One box is mostly inside the other
                if c_area > 1.8 * k_area:
                    # cand is the full physical piece, k was an early sub-slice duplicate
                    to_remove.append(k)
                else:
                    # k is the larger piece (or comparable), cand is duplicate
                    should_keep = False
                    break

        if should_keep:
            for rem in to_remove:
                kept.remove(rem)
            kept.append(cand)

    return kept


# ==============================================================================
# Two-Stage Unified Pipeline Engine
# ==============================================================================

class TwoStageVisionPipeline:
    def __init__(self, detector_weights: Path, classifier_weights: Path, device: str = "0"):
        print("\n=======================================================")
        print(" 🔍 SnapBrick Unified Two-Stage Vision Pipeline")
        print("=======================================================")
        print(f" • Stage 1 Detector:    {detector_weights.name}")
        print(f" • Stage 2 Classifier:  {classifier_weights.name}")

        self.detector = YOLO(str(detector_weights))
        self.classifier = YOLO(str(classifier_weights))
        self.device = device
        print(f" • Device:              {device}")
        print("=======================================================\n")

    def run(
        self,
        image_path: Path,
        output_dir: Path,
        conf_det: float = 0.30,
        iou_det: float = 0.40,
        imgsz_det: int = 640
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            try:
                pil_img = Image.open(str(image_path))
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise RuntimeError(f"Failed to load image via OpenCV and PIL: {e}")

        orig_h, orig_w = img_bgr.shape[:2]
        print(f"📷 Processing: {image_path.name} ({orig_w}x{orig_h} px)")

        # Sample pure table background once globally from the 4 corners of the full photo
        corner_h = max(20, int(orig_h * 0.05))
        corner_w = max(20, int(orig_w * 0.05))
        corners = [
            img_bgr[:corner_h, :corner_w],
            img_bgr[:corner_h, -corner_w:],
            img_bgr[-corner_h:, :corner_w],
            img_bgr[-corner_h:, -corner_w:]
        ]
        global_table_bgr = np.median(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)

        # Stage 1: Class-Agnostic Piece Detection with Multi-Quadrant SAHI Slicing
        t0 = time.time()
        raw_candidates = []

        # 1.1 Global Full-Image Pass (catches large plates like Plate 2x8, 1x8)
        full_results = self.detector.predict(source=img_bgr, conf=conf_det, iou=iou_det, imgsz=imgsz_det, device=self.device, verbose=False)[0]
        for box in full_results.boxes:
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            raw_candidates.append({
                "confidence": conf,
                "bbox": [max(0, x1), max(0, y1), min(orig_w - 1, x2), min(orig_h - 1, y2)]
            })

        # 1.2 Multi-Quadrant SAHI Slices (catches tiny pieces: 1x1 studs, slopes, pins)
        if orig_w > 1000 or orig_h > 1000:
            grid_x = 3
            grid_y = 3
            tile_w = int(orig_w / (grid_x - 0.25 * (grid_x - 1)))
            tile_h = int(orig_h / (grid_y - 0.25 * (grid_y - 1)))
            step_x = int(tile_w * 0.75)
            step_y = int(tile_h * 0.75)

            for gy in range(grid_y):
                for gx in range(grid_x):
                    sx1 = min(gx * step_x, max(0, orig_w - tile_w))
                    sy1 = min(gy * step_y, max(0, orig_h - tile_h))
                    sx2 = min(sx1 + tile_w, orig_w)
                    sy2 = min(sy1 + tile_h, orig_h)

                    tile_crop = img_bgr[sy1:sy2, sx1:sx2]
                    if tile_crop.size == 0:
                        continue

                    slice_results = self.detector.predict(source=tile_crop, conf=conf_det, iou=iou_det, imgsz=640, device=self.device, verbose=False)[0]
                    for box in slice_results.boxes:
                        conf = float(box.conf[0].item())
                        lx1, ly1, lx2, ly2 = [int(v) for v in box.xyxy[0].tolist()]
                        raw_candidates.append({
                            "confidence": conf,
                            "bbox": [max(0, lx1 + sx1), max(0, ly1 + sy1), min(orig_w - 1, lx2 + sx1), min(orig_h - 1, ly2 + sy1)]
                        })

        # Merge overlapping slice detections with Global NMS
        filtered_detections = apply_global_nms(raw_candidates, iou_thresh=iou_det)
        t_det = time.time() - t0
        print(f" • Stage 1: Detected {len(filtered_detections)} pieces in {t_det*1000:.1f}ms (SAHI 3x3 Slices)")

        # Visual annotation canvas
        annotated_img = img_bgr.copy()
        crops_dir = output_dir / "crops_224"
        crops_dir.mkdir(parents=True, exist_ok=True)

        inventory_items = []
        t0_cls = time.time()

        for idx, det in enumerate(filtered_detections):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            bw = x2 - x1
            bh = y2 - y1

            # CONTEXTUAL SQUARE CROP:
            # Extract a square region directly from the high-res photo centered on the piece.
            # NO BLACK BARS! Background around piece is the real table texture, matching training data!
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            side = int(max(bw, bh) * 1.25)
            half = side // 2

            crop_x1 = max(0, cx - half)
            crop_y1 = max(0, cy - half)
            crop_x2 = min(orig_w, cx + half)
            crop_y2 = min(orig_h, cy + half)

            square_roi = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            if square_roi.size == 0:
                continue

            # Ensure strict square aspect ratio by padding edges with border replication (not black bars!)
            sh, sw = square_roi.shape[:2]
            if sh != sw:
                target_dim = max(sh, sw)
                pad_top = (target_dim - sh) // 2
                pad_bottom = target_dim - sh - pad_top
                pad_left = (target_dim - sw) // 2
                pad_right = target_dim - sw - pad_left
                square_roi = cv2.copyMakeBorder(
                    square_roi, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REPLICATE
                )

            crop_224 = cv2.resize(square_roi, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            # Stage 2: Geometry Classification (Pure Neural Network Classifier)
            cls_res = self.classifier(crop_224, verbose=False)[0]
            top1_id = cls_res.probs.top1
            best_conf = cls_res.probs.top1conf.item()
            best_class = self.classifier.names[top1_id]

            # Parse Part ID and Name directly from model output
            parts = best_class.split("_")
            part_id = parts[0]
            part_name = " ".join(parts[1:])

            # Stage 3: Color Classification via Local-Table Subtraction
            # Use the TIGHT detector bbox (not the enlarged crop_224) so the
            # piece-to-background pixel ratio is maximized.
            piece_roi = img_bgr[y1:y2, x1:x2]

            # Sample LOCAL table color from a safety-gapped margin band
            # OUTSIDE the bbox — avoids piece-edge bleeding.
            gap = max(3, int(max(bw, bh) * 0.03))
            margin_w = max(20, int(max(bw, bh) * 0.18))
            ex_y1 = max(0, y1 - gap - margin_w)
            ex_y2 = min(orig_h, y2 + gap + margin_w)
            ex_x1 = max(0, x1 - gap - margin_w)
            ex_x2 = min(orig_w, x2 + gap + margin_w)

            margin_region = img_bgr[ex_y1:ex_y2, ex_x1:ex_x2]
            margin_mask = np.ones((ex_y2 - ex_y1, ex_x2 - ex_x1), dtype=bool)

            # Carve out the inner bbox + gap so we only keep pure table pixels
            inner_y1 = max(0, (y1 - gap) - ex_y1)
            inner_y2 = min(ex_y2 - ex_y1, (y2 + gap) - ex_y1)
            inner_x1 = max(0, (x1 - gap) - ex_x1)
            inner_x2 = min(ex_x2 - ex_x1, (x2 + gap) - ex_x1)
            margin_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

            margin_pixels = margin_region[margin_mask]
            if len(margin_pixels) > 10:
                local_table = np.median(margin_pixels, axis=0).astype(np.float32)
            else:
                # Fallback: sample from photo corners
                cs = max(50, min(orig_h, orig_w) // 20)
                corners = np.concatenate([
                    img_bgr[:cs, :cs].reshape(-1, 3),
                    img_bgr[:cs, -cs:].reshape(-1, 3),
                    img_bgr[-cs:, :cs].reshape(-1, 3),
                    img_bgr[-cs:, -cs:].reshape(-1, 3),
                ])
                local_table = np.median(corners, axis=0).astype(np.float32)

            color_name = classify_roi_color(piece_roi, local_table, piece_id=idx + 1)

            # Save crop for visual review
            crop_fname = f"crop_{idx+1:02d}_{part_id}_{color_name}.png"
            cv2.imwrite(str(crops_dir / crop_fname), crop_224)

            item = {
                "id": idx + 1,
                "part_id": part_id,
                "name": part_name,
                "color": color_name,
                "confidence_det": round(float(conf), 3),
                "confidence_cls": round(float(best_conf), 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_file": crop_fname
            }
            inventory_items.append(item)

            # Draw Annotation Box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 220, 100), 3)

            # Draw Label Tag
            label_text = f"#{idx+1} [{part_id}] {part_name[:16]} {color_name} ({best_conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), (20, 20, 20), -1)
            cv2.putText(annotated_img, label_text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        t_cls = time.time() - t0_cls
        print(f" • Stage 2: Classified {len(inventory_items)} crops in {t_cls*1000:.1f}ms ({t_cls/max(1, len(inventory_items))*1000:.1f}ms/crop)")

        # Save Visual & JSON
        annotated_path = output_dir / f"annotated_two_stage_{image_path.name}"
        cv2.imwrite(str(annotated_path), annotated_img)

        inventory_json = {
            "image": image_path.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_pieces_detected": len(inventory_items),
            "pieces": inventory_items,
            "performance": {
                "stage1_detection_ms": round(t_det * 1000, 1),
                "stage2_classification_total_ms": round(t_cls * 1000, 1),
                "stage2_per_crop_avg_ms": round(t_cls / max(1, len(inventory_items)) * 1000, 1)
            }
        }

        json_path = output_dir / f"inventory_two_stage_{image_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(inventory_json, f, indent=2, ensure_ascii=False)

        print("\n=======================================================")
        print(f" ✅ Two-Stage Pipeline Complete: {len(inventory_items)} pieces detected!")
        print(f" • Annotated Visual:  {annotated_path}")
        print(f" • Inventory JSON:    {json_path}")
        print(f" • Extracted Crops:   {crops_dir}")
        print("=======================================================\n")

        print("--- Resumo do Inventário Identificado ---")
        for it in inventory_items:
            print(f" #{it['id']:02d} | Part #{it['part_id']} {it['name']:<22} | Cor: {it['color']:<10} | Det: {it['confidence_det']*100:.0f}% | Cls: {it['confidence_cls']*100:.0f}%")

        return inventory_json


def main():
    parser = argparse.ArgumentParser(description="SnapBrick Unified Two-Stage Inference Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input photo (e.g. IMG_0033.jpg)")
    parser.add_argument("--detector", type=str, default="ml-core/runs/stage1_detector_poc/weights/best.pt", help="Stage 1 detector weights")
    parser.add_argument("--classifier", type=str, default="ml-core/runs/stage2_classifier/weights/best.pt", help="Stage 2 classifier weights")
    parser.add_argument("--output_dir", type=str, default="ml-core/runs/inference_two_stage", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.28, help="Stage 1 detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.35, help="Stage 1 detector IOU threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")

    args = parser.parse_args()

    pipeline = TwoStageVisionPipeline(
        detector_weights=Path(args.detector).resolve(),
        classifier_weights=Path(args.classifier).resolve(),
        device=args.device
    )

    pipeline.run(
        image_path=Path(args.image).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        conf_det=args.conf,
        iou_det=args.iou
    )


if __name__ == "__main__":
    main()
