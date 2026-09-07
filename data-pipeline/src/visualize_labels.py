#!/usr/bin/env python3
"""
SnapBrick Project - YOLO Label Visualizer
Draws 2D bounding boxes and class names directly on synthetic dataset images
to verify annotation accuracy before model training.
Supports Pillow, OpenCV, or Pure Python SVG/HTML fallback without any external dependencies!
"""

import sys
import os
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


BOX_COLORS_HEX = [
    "#FF3333", "#33FF33", "#3399FF", "#FFCC00", "#FF00FF",
    "#00FFFF", "#FF8000", "#B433FF", "#33FFB4", "#FFB4B4",
    "#A6D608", "#E67E22", "#9B59B6", "#1ABC9C", "#E74C3C"
]

BOX_COLORS_RGB = [
    (255, 50, 50), (50, 255, 50), (50, 150, 255), (255, 200, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (180, 50, 255), (50, 255, 180), (255, 180, 180),
]


def load_yaml_names(yaml_path: Path) -> Dict[int, str]:
    """Parse class names from dataset.yaml."""
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


def export_svg_html_visualization(img_path: Path, lines: List[str], names: Dict[int, str], out_path: Path):
    """Zero-dependency SVG/HTML visualizer with embedded base64 image and vector overlays."""
    with open(img_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    svg_elements = []
    w, h = 640, 640

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        c_id = int(parts[0])
        xc, yc, bw, bh = [float(v) for v in parts[1:5]]

        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        box_w = bw * w
        box_h = bh * h

        color_hex = BOX_COLORS_HEX[c_id % len(BOX_COLORS_HEX)]
        class_name = names.get(c_id, f"Class {c_id}")

        svg_elements.append(
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
            f'fill="none" stroke="{color_hex}" stroke-width="3" stroke-opacity="0.9" rx="2"/>'
        )
        svg_elements.append(
            f'<rect x="{x1:.1f}" y="{max(0, y1 - 20):.1f}" width="{len(class_name) * 8.5 + 10:.1f}" height="20" '
            f'fill="{color_hex}" rx="2"/>'
        )
        svg_elements.append(
            f'<text x="{x1 + 5:.1f}" y="{max(14, y1 - 6):.1f}" fill="#FFFFFF" '
            f'font-family="sans-serif" font-size="12px" font-weight="bold">{class_name}</text>'
        )

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <image href="data:image/png;base64,{b64_img}" x="0" y="0" width="{w}" height="{h}"/>
  {''.join(svg_elements)}
</svg>"""

    svg_path = out_path.with_suffix(".svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SnapBrick Annotation Preview - {img_path.name}</title>
  <style>
    body {{ background: #1e1e1e; color: #fff; font-family: sans-serif; text-align: center; padding: 20px; }}
    .card {{ display: inline-block; background: #2d2d2d; padding: 20px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }}
    svg {{ border-radius: 8px; max-width: 100%; height: auto; }}
    h2 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>SnapBrick Annotation Inspector</h2>
    <p>File: <code>{img_path.name}</code> | {len(lines)} Pieces Detected</p>
    {svg_content}
  </div>
</body>
</html>"""

    html_path = out_path.with_suffix(".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[SUCCESS] Zero-dependency visualizer generated:")
    print(f"  ➜ SVG:  {svg_path}")
    print(f"  ➜ HTML: {html_path} (Open in any browser)")


def visualize_annotations(img_path: Path, lbl_path: Path, output_path: Path, names: Dict[int, str]):
    """Draw bounding boxes and class labels on the image."""
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}", file=sys.stderr)
        return
    if not lbl_path.exists():
        print(f"[ERROR] Label not found: {lbl_path}", file=sys.stderr)
        return

    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_PIL:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            c_id = int(parts[0])
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]

            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            color = BOX_COLORS_RGB[c_id % len(BOX_COLORS_RGB)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            label_text = names.get(c_id, f"Class {c_id}")
            text_bg = [x1, max(0, y1 - 18), min(w, x1 + len(label_text) * 9 + 8), y1]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1 + 4, max(0, y1 - 16)), label_text, fill=(255, 255, 255))

        out_img_path = output_path.with_suffix(".png")
        img.save(out_img_path)
        print(f"[SUCCESS] Visualized {len(lines)} labels -> {out_img_path}")

    elif HAS_CV2:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            c_id = int(parts[0])
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]

            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            color = BOX_COLORS_RGB[c_id % len(BOX_COLORS_RGB)]
            bgr = (color[2], color[1], color[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)

            label_text = names.get(c_id, f"Class {c_id}")
            cv2.putText(img, label_text, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

        out_img_path = output_path.with_suffix(".png")
        cv2.imwrite(str(out_img_path), img)
        print(f"[SUCCESS] Visualized {len(lines)} labels -> {out_img_path}")

    else:
        # Fallback to Pure Python SVG & HTML
        export_svg_html_visualization(img_path, lines, names, output_path)


def main():
    parser = argparse.ArgumentParser(description="SnapBrick YOLO Annotation Visualizer")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--label", type=str, default="", help="Path to label .txt file (defaults to matching name)")
    parser.add_argument("--yaml", type=str, default="", help="Path to dataset.yaml")
    parser.add_argument("--output", type=str, default="", help="Path to output visualized image")

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    img_path = Path(args.image).resolve()

    if args.label:
        lbl_path = Path(args.label).resolve()
    else:
        rel_str = str(img_path).replace("/images/", "/labels/")
        lbl_path = Path(rel_str).with_suffix(".txt")

    if args.yaml:
        yaml_path = Path(args.yaml).resolve()
    else:
        cand = img_path.parent.parent.parent / "dataset.yaml"
        yaml_path = cand if cand.exists() else Path("dataset.yaml")

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = img_path.parent / f"annotated_{img_path.name}"

    names = load_yaml_names(yaml_path)
    visualize_annotations(img_path, lbl_path, out_path, names)


if __name__ == "__main__":
    main()
