#!/usr/bin/env python3
"""
SnapBrick Project - Headless Blender Part Renderer
Renders LDraw (.dat) parts into high-quality images with GPU acceleration,
automatic bounding-box camera framing, studio 3-point lighting, and material shaders.
"""

import sys
import os
import re
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Try importing bpy (Blender Python API)
try:
    import bpy
    import mathutils
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


# ==============================================================================
# Color Helpers
# ==============================================================================

NAMED_LEGO_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "red": (0.79, 0.05, 0.05, 1.0),       # Classic LEGO Bright Red (#C91A09)
    "blue": (0.0, 0.33, 0.75, 1.0),       # Classic LEGO Bright Blue (#0055BF)
    "yellow": (0.95, 0.80, 0.05, 1.0),   # Classic LEGO Bright Yellow (#F2CD37)
    "green": (0.02, 0.52, 0.15, 1.0),    # Classic LEGO Dark Green (#00852B)
    "black": (0.11, 0.16, 0.20, 1.0),    # Classic LEGO Black (#1B2A34)
    "white": (0.95, 0.95, 0.95, 1.0),    # Classic LEGO White (#F4F4F4)
    "orange": (0.92, 0.42, 0.04, 1.0),   # Classic LEGO Bright Orange (#FE8A18)
    "gray": (0.54, 0.57, 0.55, 1.0),     # Light Bluish Gray (#8A928D)
    "darkgray": (0.33, 0.35, 0.33, 1.0), # Dark Bluish Gray (#545955)
}

def parse_color(val: str) -> Tuple[float, float, float, float]:
    """Parse color from name ('red', 'blue', etc.), hex ('#C91A09'), or int code."""
    clean = val.strip().lower()
    if clean in NAMED_LEGO_COLORS:
        return NAMED_LEGO_COLORS[clean]
    if clean.startswith("#"):
        hex_val = clean.lstrip("#")
        if len(hex_val) == 6:
            r = int(hex_val[0:2], 16) / 255.0
            g = int(hex_val[2:4], 16) / 255.0
            b = int(hex_val[4:6], 16) / 255.0
            return (r, g, b, 1.0)
    return NAMED_LEGO_COLORS["red"]


# ==============================================================================
# LDraw Parser & Geometry Loader
# ==============================================================================

class LDrawCatalog:
    """Manages indexing and lookup of LDraw files (parts, primitives, subfiles)."""

    def __init__(self, ldraw_root: Path, main_color: Optional[Tuple[float, float, float, float]] = None):
        self.ldraw_root = ldraw_root
        self.file_index: Dict[str, Path] = {}
        self.colors: Dict[int, Tuple[float, float, float, float]] = {}
        self.default_color = main_color or NAMED_LEGO_COLORS["red"]
        self._index_catalog()
        self._load_colors()

    def _index_catalog(self):
        """Build case-insensitive filename index for fast recursive resolution."""
        search_dirs = [
            self.ldraw_root / "parts",
            self.ldraw_root / "parts" / "s",
            self.ldraw_root / "p",
            self.ldraw_root / "p" / "48",
            self.ldraw_root / "p" / "8",
            self.ldraw_root / "models",
            self.ldraw_root,
        ]
        
        nested_ldraw = self.ldraw_root / "ldraw"
        if nested_ldraw.exists():
            search_dirs.extend([
                nested_ldraw / "parts",
                nested_ldraw / "parts" / "s",
                nested_ldraw / "p",
                nested_ldraw / "p" / "48",
                nested_ldraw / "p" / "8",
                nested_ldraw / "models",
                nested_ldraw,
            ])

        for directory in search_dirs:
            if not directory.exists():
                continue
            for root, _, files in os.walk(directory):
                for file in files:
                    clean_name = file.lower()
                    full_path = Path(root) / file
                    if clean_name not in self.file_index:
                        self.file_index[clean_name] = full_path
                    rel = str(full_path.relative_to(self.ldraw_root)).replace("\\", "/").lower()
                    self.file_index[rel] = full_path

    def _load_colors(self):
        """Parse LDConfig.ldr to load standard LEGO color palette."""
        candidates = [
            self.ldraw_root / "LDConfig.ldr",
            self.ldraw_root / "ldraw" / "LDConfig.ldr",
        ]
        for cfg_path in candidates:
            if not cfg_path.exists():
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("0 !COLOUR"):
                            code_match = re.search(r"CODE\s+(\d+)", line)
                            val_match = re.search(r"VALUE\s+#([0-9a-fA-F]{6})", line)
                            alpha_match = re.search(r"ALPHA\s+(\d+)", line)
                            if code_match and val_match:
                                code = int(code_match.group(1))
                                # Ignore placeholder color 16 in LDConfig (it defaults to pale #FFFF80)
                                if code == 16:
                                    continue
                                hex_val = val_match.group(1)
                                r = int(hex_val[0:2], 16) / 255.0
                                g = int(hex_val[2:4], 16) / 255.0
                                b = int(hex_val[4:6], 16) / 255.0
                                a = 1.0
                                if alpha_match:
                                    a = int(alpha_match.group(1)) / 255.0
                                self.colors[code] = (r, g, b, a)
                break
            except Exception as e:
                print(f"[WARN] Error loading LDConfig.ldr: {e}")

        # Ensure color 16 (Main Part Color) is set to chosen default
        self.colors[16] = self.default_color

    def find_file(self, filename: str) -> Optional[Path]:
        """Resolve an LDraw subfile path."""
        norm_name = filename.replace("\\", "/").strip().lower()
        base_name = Path(norm_name).name

        if norm_name in self.file_index:
            return self.file_index[norm_name]
        if base_name in self.file_index:
            return self.file_index[base_name]

        for prefix in ["parts/", "parts/s/", "p/", "p/48/", "p/8/", "ldraw/parts/", "ldraw/p/"]:
            candidate = f"{prefix}{base_name}"
            if candidate in self.file_index:
                return self.file_index[candidate]

        return None


class LDrawMeshBuilder:
    """Parses LDraw .dat geometry recursively and builds Blender Mesh."""

    def __init__(self, catalog: LDrawCatalog, scale: float = 0.001):
        self.catalog = catalog
        self.scale = scale
        self.vertices: List[Tuple[float, float, float]] = []
        self.faces: List[List[int]] = []
        self.face_colors: List[int] = []

    def _transform_point(self, pt: Tuple[float, float, float], matrix: List[float]) -> Tuple[float, float, float]:
        """Convert LDraw (+X Right, +Y Down, +Z Towards) to Blender (+X Right, +Y Forward, +Z Up)."""
        lx, ly, lz = pt
        tx, ty, tz = matrix[0], matrix[1], matrix[2]
        a, b, c = matrix[3], matrix[4], matrix[5]
        d, e, f = matrix[6], matrix[7], matrix[8]
        g, h, i = matrix[9], matrix[10], matrix[11]

        rx = a * lx + b * ly + c * lz + tx
        ry = d * lx + e * ly + f * lz + ty
        rz = g * lx + h * ly + i * lz + tz

        bx = rx * self.scale
        by = rz * self.scale
        bz = -ry * self.scale
        return (bx, by, bz)

    def _combine_matrices(self, m_parent: List[float], m_local: List[float]) -> List[float]:
        """Combine parent 3x4 transform with local 3x4 transform."""
        tx1, ty1, tz1 = m_parent[0], m_parent[1], m_parent[2]
        a1, b1, c1 = m_parent[3], m_parent[4], m_parent[5]
        d1, e1, f1 = m_parent[6], m_parent[7], m_parent[8]
        g1, h1, i1 = m_parent[9], m_parent[10], m_parent[11]

        tx2, ty2, tz2 = m_local[0], m_local[1], m_local[2]
        a2, b2, c2 = m_local[3], m_local[4], m_local[5]
        d2, e2, f2 = m_local[6], m_local[7], m_local[8]
        g2, h2, i2 = m_local[9], m_local[10], m_local[11]

        rx = a1 * tx2 + b1 * ty2 + c1 * tz2 + tx1
        ry = d1 * tx2 + e1 * ty2 + f1 * tz2 + ty1
        rz = g1 * tx2 + h1 * ty2 + i1 * tz2 + tz1

        ra = a1 * a2 + b1 * d2 + c1 * g2
        rb = a1 * b2 + b1 * e2 + c1 * h2
        rc = a1 * c2 + b1 * f2 + c1 * i2

        rd = d1 * a2 + e1 * d2 + f1 * g2
        re = d1 * b2 + e1 * e2 + f1 * h2
        rf = d1 * c2 + e1 * f2 + f1 * i2

        rg = g1 * a2 + h1 * d2 + i1 * g2
        rh = g1 * b2 + h1 * e2 + i1 * h2
        ri = g1 * c2 + h1 * f2 + i1 * i2

        return [rx, ry, rz, ra, rb, rc, rd, re, rf, rg, rh, ri]

    def parse_file(self, file_path: Path, current_matrix: Optional[List[float]] = None, current_color: int = 16, depth: int = 0):
        """Recursively parse an LDraw file."""
        if depth > 40 or not file_path.exists():
            return

        if current_matrix is None:
            current_matrix = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[WARN] Failed to open {file_path}: {e}")
            return

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            line_type = tokens[0]

            # Type 1: Subfile
            if line_type == "1" and len(tokens) >= 15:
                color_code = int(tokens[1])
                effective_color = current_color if color_code == 16 else color_code
                sub_matrix = [float(t) for t in tokens[2:14]]
                sub_file_name = " ".join(tokens[14:])
                combined_matrix = self._combine_matrices(current_matrix, sub_matrix)
                
                resolved_path = self.catalog.find_file(sub_file_name)
                if resolved_path:
                    self.parse_file(resolved_path, combined_matrix, effective_color, depth + 1)

            # Type 3: Triangle
            elif line_type == "3" and len(tokens) >= 11:
                color_code = int(tokens[1])
                effective_color = current_color if color_code == 16 else color_code
                v1 = (float(tokens[2]), float(tokens[3]), float(tokens[4]))
                v2 = (float(tokens[5]), float(tokens[6]), float(tokens[7]))
                v3 = (float(tokens[8]), float(tokens[9]), float(tokens[10]))

                tv1 = self._transform_point(v1, current_matrix)
                tv2 = self._transform_point(v2, current_matrix)
                tv3 = self._transform_point(v3, current_matrix)

                start_idx = len(self.vertices)
                self.vertices.extend([tv1, tv2, tv3])
                self.faces.append([start_idx, start_idx + 1, start_idx + 2])
                self.face_colors.append(effective_color)

            # Type 4: Quad
            elif line_type == "4" and len(tokens) >= 14:
                color_code = int(tokens[1])
                effective_color = current_color if color_code == 16 else color_code
                v1 = (float(tokens[2]), float(tokens[3]), float(tokens[4]))
                v2 = (float(tokens[5]), float(tokens[6]), float(tokens[7]))
                v3 = (float(tokens[8]), float(tokens[9]), float(tokens[10]))
                v4 = (float(tokens[11]), float(tokens[12]), float(tokens[13]))

                tv1 = self._transform_point(v1, current_matrix)
                tv2 = self._transform_point(v2, current_matrix)
                tv3 = self._transform_point(v3, current_matrix)
                tv4 = self._transform_point(v4, current_matrix)

                start_idx = len(self.vertices)
                self.vertices.extend([tv1, tv2, tv3, tv4])
                self.faces.append([start_idx, start_idx + 1, start_idx + 2, start_idx + 3])
                self.face_colors.append(effective_color)


# ==============================================================================
# Blender Scene Setup & Rendering
# ==============================================================================

def clear_blender_scene():
    """Wipe default cube, lights, cameras, and materials."""
    if not HAS_BPY:
        return
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)


def get_or_create_lego_material(color_code: int, catalog: LDrawCatalog) -> bpy.types.Material:
    """Create realistic plastic Principled BSDF shader for a given LEGO color."""
    mat_name = f"LDraw_Color_{color_code}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    rgba = catalog.colors.get(color_code, catalog.default_color)
    
    # sRGB to Linear conversion for photorealistic color in Cycles
    r_lin = pow(rgba[0], 2.2)
    g_lin = pow(rgba[1], 2.2)
    b_lin = pow(rgba[2], 2.2)
    bsdf.inputs["Base Color"].default_value = (r_lin, g_lin, b_lin, rgba[3])
    
    # Plastic material parameters (glossy LEGO ABS)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.16
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.52
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.55
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.55
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.15
    elif "Clearcoat" in bsdf.inputs:
        bsdf.inputs["Clearcoat"].default_value = 0.15

    return mat


def setup_studio_lighting():
    """Setup high-contrast 3-point studio lighting setup (Key, Fill, Rim)."""
    # Key Light (Strong warm-white main key light)
    key_data = bpy.data.lights.new(name="Key_Light", type='AREA')
    key_data.energy = 450.0
    key_data.size = 1.5
    key_data.color = (1.0, 0.98, 0.95)
    key_obj = bpy.data.objects.new(name="Key_Light", object_data=key_data)
    key_obj.location = (2.2, -2.5, 3.0)
    bpy.context.collection.objects.link(key_obj)

    # Fill Light (Softer cool ambient fill)
    fill_data = bpy.data.lights.new(name="Fill_Light", type='AREA')
    fill_data.energy = 180.0
    fill_data.size = 3.0
    fill_data.color = (0.85, 0.92, 1.0)
    fill_obj = bpy.data.objects.new(name="Fill_Light", object_data=fill_data)
    fill_obj.location = (-2.5, -2.0, 1.8)
    bpy.context.collection.objects.link(fill_obj)

    # Rim / Back Light (High-contrast edge definition)
    rim_data = bpy.data.lights.new(name="Rim_Light", type='AREA')
    rim_data.energy = 320.0
    rim_data.size = 1.2
    rim_data.color = (1.0, 1.0, 1.0)
    rim_obj = bpy.data.objects.new(name="Rim_Light", object_data=rim_data)
    rim_obj.location = (0.0, 2.5, 3.0)
    bpy.context.collection.objects.link(rim_obj)

    # Soft ambient world background
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("StudioWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.08, 0.08, 0.08, 1.0)
        bg_node.inputs["Strength"].default_value = 0.5


def setup_camera_and_framing(target_obj: bpy.types.Object, res: int = 640):
    """Position camera aimed at center of object with auto-calculated distance."""
    cam_data = bpy.data.cameras.new(name="MainCamera")
    cam_data.lens = 55  # 55mm focal length
    cam_obj = bpy.data.objects.new(name="MainCamera", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Calculate object bounding radius
    bbox = [target_obj.matrix_world @ mathutils.Vector(corner) for corner in target_obj.bound_box]
    center = sum(bbox, mathutils.Vector((0, 0, 0))) / 8.0
    max_dist = max((v - center).length for v in bbox)
    radius = max(max_dist, 0.02)

    # Isometric/Hero angle view (Azimuth 45°, Elevation 30°)
    elevation_deg = 30.0
    azimuth_deg = 45.0
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)

    fov = cam_data.angle
    distance = (radius * 1.7) / math.sin(fov / 2.0)
    distance = max(distance, 0.15)

    cam_x = center.x + distance * math.cos(elev) * math.sin(azim)
    cam_y = center.y - distance * math.cos(elev) * math.cos(azim)
    cam_z = center.z + distance * math.sin(elev)

    cam_obj.location = (cam_x, cam_y, cam_z)

    # TrackTo constraint
    track_constraint = cam_obj.constraints.new(type='TRACK_TO')
    track_constraint.target = target_obj
    track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    track_constraint.up_axis = 'UP_Y'

    scene = bpy.context.scene
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100


def configure_render_engine(use_gpu: bool = True, engine_type: str = "CYCLES", samples: int = 64) -> str:
    """Configure render engine with GPU acceleration (Cycles/EEVEE) with CPU fallback."""
    scene = bpy.context.scene

    if engine_type.upper() == "CYCLES":
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = samples
        scene.cycles.preview_samples = min(samples, 32)
        scene.render.film_transparent = True

        gpu_configured = False
        device_summary = "CPU"
        if use_gpu:
            try:
                prefs = bpy.context.preferences
                cycles_addon = prefs.addons.get('cycles')
                if cycles_addon:
                    cprefs = cycles_addon.preferences
                    for compute_type in ['OPTIX', 'CUDA', 'HIP', 'METAL']:
                        try:
                            cprefs.compute_device_type = compute_type
                            cprefs.get_devices()
                            valid_devices = [d for d in cprefs.devices if d.type == compute_type]
                            if valid_devices:
                                for d in valid_devices:
                                    d.use = True
                                scene.cycles.device = 'GPU'
                                device_names = ", ".join(d.name for d in valid_devices)
                                device_summary = f"GPU ({compute_type} -> {device_names})"
                                gpu_configured = True
                                break
                        except Exception:
                            continue
            except Exception as e:
                print(f"[WARN] Error during GPU setup: {e}")

        if not gpu_configured:
            device_summary = "CPU (Fallback)"
            scene.cycles.device = 'CPU'
            if hasattr(scene.render, 'threads_mode'):
                scene.render.threads_mode = 'AUTO'

        return device_summary
    else:
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types, 'BLENDER_EEVEE_NEXT') else 'BLENDER_EEVEE'
        return "EEVEE"


def render_part(part_path: Path, ldraw_root: Path, output_file: Path, res: int = 640, samples: int = 64, use_gpu: bool = True, color_name: str = "red"):
    """Full pipeline: Load LDraw model, create mesh, light, frame, and render."""
    print(f"\n=======================================================")
    print(f" SnapBrick Headless Render: {part_path.name}")
    print(f"=======================================================")
    print(f" Part File: {part_path}")
    print(f" Output File: {output_file}")
    print(f" Resolution: {res}x{res}")
    print(f" Part Color: {color_name}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Clear scene
    clear_blender_scene()

    # 2. Parse LDraw part with chosen color
    main_rgba = parse_color(color_name)
    catalog = LDrawCatalog(ldraw_root, main_color=main_rgba)
    builder = LDrawMeshBuilder(catalog, scale=0.001)
    builder.parse_file(part_path)

    if not builder.vertices or not builder.faces:
        raise ValueError(f"Error: No geometry loaded from {part_path}!")

    print(f"[INFO] Parsed geometry: {len(builder.vertices)} vertices, {len(builder.faces)} faces")

    # 3. Create Blender Mesh
    mesh_name = f"Mesh_{part_path.stem}"
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(builder.vertices, [], builder.faces)
    mesh.update()

    obj = bpy.data.objects.new(f"Part_{part_path.stem}", mesh)
    bpy.context.collection.objects.link(obj)

    # Assign materials per face
    unique_colors = set(builder.face_colors)
    mat_slots = {}
    for color_code in unique_colors:
        mat = get_or_create_lego_material(color_code, catalog)
        obj.data.materials.append(mat)
        mat_slots[color_code] = len(obj.data.materials) - 1

    for idx, poly in enumerate(mesh.polygons):
        if idx < len(builder.face_colors):
            c_code = builder.face_colors[idx]
            poly.material_index = mat_slots.get(c_code, 0)

    # 4. Center object at origin (0, 0, 0)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)

    # Enable smooth shading
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        for poly in mesh.polygons:
            if hasattr(poly, 'use_smooth'):
                poly.use_smooth = True

    # 5. Setup lighting & camera
    setup_studio_lighting()
    setup_camera_and_framing(obj, res=res)

    # 6. Configure rendering
    hw_device = configure_render_engine(use_gpu=use_gpu, engine_type="CYCLES", samples=samples)
    print(f"[HARDWARE ACCELERATION] ⚡ {hw_device}")

    # 7. Render image
    scene = bpy.context.scene
    scene.render.filepath = str(output_file)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    print(f"[INFO] Rendering frame with {samples} samples...")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    t1 = time.time()
    print(f"[SUCCESS] Render finished in {t1 - t0:.2f}s | Device: {hw_device} | Output: {output_file} ({output_file.stat().st_size} bytes)")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SnapBrick LDraw Headless Blender Renderer")
    parser.add_argument("--part_id", type=str, default="3001", help="LDraw part ID (e.g., 3001) or direct .dat path")
    parser.add_argument("--ldraw_root", type=str, default="", help="Path to ldraw library directory")
    parser.add_argument("--output", type=str, default="", help="Path to output .png image")
    parser.add_argument("--color", type=str, default="red", help="LEGO Color (red, blue, yellow, green, black, white, orange, or #HEX)")
    parser.add_argument("--resolution", type=int, default=640, help="Image resolution (square width/height)")
    parser.add_argument("--samples", type=int, default=64, help="Cycles render samples")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU compute acceleration")

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    # Resolve paths relative to project
    script_dir = Path(__file__).resolve().parent
    pipeline_dir = script_dir.parent
    project_root = pipeline_dir.parent

    # LDraw root discovery
    if args.ldraw_root:
        ldraw_root = Path(args.ldraw_root).resolve()
    else:
        candidates = [
            pipeline_dir / "ldraw_lib",
            pipeline_dir / "ldraw_lib" / "ldraw",
            project_root / "data-pipeline" / "ldraw_lib",
        ]
        ldraw_root = None
        for cand in candidates:
            if cand.exists():
                ldraw_root = cand
                break
        if not ldraw_root:
            ldraw_root = pipeline_dir / "ldraw_lib"

    # Part file resolution
    part_input = args.part_id.strip()
    if os.path.isabs(part_input) and Path(part_input).exists():
        part_path = Path(part_input)
    elif (pipeline_dir / part_input).exists():
        part_path = pipeline_dir / part_input
    else:
        catalog = LDrawCatalog(ldraw_root)
        part_filename = part_input if part_input.endswith(".dat") else f"{part_input}.dat"
        resolved = catalog.find_file(part_filename)
        if resolved and resolved.exists():
            part_path = resolved
        else:
            part_path = ldraw_root / "ldraw" / "parts" / part_filename
            if not part_path.exists():
                part_path = ldraw_root / "parts" / part_filename

    if not part_path.exists():
        print(f"[ERROR] Could not find LDraw part file for '{args.part_id}' at {part_path}", file=sys.stderr)
        sys.exit(1)

    # Output file resolution
    if args.output:
        output_file = Path(args.output).resolve()
    else:
        output_file = pipeline_dir / "output" / f"test_{part_path.stem}.png"

    if not HAS_BPY:
        print("[ERROR] Blender Python API (bpy) is not available in the current Python environment.", file=sys.stderr)
        print("Please execute via Blender standalone: blender -b -P data-pipeline/src/render_single_part.py -- [OPTIONS]", file=sys.stderr)
        sys.exit(1)

    render_part(
        part_path=part_path,
        ldraw_root=ldraw_root,
        output_file=output_file,
        res=args.resolution,
        samples=args.samples,
        use_gpu=not args.no_gpu,
        color_name=args.color,
    )


if __name__ == "__main__":
    main()
