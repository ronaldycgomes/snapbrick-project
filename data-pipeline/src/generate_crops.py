#!/usr/bin/env python3
"""
SnapBrick Project - Synthetic Crop Dataset Generator (Stage 2: 224x224 Classifier)
Renders high-resolution 224x224 centered crops of catalog LEGO parts in Blender Cycles.
Applies resting poses, 3-point lighting with stud-highlighting rim lights,
domain-randomized table textures, and exports in standard PyTorch ImageFolder format.
"""

import sys
import os
import re
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import bpy
    import mathutils
    HAS_BPY = True
except ImportError:
    HAS_BPY = False

# ==============================================================================
# LEGO Colors and Textures Palette
# ==============================================================================

NAMED_LEGO_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "red": (0.79, 0.05, 0.05, 1.0),
    "blue": (0.0, 0.33, 0.75, 1.0),
    "yellow": (0.95, 0.80, 0.05, 1.0),
    "green": (0.02, 0.52, 0.15, 1.0),
    "black": (0.11, 0.16, 0.20, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
    "orange": (0.92, 0.42, 0.04, 1.0),
    "light_gray": (0.54, 0.57, 0.55, 1.0),
    "dark_gray": (0.33, 0.35, 0.33, 1.0),
    "lime": (0.65, 0.79, 0.10, 1.0),
    "tan": (0.87, 0.78, 0.61, 1.0),
    "brown": (0.35, 0.22, 0.15, 1.0),
    "dark_blue": (0.04, 0.20, 0.39, 1.0),
}

MVP_PART_CATALOG: List[Dict[str, str]] = [
    {"part_id": "3001", "name": "Brick 2x4"},
    {"part_id": "3002", "name": "Brick 2x3"},
    {"part_id": "3003", "name": "Brick 2x2"},
    {"part_id": "3004", "name": "Brick 1x2"},
    {"part_id": "3005", "name": "Brick 1x1"},
    {"part_id": "3010", "name": "Brick 1x4"},
    {"part_id": "3009", "name": "Brick 1x6"},
    {"part_id": "3008", "name": "Brick 1x8"},
    {"part_id": "3020", "name": "Plate 2x4"},
    {"part_id": "3021", "name": "Plate 2x3"},
    {"part_id": "3022", "name": "Plate 2x2"},
    {"part_id": "3023", "name": "Plate 1x2"},
    {"part_id": "3024", "name": "Plate 1x1"},
    {"part_id": "3710", "name": "Plate 1x4"},
    {"part_id": "3666", "name": "Plate 1x6"},
    {"part_id": "3460", "name": "Plate 1x8"},
    {"part_id": "3795", "name": "Plate 2x6"},
    {"part_id": "3034", "name": "Plate 2x8"},
    {"part_id": "3176", "name": "Plate 3x2 with Hole"},
    {"part_id": "3794b", "name": "Plate 1x2 with 1 Stud (Jumper)"},
    {"part_id": "4073", "name": "Plate 1x1 Round (Round Stud)"},
    {"part_id": "30503", "name": "Wedge Plate 4x4 Cut Corner (Wing)"},
    {"part_id": "99781", "name": "Bracket 1x2 - 1x1 Down"},
    {"part_id": "3040", "name": "Slope 45 2x1"},
    {"part_id": "3039", "name": "Slope 45 2x2"},
    {"part_id": "54200", "name": "Slope 30 1x1x2/3 (Cheese Slope)"},
    {"part_id": "11477", "name": "Slope Curved 2x1 No Studs"},
    {"part_id": "24201", "name": "Slope Curved 2x1 Inverted / Normal"},
    {"part_id": "32807", "name": "Brick Curved 2x1 with Curved Top"},
    {"part_id": "3062b", "name": "Brick 1x1 Round (Cylinder)"},
    {"part_id": "87087", "name": "Brick 1x1 with 1 Stud on Side (SNOT)"},
    {"part_id": "3069b", "name": "Tile 1x2 (Flat)"},
    {"part_id": "3068b", "name": "Tile 2x2 (Flat)"},
    {"part_id": "3070b", "name": "Tile 1x1 (Flat Square)"},
    {"part_id": "30039", "name": "Tile 1x2 Grille (Radiator)"},
    {"part_id": "98138", "name": "Tile 1x1 Round (Flat Round)"},
    {"part_id": "4081b", "name": "Plate 1x1 with Clip Light (Thick Ring)"},
    {"part_id": "2780", "name": "Technic Pin with Friction"},
    {"part_id": "3673", "name": "Technic Pin 1/2"},
    {"part_id": "32002", "name": "Technic Pin 3/4"},
    {"part_id": "6558", "name": "Technic Pin 3L with Friction"},
    {"part_id": "32054", "name": "Technic Pin Long with Stop Bush"},
    {"part_id": "43093", "name": "Technic Axle Pin with Friction"},
    {"part_id": "32062", "name": "Technic Axle 2L"},
    {"part_id": "3705", "name": "Technic Axle 4L"},
    {"part_id": "32123b", "name": "Technic Bush 1/2"},
    {"part_id": "3713", "name": "Technic Bush Full"},
    {"part_id": "18654", "name": "Technic Beam 1 (Pin Connector Round)"},
    {"part_id": "3700", "name": "Technic Brick 1x2 with Hole"},
    {"part_id": "32530", "name": "Technic Tile 1x2 with Two Holes"},
    {"part_id": "14704", "name": "Plate 1x2 with Small Ball Socket"},
]

# ==============================================================================
# LDraw Parser
# ==============================================================================

class LDrawCatalog:
    def __init__(self, ldraw_root: Path):
        self.ldraw_root = ldraw_root
        self.file_index: Dict[str, Path] = {}
        self._index_files()

    def _index_files(self):
        for root, _, files in os.walk(self.ldraw_root):
            for file in files:
                clean_name = file.lower()
                full_path = Path(root) / file
                if clean_name not in self.file_index:
                    self.file_index[clean_name] = full_path
                rel = str(full_path.relative_to(self.ldraw_root)).replace("\\", "/").lower()
                self.file_index[rel] = full_path

    def find_file(self, filename: str) -> Optional[Path]:
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
    def __init__(self, catalog: LDrawCatalog, scale: float = 0.001):
        self.catalog = catalog
        self.scale = scale
        self.vertices: List[Tuple[float, float, float]] = []
        self.faces: List[List[int]] = []

    def _transform_point(self, pt: Tuple[float, float, float], matrix: List[float]) -> Tuple[float, float, float]:
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

    def parse_file(self, file_path: Path, current_matrix: Optional[List[float]] = None, depth: int = 0):
        if depth > 40 or not file_path.exists():
            return
        if current_matrix is None:
            current_matrix = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    line_type = parts[0]
                    if line_type == "1" and len(parts) >= 15:
                        sub_file = " ".join(parts[14:])
                        sub_path = self.catalog.find_file(sub_file)
                        if sub_path:
                            M_rel = [float(p) for p in parts[2:14]]
                            combined = self._combine_matrices(current_matrix, M_rel)
                            self.parse_file(sub_path, combined, depth + 1)
                    elif line_type == "3" and len(parts) >= 11:
                        p1 = self._transform_point((float(parts[2]), float(parts[3]), float(parts[4])), current_matrix)
                        p2 = self._transform_point((float(parts[5]), float(parts[6]), float(parts[7])), current_matrix)
                        p3 = self._transform_point((float(parts[8]), float(parts[9]), float(parts[10])), current_matrix)
                        idx = len(self.vertices)
                        self.vertices.extend([p1, p2, p3])
                        self.faces.append([idx, idx + 1, idx + 2])
                    elif line_type == "4" and len(parts) >= 14:
                        p1 = self._transform_point((float(parts[2]), float(parts[3]), float(parts[4])), current_matrix)
                        p2 = self._transform_point((float(parts[5]), float(parts[6]), float(parts[7])), current_matrix)
                        p3 = self._transform_point((float(parts[8]), float(parts[9]), float(parts[10])), current_matrix)
                        p4 = self._transform_point((float(parts[11]), float(parts[12]), float(parts[13])), current_matrix)
                        idx = len(self.vertices)
                        self.vertices.extend([p1, p2, p3, p4])
                        self.faces.append([idx, idx + 1, idx + 2, idx + 3])
        except Exception:
            pass

# ==============================================================================
# PBR Shaders & Table Surfaces
# ==============================================================================

def clear_blender_scene():
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


def get_or_create_pbr_material(rgba: Tuple[float, float, float, float], roughness: float = 0.18):
    mat_name = f"Mat_Lego_{random.randint(1000, 9999)}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    # Linear conversion for color fidelity
    r_lin = pow(rgba[0], 2.2)
    g_lin = pow(rgba[1], 2.2)
    b_lin = pow(rgba[2], 2.2)
    bsdf.inputs['Base Color'].default_value = (r_lin, g_lin, b_lin, rgba[3])
    bsdf.inputs['Roughness'].default_value = roughness
    if 'IOR' in bsdf.inputs:
        bsdf.inputs['IOR'].default_value = 1.54
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.55
    mat.node_tree.links.new(bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat


def setup_procedural_table():
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    table = bpy.context.active_object
    table.name = "TableSurface"
    mat = bpy.data.materials.new(name="TableMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    table_type = random.choice(["wood", "gray_desk", "cutting_mat", "warm_desk"])
    if table_type == "wood":
        bsdf.inputs['Base Color'].default_value = (random.uniform(0.30, 0.45), random.uniform(0.18, 0.28), random.uniform(0.10, 0.18), 1.0)
        bsdf.inputs['Roughness'].default_value = random.uniform(0.45, 0.70)
    elif table_type == "gray_desk":
        val = random.uniform(0.38, 0.52)
        bsdf.inputs['Base Color'].default_value = (val, val, val, 1.0)
        bsdf.inputs['Roughness'].default_value = random.uniform(0.50, 0.75)
    elif table_type == "cutting_mat":
        bsdf.inputs['Base Color'].default_value = (random.uniform(0.15, 0.25), random.uniform(0.28, 0.38), random.uniform(0.20, 0.30), 1.0)
        bsdf.inputs['Roughness'].default_value = random.uniform(0.40, 0.65)
    else:  # warm beige / light surface
        bsdf.inputs['Base Color'].default_value = (random.uniform(0.55, 0.65), random.uniform(0.50, 0.60), random.uniform(0.42, 0.52), 1.0)
        bsdf.inputs['Roughness'].default_value = random.uniform(0.40, 0.60)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    table.data.materials.append(mat)
    return table


def kelvin_to_rgb(temp_k: float) -> Tuple[float, float, float]:
    """Calculate sRGB components from blackbody temperature in Kelvin (2000K - 10000K)."""
    temp = temp_k / 100.0
    if temp <= 66:
        r = 255.0
    else:
        r = 329.698727446 * (max(1.0, temp - 60) ** -0.1332047592)
        r = max(0.0, min(255.0, r))

    if temp <= 66:
        g = 99.4708025861 * math.log(max(1.0, temp)) - 161.1195681661
        g = max(0.0, min(255.0, g))
    else:
        g = 288.1221695283 * (max(1.0, temp - 60) ** -0.0755148492)
        g = max(0.0, min(255.0, g))

    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(max(1.0, temp - 10)) - 305.0447927307
        b = max(0.0, min(255.0, b))

    return (r / 255.0, g / 255.0, b / 255.0)


# ==============================================================================
# Crop Generator Engine
# ==============================================================================

class CropDatasetGenerator:
    def __init__(self, ldraw_root: Path, output_dir: Path, use_gpu: bool = True, samples: int = 24):
        self.ldraw_root = ldraw_root
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.samples = samples
        self.catalog = LDrawCatalog(ldraw_root)
        self.mesh_cache: Dict[str, LDrawMeshBuilder] = {}
        self._setup_cycles_engine()

    def _setup_cycles_engine(self):
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = self.samples
        scene.cycles.preview_samples = min(self.samples, 8)
        scene.render.film_transparent = False

        if self.use_gpu:
            try:
                prefs = bpy.context.preferences
                cycles_addon = prefs.addons.get('cycles')
                if cycles_addon:
                    cprefs = cycles_addon.preferences
                    for compute_type in ['OPTIX', 'CUDA']:
                        try:
                            cprefs.compute_device_type = compute_type
                            cprefs.get_devices()
                            valid = [d for d in cprefs.devices if d.type == compute_type]
                            if valid:
                                for d in valid:
                                    d.use = True
                                scene.cycles.device = 'GPU'
                                print(f"[INFO] Cycles hardware acceleration enabled: {compute_type}")
                                return
                        except Exception:
                            continue
            except Exception as e:
                print(f"[WARN] Cycles GPU configuration fallback to CPU: {e}")
        scene.cycles.device = 'CPU'

    def preload_part(self, part_id: str) -> bool:
        if part_id in self.mesh_cache:
            return True
        part_filename = part_id if part_id.endswith(".dat") else f"{part_id}.dat"
        part_file = self.catalog.find_file(part_filename)
        if not part_file:
            print(f"[WARN] Part geometry not found: {part_id}")
            return False
        builder = LDrawMeshBuilder(self.catalog, scale=0.001)
        builder.parse_file(part_file)
        if builder.vertices and builder.faces:
            self.mesh_cache[part_id] = builder
            return True
        return False

    def render_crop(self, part_id: str, out_path: Path, res: int = 224):
        clear_blender_scene()
        scene = bpy.context.scene
        builder = self.mesh_cache[part_id]

        # 0. World Environment Ambient Lighting & Tone Mapping
        world = scene.world
        if not world:
            world = bpy.data.worlds.new("StudioWorld")
            scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
            bg_node.inputs["Strength"].default_value = 0.3

        if hasattr(scene, "view_settings"):
            try:
                scene.view_settings.view_transform = 'Filmic'
                scene.view_settings.look = 'Medium High Contrast'
            except Exception:
                pass

        # 1. Create table
        setup_procedural_table()

        # 2. Build piece mesh
        mesh = bpy.data.meshes.new(f"Mesh_{part_id}")
        mesh.from_pydata(builder.vertices, [], builder.faces)
        mesh.update()
        obj = bpy.data.objects.new(f"Part_{part_id}", mesh)
        bpy.context.collection.objects.link(obj)

        # Color & PBR
        _, rgba = random.choice(list(NAMED_LEGO_COLORS.items()))
        mat = get_or_create_pbr_material(rgba, roughness=random.uniform(0.12, 0.28))
        obj.data.materials.append(mat)

        # Center geometry
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        # 3. Physics resting poses (True physical equilibrium on surfaces)
        is_plate_or_tile = part_id in [
            "3020", "3021", "3022", "3023", "3024", "3710", "3666", "3460",
            "3795", "3034", "3176", "3794b", "4073", "30503", "3069b", "3068b",
            "3070b", "30039", "98138", "14704"
        ]
        is_slope = part_id in ["3040", "3039", "54200", "11477", "24201", "32807"]
        is_bush = part_id in ["32123b", "3713", "18654", "3062b"]
        is_pin_or_axle = part_id in ["6558", "43093", "3705", "32062", "2780", "3673", "32002", "32054"]

        if is_plate_or_tile:
            # 85% studs-up, 15% upside down
            rx = 0.0 if random.random() < 0.85 else math.pi
            ry = 0.0
            rx += math.radians(random.uniform(-2.0, 2.0))
            ry += math.radians(random.uniform(-2.0, 2.0))
            rz = random.uniform(0.0, 2.0 * math.pi)
        elif is_slope:
            # Slopes resting on bottom face or back
            if random.random() < 0.80:
                rx, ry = 0.0, 0.0
            else:
                rx = random.choice([math.pi / 2.0, -math.pi / 2.0, math.pi])
                ry = 0.0
            rx += math.radians(random.uniform(-2.0, 2.0))
            ry += math.radians(random.uniform(-2.0, 2.0))
            rz = random.uniform(0.0, 2.0 * math.pi)
        elif is_bush:
            # Cylindrical Technic Bush / Pin Connector:
            # 65% upright resting on circular base (looking down at axle/stud hole from above),
            # 35% resting on cylinder side rolling on table
            if random.random() < 0.65:
                rx = 0.0 if random.random() < 0.80 else math.pi
                ry = 0.0
            else:
                rx = math.pi / 2.0
                ry = 0.0
            rx += math.radians(random.uniform(-2.0, 2.0))
            ry += math.radians(random.uniform(-2.0, 2.0))
            rz = random.uniform(0.0, 2.0 * math.pi)
        elif is_pin_or_axle:
            # Technic pins and axles ALWAYS lie flat on surface!
            rx = math.pi / 2.0
            ry = 0.0
            rx += math.radians(random.uniform(-2.0, 2.0))
            ry += math.radians(random.uniform(-2.0, 2.0))
            rz = random.uniform(0.0, 2.0 * math.pi)
        else:
            # Bricks and other parts
            if random.random() < 0.75:
                rx, ry = 0.0, 0.0
            elif random.random() < 0.90:
                rx, ry = math.pi, 0.0
            else:
                rx, ry = math.pi / 2.0, 0.0
            rx += math.radians(random.uniform(-2.0, 2.0))
            ry += math.radians(random.uniform(-2.0, 2.0))
            rz = random.uniform(0.0, 2.0 * math.pi)

        obj.rotation_euler = (rx, ry, rz)
        bpy.context.view_layer.update()

        # Rest on table
        bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        min_z = min(c.z for c in bbox)
        obj.location = (0.0, 0.0, -min_z + 0.0003)
        bpy.context.view_layer.update()

        # 4. Framing Camera (Auto-framing in 224x224)
        bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        center = sum(bbox, mathutils.Vector((0, 0, 0))) / len(bbox)
        radius = max((c - center).length for c in bbox)
        radius = max(radius, 0.005)

        cam_data = bpy.data.cameras.new("CropCam")
        cam_data.lens = random.uniform(45.0, 55.0)
        cam_data.sensor_width = 36.0
        # CRITICAL: 1mm near clipping plane so macro crops are never clipped!
        cam_data.clip_start = 0.001
        cam_data.clip_end = 10.0

        camera = bpy.data.objects.new("CropCam", cam_data)
        bpy.context.collection.objects.link(camera)
        scene.camera = camera

        # Smartphone perspective (elevation 30 to 82 degrees, capturing steep top-down and angled profiles)
        elevation = math.radians(random.uniform(30.0, 82.0))
        azimuth = random.uniform(0.0, 2.0 * math.pi)

        # Distance formula to fill ~75-85% of 224x224
        fov = cam_data.angle
        fit_mult = random.uniform(1.30, 1.60)
        distance = (radius * fit_mult) / math.sin(fov / 2.0)
        distance = max(distance, 0.02)

        cam_x = center.x + distance * math.cos(elevation) * math.sin(azimuth)
        cam_y = center.y - distance * math.cos(elevation) * math.cos(azimuth)
        cam_z = center.z + distance * math.sin(elevation)
        camera.location = (cam_x, cam_y, cam_z)

        # Track center target
        cam_target = bpy.data.objects.new("CamTarget", None)
        bpy.context.collection.objects.link(cam_target)
        cam_target.location = center

        track = camera.constraints.new(type='TRACK_TO')
        track.target = cam_target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'

        # 5. Studio 3-Point Lighting with Stud-Highlighting Rim Light & Blackbody Color Temperature
        # Key Light (Randomized Kelvin 2500K - 7000K: warm incandescent to cool daylight)
        key_data = bpy.data.lights.new("KeyLight", type='AREA')
        key_data.energy = random.uniform(220.0, 320.0)
        key_data.size = 1.2
        key_temp = random.uniform(2600.0, 6800.0)
        key_data.color = kelvin_to_rgb(key_temp)
        key_light = bpy.data.objects.new("KeyLight", key_data)
        bpy.context.collection.objects.link(key_light)
        key_light.location = (1.8, -2.0, 2.5)
        k_track = key_light.constraints.new(type='TRACK_TO')
        k_track.target = cam_target
        k_track.track_axis = 'TRACK_NEGATIVE_Z'
        k_track.up_axis = 'UP_Y'

        # Fill Light (Complementary diffuse fill)
        fill_data = bpy.data.lights.new("FillLight", type='AREA')
        fill_data.energy = random.uniform(90.0, 140.0)
        fill_data.size = 2.0
        fill_temp = random.uniform(4000.0, 7500.0)
        fill_data.color = kelvin_to_rgb(fill_temp)
        fill_light = bpy.data.objects.new("FillLight", fill_data)
        bpy.context.collection.objects.link(fill_light)
        fill_light.location = (-2.0, -1.5, 1.8)
        f_track = fill_light.constraints.new(type='TRACK_TO')
        f_track.target = cam_target
        f_track.track_axis = 'TRACK_NEGATIVE_Z'
        f_track.up_axis = 'UP_Y'

        # Rim / Stud Light (Highlights stud circles and chamfered edges)
        rim_data = bpy.data.lights.new("RimLight", type='AREA')
        rim_data.energy = random.uniform(150.0, 240.0)
        rim_data.size = 0.8
        rim_data.color = (1.0, 1.0, 1.0)
        rim_light = bpy.data.objects.new("RimLight", rim_data)
        bpy.context.collection.objects.link(rim_light)
        rim_light.location = (0.0, 2.2, 2.2)
        r_track = rim_light.constraints.new(type='TRACK_TO')
        r_track.target = cam_target
        r_track.track_axis = 'TRACK_NEGATIVE_Z'
        r_track.up_axis = 'UP_Y'

        # 6. Render
        scene.render.resolution_x = res
        scene.render.resolution_y = res
        scene.render.resolution_percentage = 100
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_path)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        bpy.ops.render.render(write_still=True)

# ==============================================================================
# Preview Collage Generator
# ==============================================================================

def generate_preview_grid(image_paths: List[Path], out_grid_path: Path, labels: List[str], cell_size: int = 224, cols: int = 5):
    try:
        from PIL import Image, ImageDraw
        n = len(image_paths)
        if n == 0:
            return
        actual_cols = min(n, cols)
        rows = (n + actual_cols - 1) // actual_cols

        cell_w = cell_size + 16
        cell_h = cell_size + 40
        grid_img = Image.new("RGB", (actual_cols * cell_w, rows * cell_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(grid_img)

        for i, (img_p, lbl) in enumerate(zip(image_paths, labels)):
            if not img_p.exists():
                continue
            r = i // actual_cols
            c = i % actual_cols
            x = c * cell_w + 8
            y = r * cell_h + 8

            with Image.open(img_p) as im:
                im_resized = im.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                grid_img.paste(im_resized, (x, y))

            draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(100, 100, 100))
            draw.text((x + 4, y + cell_size + 10), lbl[:22], fill=(230, 230, 230))

        out_grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(str(out_grid_path), quality=95)
        print(f"\n🖼️ Preview Collage Grid salvo em: {out_grid_path}")
    except Exception as e:
        print(f"[WARN] Preview collage could not be generated: {e}")

# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SnapBrick Synthetic Crop Generator (224x224)")
    parser.add_argument("--ldraw_root", type=str, default="", help="Path to ldraw library directory")
    parser.add_argument("--output_dir", type=str, default="", help="Destination dataset directory")
    parser.add_argument("--parts", nargs="+", default=["all"], help="Specific Part IDs (e.g. 3070b 54200 3024) or 'all'")
    parser.add_argument("--samples_per_part", type=int, default=50, help="Number of crop variations per part")
    parser.add_argument("--res", type=int, default=224, help="Crop resolution (default 224)")
    parser.add_argument("--samples", type=int, default=20, help="Cycles render samples (default 20)")
    parser.add_argument("--val_ratio", type=float, default=0.20, help="Fraction of crops assigned to validation split")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--preview", action="store_true", help="Generate all_crops_preview.jpg collage")

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

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

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = project_root / "ml-core" / "dataset_crops"

    if not HAS_BPY:
        print("[ERROR] Blender Python API (bpy) is not available in the current environment.", file=sys.stderr)
        print("Please run via Blender: blender -b -P data-pipeline/src/generate_crops.py -- [OPTIONS]", file=sys.stderr)
        sys.exit(1)

    # Determine part IDs to render
    if "all" in args.parts:
        target_catalog = MVP_PART_CATALOG
    else:
        target_catalog = []
        for p in args.parts:
            found = next((c for c in MVP_PART_CATALOG if c["part_id"] == p), None)
            if found:
                target_catalog.append(found)
            else:
                target_catalog.append({"part_id": p, "name": f"Part {p}"})

    print("\n=======================================================")
    print(" 🧱 SnapBrick: High-Resolution Crop Dataset Generator")
    print("=======================================================")
    print(f" • Output Root:        {output_dir}")
    print(f" • Parts to Render:    {len(target_catalog)} classes")
    print(f" • Crops per Part:     {args.samples_per_part}")
    print(f" • Resolution:         {args.res}x{args.res}")
    print(f" • Cycles Samples:     {args.samples}")
    print(f" • GPU Acceleration:   {'Disabled' if args.no_gpu else 'Enabled (CUDA/OptiX)'}")
    print("=======================================================\n")

    generator = CropDatasetGenerator(
        ldraw_root=ldraw_root,
        output_dir=output_dir,
        use_gpu=not args.no_gpu,
        samples=args.samples
    )

    t0 = time.time()
    total_rendered = 0
    preview_imgs = []
    preview_labels = []

    for item in target_catalog:
        p_id = item["part_id"]
        p_name = item["name"]
        class_folder_name = f"{p_id}_{p_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"

        if not generator.preload_part(p_id):
            continue

        num_val = max(1, int(args.samples_per_part * args.val_ratio)) if args.samples_per_part > 3 else 0
        num_train = args.samples_per_part - num_val

        splits = [("train", num_train), ("val", num_val)]
        part_rendered = 0

        for split_name, count in splits:
            if count <= 0:
                continue
            split_dir = output_dir / split_name / class_folder_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for i in range(1, count + 1):
                crop_path = split_dir / f"{p_id}_crop_{part_rendered + 1:04d}.png"
                t_render = time.time()
                generator.render_crop(p_id, crop_path, res=args.res)
                dt = time.time() - t_render
                part_rendered += 1
                total_rendered += 1

                if len(preview_imgs) < 25 and (part_rendered <= 3):
                    preview_imgs.append(crop_path)
                    preview_labels.append(f"{p_id} {p_name}")

                print(f" • [{p_id}] {split_name}/{crop_path.name} rendered in {dt:.2f}s")

    total_time = time.time() - t0
    print("\n=======================================================")
    print(f" ✅ Crop Dataset Generated: {total_rendered} crops in {total_time:.1f}s!")
    print(f" Destination: {output_dir}")
    print("=======================================================\n")

    # Generate preview collage if requested or during small batches
    if args.preview or total_rendered <= 30:
        preview_path = output_dir / "crops_preview_test.jpg"
        generate_preview_grid(preview_imgs, preview_path, preview_labels, cell_size=args.res)

if __name__ == "__main__":
    main()
