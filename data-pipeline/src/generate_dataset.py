#!/usr/bin/env python3
"""
SnapBrick Project - Synthetic Dataset Generator with Domain Randomization
Generates multi-object LEGO scenes with procedural backgrounds, varied lighting,
camera angles, and automated YOLOv11 annotations (images + labels + dataset.yaml).
"""

import sys
import os
import re
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Try importing bpy (Blender Python API) and bpy_extras
try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


# ==============================================================================
# LEGO Palette & Part Catalog (MVP Classes)
# ==============================================================================

NAMED_LEGO_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "red": (0.79, 0.05, 0.05, 1.0),          # Bright Red / Ferrari Red (#C91A09)
    "blue": (0.0, 0.33, 0.75, 1.0),          # Bright Blue (#0055BF)
    "yellow": (0.95, 0.80, 0.05, 1.0),      # Bright Yellow (#F2CD37)
    "green": (0.02, 0.52, 0.15, 1.0),       # Dark Green (#00852B)
    "black": (0.11, 0.16, 0.20, 1.0),       # Black / Dodge & Mercedes (#1B2A34)
    "white": (0.95, 0.95, 0.95, 1.0),       # White / Hedwig (#F4F4F4)
    "orange": (0.92, 0.42, 0.04, 1.0),      # Bright Orange / Up House (#FE8A18)
    "gray": (0.54, 0.57, 0.55, 1.0),        # Light Bluish Gray / London Skyline (#8A928D)
    "darkgray": (0.33, 0.35, 0.33, 1.0),    # Dark Bluish Gray (#545955)
    "gold": (0.83, 0.65, 0.17, 1.0),        # Warm Gold / Pearl Gold (Infinity Gauntlet #CC9C2B)
    "silver": (0.75, 0.75, 0.75, 1.0),      # Metallic Silver (#C0C0C0)
    "darkred": (0.45, 0.05, 0.05, 1.0),     # Dark Red (#720E0F)
    "darkblue": (0.04, 0.20, 0.39, 1.0),    # Dark Blue (#0A3463)
    "lime": (0.65, 0.82, 0.12, 1.0),        # Lime (#A5CA18)
    "tan": (0.87, 0.78, 0.55, 1.0),         # Tan / Brick Yellow (#DEC69C)
    "brown": (0.35, 0.18, 0.08, 1.0),       # Reddish Brown (#583927)
}

# 32 Unified Classes for SnapBrick (Classic System, User Spare Parts Photo IMG_0032, and Technic)
MVP_PART_CATALOG: List[Dict[str, str]] = [
    # --- Peças Clássicas System (Bricks) ---
    {"part_id": "3001", "name": "Brick 2x4"},
    {"part_id": "3002", "name": "Brick 2x3"},
    {"part_id": "3003", "name": "Brick 2x2"},
    {"part_id": "3004", "name": "Brick 1x2"},
    {"part_id": "3005", "name": "Brick 1x1"},
    {"part_id": "3010", "name": "Brick 1x4"},
    {"part_id": "3009", "name": "Brick 1x6"},
    {"part_id": "3008", "name": "Brick 1x8"},

    # --- Placas System (Plates) ---
    {"part_id": "3020", "name": "Plate 2x4"},
    {"part_id": "3021", "name": "Plate 2x3"},
    {"part_id": "3022", "name": "Plate 2x2"},
    {"part_id": "3023", "name": "Plate 1x2"},
    {"part_id": "3024", "name": "Plate 1x1"},
    {"part_id": "3710", "name": "Plate 1x4"},
    {"part_id": "3666", "name": "Plate 1x6"},
    {"part_id": "3460", "name": "Plate 1x8"},
    {"part_id": "3794b", "name": "Plate 1x2 with 1 Stud (Jumper)"},
    {"part_id": "4073", "name": "Plate 1x1 Round (Round Stud)"},
    {"part_id": "30503", "name": "Wedge Plate 4x4 Cut Corner (Wing)"},
    {"part_id": "99781", "name": "Bracket 1x2 - 1x1 Down"},

    # --- Rampas e Placas Lisas (Slopes & Tiles) ---
    {"part_id": "3040", "name": "Slope 45 2x1"},
    {"part_id": "3039", "name": "Slope 45 2x2"},
    {"part_id": "54200", "name": "Slope 30 1x1x2/3 (Cheese Slope)"},
    {"part_id": "3069b", "name": "Tile 1x2 (Flat)"},
    {"part_id": "3068b", "name": "Tile 2x2 (Flat)"},
    {"part_id": "3070b", "name": "Tile 1x1 (Flat Square)"},
    {"part_id": "98138", "name": "Tile 1x1 Round (Flat Round)"},

    # --- Peças Technic (Ferrari, Mercedes F1, Mercedes G500) ---
    {"part_id": "2780", "name": "Technic Pin with Friction"},
    {"part_id": "3673", "name": "Technic Pin 1/2"},
    {"part_id": "32062", "name": "Technic Axle 2L"},
    {"part_id": "32123b", "name": "Technic Bush 1/2"},
    {"part_id": "3700", "name": "Technic Brick 1x2 with Hole"},
]


# ==============================================================================
# LDraw Catalog & Mesh Parser
# ==============================================================================

class LDrawCatalog:
    """Manages indexing and lookup of LDraw files."""

    def __init__(self, ldraw_root: Path):
        self.ldraw_root = ldraw_root
        self.file_index: Dict[str, Path] = {}
        self.colors: Dict[int, Tuple[float, float, float, float]] = {}
        self._index_catalog()
        self._load_colors()

    def _index_catalog(self):
        """Build case-insensitive filename index for fast sub-part resolution."""
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
                                if code == 16:
                                    continue
                                hex_val = val_match.group(1)
                                r = int(hex_val[0:2], 16) / 255.0
                                g = int(hex_val[2:4], 16) / 255.0
                                b = int(hex_val[4:6], 16) / 255.0
                                a = int(alpha_match.group(1)) / 255.0 if alpha_match else 1.0
                                self.colors[code] = (r, g, b, a)
                break
            except Exception as e:
                print(f"[WARN] Error loading LDConfig.ldr: {e}")

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
    """Parses LDraw .dat geometry recursively and stores vertex/face structure."""

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

            if line_type == "1" and len(tokens) >= 15:
                color_code = int(tokens[1])
                effective_color = current_color if color_code == 16 else color_code
                sub_matrix = [float(t) for t in tokens[2:14]]
                sub_file_name = " ".join(tokens[14:])
                combined_matrix = self._combine_matrices(current_matrix, sub_matrix)
                
                resolved_path = self.catalog.find_file(sub_file_name)
                if resolved_path:
                    self.parse_file(resolved_path, combined_matrix, effective_color, depth + 1)

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
# Materials & Shaders (Plastics & Domain Randomization Backgrounds)
# ==============================================================================

def get_or_create_lego_material(rgba: Tuple[float, float, float, float], name_suffix: str = "") -> bpy.types.Material:
    """Create realistic plastic Principled BSDF shader for LEGO pieces."""
    mat_name = f"LEGO_Plastic_{name_suffix}_{int(rgba[0]*255)}_{int(rgba[1]*255)}_{int(rgba[2]*255)}"
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

    r_lin = pow(rgba[0], 2.2)
    g_lin = pow(rgba[1], 2.2)
    b_lin = pow(rgba[2], 2.2)
    bsdf.inputs["Base Color"].default_value = (r_lin, g_lin, b_lin, rgba[3])

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = random.uniform(0.12, 0.22)
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


def create_procedural_table_material(style: Optional[str] = None) -> bpy.types.Material:
    """Create varied procedural materials for table/floor surfaces to bridge Sim-to-Real gap."""
    styles = ["wood", "granite", "fabric", "matte_color"]
    chosen_style = style if style in styles else random.choice(styles)

    mat = bpy.data.materials.new(name=f"Surface_{chosen_style}_{random.randint(1000, 9999)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    if chosen_style == "wood":
        # Wood texture with wave node and noise
        wave = nodes.new(type="ShaderNodeTexWave")
        wave.wave_type = 'RINGS'
        wave.inputs["Scale"].default_value = random.uniform(4.0, 10.0)
        wave.inputs["Distortion"].default_value = random.uniform(2.0, 6.0)
        wave.inputs["Detail"].default_value = 4.0
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

        ramp = nodes.new(type="ShaderNodeValToRGB")
        # Warm wood palette
        ramp.color_ramp.elements[0].color = (0.22, 0.12, 0.06, 1.0)
        ramp.color_ramp.elements[1].color = (0.65, 0.45, 0.28, 1.0)
        links.new(wave.outputs["Color"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.6)

    elif chosen_style == "granite":
        # Granite / Stone surface with Voronoi
        voronoi = nodes.new(type="ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = random.uniform(20.0, 60.0)
        links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])

        ramp = nodes.new(type="ShaderNodeValToRGB")
        c1 = random.uniform(0.15, 0.3)
        c2 = random.uniform(0.7, 0.95)
        ramp.color_ramp.elements[0].color = (c1, c1, c1, 1.0)
        ramp.color_ramp.elements[1].color = (c2, c2, c2, 1.0)
        links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.2, 0.5)

    elif chosen_style == "fabric":
        # Fabric / Carpet / Blanket texture
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = random.uniform(80.0, 200.0)
        noise.inputs["Detail"].default_value = 6.0
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        base_r = random.uniform(0.1, 0.8)
        base_g = random.uniform(0.1, 0.8)
        base_b = random.uniform(0.1, 0.8)

        ramp = nodes.new(type="ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (base_r * 0.7, base_g * 0.7, base_b * 0.7, 1.0)
        ramp.color_ramp.elements[1].color = (base_r, base_g, base_b, 1.0)
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.7, 0.95)

    else:
        # Smooth neutral or painted mat (white, grey, pastel, dark)
        base_color = (
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
            1.0
        )
        bsdf.inputs["Base Color"].default_value = base_color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.7)

    return mat


# ==============================================================================
# Scene Setup, Randomization & Bounding Box Calculation
# ==============================================================================

def clear_blender_scene():
    """Wipe default objects, meshes, materials, lights, and cameras."""
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


def setup_ground_plane():
    """Create a large ground table with procedural material."""
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "Ground_Table"
    
    mat = create_procedural_table_material()
    plane.data.materials.append(mat)
    return plane


def setup_random_lighting():
    """Create randomized 2-3 light setup (Key + Fill/Environment) with temperature variations."""
    # 1. Main Key Light (Point or Area)
    key_type = random.choice(['AREA', 'POINT'])
    key_data = bpy.data.lights.new(name="Rand_Key_Light", type=key_type)
    key_data.energy = random.uniform(250.0, 650.0)
    if key_type == 'AREA':
        key_data.size = random.uniform(0.8, 2.0)
    
    # Random Kelvin temperature (Warm 2800K to Daylight 6500K)
    warmth = random.uniform(0.0, 1.0)
    key_data.color = (
        1.0,
        0.85 + 0.15 * warmth,
        0.70 + 0.30 * warmth
    )
    
    key_obj = bpy.data.objects.new(name="Rand_Key_Light", object_data=key_data)
    # Position in upper hemisphere
    angle_rad = random.uniform(0, 2 * math.pi)
    dist = random.uniform(1.2, 2.5)
    z_pos = random.uniform(1.5, 3.0)
    key_obj.location = (dist * math.cos(angle_rad), dist * math.sin(angle_rad), z_pos)
    bpy.context.collection.objects.link(key_obj)

    # 2. Secondary Fill Light
    fill_data = bpy.data.lights.new(name="Rand_Fill_Light", type='AREA')
    fill_data.energy = random.uniform(80.0, 220.0)
    fill_data.size = random.uniform(2.0, 4.0)
    fill_data.color = (random.uniform(0.8, 1.0), random.uniform(0.85, 1.0), random.uniform(0.9, 1.0))
    fill_obj = bpy.data.objects.new(name="Rand_Fill_Light", object_data=fill_data)
    fill_obj.location = (-key_obj.location.x * 0.8, -key_obj.location.y * 0.8, random.uniform(1.2, 2.5))
    bpy.context.collection.objects.link(fill_obj)

    # 3. Ambient World Color & Strength
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (
            random.uniform(0.05, 0.2),
            random.uniform(0.05, 0.2),
            random.uniform(0.05, 0.2),
            1.0
        )
        bg_node.inputs["Strength"].default_value = random.uniform(0.3, 0.8)


def setup_random_camera(cluster_center: mathutils.Vector, cluster_radius: float, res: int = 640) -> bpy.types.Object:
    """Setup camera with randomized elevation, azimuth, and focal length framed on parts."""
    cam_data = bpy.data.cameras.new(name="RenderCamera")
    cam_data.lens = random.uniform(35.0, 70.0)
    cam_obj = bpy.data.objects.new(name="RenderCamera", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Random elevation (35° oblique to 85° almost top-down)
    elev_deg = random.uniform(35.0, 85.0)
    azim_deg = random.uniform(0.0, 360.0)
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    fov = cam_data.angle
    effective_radius = max(cluster_radius * random.uniform(1.3, 1.9), 0.12)
    distance = effective_radius / math.sin(fov / 2.0)
    distance = max(distance, 0.35)

    cam_x = cluster_center.x + distance * math.cos(elev) * math.sin(azim)
    cam_y = cluster_center.y - distance * math.cos(elev) * math.cos(azim)
    cam_z = cluster_center.z + distance * math.sin(elev)
    cam_obj.location = (cam_x, cam_y, cam_z)

    # Track camera to cluster center
    track = cam_obj.constraints.new(type='TRACK_TO')
    
    empty = bpy.data.objects.new("CamTarget", None)
    bpy.context.collection.objects.link(empty)
    empty.location = cluster_center + mathutils.Vector((
        random.uniform(-0.02, 0.02),
        random.uniform(-0.02, 0.02),
        0.0
    ))
    track.target = empty
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    scene = bpy.context.scene
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100

    return cam_obj


def calculate_yolo_bbox(scene: bpy.types.Scene, camera: bpy.types.Object, obj: bpy.types.Object, class_id: int) -> Optional[str]:
    """
    Project 3D bounding box to 2D normalized screen space and output YOLO format.
    Format: <class_id> <x_center> <y_center> <width> <height>
    """
    corners_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    
    u_coords = []
    v_coords = []
    in_front = False

    for pt in corners_world:
        coords_2d = world_to_camera_view(scene, camera, pt)
        if coords_2d.z > 0:
            in_front = True
        u_coords.append(coords_2d.x)
        v_coords.append(coords_2d.y)

    if not in_front:
        return None

    u_min = min(u_coords)
    u_max = max(u_coords)
    v_min = min(v_coords)
    v_max = max(v_coords)

    # Clamp to image screen [0.0, 1.0]
    x_min = max(0.0, min(1.0, u_min))
    x_max = max(0.0, min(1.0, u_max))
    y_min = max(0.0, min(1.0, 1.0 - v_max))  # Invert Y for image origin (top-left)
    y_max = max(0.0, min(1.0, 1.0 - v_min))

    width = x_max - x_min
    height = y_max - y_min

    # Discard if box is outside view or too small (noise)
    if width <= 0.01 or height <= 0.01 or (width * height < 0.0003):
        return None

    x_center = x_min + (width / 2.0)
    y_center = y_min + (height / 2.0)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ==============================================================================
# Scene Generation Pipeline
# ==============================================================================

class SyntheticSceneGenerator:
    """Manages parsing, caching, and composition of multi-part LEGO synthetic scenes."""

    def __init__(self, ldraw_root: Path, classes: List[Dict[str, str]], use_gpu: bool = True, samples: int = 48):
        self.ldraw_root = ldraw_root
        self.classes = classes
        self.use_gpu = use_gpu
        self.samples = samples
        self.catalog = LDrawCatalog(ldraw_root)
        self.mesh_cache: Dict[str, LDrawMeshBuilder] = {}
        self._preload_catalog_meshes()

    def _preload_catalog_meshes(self):
        """Parse all MVP catalog pieces into memory cache once."""
        print(f"[PRELOAD] Caching {len(self.classes)} MVP part geometries from LDraw...")
        for item in self.classes:
            part_id = item["part_id"]
            part_filename = part_id if part_id.endswith(".dat") else f"{part_id}.dat"
            part_file = self.catalog.find_file(part_filename)
            if not part_file or not part_file.exists():
                print(f"[WARN] Part file not found for {part_id} ({item['name']})")
                continue

            builder = LDrawMeshBuilder(self.catalog, scale=0.001)
            builder.parse_file(part_file)
            if builder.vertices and builder.faces:
                self.mesh_cache[part_id] = builder
            else:
                print(f"[WARN] Failed to load geometry for {part_id}")

        print(f"[PRELOAD] Successfully loaded {len(self.mesh_cache)} / {len(self.classes)} part meshes into cache.")

    def generate_single_scene(self, output_img_path: Path, output_lbl_path: Path, min_parts: int = 3, max_parts: int = 10, res: int = 640):
        """Generate, render, and annotate one complete synthetic image."""
        clear_blender_scene()

        # 1. Ground plane & Lighting
        setup_ground_plane()
        setup_random_lighting()

        # 2. Pick N random parts
        available_part_ids = list(self.mesh_cache.keys())
        if not available_part_ids:
            raise RuntimeError("No cached parts available to generate scene!")

        num_parts = random.randint(min_parts, max_parts)
        chosen_part_ids = [random.choice(available_part_ids) for _ in range(num_parts)]

        placed_objects: List[Tuple[bpy.types.Object, int]] = []
        occupied_positions: List[Tuple[float, float, float]] = []

        # 3. Place parts with randomized positions and resting poses
        scatter_radius = 0.08 + (num_parts * 0.012)

        for p_id in chosen_part_ids:
            builder = self.mesh_cache[p_id]
            class_id = next((i for i, c in enumerate(self.classes) if c["part_id"] == p_id), 0)

            mesh_name = f"Mesh_{p_id}_{random.randint(10000, 99999)}"
            mesh = bpy.data.meshes.new(mesh_name)
            mesh.from_pydata(builder.vertices, [], builder.faces)
            mesh.update()

            obj = bpy.data.objects.new(f"LegoPart_{p_id}", mesh)
            bpy.context.collection.objects.link(obj)

            # Pick random official LEGO color
            color_name, rgba = random.choice(list(NAMED_LEGO_COLORS.items()))
            mat = get_or_create_lego_material(rgba, name_suffix=color_name)
            obj.data.materials.append(mat)

            # Center object origin
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

            # Random natural resting pose
            pitch_candidates = [0.0, math.pi / 2.0, -math.pi / 2.0, math.pi]
            roll_candidates = [0.0, math.pi / 2.0, -math.pi / 2.0, math.pi]
            
            rx = random.choice(pitch_candidates) + math.radians(random.uniform(-8.0, 8.0))
            ry = random.choice(roll_candidates) + math.radians(random.uniform(-8.0, 8.0))
            rz = random.uniform(0.0, 2.0 * math.pi)
            obj.rotation_euler = (rx, ry, rz)

            # Find non-overlapping (X, Y) spot on table
            max_attempts = 35
            placed_pos = None
            min_dist = 0.035

            for _ in range(max_attempts):
                cand_r = random.uniform(0.0, scatter_radius)
                cand_theta = random.uniform(0.0, 2.0 * math.pi)
                cand_x = cand_r * math.cos(cand_theta)
                cand_y = cand_r * math.sin(cand_theta)

                if all(math.hypot(cand_x - ox, cand_y - oy) > min_dist for ox, oy, _ in occupied_positions):
                    placed_pos = (cand_x, cand_y)
                    break

            if placed_pos is None:
                cand_r = random.uniform(0.0, scatter_radius * 1.3)
                cand_theta = random.uniform(0.0, 2.0 * math.pi)
                placed_pos = (cand_r * math.cos(cand_theta), cand_r * math.sin(cand_theta))

            # Adjust Z so lowest vertex touches table
            bpy.context.view_layer.update()
            bbox_corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            min_z = min(c.z for c in bbox_corners)
            z_offset = -min_z + 0.0005

            obj.location = (placed_pos[0], placed_pos[1], z_offset)
            occupied_positions.append((placed_pos[0], placed_pos[1], z_offset))
            placed_objects.append((obj, class_id))

        # 4. Camera framing & Domain Randomization
        bpy.context.view_layer.update()
        all_corners = []
        for obj, _ in placed_objects:
            all_corners.extend([obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box])
        
        cluster_center = sum(all_corners, mathutils.Vector((0, 0, 0))) / len(all_corners)
        cluster_radius = max((v - cluster_center).length for v in all_corners)
        
        camera = setup_random_camera(cluster_center, cluster_radius, res=res)
        bpy.context.view_layer.update()

        # 5. Extract YOLO Labels
        scene = bpy.context.scene
        yolo_lines = []
        for obj, c_id in placed_objects:
            line = calculate_yolo_bbox(scene, camera, obj, c_id)
            if line:
                yolo_lines.append(line)

        # 6. Configure Cycles Renderer
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = self.samples
        scene.cycles.preview_samples = min(self.samples, 16)
        scene.render.film_transparent = False

        if self.use_gpu:
            try:
                prefs = bpy.context.preferences
                cycles_addon = prefs.addons.get('cycles')
                if cycles_addon:
                    cprefs = cycles_addon.preferences
                    for compute_type in ['OPTIX', 'CUDA', 'HIP', 'METAL']:
                        cprefs.compute_device_type = compute_type
                        cprefs.get_devices()
                        valid_devices = [d for d in cprefs.devices if d.type == compute_type]
                        if valid_devices:
                            for d in valid_devices:
                                d.use = True
                            scene.cycles.device = 'GPU'
                            break
            except Exception:
                scene.cycles.device = 'CPU'

        # 7. Render & Save Image & Label
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        output_lbl_path.parent.mkdir(parents=True, exist_ok=True)

        scene.render.filepath = str(output_img_path)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'

        bpy.ops.render.render(write_still=True)

        with open(output_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")


# ==============================================================================
# Batch Dataset Generator & dataset.yaml Exporter
# ==============================================================================

def export_yolo_yaml(dataset_root: Path, classes: List[Dict[str, str]]):
    """Generate Ultralytics YOLOv11 dataset.yaml configuration file."""
    yaml_content = [
        f"# SnapBrick Synthetic Dataset for YOLOv11",
        f"# Auto-generated by data-pipeline/src/generate_dataset.py",
        f"",
        f"path: {dataset_root.resolve().as_posix()}",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        f"",
        f"nc: {len(classes)}",
        f"names:",
    ]
    for idx, item in enumerate(classes):
        yaml_content.append(f"  {idx}: \"{item['name']} ({item['part_id']})\"")

    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_content) + "\n")
    print(f"[DATASET] Wrote YOLO configuration to {yaml_path}")


def run_batch_generation(
    ldraw_root: Path,
    output_dir: Path,
    num_train: int = 100,
    num_val: int = 20,
    num_test: int = 10,
    min_parts: int = 3,
    max_parts: int = 10,
    res: int = 640,
    samples: int = 48,
    use_gpu: bool = True
):
    """Execute complete dataset generation pipeline across train, val, and test splits."""
    print("\n=======================================================")
    print(" 🧱 SnapBrick Synthetic Dataset Generator (YOLOv11)")
    print("=======================================================")
    print(f" Output Root:     {output_dir}")
    print(f" Train / Val / Test: {num_train} / {num_val} / {num_test} images")
    print(f" Resolution:      {res}x{res} | Samples: {samples}")
    print(f" Parts per Scene: {min_parts} to {max_parts}")
    print(f" GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print("=======================================================\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    export_yolo_yaml(output_dir, MVP_PART_CATALOG)

    generator = SyntheticSceneGenerator(
        ldraw_root=ldraw_root,
        classes=MVP_PART_CATALOG,
        use_gpu=use_gpu,
        samples=samples
    )

    splits = [
        ("train", num_train),
        ("val", num_val),
        ("test", num_test),
    ]

    global_img_idx = 1
    total_imgs = num_train + num_val + num_test

    t_start = time.time()

    for split_name, count in splits:
        if count <= 0:
            continue
        print(f"\n--- Generating '{split_name}' Split ({count} images) ---")
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, count + 1):
            file_stem = f"frame_{global_img_idx:06d}"
            img_path = img_dir / f"{file_stem}.png"
            lbl_path = lbl_dir / f"{file_stem}.txt"

            t0 = time.time()
            generator.generate_single_scene(
                output_img_path=img_path,
                output_lbl_path=lbl_path,
                min_parts=min_parts,
                max_parts=max_parts,
                res=res
            )
            dt = time.time() - t0

            lbl_count = 0
            if lbl_path.exists():
                with open(lbl_path, "r") as lf:
                    lbl_count = len([l for l in lf if l.strip()])

            print(f"[{global_img_idx}/{total_imgs}] {split_name}/{file_stem}.png -> {lbl_count} pieces annotated in {dt:.2f}s")
            global_img_idx += 1

    total_time = time.time() - t_start
    print("\n=======================================================")
    print(f" ✅ Synthetic Dataset successfully generated in {total_time:.1f}s!")
    print(f" Location: {output_dir}")
    print(f" Manifest: {output_dir / 'dataset.yaml'}")
    print("=======================================================\n")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SnapBrick Synthetic Dataset Generator for YOLOv11")
    parser.add_argument("--ldraw_root", type=str, default="", help="Path to ldraw library directory")
    parser.add_argument("--output_dir", type=str, default="", help="Destination directory for dataset")
    parser.add_argument("--num_train", type=int, default=10, help="Number of training images to generate")
    parser.add_argument("--num_val", type=int, default=2, help="Number of validation images to generate")
    parser.add_argument("--num_test", type=int, default=2, help="Number of test images to generate")
    parser.add_argument("--min_parts", type=int, default=3, help="Minimum pieces per scene")
    parser.add_argument("--max_parts", type=int, default=8, help="Maximum pieces per scene")
    parser.add_argument("--resolution", type=int, default=640, help="Image square resolution")
    parser.add_argument("--samples", type=int, default=48, help="Cycles render samples per image")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU compute acceleration")

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

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = project_root / "ml-core" / "dataset"

    if not HAS_BPY:
        print("[ERROR] Blender Python API (bpy) is not available in the current Python environment.", file=sys.stderr)
        print("Please run via Blender: blender -b -P data-pipeline/src/generate_dataset.py -- [OPTIONS]", file=sys.stderr)
        sys.exit(1)

    run_batch_generation(
        ldraw_root=ldraw_root,
        output_dir=output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        min_parts=args.min_parts,
        max_parts=args.max_parts,
        res=args.resolution,
        samples=args.samples,
        use_gpu=not args.no_gpu
    )


if __name__ == "__main__":
    main()
