#!/usr/bin/env python3
"""
SnapBrick Diagnostic - Blender GPU & CUDA/OptiX Verification Script
Checks if NVIDIA RTX 5070 (or any dedicated GPU) is recognized and utilized by Blender Cycles in WSL 2.
"""

import sys
import time

try:
    import bpy
except ImportError:
    print("[ERROR] bpy (Blender Python API) is not available. Run this with: blender -b -P data-pipeline/src/check_gpu.py")
    sys.exit(1)


def check_gpu_support():
    print("=" * 65)
    print(" 🚀 SnapBrick - Diagnóstico de GPU / Blender Cycles no WSL 2")
    print("=" * 65)
    print(f"Blender Version: {bpy.app.version_string}")

    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get('cycles')

    if not cycles_addon:
        print("[ERROR] Cycles addon não encontrado no Blender!")
        return

    cprefs = cycles_addon.preferences

    print("\n🔍 Investigando backends de computação disponíveis:")
    detected_gpus = []

    for backend in ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']:
        try:
            cprefs.compute_device_type = backend
            cprefs.get_devices()
            devices = cprefs.devices
            matching = [d for d in devices if d.type == backend]

            if matching:
                print(f"\n  ✅ Backend [{backend}]: {len(matching)} dispositivo(s) encontrado(s):")
                for d in matching:
                    print(f"     - Nome: {d.name}")
                    print(f"       Tipo: {d.type}")
                    print(f"       ID: {d.id}")
                    detected_gpus.append((backend, d.name))
            else:
                print(f"  ⚪ Backend [{backend}]: Nenhum dispositivo compatível.")
        except Exception as e:
            print(f"  ❌ Backend [{backend}]: Erro ao consultar ({e})")

    print("\n" + "-" * 65)

    if not detected_gpus:
        print("⚠️ NENHUMA GPU DEDICADA DETECTADA PELO BLENDER NO WSL 2!")
        print("\nPossíveis motivos e soluções:")
        print(" 1. Driver NVIDIA no Windows:")
        print("    Certifique-se de ter o driver NVIDIA Game Ready ou Studio atualizado no Windows.")
        print(" 2. Bibliotecas CUDA no WSL 2:")
        print("    Verifique se o comando 'nvidia-smi' funciona no terminal do WSL 2.")
        print(" 3. Pacotes de aceleração:")
        print("    sudo apt install -y libcuda1 libnvidia-compute-535 (ou versão do seu driver).")
        return

    # Se encontramos GPU, vamos rodar um teste rápido de renderização
    best_backend, gpu_name = detected_gpus[0]
    print(f"🎯 GPU Principal Identificada: {gpu_name} via {best_backend}")
    print("\n🧪 Executando teste prático de renderização com a GPU...")

    # Configura cena simples para benchmark
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cprefs.compute_device_type = best_backend
    cprefs.get_devices()
    for d in cprefs.devices:
        if d.type == best_backend:
            d.use = True

    scene.cycles.device = 'GPU'
    scene.cycles.samples = 128
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.filepath = "/tmp/gpu_test_render.png"

    # Criar cubo de teste
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 64
    scene.render.filepath = "/tmp/gpu_test_render.png"

    bpy.ops.mesh.primitive_cube_add(size=2.0)
    cam = bpy.data.cameras.new("TestCam")
    cam_obj = bpy.data.objects.new("TestCam", cam)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.location = (4.0, -4.0, 3.0)
    cam_obj.rotation_euler = (1.1, 0.0, 0.8)

    light_data = bpy.data.lights.new(name="TestLight", type='POINT')
    light_data.energy = 500.0
    light_obj = bpy.data.objects.new(name="TestLight", object_data=light_data)
    light_obj.location = (3.0, -2.0, 4.0)
    bpy.context.collection.objects.link(light_obj)

    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    t1 = time.time()

    print(f"\n✨ SUCESSO! Renderização de teste finalizada em {t1 - t0:.2f} segundos na GPU!")
    print(f"   Motor: Blender Cycles ({best_backend})")
    print(f"   Dispositivo: {gpu_name}")
    print("=" * 65)


if __name__ == "__main__":
    check_gpu_support()
