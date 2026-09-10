#!/usr/bin/env bash
# ==============================================================================
# SnapBrick - Geração do Dataset Sintético Completo de Crops 224x224 (Task 2)
# Renderiza as 50 classes do catálogo com poses de repouso, PBR e fundos variados.
# Split: 80% train / 20% val (Estrutura PyTorch ImageFolder)
# Tempo estimado na RTX 5070: ~5 a 7 minutos (3.000 crops)
# ==============================================================================

set -euo pipefail

OUTPUT_DIR="ml-core/dataset_crops"
SAMPLES_PER_PART=${1:-120}   # 120 crops por classe (96 train / 24 val) -> ~6.120 crops total
CYCLES_SAMPLES=20           # 20 samples em 224x224 com OptiX denoiser

echo "======================================================="
echo " 🧱 SnapBrick: Gerando Dataset Completo de Crops (224x224)"
echo "======================================================="
echo " • Diretório Destino:  $OUTPUT_DIR"
echo " • Peças no Catálogo:  51 classes (com 14704 e poses reais no solo)"
echo " • Crops por Classe:   $SAMPLES_PER_PART (total: $(( 51 * SAMPLES_PER_PART )) crops)"
echo " • Resolução:          224x224 (Padrão SOTA ViT/EfficientNet/YOLOv11-cls)"
echo " • GPU Acceleration:   Ativa (CUDA/OptiX na RTX 5070)"
echo "======================================================="

mkdir -p "$OUTPUT_DIR"

blender -b -P data-pipeline/src/generate_crops.py -- \
    --parts all \
    --samples_per_part "$SAMPLES_PER_PART" \
    --output_dir "$OUTPUT_DIR" \
    --res 224 \
    --samples "$CYCLES_SAMPLES" \
    --val_ratio 0.20

echo -e "\n==> Gerando painel de verificação final (preview)..."
python3 -c "
import sys
from pathlib import Path
from PIL import Image, ImageDraw

out_dir = Path('$OUTPUT_DIR')
img_paths = sorted(list(out_dir.glob('val/**/*.png')))
if not img_paths:
    img_paths = sorted(list(out_dir.glob('**/*.png')))

if not img_paths:
    print('[WARN] Nenhuma imagem encontrada.')
    sys.exit(0)

# Pega 1 exemplo de cada classe (até 50 classes)
seen_classes = set()
selected = []
for p in img_paths:
    cls_name = p.parent.name
    if cls_name not in seen_classes:
        seen_classes.add(cls_name)
        selected.append(p)
    if len(selected) >= 50:
        break

cols = 10
cell_size = 140
cell_w = cell_size + 10
cell_h = cell_size + 24
n = len(selected)
rows = (n + cols - 1) // cols

grid_img = Image.new('RGB', (cols * cell_w, rows * cell_h), color=(25, 25, 25))
draw = ImageDraw.Draw(grid_img)

for i, p in enumerate(selected):
    r = i // cols
    c = i % cols
    x = c * cell_w + 5
    y = r * cell_h + 5
    with Image.open(p) as im:
        im_resized = im.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        grid_img.paste(im_resized, (x, y))
    draw.rectangle([x, y, x + cell_size, y + cell_size], outline=(80, 80, 80))
    label = p.parent.name.split('_')[0]
    draw.text((x + 4, y + cell_size + 4), label, fill=(220, 220, 220))

preview_path = out_dir / 'full_dataset_catalog_preview.jpg'
grid_img.save(str(preview_path), quality=92)
print(f'🖼️ Painel do catálogo completo salvo em: {preview_path}')
"

echo -e "\n======================================================="
echo " ✅ Dataset Completo de Crops 224x224 Gerado com Sucesso!"
echo " Estrutura pronta para o Treinamento do Classificador de Estágio 2:"
echo "   - $OUTPUT_DIR/train/"
echo "   - $OUTPUT_DIR/val/"
echo "======================================================="
