#!/usr/bin/env bash
# ==============================================================================
# SnapBrick - Pipeline Unificado Two-Stage (Task 4 & 5 / Issues #24 e #25)
# Executa Detecção (Estágio 1) + Classificação de Crops (Estágio 2) + Cores (CIE Lab)
# Entrada: Foto física real do iPhone (IMG_0033.jpg ou IMG_0032.jpg)
# ==============================================================================

set -euo pipefail

IMAGE=${1:-"ml-core/dataset/images/IMG_0033.jpg"}
OUTPUT_DIR="ml-core/runs/inference_two_stage"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "data-pipeline/.venv" ]; then
    source data-pipeline/.venv/bin/activate
fi

python3 ml-core/src/detect_pipeline.py \
    --image "$IMAGE" \
    --detector ml-core/runs/stage1_detector_poc/weights/best.pt \
    --classifier ml-core/runs/stage2_classifier/weights/best.pt \
    --output_dir "$OUTPUT_DIR" \
    --conf 0.28 \
    --iou 0.30

echo -e "\n======================================================="
echo " ✅ Validação Ponta-a-Ponta Two-Stage Concluída!"
echo " Imagem anotada: $OUTPUT_DIR/annotated_two_stage_$(basename "$IMAGE")"
echo " Inventário JSON: $OUTPUT_DIR/inventory_two_stage_$(basename "${IMAGE%.*}").json"
echo "======================================================="
