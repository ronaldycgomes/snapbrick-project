#!/usr/bin/env bash
# ==============================================================================
# SnapBrick - Treinamento do Classificador de Estágio 2 (Task 3 / Issue #23)
# Modelo: YOLOv11m-cls (ou yolo11n-cls)
# Entrada: Crops 224x224 com Color-Blind Augmentation (Hue Jitter, 360° Rot)
# Tempo estimado na RTX 5070: ~2 a 3 minutos
# ==============================================================================

set -euo pipefail

echo "======================================================="
echo " 🧠 SnapBrick: Treinamento do Classificador de Estágio 2"
echo "======================================================="

# Ativar venv se existir
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "data-pipeline/.venv" ]; then
    source data-pipeline/.venv/bin/activate
fi

# Limpar caches antigos para reindexar as 51 classes completas
rm -f ml-core/dataset_crops/*.cache ml-core/dataset_crops/train.cache ml-core/dataset_crops/val.cache 2>/dev/null || true

python3 ml-core/src/train_classifier.py \
    --data ml-core/dataset_crops \
    --model yolo11m-cls.pt \
    --epochs 35 \
    --batch 32 \
    --workers 4 \
    --name stage2_classifier

echo -e "\n======================================================="
echo " ✅ Treinamento do Estágio 2 Concluído com Sucesso!"
echo " Pesos salvos em: ml-core/runs/stage2_classifier/weights/best.pt"
echo "======================================================="
