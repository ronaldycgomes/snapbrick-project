#!/usr/bin/env bash
# ==============================================================================
# SnapBrick - Validação Rápida do Estágio 2 nos Crops Físicos Reais (IMG_0033)
# Avalia os pesos treinados em best.pt diretamente nos 12 recortes do iPhone
# ==============================================================================

set -euo pipefail

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "data-pipeline/.venv" ]; then
    source data-pipeline/.venv/bin/activate
fi

python3 ml-core/src/validate_stage2_real_crops.py
