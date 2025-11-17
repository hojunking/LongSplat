#!/bin/bash

# ============================================
# Pose Visualization for All Free Scenes
# Baseline + Ours
# ============================================

SCENES=(grass hydrant lab pillar road sky stair)

DATA_ROOT=./data/compress-x/free
BASELINE_ROOT=./pose-vis
OURS_ROOT=./outputs/free_ema
OUT_DIR=./results_vis

mkdir -p ${OUT_DIR}

echo "==============================================="
echo "🚀 Starting Pose Visualization for All Scenes"
echo "==============================================="

for S in "${SCENES[@]}"; do
    echo ""
    echo "-----------------------------------------------"
    echo "📌 Processing scene: $S"
    echo "-----------------------------------------------"

    DATA_PATH=${DATA_ROOT}/${S}/

    # =============================
    # Baseline path
    # =============================
    BASELINE_PATH=${BASELINE_ROOT}/${S}_qp37/
    BASELINE_OUT=${OUT_DIR}/${S}-longsplat-baseline.png

    echo "➡️  Baseline:"
    echo "    -s $DATA_PATH"
    echo "    -m $BASELINE_PATH"
    echo "    → $BASELINE_OUT"

    python vispose_longsplat.py \
        -s ${DATA_PATH} \
        -m ${BASELINE_PATH} \
        --output_path ${BASELINE_OUT}

    # =============================
    # Ours (CompGS) path
    # =============================
    OURS_PATH=${OURS_ROOT}/${S}_qp37_compgs_mom095_dmu05/
    OURS_OUT=${OUT_DIR}/${S}-longsplat-ours.png

    echo "➡️  Ours:"
    echo "    -s $DATA_PATH"
    echo "    -m $OURS_PATH"
    echo "    → $OURS_OUT"

    python vispose_longsplat.py \
        -s ${DATA_PATH} \
        -m ${OURS_PATH} \
        --output_path ${OURS_OUT}

done

echo ""
echo "==============================================="
echo "🎉 All pose visualizations completed!"
echo "==============================================="
