#!/bin/bash

# ===============================================
# Usage:
#   bash evaluate_scene.sh <SCENE_PATH> <SHEET_NAME>
# Example:
#   bash evaluate_scene.sh /workdir/outputs/free_prev/grass_qp37 my_sheet
# ===============================================

SCENE_PATH="outputs/free_ema/sky_qp37_compgs_ours_try3"
SHEET_NAME="PC2"
echo "📌 Evaluating scene: ${SCENE_PATH}"

# Extract scene name (grass_qp37 → grass , qp37)
SCENE_NAME=$(basename "$SCENE_PATH")
SCENE=$(echo "$SCENE_NAME" | cut -d'_' -f1)
QP=$(echo "$SCENE_NAME" | cut -d'_' -f2)

echo "  → Scene = ${SCENE}"
echo "  → QP = ${QP}"

MODEL_PATH="${SCENE_PATH}/test/ours_40000"
RENDER_DIR="${MODEL_PATH}/renders"
JSON_PATH="${RENDER_DIR}/results_comp.json"
TEST_TXT="${MODEL_PATH}/test.txt"
POSE_PATH="${MODEL_PATH}/poses/pose_eval.txt"

GT_BASE="/workdir/data/compress-o/free/${QP}"
GT_DIR="${GT_BASE}/${SCENE}/images_2"

echo "📁 GT DIR: ${GT_DIR}"

# =====================================================
# 1. Photometric Evaluation
# =====================================================

python get_phtometric.py "${RENDER_DIR}" "${GT_DIR}" "${JSON_PATH}"

# =====================================================
# 3. Upload to Google Sheets
# =====================================================

echo "📤 Uploading to Google Sheets..."

python gspread/gspread-results.py \
    "${JSON_PATH}" \
    "${POSE_PATH}" \
    "${MODEL_PATH}" \
    "${SHEET_NAME}"

echo "🎉 Done!"