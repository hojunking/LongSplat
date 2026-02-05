#!/bin/bash

# ============================================
# 실험 설정
# ============================================

# SCENES_HIKE=("forest1" "forest2" "forest3" "garden1" "garden2" "garden3" "indoor" "playground" "university1" "university2" "university3" "university4")
# SCENES_HIKE=( "playground" "university1" "university2" "university3" "university4")
SCENES_HIKE=("garden2")

DATA_COMP_O="./data/compress-o/hike/qp37_half"   # 입력 (압축된 qp37_half)
DATA_ORIG="./data/compress-x/hike_half"         # 원본 이미지 경로
OUTPUT_BASE="./outputs/hike/compressed_ours"          # 모델 저장 base
QP_TAG="qp37"                                   # 태그용
SHEET_NAME="supple"                             # ✅ 구글 시트 이름 (원하는 이름으로 바꿔도 됨)

ulimit -n 4096

echo "=========================================="
echo "🚀 Running Hike CompGS (ema_revise) experiments..."
echo "=========================================="

for SCENE in "${SCENES_HIKE[@]}"; do
  SCENE_TAG="${QP_TAG}_${SCENE}"
  MODEL_PATH="${OUTPUT_BASE}/${SCENE_TAG}"

  echo ""
  echo "=========================================="
  echo "📂 Processing: ${SCENE_TAG}"
  echo "=========================================="

  PORT=$((39200 + RANDOM % 1000))
  echo "🔌 Using port: ${PORT}"

  # 1️⃣ Training
  echo ""
  echo "🔵 [1/3] Training ${SCENE_TAG}..."
  python train_compgs_ema_revise.py --eval \
      -s ${DATA_COMP_O}/${SCENE} \
      -m ${MODEL_PATH} \
      -r 4 \
      --port ${PORT} \
      --mode hike

  if [ $? -ne 0 ]; then
    echo "❌ Training failed for ${SCENE_TAG}, skipping..."
    continue
  fi

  # 2️⃣ Rendering
  echo ""
  echo "🟢 [2/3] Rendering ${SCENE_TAG}..."
  python render.py \
      -m ${MODEL_PATH} \
      --original_images_path ${DATA_ORIG}/${SCENE}/images/

  if [ $? -ne 0 ]; then
    echo "❌ Rendering failed for ${SCENE_TAG}, skipping..."
    continue
  fi

  # 3️⃣ Metrics
  echo ""
  echo "🟣 [3/3] Evaluating metrics for ${SCENE_TAG}..."
  python metrics.py -m ${MODEL_PATH}

  if [ $? -ne 0 ]; then
    echo "❌ Metrics failed for ${SCENE_TAG}, skipping..."
    continue
  fi

  # 4️⃣ GSpread 업로드 (GSplat 시트에 업로드)
  JSON_PATH="${MODEL_PATH}/results.json"
  POSE_PATH="${MODEL_PATH}/test/ours_40000/poses/pose_eval.txt"

  if [ -f "$JSON_PATH" ]; then
    echo ""
    echo "📤 Uploading ${SCENE_TAG} results to GSpread (${SHEET_NAME})..."
    python gspread/gspread-results.py \
        "${JSON_PATH}" \
        "${POSE_PATH}" \
        "${MODEL_PATH}" \
        "${SHEET_NAME}"
  else
    echo "⚠️ No results.json found for ${SCENE_TAG}, skipping upload."
  fi

  echo ""
  echo "✅ Finished ${SCENE_TAG}"
  echo "------------------------------------------"
done
