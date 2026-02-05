#!/bin/bash

# scene0000_00
# scene0000_02
# scene0002_00
# scene0024_00
# scene0517_01
# scene0554_01

SCENES=(
  # scene0000_02
  # scene0002_00
  # scene0024_00
  scene0517_01
  scene0554_01
)




SHEET_NAME="PC2"   # ← 업로드할 구글시트 이름

for SCENE in "${SCENES[@]}"; do
  echo "=========================================="
  echo "📂 Processing: ${SCENE}"
  echo "=========================================="

  ######실험돌릴 때마다 경로 수정!!!!!!!!!!!!!!!!!!!!!!!!
  # scannet_base -> scannet_base_v2이렇게
  MODEL_PATH="./outputs/scannet_base_t2/${SCENE}"

  # 1️⃣ Training / Eval
  echo ""
  echo "🔵 [1/4] Training (eval mode)..."
  python train.py --eval \
    -s ./data/scannet/${SCENE}/ \
    -m ${MODEL_PATH} \
    --mode free \
    --port 33888

  [ $? -ne 0 ] && echo "❌ Training failed for ${SCENE}, skipping..." && continue

  # 2️⃣ Rendering
  echo ""
  echo "🟢 [2/4] Rendering..."
  python render.py \
    -m ${MODEL_PATH} \
    --original_images_path ./data/scannet_origin/${SCENE}/images

  [ $? -ne 0 ] && echo "❌ Rendering failed for ${SCENE}, skipping..." && continue

  # 3️⃣ Metrics
  echo ""
  echo "🟣 [3/4] Evaluating metrics..."
  python metrics.py \
    -m ${MODEL_PATH}

  [ $? -ne 0 ] && echo "❌ Metrics failed for ${SCENE}, skipping..." && continue

  # 4️⃣ GSpread 업로드
  echo ""
  echo "📤 [4/4] Uploading results to GSpread..."

  JSON_PATH="${MODEL_PATH}/results.json"
  POSE_PATH="${MODEL_PATH}/test/ours_40000/poses/pose_eval.txt"

  if [ -f "${JSON_PATH}" ]; then
    python gspread/gspread-results.py \
      "${JSON_PATH}" \
      "${POSE_PATH}" \
      "${MODEL_PATH}" \
      "${SHEET_NAME}"
  else
    echo "⚠️ No results.json found for ${SCENE}, skipping upload."
  fi

  echo ""
  echo "✅ Finished ${SCENE}"
  echo "------------------------------------------"
done
