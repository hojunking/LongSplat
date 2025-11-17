#!/bin/bash

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

ulimit -n 4096
port=$(rand 10000 30000)

# ============================================
# 1) Hike Baseline 실험 (경로 수정 버전)
# ============================================
# SCENES_HIKE=("forest1" "forest2" "forest3" "garden1" "garden2" "garden3" "indoor" "playground" "university1" "university2" "university3" "university4")
SCENES_HIKE=("forest3" "garden1" "garden2" "garden3" "indoor" "playground" "university1" "university2" "university3" "university4")

HIKE_DATA_ROOT="/workdir/data/compress-x/hike_half"
OUTPUT_ROOT="outputs/hike"
SHEET_NAME="supple"   # ✅ gspread 시트 이름 (원하는 이름으로 바꿔도 됨)

echo "=========================================="
echo "🚀 Running HIKE baseline experiments..."
echo "=========================================="

for scene in "${SCENES_HIKE[@]}"; do

    MODEL_PATH="${OUTPUT_ROOT}/${scene}"

    echo ""
    echo "▶️ HIKE Baseline: $scene"
    echo "   - DATA : ${HIKE_DATA_ROOT}/${scene}"
    echo "   - MODEL: ${MODEL_PATH}"

    # 1️⃣ Training
    python train.py --eval \
        -s ${HIKE_DATA_ROOT}/${scene} \
        -m ${MODEL_PATH} \
        -r 4 \
        --port ${port} \
        --mode hike

    if [ $? -ne 0 ]; then
      echo "❌ Training failed for ${scene}, skipping..."
      continue
    fi

    # 2️⃣ Rendering
    python render.py \
        -m ${MODEL_PATH} \
        --original_images_path ${HIKE_DATA_ROOT}/${scene}/images

    if [ $? -ne 0 ]; then
      echo "❌ Rendering failed for ${scene}, skipping..."
      continue
    fi

    # 3️⃣ Metrics
    python metrics.py \
        -m ${MODEL_PATH}

    if [ $? -ne 0 ]; then
      echo "❌ Metrics failed for ${scene}, skipping..."
      continue
    fi

    # 4️⃣ GSpread 업로드
    JSON_PATH="${MODEL_PATH}/results.json"
    POSE_PATH="${MODEL_PATH}/test/ours_40000/poses/pose_eval.txt"  # 🔸경로 구조 다르면 여기만 수정

    if [ -f "${JSON_PATH}" ]; then
      echo "📤 Uploading ${scene} results to GSpread (${SHEET_NAME})..."
      python gspread/gspread-results.py \
          "${JSON_PATH}" \
          "${POSE_PATH}" \
          "${MODEL_PATH}" \
          "${SHEET_NAME}"
    else
      echo "⚠️ No results.json found for ${scene}, skipping upload."
    fi

    echo "✅ Finished ${scene}"
    echo "------------------------------------------"

done

# 이하 CompGS/GSplat 블록은 그대로 두면 됨
