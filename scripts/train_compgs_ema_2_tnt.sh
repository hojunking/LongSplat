#!/bin/bash
# ============================================
# 실험 설정
# ============================================

SCENES=("Family" "Church" "Barn" "Museum" "Horse" "Ballroom" "Francis" "Ignatius")

QP_LEVELS=("qp27")  # QP37 먼저, QP32 나중
COMPRESSED_DATA="/workdir/data/compress-o/tnt"
ORIGINAL_DATA="/workdir/data/compress-x/tnt"
OUTPUT_BASE="outputs/tnt_qp27_freemode_rx"
SHEET_NAME="rebuttal"

# ============================================
# 루프 시작
# ============================================
for SCENE in "${SCENES[@]}"; do
  for QP in "${QP_LEVELS[@]}"; do
    SCENE_QP="${SCENE}_${QP}_compgs"
    COMP_PATH="${COMPRESSED_DATA}/${QP}/${SCENE}"
    MODEL_PATH="${OUTPUT_BASE}/${SCENE_QP}"

    echo "=========================================="
    echo "📂 Processing: ${SCENE_QP}"
    echo "=========================================="

    # 1️⃣ Training 
    echo ""
    echo "🔵 [1/3] Training with ${QP} images..."
    python train_compgs_ema_revise.py --eval \
        -s ${COMP_PATH} \
        -m ${MODEL_PATH} \
        --mode free \
        --d_mu 0.5  \
        --port $((12345 + RANDOM % 1000)) \
        --scene_name ${SCENE} \
        --qp_level ${QP}  \
        --trust_momentum 0.95

    [ $? -ne 0 ] && echo "❌ Training failed for ${SCENE_QP}, skipping..." && continue

    # 2️⃣ Rendering
    echo ""
    echo "🟢 [2/3] Rendering ${SCENE_QP}..."
    python render.py \
        -m ${MODEL_PATH} \
        --original_images_path ${ORIGINAL_DATA}/${SCENE}/images

    [ $? -ne 0 ] && echo "❌ Rendering failed for ${SCENE_QP}, skipping..." && continue

    # 3️⃣ Metrics
    echo ""
    echo "🟣 [3/3] Evaluating metrics for ${SCENE_QP}..."
    python metrics.py -m ${MODEL_PATH}

    [ $? -ne 0 ] && echo "❌ Metrics failed for ${SCENE_QP}, skipping..." && continue

    # 4️⃣ GSpread 업로드 (결과 시트 자동 업로드)
    JSON_PATH="${MODEL_PATH}/results.json"
    POSE_PATH="${MODEL_PATH}/test/ours_40000/poses/pose_eval.txt"

    if [ -f "$JSON_PATH" ]; then
      echo "📤 Uploading ${SCENE_QP} results to GSpread (${SHEET_NAME})..."
      python gspread/gspread-results.py \
          "${JSON_PATH}" \
          "${POSE_PATH}" \
          "${MODEL_PATH}" \
          "${SHEET_NAME}"
    else
      echo "⚠️ No results.json found for ${SCENE_QP}, skipping upload."
    fi

    echo ""
    echo "✅ Finished ${SCENE_QP}"
    echo "------------------------------------------"
  done
done




# python train_imp.py --eval -s ./data/compress-o/free/qp37/grass/ -m ./outputs/free/grass_qp37_imp1/ -r 2 --mode free --port 38290

# python render.py  -m ./outputs/free/grass_qp37_imp1/ --original_images_path ./data/compress-x/free/grass/images 

# python metrics.py -m ./outputs/free/grass_qp37_imp1/ 