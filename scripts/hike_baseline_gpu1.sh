# ============================================
# 실험 설정
# ============================================

SCENES=( "garden3" "indoor" "playground" "university1" "university2" "university3" "university4")
# "garden3" "indoor" "playground" "university1" "university2" "university3" "university4")

QP_LEVELS=("qp47")
COMPRESSED_DATA="/workdir/data/compress-o/hike_half/"
ORIGINAL_DATA="/workdir/data/compress-x/hike_half/"
OUTPUT_BASE=""
SHEET_NAME="rebuttal"

# ============================================
# 루프 시작
# ============================================
for SCENE in "${SCENES[@]}"; do
  for QP in "${QP_LEVELS[@]}"; do
    OUTPUT_BASE="outputs/hike_${QP}_baseline"
    SCENE_QP="${SCENE}_${QP}_baseline"
    COMP_PATH="${COMPRESSED_DATA}/${QP}/${SCENE}"
    MODEL_PATH="${OUTPUT_BASE}/${SCENE_QP}"

    echo "=========================================="
    echo "📂 Processing: ${SCENE_QP}"
    echo "=========================================="

    # 1️⃣ Training 
    echo ""
    echo "🔵 [1/3] Training with ${QP} images..."
    python train.py --eval \
        -s ${COMP_PATH} \
        -m ${MODEL_PATH} \
        -r 4 \
        --mode hike \
        --port $((12345 + RANDOM % 1000))
    
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
