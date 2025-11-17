#!/bin/bash
# ============================================
# 실험 설정
# ============================================
# SCENES=("IMG_0405" "IMG_0406")
SCENES=("IMG_0406")
# SCENES=("IMG_0406")

QP_LEVELS=("QP37")  # QP37 먼저, QP32 나중
COMPRESSED_DATA="/workdir/data/compo-youtube"
ORIGINAL_DATA="/workdir/data/compx-youtube"
OUTPUT_BASE="outputs/realworld_ours"
SHEET_NAME="PC2"

# ============================================
# 루프 시작
# ============================================
for SCENE in "${SCENES[@]}"; do
  for QP in "${QP_LEVELS[@]}"; do
    # SCENE_QP="${SCENE}_${QP}_compgs_mom095_dmu001"
    SCENE_QP="${SCENE}_${QP}"
    COMP_PATH="${COMPRESSED_DATA}/${SCENE}/${QP}"
    MODEL_PATH="${OUTPUT_BASE}/${SCENE_QP}"

    echo "=========================================="
    echo "📂 Processing: ${SCENE_QP}"
    echo "=========================================="

    # 1️⃣ Training 
    echo ""
    echo "🔵 [1/3] Training with ${QP} images..."
    # python train_compgs_ema_revise.py --eval \
    #     -s ${COMP_PATH} \
    #     -m ${MODEL_PATH}_compgs_mom095_dmu01 \
    #     -r 4 --mode custom \
    #     --d_mu 0.1  \
    #     --port $((12345 + RANDOM % 1000)) \
    #     --scene_name ${SCENE} \
    #     --qp_level ${QP}  \
    #     --trust_momentum 0.95

    # python train_compgs_ema_revise.py --eval \
    #     -s ${COMP_PATH} \
    #     -m ${MODEL_PATH}_compgs_mom095_dmu03 \
    #     -r 4 --mode custom \
    #     --d_mu 0.3  \
    #     --port $((12345 + RANDOM % 1000)) \
    #     --scene_name ${SCENE} \
    #     --qp_level ${QP}  \
    #     --trust_momentum 0.95



    [ $? -ne 0 ] && echo "❌ Training failed for ${SCENE_QP}, skipping..." && continue

    # 2️⃣ Rendering
    echo ""
    echo "🟢 [2/3] Rendering ${SCENE_QP}..."
    python render.py \
        -m ${MODEL_PATH}_compgs_mom095_dmu01 \
        --original_images_path ${ORIGINAL_DATA}/${SCENE}_frames/images

    [ $? -ne 0 ] && echo "❌ Rendering failed for ${SCENE_QP}, skipping..." && continue

    # 3️⃣ Metrics
    echo ""
    echo "🟣 [3/3] Evaluating metrics for ${SCENE_QP}..."
    python metrics.py -m ${MODEL_PATH}_compgs_mom095_dmu01

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