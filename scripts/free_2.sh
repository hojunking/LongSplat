#!/bin/bash

# Scenes: road, sky, stair
SCENES=("road" "sky" "stair")

# 데이터 경로
COMPRESSED_DATA="/workdir/data/compress-o/free_dataset/qp37"
ORIGINAL_DATA="/workdir/data/compress-x/free_dataset"

# 출력 경로
OUTPUT_BASE="outputs/free"

# gspread 설정
SHEET_NAME="experiments_compression"

echo "=========================================="
echo "🚀 Starting Experiments (Part 2/2)"
echo "Scenes: road, sky, stair"
echo "=========================================="
echo ""

for SCENE in "${SCENES[@]}"; do
    echo "=========================================="
    echo "📂 Processing Scene: $SCENE"
    echo "=========================================="
    
    # ============================================
    # 1. 압축 이미지로 학습
    # ============================================
    echo ""
    echo "🔵 [1/6] Training with COMPRESSED images (qp37)..."
    python train.py --eval \
        -s ${COMPRESSED_DATA}/${SCENE} \
        -m ${OUTPUT_BASE}/${SCENE}_qp37 \
        -r 2 --mode free --images images \
        --port $((12345 + RANDOM % 1000))
    
    if [ $? -ne 0 ]; then
        echo "❌ Training failed for compressed ${SCENE}"
        continue
    fi
    echo "✅ Compressed training completed for ${SCENE}"
    
    # ============================================
    # 2. 원본 이미지로 학습
    # ============================================
    echo ""
    echo "🟢 [2/6] Training with ORIGINAL images..."
    python train.py --eval \
        -s ${ORIGINAL_DATA}/${SCENE} \
        -m ${OUTPUT_BASE}/${SCENE} \
        -r 2 --mode free --images images \
        --port $((13345 + RANDOM % 1000))
    
    if [ $? -ne 0 ]; then
        echo "❌ Training failed for original ${SCENE}"
        continue
    fi
    echo "✅ Original training completed for ${SCENE}"
    
    # ============================================
    # 3. 렌더링 (둘 다 원본 GT 사용)
    # ============================================
    echo ""
    echo "🎨 [3/6] Rendering compressed model (with original GT)..."
    python render.py \
        -m ${OUTPUT_BASE}/${SCENE}_qp37 \
        --original_images_path ${ORIGINAL_DATA}/${SCENE}/images
    
    if [ $? -ne 0 ]; then
        echo "❌ Rendering failed for compressed ${SCENE}"
    else
        echo "✅ Compressed rendering completed for ${SCENE}"
    fi
    
    echo ""
    echo "🎨 [3/6] Rendering original model (with original GT)..."
    python render.py \
        -m ${OUTPUT_BASE}/${SCENE} \
        --original_images_path ${ORIGINAL_DATA}/${SCENE}/images
    
    if [ $? -ne 0 ]; then
        echo "❌ Rendering failed for original ${SCENE}"
    else
        echo "✅ Original rendering completed for ${SCENE}"
    fi
    
    # ============================================
    # 4. Metrics 계산
    # ============================================
    echo ""
    echo "📊 [4/6] Computing metrics..."
    
    echo "  Metrics for compressed model..."
    python metrics.py -m ${OUTPUT_BASE}/${SCENE}_qp37
    
    echo ""
    echo "  Metrics for original model..."
    python metrics.py -m ${OUTPUT_BASE}/${SCENE}
    
    # ============================================
    # 5. gspread - 압축 모델 결과 업로드
    # ============================================
    echo ""
    echo "📤 [5/6] Uploading compressed results to gspread..."
    JSON_PATH_COMP="${OUTPUT_BASE}/${SCENE}_qp37/results.json"
    POSE_PATH_COMP="${OUTPUT_BASE}/${SCENE}_qp37/test/ours_40000/poses/pose_eval.txt"
    
    if [ -f "$JSON_PATH_COMP" ]; then
        python gspread/gspread-results.py \
            "${JSON_PATH_COMP}" \
            "${POSE_PATH_COMP}" \
            "${OUTPUT_BASE}/${SCENE}_qp37" \
            "${SHEET_NAME}"
        echo "✅ Compressed results uploaded to gspread"
    else
        echo "⚠️ Compressed results.json not found, skipping gspread upload"
    fi
    
    # ============================================
    # 6. gspread - 원본 모델 결과 업로드
    # ============================================
    echo ""
    echo "📤 [6/6] Uploading original results to gspread..."
    JSON_PATH_ORIG="${OUTPUT_BASE}/${SCENE}/results.json"
    POSE_PATH_ORIG="${OUTPUT_BASE}/${SCENE}/test/ours_40000/poses/pose_eval.txt"
    
    if [ -f "$JSON_PATH_ORIG" ]; then
        python gspread/gspread-results.py \
            "${JSON_PATH_ORIG}" \
            "${POSE_PATH_ORIG}" \
            "${OUTPUT_BASE}/${SCENE}" \
            "${SHEET_NAME}"
        echo "✅ Original results uploaded to gspread"
    else
        echo "⚠️ Original results.json not found, skipping gspread upload"
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ Completed: $SCENE"
    echo "=========================================="
    echo ""
    
done

echo ""
echo "=========================================="
echo "🎉 Part 2 Completed!"
echo "Processed: road, sky, stair"
echo "=========================================="
echo ""
echo "📊 Results Summary:"
for SCENE in "${SCENES[@]}"; do
    echo ""
    echo "Scene: $SCENE"
    if [ -f "${OUTPUT_BASE}/${SCENE}_qp37/results.json" ]; then
        echo "  Compressed: ${OUTPUT_BASE}/${SCENE}_qp37/results.json"
    fi
    if [ -f "${OUTPUT_BASE}/${SCENE}/results.json" ]; then
        echo "  Original:   ${OUTPUT_BASE}/${SCENE}/results.json"
    fi
done