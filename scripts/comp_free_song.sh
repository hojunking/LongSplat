

#!/bin/bash
SCENES=("lab" "grass" "hydrant")
#
#SCENES=("hydrant" "lab" "pillar" "road" "sky" "stair")
QP_LEVELS=("qp37")
COMPRESSED_DATA="/workdir/data/compress-o/free/qp37"
ORIGINAL_DATA="/workdir/data/compress-x/free"
OUTPUT_BASE="outputs_song/free/dropout02-re"
SHEET_NAME="PC2"

for SCENE in "${SCENES[@]}"; do
    for QP in "${QP_LEVELS[@]}"; do
        SCENE_QP="${SCENE}_${QP}"
        COMP_PATH="${COMPRESSED_DATA}/${SCENE}"
        
        python train_dropout.py --eval \
            -s ${COMP_PATH} \
            -m ${OUTPUT_BASE}/${SCENE_QP} \
            -r 2 --mode free \
            --d_mu 0.2  \
            --port $((12345 + RANDOM % 1000))
        
        # [ $? -ne 0 ] && continue
        
        python render.py \
            -m ${OUTPUT_BASE}/${SCENE_QP} \
            --original_images_path ${ORIGINAL_DATA}/${SCENE}/images
        
        [ $? -ne 0 ] && continue
        
        python metrics.py -m ${OUTPUT_BASE}/${SCENE_QP}
        
        JSON_PATH="${OUTPUT_BASE}/${SCENE_QP}/results.json"
        POSE_PATH="${OUTPUT_BASE}/${SCENE_QP}/test/ours_40000/poses/pose_eval.txt"
        
        if [ -f "$JSON_PATH" ]; then
            python gspread/gspread-results.py \
                "${JSON_PATH}" \
                "${POSE_PATH}" \
                "${OUTPUT_BASE}/${SCENE_QP}" \
                "${SHEET_NAME}"
        fi
    done
done

#!/bin/bash
SCENES=("lab" "grass" "hydrant")
#
#SCENES=("hydrant" "lab" "pillar" "road" "sky" "stair")
QP_LEVELS=("qp37")
COMPRESSED_DATA="/workdir/data/compress-o/free/qp37"
ORIGINAL_DATA="/workdir/data/compress-x/free"
OUTPUT_BASE="outputs_song/free/pose-grad02"
SHEET_NAME="PC2"

for SCENE in "${SCENES[@]}"; do
    for QP in "${QP_LEVELS[@]}"; do
        SCENE_QP="${SCENE}_${QP}"
        COMP_PATH="${COMPRESSED_DATA}/${SCENE}"
        
        python train_pose.py --eval \
            -s ${COMP_PATH} \
            -m ${OUTPUT_BASE}/${SCENE_QP} \
            -r 2 --mode free \
            --p_mu 0.2  \
            --port $((12345 + RANDOM % 1000))
        
        # [ $? -ne 0 ] && continue
        
        python render.py \
            -m ${OUTPUT_BASE}/${SCENE_QP} \
            --original_images_path ${ORIGINAL_DATA}/${SCENE}/images
        
        [ $? -ne 0 ] && continue
        
        python metrics.py -m ${OUTPUT_BASE}/${SCENE_QP}
        
        JSON_PATH="${OUTPUT_BASE}/${SCENE_QP}/results.json"
        POSE_PATH="${OUTPUT_BASE}/${SCENE_QP}/test/ours_40000/poses/pose_eval.txt"
        
        if [ -f "$JSON_PATH" ]; then
            python gspread/gspread-results.py \
                "${JSON_PATH}" \
                "${POSE_PATH}" \
                "${OUTPUT_BASE}/${SCENE_QP}" \
                "${SHEET_NAME}"
        fi
    done
done
