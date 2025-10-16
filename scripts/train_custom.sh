# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}
ulimit -n 4096

port=$(rand 10000 30000)

SCENES=(
    "/workdir/dataset/DL3DV10K/compress-o/vggt/images_video2_qp37"
    "/workdir/dataset/DL3DV10K/compress-o/vggt/images_video4_qp37"
    "/workdir/dataset/DL3DV10K/compress-o/vggt/images_video5_qp37"
    "/workdir/dataset/DL3DV10K/compress-o/vggt/images_video6_qp37"
    "/workdir/dataset/DL3DV10K/compress-o/vggt/images_video7_qp37"
    "/workdir/dataset/DL3DV10K/compress-o/vggt/images_video9_qp37"
)

PREFIX="outputs"
MIDDLE_PATH="DL3DV10K/compress-o/vggt/adaptive03"
SAVE_PATH="${PREFIX}/${MIDDLE_PATH}"
SHEET_NAME="gspread"

for SCENE_PATH in "${SCENES[@]}"; do
    EXP_NAME=$(basename "${SCENE_PATH}")  # 폴더명 추출

    echo "==============================================="
    echo "[Processing Scene] ${EXP_NAME}"
    echo "==============================================="

    # -----------------------------------------------
    # 1) Training
    # -----------------------------------------------

    timestamp=$(date "+%Y-%m-%d_%H:%M:%S")
    python train.py \
    -s "${SCENE_PATH}" \
    -m "${SAVE_PATH}/${EXP_NAME}" \
    --eval \
    --port $port \
    --qp_json "${SCENE_PATH}/qp.json" \
    --lower_bound 0.3
    
    python render.py -m "${SAVE_PATH}/${EXP_NAME}" --skip_train
    python metrics.py -m "${SAVE_PATH}/${EXP_NAME}" --qp

    JSON_PATH="${SAVE_PATH}/${EXP_NAME}/results.json"
    POSE_PATH="${SAVE_PATH}/${EXP_NAME}/test/ours_40000/poses/pose_eval.txt"
    python gspread/gspread-results.py "${JSON_PATH}" "${POSE_PATH}" "${SAVE_PATH}/${EXP_NAME}" "${SHEET_NAME}"
    
    echo "Completed processing for ${EXP_NAME}"
    echo ""
done

echo "==============================================="
echo "All listed scenes processed!"
echo "==============================================="