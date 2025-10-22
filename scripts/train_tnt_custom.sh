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
scene=("Barn" "Family" "Museum")
# "Museum"
SHEET_NAME="gspread"
for scene in "${scene[@]}"; do
    SAVE_PATH="outputs/tanks/base_init_frame=5_2/$scene/"
    
    python train.py --eval -s "./data/Tanks/$scene" -m "$SAVE_PATH" --port $port --mode tanks --init_frame_num 5
    python render.py -m "$SAVE_PATH"
    python metrics.py -m "$SAVE_PATH"
    
    # gspread 부분 추가
    JSON_PATH="${SAVE_PATH}/results.json"
    POSE_PATH="${SAVE_PATH}/test/ours_40000/poses/pose_eval.txt"
    python gspread/gspread-results.py "${JSON_PATH}" "${POSE_PATH}" "${SAVE_PATH}" "${SHEET_NAME}"


done