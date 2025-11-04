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
#scene=("Ballroom" "Barn" "Church" "Family" "Francis" "Horse" "Ignatius")
scene=("Church")

# "Museum"
SHEET_NAME="PC1"
for scene in "${scene[@]}"; do
    # SAVE_PATH="output/tanks/compress-x/$scene/"
    SAVE_PATH="output/tanks/compress-o/$scene/"
    
    python train.py --eval -s "./data/compress-x/tnt/$scene" -m "$SAVE_PATH" --port $port --mode tanks
    python render.py -m "$SAVE_PATH" --original_images_path "./data/compress-x/tnt/$scene/images"
    python metrics.py -m "$SAVE_PATH"
    
    # gspread 부분 추가
    JSON_PATH="${SAVE_PATH}/results.json"
    POSE_PATH="${SAVE_PATH}/test/ours_40000/poses/pose_eval.txt"
    python gspread/gspread-results.py "${JSON_PATH}" "${POSE_PATH}" "${SAVE_PATH}" "${SHEET_NAME}"
done