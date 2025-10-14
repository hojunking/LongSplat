# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# function rand(){
#     min=$1
#     max=$(($2-$min+1))
#     num=$(date +%s%N)
#     echo $(($num%$max+$min))  
# }
# ulimit -n 4096

# port=$(rand 10000 30000)

# scene=("images_video2_qp37" "images_video4_qp37" "images_video5_qp37" "images_video6_qp37" "images_video7_qp37" "images_video9_qp37")

# for scene in "${scene[@]}"; do
#     timestamp=$(date "+%Y-%m-%d_%H:%M:%S")
#     #python train.py --eval -s "./dataset/DL3DV10K/compress-o/vggt/$scene" -m "outputs/DL3DV10K/compress-o/vggt/$scene/$timestamp" --port $port
#     python render.py -m "outputs/DL3DV10K/compress-o/vggt/$scene/$timestamp" --skip_train
#     python metrics.py -m "outputs/DL3DV10K/compress-o/vggt/$scene/$timestamp" --qp
# done

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}
ulimit -n 4096

port=$(rand 10000 30000)

scene=("images_video2_qp37" "images_video4_qp37" "images_video5_qp37" "images_video6_qp37" "images_video7_qp37" "images_video9_qp37")

for scene in "${scene[@]}"; do
    timestamp=$(date "+%Y-%m-%d_%H:%M:%S")
    python train.py \
    --eval \
    -s "./dataset/DL3DV10K/compress-o/vggt/$scene" \
    -m "outputs/DL3DV10K/compress-o/vggt/adaptive05/$scene" \
    --port $port \
    --qp_json "./dataset/DL3DV10K/compress-o/vggt/$scene/qp.json"
    
    python render.py -m "outputs/DL3DV10K/compress-o/vggt/adaptive05/$scene" --skip_train
    python metrics.py -m "outputs/DL3DV10K/compress-o/vggt/adaptive05/$scene" --qp
done