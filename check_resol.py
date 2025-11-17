from PIL import Image

# 이미지 경로
# image_path = '/home/knuvi/Desktop/song/LongSplat/output/tanks/compress-o/Church_qp32/test/ours_40000/renders/009404.png'
image_path ='/home/knuvi/Desktop/song/LongSplat/outputs/free_ema/stair_qp37_compgs_ablation_module1only_mom095/test/ours_40000/renders/DSC06389.png'
# 이미지 열기
img = Image.open(image_path)

# 해상도 출력
print(f"Image resolution: {img.size}")  # (width, height) 형식으로 출력


print("nope-nerf yaml파일에 넣을 때는 위에 출력 결과에서 반대 순서로 넣어야 함.")