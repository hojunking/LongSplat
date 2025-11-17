
import cv2
import os

# --- 현재 경로 기준 ---
video_path = "/workdir/outputs/free_ema/grass_qp37_compgs_ablation_module1only_mom095/test/ours_40000/videos/poses.mp4"
output_path = "./poses_lastframe.png"

# --- 비디오 열기 ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Cannot open video file: {os.path.abspath(video_path)}")

# --- 총 프레임 수 확인 ---
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞 Total frames: {frame_count}")

# --- 마지막 프레임으로 이동 ---
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)

# --- 프레임 읽기 ---
ret, frame = cap.read()
if not ret:
    raise RuntimeError("❌ Failed to read the last frame.")

# --- 저장 ---
cv2.imwrite(output_path, frame)
print(f"✅ Saved last frame to: {os.path.abspath(output_path)}")

cap.release()
