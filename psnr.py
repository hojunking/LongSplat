import os
import cv2
import numpy as np
import torch
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from math import log10

# =============================
# 경로 설정
# =============================
PRED_DIR = "renders"
GT_DIR   = "data/compress-x/free/grass/images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# PSNR 함수
# =============================
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * log10(max_pixel / np.sqrt(mse))

# =============================
# 메인
# =============================
pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".png")])
gt_files   = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".JPG")])

gt_set = set([os.path.splitext(f)[0] for f in gt_files])

psnr_list, ssim_list, lpips_list = [], [], []

for pred_file in pred_files:
    name = os.path.splitext(pred_file)[0]
    gt_name = name + ".JPG"

    if name not in gt_set:
        print(f"❌ GT 없음: {pred_file}")
        continue

    pred_path = os.path.join(PRED_DIR, pred_file)
    gt_path   = os.path.join(GT_DIR, gt_name)

    pred_img = cv2.imread(pred_path)
    gt_img   = cv2.imread(gt_path)

    if pred_img is None or gt_img is None:
        print(f"❌ 로딩 실패: {pred_file}")
        continue

    # 해상도 맞추기
    h = min(pred_img.shape[0], gt_img.shape[0])
    w = min(pred_img.shape[1], gt_img.shape[1])
    pred_img = cv2.resize(pred_img, (w, h))
    gt_img   = cv2.resize(gt_img, (w, h))

    # BGR→RGB
    pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    gt_rgb   = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    # numpy PSNR
    psnr_val = compute_psnr(pred_rgb, gt_rgb)
    psnr_list.append(psnr_val)

    # torch tensor 변환
    pred_t = torch.tensor(pred_rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.
    gt_t   = torch.tensor(gt_rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.

    # SSIM — metrics.py와 동일 (torch 기반)
    ssim_val = ssim(pred_t, gt_t).item()
    ssim_list.append(ssim_val)

    # LPIPS — metrics.py와 동일 (함수 기반)
    lpips_val = lpips(pred_t, gt_t, net_type='vgg').item()
    lpips_list.append(lpips_val)

    print(f"[{name}] PSNR={psnr_val:.4f} | SSIM={ssim_val:.4f} | LPIPS={lpips_val:.4f}")

print("\n===== 전체 평균 =====")
print(f"Mean PSNR : {np.mean(psnr_list):.4f}")
print(f"Mean SSIM : {np.mean(ssim_list):.4f}")
print(f"Mean LPIPS: {np.mean(lpips_list):.4f}")