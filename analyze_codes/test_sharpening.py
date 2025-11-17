#!/usr/bin/env python3
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# 1️⃣ Soft Sharpen 함수 정의 (밝기 채널만 강화)
# ============================================
def soft_sharpen_image(img_torch):
    """Y 채널 기반 Soft Sharpen (PSNR 유지형)"""
    img_np = (img_torch.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)

    # 🔸 부드러운 커널 (엣지 강조 강도 낮춤)
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3.0, -0.5],
                       [0, -0.5, 0]])

    yuv[...,0] = cv2.filter2D(yuv[...,0], -1, kernel)  # Y(밝기) 채널만 sharpen
    img_np = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    img_torch_new = torch.tensor(img_np.transpose(2,0,1)/255., dtype=torch.float32, device=img_torch.device)
    return img_torch_new.clamp(0,1)

# ============================================
# 2️⃣ 경로 설정
# ============================================
comp_path = "/workdir/data/compress-o/free/qp37/grass/images/DSC07880.JPG"
orig_path = "/workdir/data/compress-x/free/grass/images/DSC07880.JPG"
output_dir = "/workdir/analyze_codes/plots"
os.makedirs(output_dir, exist_ok=True)

# ============================================
# 3️⃣ 이미지 로드 및 크기 정규화
# ============================================
img_bgr_comp = cv2.imread(comp_path)
img_bgr_orig = cv2.imread(orig_path)
if img_bgr_comp is None or img_bgr_orig is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

img_rgb_comp = cv2.cvtColor(img_bgr_comp, cv2.COLOR_BGR2RGB)
img_rgb_orig = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)

h = min(img_rgb_comp.shape[0], img_rgb_orig.shape[0])
w = min(img_rgb_comp.shape[1], img_rgb_orig.shape[1])
img_rgb_comp = img_rgb_comp[:h, :w]
img_rgb_orig = img_rgb_orig[:h, :w]

img_torch_comp = torch.tensor(img_rgb_comp.transpose(2,0,1)/255., dtype=torch.float32)
img_torch_orig = torch.tensor(img_rgb_orig.transpose(2,0,1)/255., dtype=torch.float32)

# ============================================
# 4️⃣ Soft Sharpen 적용
# ============================================
img_soft = soft_sharpen_image(img_torch_comp)

# ============================================
# 5️⃣ PSNR / SSIM 계산
# ============================================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

np_comp = img_rgb_comp.astype(np.float32) / 255.0
np_orig = img_rgb_orig.astype(np.float32) / 255.0
np_soft = img_soft.permute(1,2,0).cpu().numpy()

psnr_comp = psnr(np_comp, np_orig)
psnr_soft = psnr(np_soft, np_orig)
ssim_comp = ssim(np_comp, np_orig)
ssim_soft = ssim(np_soft, np_orig)

print(f"📊 PSNR/SSIM vs Original:")
print(f" - Compressed: PSNR={psnr_comp:.2f}, SSIM={ssim_comp:.4f}")
print(f" - Soft-Sharp: PSNR={psnr_soft:.2f}, SSIM={ssim_soft:.4f}")

# ============================================
# 6️⃣ 시각화 및 저장
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(img_rgb_orig)
axes[0].set_title("Original (Uncompressed)")
axes[0].axis("off")

axes[1].imshow(img_rgb_comp)
axes[1].set_title(f"Compressed\nPSNR={psnr_comp:.2f}, SSIM={ssim_comp:.3f}")
axes[1].axis("off")

axes[2].imshow(np_soft)
axes[2].set_title(f"Soft-Sharpened\nPSNR={psnr_soft:.2f}, SSIM={ssim_soft:.3f}")
axes[2].axis("off")

plt.tight_layout()

base_name = os.path.splitext(os.path.basename(comp_path))[0]
out_path = os.path.join(output_dir, f"{base_name}_softsharpen_compare.png")
plt.savefig(out_path, dpi=300)
plt.show()

print("✅ Saved results to:", out_path)
