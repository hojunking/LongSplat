import os
import cv2
import numpy as np
from tqdm import tqdm

GT_DIR = "/workdir/outputs/free_ema/grass_qp37_compgs_mom095_dmu05/test/ours_40000/gt"
RENDER_DIR = "/workdir/outputs/free_ema/grass_qp37_compgs_mom095_dmu05/test/ours_40000/renders"
OUT_DIR = "/workdir/foreccv/grass_qp37_compgs_mom095_dmu05/diff_maps"

os.makedirs(OUT_DIR, exist_ok=True)

# 공통 파일 이름
gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".png")])

for fname in tqdm(gt_files):
    gt_path = os.path.join(GT_DIR, fname)
    render_path = os.path.join(RENDER_DIR, fname)

    if not os.path.exists(render_path):
        print(f"[WARN] Missing render: {fname}")
        continue

    # BGR, uint8
    gt = cv2.imread(gt_path)
    render = cv2.imread(render_path)

    if gt.shape != render.shape:
        print(f"[WARN] Shape mismatch: {fname}")
        continue

    # float 변환
    gt = gt.astype(np.float32)
    render = render.astype(np.float32)

    # 픽셀 단위 절대 차이
    diff = np.abs(gt - render)        # (H, W, 3)
    diff_map = diff.mean(axis=2)      # (H, W)

    # 1) 고정 기준으로 클립
    MAX_ERR = 40.0
    diff_clipped = np.clip(diff_map, 0, MAX_ERR)

    # 2) gamma 강조
    diff_norm = diff_clipped / MAX_ERR
    diff_gamma = diff_norm ** 2.5

    # 3) colormap
    diff_vis = (diff_gamma * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)


    # 저장
    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, diff_color)
