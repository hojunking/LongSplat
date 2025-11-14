import os
import json
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from tqdm import tqdm

def load_and_match_images(render_dir, gt_dir):
    render_dir = Path(render_dir)
    gt_dir = Path(gt_dir)

    render_files = sorted([f for f in os.listdir(render_dir) if f.lower().endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith((".jpg", ".jpeg"))])

    matched = []
    gt_dict = {Path(f).stem: f for f in gt_files}

    for r in render_files:
        stem = Path(r).stem
        if stem in gt_dict:
            matched.append((render_dir / r, gt_dir / gt_dict[stem]))

    return matched

def load_pair(render_path, gt_path):
    render = Image.open(render_path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB")

    if render.size != gt.size:
        gt = gt.resize(render.size, Image.BICUBIC)

    render_t = tf.to_tensor(render).unsqueeze(0).cuda()
    gt_t = tf.to_tensor(gt).unsqueeze(0).cuda()

    return render_t, gt_t

def evaluate_render_vs_gt(render_dir, gt_dir, output_json):

    matched_files = load_and_match_images(render_dir, gt_dir)

    if len(matched_files) == 0:
        print("❌ 매칭되는 파일이 없습니다.")
        return

    ssims, psnrs, lpipss = [], [], []

    print(f"총 매칭 파일 수: {len(matched_files)}\n")

    for render_path, gt_path in tqdm(matched_files, desc="Evaluation"):
        render_t, gt_t = load_pair(render_path, gt_path)

        s = ssim(render_t, gt_t)
        p = psnr(render_t, gt_t)
        l = lpips(render_t, gt_t, net_type='vgg')

        ssims.append(float(s))
        psnrs.append(float(p))
        lpipss.append(float(l))

    # ⭐ gspread 호환 포맷
    results = {
        "ours_40000": {
            "SSIM": sum(ssims) / len(ssims),
            "PSNR": sum(psnrs) / len(psnrs),
            "LPIPS": sum(lpipss) / len(lpipss),
        }
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as fp:
        json.dump(results, fp, indent=2)

    print("\n✔ 평가 완료!")
    print("결과 저장:", str(output_json))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python evaluate_photometrics.py <render_dir> <gt_dir> <output_json>")
        sys.exit(1)

    render_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    output_json = sys.argv[3]

    evaluate_render_vs_gt(render_dir, gt_dir, output_json)
