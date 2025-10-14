#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import subprocess
import numpy as np
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

# NEW: Render(PNG) vs Original GT(JPG) 비교를 위한 새로운 함수
def readImages_jpg_gt(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        if '.ipynb' in fname or not fname.lower().endswith('.png'):
            continue
        
        render_path = renders_dir / fname
        
        gt_basename = os.path.splitext(fname)[0]
        gt_filename = gt_basename + ".jpg"
        gt_path = gt_dir / gt_filename

        if os.path.exists(gt_path):
            try:
                render = Image.open(render_path)
                gt = Image.open(gt_path)
                
                renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
                gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
                image_names.append(fname)
            except Exception as e:
                print(f"Warning: Could not read image pair ({fname}, {gt_filename}). Error: {e}")
    return renders, gts, image_names

def evaluate(model_paths, use_qp_logic):

    ORIGINAL_GT_BASE_PATH = "/workdir/dataset/DL3DV10K/compress-x" #GT 참조 scene path

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                method_dir = test_dir / method
                renders_dir = method_dir / "renders"

                # NEW LOGIC START: QP 모드와 일반 모드를 분기 처리
                eval_tasks = []
                if use_qp_logic and "_qp" in scene_dir:
                    print("INFO: [QP Mode] Dual evaluation started.")
                    # 1. 원본 GT와 비교하는 태스크 추가
                    qp_index = scene_dir.find("_qp")
                    base_scene_name = Path(scene_dir[:qp_index]).name
                    #original_gt_dir = Path(ORIGINAL_GT_BASE_PATH) / base_scene_name / "color"
                    original_gt_dir = Path(ORIGINAL_GT_BASE_PATH) / base_scene_name
                    
                    eval_tasks.append( (original_gt_dir, "_vs_OrigGT", "jpg_gt") )
                    
                    # 2. 압축 GT와 비교하는 태스크 추가
                    compressed_gt_dir = method_dir / "gt"
                    eval_tasks.append( (compressed_gt_dir, "_vs_CompGT", "png_gt") )
                else:
                    # 3. 일반 모드 (기존 방식) 태스크 추가
                    print("INFO: [Default Mode] Single evaluation started.")
                    default_gt_dir = method_dir / "gt"
                    eval_tasks.append( (default_gt_dir, "", "png_gt") )

                for gt_dir, suffix, gt_type in eval_tasks:
                    method_key = method + suffix
                    print(f" -> Evaluating with GT: {gt_dir}")
                    
                    # method key 초기화
                    full_dict[scene_dir][method_key] = {}
                    per_view_dict[scene_dir][method_key] = {}

                    # GT 타입에 따라 적절한 함수를 호출
                    if gt_type == "jpg_gt":
                        renders, gts, image_names = readImages(renders_dir, gt_dir)
                    else: # "png_gt" 또는 기본값
                        renders, gts, image_names = readImages(renders_dir, gt_dir)

                    if not renders:
                        print(" -> No matching images found. Skipping.")
                        continue

                    # 이미지를 불러오지 못했으면 다음 태스크로 넘어감
                    if not renders:
                        continue

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
                    print("")

                    full_dict[scene_dir][method_key].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                             "PSNR": torch.tensor(psnrs).mean().item(),
                                                             "LPIPS": torch.tensor(lpipss).mean().item()})
                    per_view_dict[scene_dir][method_key].update({"SSIM": {name: s.item() for s, name in zip(ssims, image_names)},
                                                                 "PSNR": {name: p.item() for p, name in zip(psnrs, image_names)},
                                                                 "LPIPS": {name: l.item() for l, name in zip(lpipss, image_names)}})
                # NEW LOGIC END

            # JSON 저장 로직은 수정할 필요 없이 그대로 작동합니다.
            with open(str(Path(scene_dir) / "results.json"), 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(str(Path(scene_dir) / "per_view.json"), 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}. Error: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--qp', action='store_true', 
                        help="If specified, QP-models are evaluated against both original and compressed GTs.")
    args = parser.parse_args()
    evaluate(args.model_paths, args.qp)
