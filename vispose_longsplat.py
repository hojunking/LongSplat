###
# python vispose_longsplat.py -s ./data/compress-x/free/road/ -m ./pose-vis/road_qp37/ --output_path ./results_vis/road-longsplat-baseline.png
#  python vispose_longsplat.py -s ./data/compress-x/free/road/ -m ./outputs/free_ema/road_qp37_compgs_mom095_dmu05/  --output_path ./results_vis/road-longsplat-ours.png    

#!/usr/bin/env python3
"""
Simplified script to generate ONLY the final pose visualization image
Shows all camera positions in one image
"""

import os
import torch
import cv2
import json
from argparse import ArgumentParser

# LongSplat imports
from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.visualize_utils import vis_pose
import numpy as np

def load_scene_and_cameras(dataset: ModelParams, iteration: int):
    """Load the scene and camera data"""
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.voxel_size,
            dataset.update_depth, dataset.update_init_factor,
            dataset.update_hierachy_factor, dataset.use_feat_bank,
            dataset.appearance_dim, dataset.ratio,
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist
        )
        dataset.load_pose = True
        scene = Scene(dataset, gaussians, load_iteration=iteration)
        
    return scene


# def generate_final_pose_image(views, output_path):
#     """
#     Generate final pose visualization image with all cameras
    
#     Args:
#         views: List of camera views
#         output_path: Full path to save the image
#     """
#     # Check if GT poses are available
#     has_gt_poses = any(v.T_gt is not None for v in views)
#     if not has_gt_poses:
#         print("⚠️  Warning: No GT poses available")
#         return None
    
#     print(f"📸 Generating final pose visualization with all {len(views)} cameras...")
    
#     # Generate pose visualization with ALL cameras
#     pose_img = vis_pose(views)
#     pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
    
#     # Save the image
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     cv2.imwrite(output_path, pose_img)
    
#     print(f"✅ Saved final pose image to: {output_path}")
    
#     return output_path

def generate_final_pose_image(views, output_path):
    """
    Generate final pose visualization image with all cameras
    """
    # Check if GT poses are available
    has_gt_poses = any(v.T_gt is not None for v in views)
    if not has_gt_poses:
        print("⚠️  Warning: No GT poses available")
        return None
    
    # ==================== 디버깅 ====================
    print("\n" + "="*80)
    print("🔍 [LONGSPLAT] GT POSE COMPUTATION (First 3 poses)")
    print("="*80)
    
    for idx in range(min(3, len(views))):
        view = views[idx]
        print(f"\n--- Pose {idx} (image: {view.image_name}) ---")
        
        if hasattr(view, 'R_gt') and hasattr(view, 'T_gt'):
            R_gt = view.R_gt
            T_gt = view.T_gt
            
            print(f"R_gt type: {type(R_gt)}")
            print(f"R_gt:\n{R_gt.cpu().numpy() if isinstance(R_gt, torch.Tensor) else R_gt}")
            print(f"T_gt: {T_gt.cpu().numpy() if isinstance(T_gt, torch.Tensor) else T_gt}")
            
            # LongSplat's vis_pose does this:
            # pose_gt = np.linalg.inv(getWorld2View_np(camera.R_gt, camera.T_gt))
            
            # Import getWorld2View_np from utils
            from utils.graphics_utils import getWorld2View_np
            
            w2c = getWorld2View_np(R_gt, T_gt)
            print(f"\nW2C matrix (from getWorld2View_np):\n{w2c}")
            
            c2w = np.linalg.inv(w2c)
            print(f"\nC2W matrix (after inverse):\n{c2w}")
            print(f"C2W position: {c2w[:3, 3]}")
    
    print("\n" + "="*80 + "\n")
    # ================================================
    
    print(f"📸 Generating final pose visualization with all {len(views)} cameras...")
    
    # Generate pose visualization with ALL cameras
    pose_img = vis_pose(views)
    pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, pose_img)
    
    print(f"✅ Saved final pose image to: {output_path}")
    
    return output_path


import glob

def inspect_model_path(model_path):
    print("\n" + "="*80)
    print(f"📂 Inspecting model path: {model_path}")
    print("="*80)

    paths_to_check = [
        "cameras.json",
        "cam_info.json",
        "train.json",
        "test.json",
        "cameras/*.json",
        "cam_info/*.json",
        "point_cloud/*/point_cloud.ply",
        "model/*/*.pth",
        "model/*.pth"
    ]

    found_any = False
    for pattern in paths_to_check:
        full_pattern = os.path.join(model_path, pattern)
        matched = glob.glob(full_pattern)

        if matched:
            found_any = True
            for m in matched:
                print(f" ✔ Found: {m}")
        else:
            print(f" ✘ Missing: {full_pattern}")

    if not found_any:
        print("⚠️ No expected files found in this model_path. Maybe wrong folder?")
    print("="*80 + "\n")

def main():
    parser = ArgumentParser(description="Generate final pose visualization image")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int,
                       help="Iteration to load (-1 for latest)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output image path (default: model_path/train/ours_X/pose_final.png)")
    parser.add_argument("--use_test", action="store_true",
                       help="Use test cameras instead of train cameras")
    
    args = get_combined_args(parser)
    
    # Ensure attributes exist
    if not hasattr(args, 'output_path'):
        args.output_path = None
    if not hasattr(args, 'use_test'):
        args.use_test = False
    
    # Load scene and cameras
    print(f"🔄 Loading scene from: {args.model_path}")


    # 🔥 추가: model_path 내부 파일 검사
    inspect_model_path(args.model_path)



    dataset = model.extract(args)
    scene = load_scene_and_cameras(dataset, args.iteration)
    
    # Get cameras
    if args.use_test:
        views = scene.getTestCameras()
        camera_type = "test"
    else:
        views = scene.getTrainCameras()
        camera_type = "train"
    
    print(f"📷 Loaded {len(views)} {camera_type} cameras")
    
    # Set output path
    if args.output_path is None:
        iteration_str = f"ours_{scene.loaded_iter}" if scene.loaded_iter >= 0 else "latest"
        output_dir = os.path.join(args.model_path, camera_type, iteration_str)
        args.output_path = os.path.join(output_dir, "pose_final.png")
    
    # Generate final pose image
    saved_path = generate_final_pose_image(views, args.output_path)
    
    if saved_path:
        print(f"\n🎉 Done!")
        print(f"   Image saved to: {saved_path}")


if __name__ == "__main__":
    main()