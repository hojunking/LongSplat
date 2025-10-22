# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, generate_neural_gaussians
from utils.visualize_utils import vis_depth, vis_pose, eval_pose_metrics
from utils.pose_utils import update_pose, smooth_poses_spline, save_transforms
from scene.cameras import Camera
from utils.loss_utils import l1_loss
import cv2
import imageio
from utils.colmap_utils import save_points3D_text, save_imagestxt, save_cameras

def pose_estimation_test(gaussians_pose, view, pipe, bg):
    print("Pose estimation Cam%s" % view.uid)
    pose_iteration = 500

    pose_optimizer = torch.optim.Adam([{"params": [view.cam_trans_delta], "lr": 0.01}, {"params": [view.cam_rot_delta], "lr": 0.01}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=pose_iteration)
    gt_image = view.original_image.cuda()

    progress_bar = tqdm(range(0, pose_iteration), desc="Pose estiamtion progress")
    for iteration in range(pose_iteration):
        voxel_visible_mask = prefilter_voxel(view, gaussians_pose, pipe, bg)
        image = render(view, gaussians_pose, pipe, bg, visible_mask=voxel_visible_mask, retain_grad=True)["render"]
        
        Ll1 = l1_loss(image, gt_image)
        loss = Ll1
        loss.backward()

        with torch.no_grad():
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_pose(view)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
                progress_bar.update(10)
       
    progress_bar.close()

def render_nvs(model_path, name, iteration, views, gaussians, pipeline, background):
    nvs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "nvs")
    videos_path = os.path.join(model_path, name, "ours_{}".format(iteration), "videos")
    
    if not os.path.exists(nvs_path):
        os.makedirs(nvs_path)
    if not os.path.exists(videos_path):
        os.makedirs(videos_path)

    poses_list = []
    for view in views:
        poses_list.append(view.view_world_transform.transpose(0, 1).detach().cpu().numpy())
    poses_list = np.array(poses_list)
    nvs_num = len(poses_list)

    poses_list = np.array(poses_list)
    nvs_pose_list = smooth_poses_spline(poses_list)
    nvs_pose_list = torch.from_numpy(nvs_pose_list).cuda()
    nvs_pose_list = nvs_pose_list.inverse()
    FoVx = views[0].FoVx
    FoVy = views[0].FoVy
    nvs_image_list = []
    nvs_depth_list = []
    gt_list = []
    nvs_views = []
    name_list = []
    for i in tqdm(range(nvs_num), desc="Rendering NVS progress"):
        nvs_view = Camera(colmap_id=i, R=None, T=None, R_gt=None, T_gt=None, FoVx=FoVx, FoVy=FoVy, 
                         image=views[0].original_image, gt_alpha_mask=None, image_name=None, uid=None)
        nvs_view.update_RT(nvs_pose_list[i, :3, :3].transpose(0, 1), nvs_pose_list[i, :3, 3])
        nvs_view.to_final()
        voxel_visible_mask = prefilter_voxel(nvs_view, gaussians, pipeline, background)
        rendering = render(nvs_view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, retain_grad=False)
        torchvision.utils.save_image(rendering["render"], os.path.join(nvs_path, '{0:05d}'.format(i) + ".png"))
        render_img = torch.clamp(rendering["render"], min=0., max=1.)
        render_img = (render_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = nvs_view.original_image[0:3, :, :]
        gt = (gt.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        gt_list.append(gt)
        depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)
        nvs_depth_list.append(depth_map)
        render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
        nvs_image_list.append(render_img)
        nvs_views.append(nvs_view)
        name_list.append('{0:05d}'.format(i))
    imageio.mimwrite(os.path.join(videos_path, 'gt.mp4'), np.stack(gt_list), fps=30, quality=6, output_params=["-f", "mp4"])
    imageio.mimwrite(os.path.join(videos_path, 'nvs_rgb.mp4'), np.stack(nvs_image_list), fps=30, quality=6, output_params=["-f", "mp4"])
    imageio.mimwrite(os.path.join(videos_path, 'nvs_depth.mp4'), np.stack(nvs_depth_list), fps=30, quality=6, output_params=["-f", "mp4"])

# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

# 수정
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               use_compressed_gt=True, original_images_path=None):
    """
    Args:
        use_compressed_gt: True면 학습에 사용된 압축 이미지 사용 (기본값)
                          False면 original_images_path에서 원본 이미지 로드
        original_images_path: 원본 이미지 경로 (use_compressed_gt=False일 때 필수)
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    poses_path = os.path.join(model_path, name, "ours_{}".format(iteration), "poses")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    videos_path = os.path.join(model_path, name, "ours_{}".format(iteration), "videos")
    nvs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "nvs")

    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
    if not os.path.exists(poses_path):
        os.makedirs(poses_path)
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    if not os.path.exists(videos_path):
        os.makedirs(videos_path)
    if not os.path.exists(nvs_path):
        os.makedirs(nvs_path)

    name_list = []
    per_view_dict = {}
    t_list = []
    poses_list = []
    pose_imgs_list = []
    render_imgs_list = []
    render_depth_list = []
    gt_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        rendering = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        poses_list.append(view.view_world_transform.transpose(0, 1).detach().cpu().numpy())

        # 기존: original_image 가져오기
        # gt = view.original_image[0:3, :, :]

        # ===== 수정된 GT 이미지 로드 로직 =====
        if original_images_path is not None:
            # 원본(압축 안한) 이미지 사용
            if idx == 0:
                print(f"📍 Using original UNCOMPRESSED images from: {original_images_path}")
            
            original_image_file = None
            
            # 1. view.image_name으로 찾기
            if view.image_name is not None:
                test_file = os.path.join(original_images_path, view.image_name)
                if os.path.exists(test_file):
                    original_image_file = test_file
                else:
                    # 확장자 변경 시도
                    base_name = os.path.splitext(view.image_name)[0]
                    for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
                        test_file = os.path.join(original_images_path, base_name + ext)
                        if os.path.exists(test_file):
                            original_image_file = test_file
                            break
            
            # 2. image_name이 없거나 못 찾으면 idx로 시도
            if original_image_file is None:
                for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
                    test_file = os.path.join(original_images_path, f"{idx:05d}{ext}")
                    if os.path.exists(test_file):
                        original_image_file = test_file
                        break
            
            # 3. 원본 이미지 로드 및 리사이즈
            if original_image_file is not None and os.path.exists(original_image_file):
                from PIL import Image
                import torchvision.transforms.functional as TF
                original_img = Image.open(original_image_file)
                
                # view의 해상도에 맞게 리사이즈
                target_width = view.original_image.shape[2]
                target_height = view.original_image.shape[1]
                original_img = original_img.resize((target_width, target_height), Image.LANCZOS)
                gt = TF.to_tensor(original_img)[:3, :, :].cuda()
                
                if idx == 0:
                    print(f"  Loaded original image: {original_image_file}")
                    print(f"  Resized to: {target_width}x{target_height}")
            else:
                print(f"WARNING: Original image not found for idx {idx}")
                print(f"  Searched: {original_images_path}")
                print(f"  Falling back to compressed image")
                gt = view.original_image[0:3, :, :]
        else:
            # 기본 동작: 학습에 사용된 압축 이미지 사용
            if idx == 0:
                print("📍 Using COMPRESSED images as GT")
            gt = view.original_image[0:3, :, :]
        # ===== 수정 끝 =====



        # 파일 이름 저장 - 기본
        # name_list.append('{0:05d}'.format(idx))
        # ===== 파일명 결정 (수정) =====
        if view.image_name is not None and view.image_name != "":
            # 원본 이미지 이름 사용 (확장자 제거)
            base_name = os.path.splitext(view.image_name)[0]
            save_name = base_name
            if idx == 0:
                print(f"💾 Saving with original filenames (e.g., {save_name}.png)")
        else:
            # image_name이 없으면 인덱스 사용
            save_name = '{0:05d}'.format(idx)
            if idx == 0:
                print(f"💾 Saving with index filenames (e.g., {save_name}.png)")
        
        name_list.append(save_name)
        # ===== 파일명 결정 끝 =====


        ### 
        image_basename = os.path.splitext(view.image_name)[0]
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, image_basename + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, image_basename + ".png"))
        depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
        np.save(os.path.join(depth_path, view.image_name + '.npy'), rendering['depth'][0].detach().cpu().numpy())
        cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth_map)

        if view.T_gt is not None and idx > 1:
            pose_img = vis_pose(views[0:idx+1])
            pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
            pose_imgs_list.append(pose_img)
        
        render_img = torch.clamp(rendering["render"], min=0., max=1.)
        render_img = (render_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = (gt.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        gt_list.append(gt)
        render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
        render_imgs_list.append(render_img)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)
        render_depth_list.append(depth_map)
    
    # Only evaluate pose metrics if GT poses are available
    has_gt_poses = any(view.T_gt is not None for view in views)
    if has_gt_poses:
        eval_pose_metrics(views, poses_path)
    else:
        print("No GT poses available, skipping pose metrics evaluation")

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    if len(render_imgs_list) > 0:
        if len(pose_imgs_list) > 0:
            imageio.mimwrite(os.path.join(videos_path, 'poses.mp4'), np.stack(pose_imgs_list), fps=30, quality=6)
        imageio.mimwrite(os.path.join(videos_path, 'render.mp4'), np.stack(render_imgs_list), fps=30, quality=6)
        imageio.mimwrite(os.path.join(videos_path, 'depth.mp4'), np.stack(render_depth_list), fps=30, quality=6)

    if name == "train":        
        colmap_path = os.path.join(model_path, name)
        focals = [views[0].intrinsic[0, 0].detach().cpu().numpy()] * len(views)
        focals = np.array(focals)[..., None]
        principal_points = [views[0].intrinsic[:2, 2].detach().cpu().numpy()] * len(views)
        principal_points = np.array(principal_points)
        image_shape = views[0].original_image.shape
        world2cam_np = []
        for cam in views:
            Rt = np.eye(4)
            Rt[:3, :3] = cam.R.t().cpu().numpy()
            Rt[:3, 3] = cam.T.cpu().numpy()
            world2cam_np.append(Rt)
        world2cam_np = np.array(world2cam_np)
        name_list = np.array(name_list)
        save_cameras(focals, principal_points, colmap_path, imgs_shape=image_shape)
        save_imagestxt(world2cam_np, colmap_path, name_list)        

# def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
#                               dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
#         dataset.load_pose = True
#         scene = Scene(dataset, gaussians, load_iteration=iteration)
#         gaussians.eval()

#         bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#         if not os.path.exists(dataset.model_path):
#             os.makedirs(dataset.model_path)
        
#     if not skip_train:
#         with torch.no_grad():
#             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
#             render_nvs(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
#     if not skip_test:
#         for idx, viewpoint in enumerate(scene.getTestCameras()):
#             if "hike_dataset" in dataset.model_path:
#                 test_frame_every = 10
#             elif "Tanks" in dataset.model_path:
#                 test_frame_every = 2 if "Family" in dataset.model_path else 8
#             else:
#                 test_frame_every = 8
#             next_train_idx = viewpoint.uid * test_frame_every - idx
#             if next_train_idx > len(scene.getTrainCameras()) - 1:
#                 next_train_idx = len(scene.getTrainCameras()) - 1
#             ref_viewpoint = scene.getTrainCameras()[next_train_idx]            
#             viewpoint.update_RT(ref_viewpoint.R, ref_viewpoint.T)
#             pose_estimation_test(gaussians, viewpoint, pipeline, background)
#             save_transforms(scene.getTestCameras().copy(), os.path.join(scene.model_path, "cameras_all_test.json"))
#         with torch.no_grad():
#             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, 
                original_images_path : str = None):  # use_compressed_gt 파라미터 제거
    """
    Args:
        dataset: 모델 파라미터
        iteration: 로드할 iteration
        pipeline: 파이프라인 파라미터
        skip_train: train set 렌더링 건너뛰기
        skip_test: test set 렌더링 건너뛰기
        original_images_path: 원본 이미지 경로. None이면 압축 이미지 사용.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, 
                                  dataset.update_depth, dataset.update_init_factor, 
                                  dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                  dataset.appearance_dim, dataset.ratio, 
                                  dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        dataset.load_pose = True
        scene = Scene(dataset, gaussians, load_iteration=iteration)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)


    if not skip_train:
        with torch.no_grad():
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                       gaussians, pipeline, background, 
                       original_images_path=original_images_path)

            render_nvs(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
    if not skip_test:
        for idx, viewpoint in enumerate(scene.getTestCameras()):
            if "hike_dataset" in dataset.model_path:
                test_frame_every = 10
            elif "Tanks" in dataset.model_path:
                test_frame_every = 2 if "Family" in dataset.model_path else 8
            else:
                test_frame_every = 8
            next_train_idx = viewpoint.uid * test_frame_every - idx
            if next_train_idx > len(scene.getTrainCameras()) - 1:
                next_train_idx = len(scene.getTrainCameras()) - 1
            ref_viewpoint = scene.getTrainCameras()[next_train_idx]            
            viewpoint.update_RT(ref_viewpoint.R, ref_viewpoint.T)
            pose_estimation_test(gaussians, viewpoint, pipeline, background)
            save_transforms(scene.getTestCameras().copy(), os.path.join(scene.model_path, "cameras_all_test.json"))
        with torch.no_grad():
            render_set(dataset.model_path, "test", scene.loaded_iter,
                    scene.getTestCameras(), gaussians, pipeline, background,
                    original_images_path=original_images_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")


    # ===== 수정된 인자 (단순화) =====
    parser.add_argument("--original_images_path", type=str, default=None,
                       help="Path to original uncompressed images. If not provided, use compressed images as GT.")
    # ===== 끝 =====
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)


    # ===== render_sets 호출 시 파라미터 전달 =====
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test,
                original_images_path=args.original_images_path)



# (이전) train 명령어
# python train.py --eval -s ./data/free_comp/grass -m outputs/free_comp/grass/baseline_images2_qp37 -r 2 --port 12345 --mode free --images images_2/qp37


# Render + Metrics (압축 GT)
# python render.py -m outputs/free_comp/grass/baseline
# python metrics.py -m outputs/free_comp/grass/baseline

# Render + Metrics (원본 GT)
# python render.py -m outputs/free_comp/grass/baseline_qp37 --original_images_path ./data/free/grass/images
# python metrics.py -m outputs/free_comp/grass/baseline

# 압축 이미지를 GT로 해서 평가
# python render.py -m outputs/free_comp/grass/baseline

# 원본 (압축 안한) 이미지를 GT로 해서 평가
# python render.py -m outputs/free_comp/grass/baseline --original_images_path ./data/free_original/grass/images