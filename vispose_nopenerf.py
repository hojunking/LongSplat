# python vispose_nopenerf.py     --checkpoint ./vis_test/model_pose.pt     --gt_images_bin ./data/compress-x/free/road/sparse/0/images.bin     --output_path ./pose.png   --zoom 2
#  python vispose_nopenerf.py --checkpoint ./vis_test/model_pose-sky-nopenerf.pt --gt_images_bin ./data/compress-x/free/road/sparse/0/images.bin --output_path ./results_vis/sky-nopenerf-baseline.png  --zoom 2



#  python vispose_nopenerf.py --checkpoint ./vis_test/model_pose-sky-nopenerf.pt --gt_images_bin ./data/compress-x/free/sky/sparse/0/images.bin --output_path ./results_vis/sky-nopenerf-baseline.png  --zoom 1.8


#!/usr/bin/env python3
"""
Pose visualization using EVO library - EXACT LongSplat reproduction
Uses the same evo.tools.plot functions as LongSplat
"""

import os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import struct
import collections
import copy

# EVO imports (same as LongSplat)
from evo.core.trajectory import PosePath3D
from evo.tools import plot

# COLMAP data structures
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    """Read COLMAP images.bin file"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                          format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                      format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                  tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = BaseImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def axis_angle_to_rotation_matrix(axis_angle):
    """Convert axis-angle to rotation matrix"""
    batch_size = axis_angle.shape[0]
    theta = torch.norm(axis_angle, dim=1, keepdim=True)
    
    theta_safe = torch.where(theta > 1e-8, theta, torch.ones_like(theta))
    axis = axis_angle / theta_safe
    
    K = torch.zeros(batch_size, 3, 3, dtype=axis_angle.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    
    eye = torch.eye(3, dtype=axis_angle.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    R = eye + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
    
    zero_rotation = (theta.squeeze(-1) < 1e-8)
    R[zero_rotation] = eye[zero_rotation]
    
    return R

def getWorld2View_np(R, T):
    """
    EXACT copy of LongSplat's getWorld2View_np function
    """
    pose = np.eye(4)
    if isinstance(R, torch.Tensor):
        pose[0:3, 0:3] = R.t().cpu().numpy()  # ✅ .t() 추가!
        pose[0:3, 3] = T.cpu().numpy()
    else:
        pose[0:3, 0:3] = R.T  # ✅ .T 추가!
        pose[0:3, 3] = T
    return pose


def load_gt_poses(images_bin_path, test_every=9):
    """Load GT poses from COLMAP images.bin"""
    print(f"📂 Loading GT poses from: {images_bin_path}")
    
    images = read_images_binary(images_bin_path)
    sorted_images = sorted(images.values(), key=lambda x: x.name)
    
    print(f"   Loaded {len(sorted_images)} total GT poses (train + test)")
    
    # Filter out test frames
    if test_every <= 1:
        # test_every=1 means no test frames, use all frames
        train_images = sorted_images
        print(f"   Using all {len(train_images)} frames (no test split)")
    else:
        train_indices = [i for i in range(len(sorted_images)) if i % test_every != 0]
        train_images = [sorted_images[i] for i in train_indices]
        print(f"   Filtered to {len(train_images)} train GT poses (removed every {test_every}th frame)")
    
    poses_se3 = []
    
    # ==================== 디버깅 ====================
    print("\n" + "="*80)
    print("🔍 [NOPENERF] GT POSE COMPUTATION (First 3 poses)")
    print("="*80)
    
    for idx, img in enumerate(train_images[:3]):
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        
        print(f"\n--- Pose {idx} (image: {img.name}) ---")
        print(f"R (from COLMAP):\n{R}")
        print(f"t (from COLMAP): {t}")
        
        # Current method: R as-is, inverse
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        
        print(f"\nW2C matrix:\n{w2c}")
        
        c2w = np.linalg.inv(w2c)
        
        print(f"\nC2W matrix (after inverse):\n{c2w}")
        print(f"C2W position: {c2w[:3, 3]}")
        
        poses_se3.append(c2w)
    
    # 나머지 포즈들
    for img in train_images[3:]:
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        poses_se3.append(c2w)
    
    print("\n" + "="*80 + "\n")
    # ================================================
    
    return poses_se3

def load_learned_poses(checkpoint_path):
    """Load learned poses from nope-nerf checkpoint"""
    print(f"📂 Loading learned poses from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ckpt['model']
    
    axis_angle = model['r']  # [N, 3]
    translations = model['t']  # [N, 3]
    
    rotations = axis_angle_to_rotation_matrix(axis_angle)  # [N, 3, 3]
    
    print(f"   Loaded {axis_angle.shape[0]} learned poses")
    
    # Convert to SE3 matrices using LongSplat's convention
    poses_se3 = []
    for i in range(len(rotations)):
        R = rotations[i]  # torch tensor
        T = translations[i]  # torch tensor
        
        # Use LongSplat's getWorld2View_np, then inverse to get C2W
        w2c = getWorld2View_np(R, T)
        c2w = np.linalg.inv(w2c)
        
        poses_se3.append(c2w)
    
    return poses_se3


def plot_pose_longsplat(ref_poses, est_poses, figsize=(6, 6), elev=10., azim=45, zoom_factor=0.5):
    """
    EXACT reproduction of LongSplat's plot_pose function
    
    Args:
        figsize: Figure size in inches (default: (6, 6))
        elev: Elevation angle in degrees (default: 10)
        azim: Azimuth angle in degrees (default: 45)
        zoom_factor: Axis range multiplier (smaller = more zoomed in, default: 0.5)
    """
    plt.rc('legend', fontsize=20)
    
    print(f"   Using all {len(ref_poses)} poses for visualization")
    
    # Create trajectories
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    
    # Re-enable alignment
    traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    
    # Create figure with custom size
    fig = plt.figure(figsize=figsize)
    traj_by_label = {
        "Ours": traj_est_aligned,
        "Ground-truth": traj_ref
    }
    
    plot_mode = plot.PlotMode.xyz
    ax = fig.add_subplot(111, projection="3d")
    
    # Set background color to match LongSplat
    ax.set_facecolor('#eaeaf2')
    fig.patch.set_facecolor('#eaeaf2')
    
    # Remove tick LABELS but keep tick marks visible
    ax.xaxis.set_tick_params(labelbottom=False, length=6, width=1, color='black')
    ax.yaxis.set_tick_params(labelleft=False, length=6, width=1, color='black')
    ax.zaxis.set_tick_params(labelleft=False, length=6, width=1, color='black')
    
    # Grid styling to match LongSplat
    ax.grid(True, color='white', linewidth=0.8, linestyle='-', alpha=1.0)
    
    colors = ['r', 'b']
    styles = ['-', '--']
    
    # Plot trajectories using evo with thicker lines
    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj, styles[idx], colors[idx], label, alpha=1.0)

        # --- Add: make GT line thicker ---
        # if label == "Ground-truth":        # 또는 idx == 1
        ax.lines[-1].set_linewidth(2)  # 원하는 값: 2~5 추천

    # Zoom in by reducing axis limits (make trajectories more compact)
    # Get the current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    # Reduce the range to zoom in (smaller range = more zoomed in)
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    z_center = (zlim[0] + zlim[1]) / 2
    
    x_range = (xlim[1] - xlim[0]) * zoom_factor / 2
    y_range = (ylim[1] - ylim[0]) * zoom_factor / 2
    z_range = (zlim[1] - zlim[0]) * zoom_factor / 2
    
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range)
    ax.set_zlim(z_center - z_range, z_center + z_range)
    
    # Move legend to upper right
    # ax.legend(loc='upper right', frameon=True)
    ax.legend(
        loc='upper right',
        frameon=True,
        bbox_to_anchor=(1.0, 0.99),
        fontsize=26,           # 텍스트 크기

        borderpad=0.6     # ← 기본 0.4 → 1.2로 키우면 박스 커짐
    )

    # View angle (adjustable)
    ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    
    # Convert to image (same as LongSplat)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
    plt.close(fig)
    
    return img


def main():
    parser = ArgumentParser(description="Visualize nope-nerf poses using EVO (LongSplat style)")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model_pose_*.pt")
    parser.add_argument("--gt_images_bin", type=str, required=True,
                       help="Path to COLMAP images.bin")
    parser.add_argument("--output_path", type=str, default="pose_evo.png",
                       help="Output image path")
    parser.add_argument("--test_every", type=int, default=2,
                       help="Test frame interval")
    parser.add_argument("--figsize", type=float, default=6.0,
                       help="Figure size in inches (default: 6.0)")
    parser.add_argument("--elev", type=float, default=10.0,
                       help="Elevation angle in degrees (default: 10)")
    parser.add_argument("--azim", type=float, default=45.0,
                       help="Azimuth angle in degrees (default: 45)")
    parser.add_argument("--zoom", type=float, default=0.5,
                       help="Zoom factor - smaller = more zoomed in (default: 0.5)")
    
    args = parser.parse_args()
    
    # Load poses
    gt_poses_se3 = load_gt_poses(args.gt_images_bin, test_every=args.test_every)
    learned_poses_se3 = load_learned_poses(args.checkpoint)
    
    # Match counts
    min_len = min(len(gt_poses_se3), len(learned_poses_se3))
    if len(gt_poses_se3) != len(learned_poses_se3):
        print(f"⚠️  Using first {min_len} poses")
        gt_poses_se3 = gt_poses_se3[:min_len]
        learned_poses_se3 = learned_poses_se3[:min_len]
    
    print(f"📊 Aligning trajectories with scale correction...")
    
    # Generate visualization (exactly like LongSplat)
    figsize = (args.figsize, args.figsize)
    img = plot_pose_longsplat(gt_poses_se3, learned_poses_se3, 
                              figsize=figsize,
                              elev=args.elev,
                              azim=args.azim,
                              zoom_factor=args.zoom)
    
    # Convert RGB to BGR and save
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    cv2.imwrite(args.output_path, img_bgr)
    
    print(f"✅ Saved to: {args.output_path}")
    print(f"\n🎉 Done! This should be IDENTICAL to LongSplat's visualization.")


if __name__ == "__main__":
    main()