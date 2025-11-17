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


def rotation_translation_to_se3(R, t):
    """Convert rotation matrix and translation to SE3 matrix"""
    se3 = np.eye(4)
    se3[:3, :3] = R
    se3[:3, 3] = t
    return se3


def load_gt_poses(images_bin_path, test_every=9):
    """Load GT poses from COLMAP images.bin and filter out test frames"""
    print(f"📂 Loading GT poses from: {images_bin_path}")
    
    images = read_images_binary(images_bin_path)
    sorted_images = sorted(images.values(), key=lambda x: x.name)
    
    print(f"   Loaded {len(sorted_images)} total GT poses (train + test)")
    
    # Filter out test frames
    train_indices = [i for i in range(len(sorted_images)) if i % test_every != 0]
    train_images = [sorted_images[i] for i in train_indices]
    
    print(f"   Filtered to {len(train_images)} train GT poses (removed every {test_every}th frame)")
    
    poses_se3 = []
    
    for img in train_images:
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        # Convert to camera-to-world (C2W) for visualization
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        poses_se3.append(c2w)
    
    return poses_se3


def load_learned_poses(checkpoint_path):
    """Load learned poses from nope-nerf checkpoint"""
    print(f"📂 Loading learned poses from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ckpt['model']
    
    axis_angle = model['r']
    translations = model['t']
    
    rotations = axis_angle_to_rotation_matrix(axis_angle)
    
    print(f"   Loaded {axis_angle.shape[0]} learned poses")
    
    # Convert to SE3 matrices (C2W)
    poses_se3 = []
    for i in range(len(rotations)):
        R = rotations[i].numpy()
        t = translations[i].numpy()
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        poses_se3.append(c2w)
    
    return poses_se3


def plot_pose_longsplat(ref_poses, est_poses):
    """
    EXACT reproduction of LongSplat's plot_pose function
    """
    plt.rc('legend', fontsize=20)
    
    ref_poses = [pose for pose in ref_poses]
    est_poses = [pose for pose in est_poses]
    
    # Create trajectories
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    
    # NO ALIGNMENT - use poses as-is
    traj_est_aligned = traj_est  # Don't align
    
    # Create figure
    fig = plt.figure()
    traj_by_label = {
        "Ours": traj_est_aligned,
        "Ground-truth": traj_ref
    }
    
    plot_mode = plot.PlotMode.xyz
    ax = fig.add_subplot(111, projection="3d")
    
    # Remove tick LABELS but keep tick marks visible
    ax.xaxis.set_tick_params(labelbottom=False, length=6, width=1, color='black')
    ax.yaxis.set_tick_params(labelleft=False, length=6, width=1, color='black')
    ax.zaxis.set_tick_params(labelleft=False, length=6, width=1, color='black')
    
    # White grid
    ax.grid(True, color='white', linewidth=1, alpha=0.8)
    
    colors = ['r', 'b']
    styles = ['-', '--']
    
    # Plot trajectories using evo
    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj, styles[idx], colors[idx], label)
    
    # View angle (same as LongSplat)
    ax.view_init(elev=10., azim=45)
    
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
    parser.add_argument("--test_every", type=int, default=9,
                       help="Test frame interval")
    
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
    
    print(f"📊 Using poses WITHOUT alignment (no scale correction)...")
    
    # Generate visualization (exactly like LongSplat)
    img = plot_pose_longsplat(gt_poses_se3, learned_poses_se3)
    
    # Convert RGB to BGR and save
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    cv2.imwrite(args.output_path, img_bgr)
    
    print(f"✅ Saved to: {args.output_path}")
    print(f"\n🎉 Done! This should be IDENTICAL to LongSplat's visualization.")


if __name__ == "__main__":
    main()