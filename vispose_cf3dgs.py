#!/usr/bin/env python3
"""
CF-3DGS Pose Visualization (LongSplat-style, NOPENERF zoom + legend identical)
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

from evo.core.trajectory import PosePath3D
from evo.tools import plot


# ----------------------------------------------------------
# 1) COLMAP Loader
# ----------------------------------------------------------

BaseImage = collections.namedtuple(
    "Image", ["id","qvec","tvec","camera_id","name","xys","point3D_ids"]
)

def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg):
            props = read_next_bytes(fid, 64, "idddddddi")
            img_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]

            name = ""
            c = read_next_bytes(fid, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = read_next_bytes(fid, 1, "c")[0]

            npts = read_next_bytes(fid, 8, "Q")[0]
            _ = read_next_bytes(fid, 24*npts, "ddq"*npts)

            images[img_id] = BaseImage(img_id, qvec, tvec, cam_id, name, None, None)
    return images

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*w*y+2*x*z],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*w*x+2*y*z,   1-2*x*x-2*y*y]
    ])

def load_colmap_gt(images_bin):
    images = read_images_binary(images_bin)
    imgs = sorted(images.values(), key=lambda x: x.name)

    poses = []
    for im in imgs:
        R = qvec2rotmat(im.qvec)
        t = im.tvec

        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3,3] = t

        c2w = np.linalg.inv(w2c)
        poses.append(c2w)

    return np.stack(poses, axis=0)


# ----------------------------------------------------------
# 2) CF-3DGS pred loader
# ----------------------------------------------------------

def load_cf_pred(pth):
    ckpt = torch.load(pth, map_location='cpu')
    gt = ckpt["poses_gt"].numpy()
    pred = ckpt["poses_pred"].numpy()
    return gt, pred


# ----------------------------------------------------------
# 3) Visualization (NOPENERF zoom + identical legend)
# ----------------------------------------------------------

def plot_pose_longsplat(ref, est, figsize=(6,6), elev=10, azim=45, zoom_factor=0.5):

    plt.rc('legend', fontsize=20)   # NOPENERF와 100% 동일하게 설정

    traj_ref = PosePath3D(poses_se3=ref)
    traj_est = PosePath3D(poses_se3=est)

    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Background same as NOPENERF
    ax.set_facecolor('#e7e7eb')
    fig.patch.set_facecolor('#eaeaf2')

    # Tick labels removed, but tick marks preserved (NOPENERF behavior)
    ax.xaxis.set_tick_params(labelbottom=False, length=6, width=1, color='black')
    ax.yaxis.set_tick_params(labelleft=False,  length=6, width=1, color='black')
    ax.zaxis.set_tick_params(labelleft=False,  length=6, width=1, color='black')

    ax.grid(True, color='white', linewidth=0.8, linestyle='-', alpha=1.0)

    # ========== EXACT same plotting order/style ==========
    plot.traj(ax, plot.PlotMode.xyz, traj_est_aligned, '-', 'r', "Ours")
    ax.lines[-1].set_linewidth(2)

    plot.traj(ax, plot.PlotMode.xyz, traj_ref, '--', 'b', "Ground-truth")
    ax.lines[-1].set_linewidth(2)
    # ======================================================

    # === NOPENERF zoom: shrink axis ranges ===
    xlim = ax.get_xlim(); ylim = ax.get_ylim(); zlim = ax.get_zlim()
    xc = (xlim[0]+xlim[1])/2; yc = (ylim[0]+ylim[1])/2; zc = (zlim[0]+zlim[1])/2
    xr = (xlim[1]-xlim[0]) * zoom_factor / 2
    yr = (ylim[1]-ylim[0]) * zoom_factor / 2
    zr = (zlim[1]-zlim[0]) * zoom_factor / 2

    ax.set_xlim(xc-xr, xc+xr)
    ax.set_ylim(yc-yr, yc+yr)
    ax.set_zlim(zc-zr, zc+zr)

    # ========== NOPENERF legend ==========
    ax.legend(loc='upper right', frameon=True)

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    # Convert to numpy image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
    plt.close(fig)
    return img


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    p = ArgumentParser()
    p.add_argument("--pose_path", required=True)
    p.add_argument("--images_bin", required=True)
    p.add_argument("--output_path", default="pose_cf3dgs.png")
    p.add_argument("--figsize", type=float, default=6.0)
    p.add_argument("--elev", type=float, default=10.0)
    p.add_argument("--azim", type=float, default=45.0)
    p.add_argument("--zoom", type=float, default=0.5)
    args = p.parse_args()

    gt_cf, pred_cf = load_cf_pred(args.pose_path)
    gt_colmap = load_colmap_gt(args.images_bin)

    N = min(len(gt_cf), len(gt_colmap))
    ref = gt_colmap[:N]
    est = pred_cf[:N]

    img = plot_pose_longsplat(
        ref, est,
        figsize=(args.figsize, args.figsize),
        elev=args.elev,
        azim=args.azim,
        zoom_factor=args.zoom
    )

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    cv2.imwrite(args.output_path, img_bgr)

    print(f"✅ Saved: {args.output_path}")


if __name__ == "__main__":
    main()
