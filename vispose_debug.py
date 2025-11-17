#!/usr/bin/env python3
import os
import torch
import numpy as np
import struct
import collections
from argparse import ArgumentParser

# ================================================================
# 0) 기본 도구
# ================================================================
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

def read_next(fid, n, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(n))

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])

def getWorld2View_np(R, T):
    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = T
    return pose

# LongSplat 좌표계 변환 매트릭스
R_fix = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

# ================================================================
# 1) COLMAP images.bin → GT poses (LongSplat 스타일)
# ================================================================
def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = read_next(f, 8, "Q")[0]
        for _ in range(n):
            props = read_next(f, 64, "idddddddi")
            iid = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]

            name = ""
            c = read_next(f, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = read_next(f, 1, "c")[0]

            npts = read_next(f, 8, "Q")[0]
            _ = read_next(f, 24 * npts, "ddq" * npts)

            images[iid] = BaseImage(iid, qvec, tvec, cam_id, name, None, None)
    return images

def load_colmap_gt(images_bin):
    imgs = sorted(read_images_bin(images_bin).values(), key=lambda x: x.name)
    poses = []
    for im in imgs:
        R = qvec2rotmat(im.qvec)
        T = im.tvec
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.stack(poses, axis=0)


# ================================================================
# 2) CF-3DGS ep00_init → GT/PRED pose(C2W)
# ================================================================
def load_cf3dgs(pth):
    ckpt = torch.load(pth, map_location='cpu')
    gt = ckpt["poses_gt"].cpu().numpy()
    pred = ckpt["poses_pred"].cpu().numpy()
    return gt, pred


# ================================================================
# 3) CF → LongSplat 좌표계 변환
# ================================================================
def convert_cf_to_longsplat(poses):
    out = poses.copy()
    for i in range(len(poses)):
        out[i,:3,:3] = poses[i,:3,:3] @ R_fix
        out[i,:3, 3] = poses[i,:3, 3] @ R_fix
    return out


# ================================================================
# 4) 디버깅 출력
# ================================================================
def print_pose(name, P):
    print(f"\n=== {name} ===")
    print(P)
    print(f"position: {P[:3,3]}")
    print()


def main():
    parser = ArgumentParser()
    parser.add_argument("--pose_path", required=True)
    parser.add_argument("--images_bin", required=True)
    parser.add_argument("--index", type=int, default=0,
                        help="Print pose at this index (default 0)")
    args = parser.parse_args()

    # --- Load all poses ---
    colmap_gt = load_colmap_gt(args.images_bin)
    cf_gt, cf_pred = load_cf3dgs(args.pose_path)

    idx = args.index
    print(f"\n🔍 DEBUGGING POSE INDEX = {idx}\n")

    # --- (A) COLMAP GT pose ---
    P_colmap = colmap_gt[idx]

    # --- (B) CF3DGS GT pose ---
    P_cf_gt = cf_gt[idx]

    # --- (C) CF3DGS Pred pose ---
    P_cf_pred = cf_pred[idx]

    # --- (D) CF GT (LongSplat 변환 후) ---
    P_cf_gt_ls = convert_cf_to_longsplat(np.array([P_cf_gt]))[0]

    # --- (E) CF Pred (LongSplat 변환 후) ---
    P_cf_pred_ls = convert_cf_to_longsplat(np.array([P_cf_pred]))[0]

    # --- 출력 ---
    print_pose("COLMAP_GT (C2W)", P_colmap)
    print_pose("CF3DGS_GT (C2W)", P_cf_gt)
    print_pose("CF3DGS_PRED (C2W)", P_cf_pred)
    print_pose("CF3DGS_GT → LongSplat(Z-up)", P_cf_gt_ls)
    print_pose("CF3DGS_PRED → LongSplat(Z-up)", P_cf_pred_ls)

    # 차이도 출력
    print("=== Norm Differences ===")
    print("|| COLMAP_GT - CF_GT || =", np.linalg.norm(P_colmap - P_cf_gt))
    print("|| COLMAP_GT - CF_PRED || =", np.linalg.norm(P_colmap - P_cf_pred))
    print("|| COLMAP_GT - CF_GT_LS || =", np.linalg.norm(P_colmap - P_cf_gt_ls))
    print("|| COLMAP_GT - CF_PRED_LS || =", np.linalg.norm(P_colmap - P_cf_pred_ls))


if __name__ == "__main__":
    main()
