import os
import numpy as np
import cv2


def load_render_depth(path):
    depth = np.load(path).astype(np.float32)
    print(f"[LOAD RENDER] {os.path.basename(path)} shape={depth.shape}, dtype={depth.dtype}")
    return depth


def load_gt_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError(f"Failed to load {path}")

    print(f"[LOAD GT] {os.path.basename(path)} shape={depth.shape}, dtype={depth.dtype}")

    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0  # mm → m
    else:
        depth = depth.astype(np.float32)

    return depth


def compute_abs_depth_error_all_npy(render_dir, gt_dir, max_depth=10.0):
    errors = []

    render_files = sorted([f for f in os.listdir(render_dir) if f.endswith(".npy")])
    print(f"[INFO] Found {len(render_files)} render depth files (.npy)")

    for rfile in render_files:
        idx = os.path.splitext(rfile)[0]   # e.g., "4"
        gt_file = f"{idx}.png"

        render_path = os.path.join(render_dir, rfile)
        gt_path = os.path.join(gt_dir, gt_file)

        if not os.path.exists(gt_path):
            print(f"[SKIP] GT not found for {rfile}")
            continue

        render_depth = load_render_depth(render_path)
        gt_depth = load_gt_depth(gt_path)

        if render_depth.shape != gt_depth.shape:
            print(
                f"[WARN] Shape mismatch {idx}: "
                f"render={render_depth.shape}, gt={gt_depth.shape}"
            )
            continue

        mask = (
            (gt_depth > 0) &
            (render_depth > 0) 
            & (gt_depth < max_depth)
        )

        valid_pixels = mask.sum()
        print(f"[MASK] {idx}: valid pixels = {valid_pixels}")

        if valid_pixels == 0:
            print(f"[SKIP] {idx}: no valid depth pixels")
            continue

        frame_err = np.abs(render_depth[mask] - gt_depth[mask]).mean()
        print(f"[FRAME ERR] {idx}: L1 = {frame_err:.6f} m")

        errors.append(frame_err)

    if len(errors) == 0:
        raise RuntimeError("No valid frames found")

    mean_l1 = float(np.mean(errors))
    print("===================================")
    print(f"[RESULT] Mean Absolute Depth Error (m): {mean_l1:.6f}")
    print("===================================")

    return mean_l1, errors


# =========================
# paths
# =========================
render_dir = "./outputs/scannet_base_t2/scene0000_02/test/ours_40000/depths"
gt_dir = "./data/scannet_origin/scene0000_02/depth"

mean_l1, per_frame_l1 = compute_abs_depth_error_all_npy(render_dir, gt_dir)
