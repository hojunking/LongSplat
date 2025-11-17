# pose_weighting.py
import torch, math, numpy as np
from collections import deque

# --- global buffers (inlier/geometry history) ---
prev_geom_changes = deque(maxlen=3)
prev_inlier_ratios = deque(maxlen=5)


def compute_geometry_change(viewpoint_cam):
    """keypoint 기반 geometry 변화량 (normalized)"""
    if hasattr(viewpoint_cam, "kp0") and hasattr(viewpoint_cam, "kp1"):
        disp = torch.norm(viewpoint_cam.kp0 - viewpoint_cam.kp1, dim=-1).mean().item()
        H, W = viewpoint_cam.original_image.shape[1:3]
        return disp / max(H, W)
    return 0.0


def compute_inlier_stability(viewpoint_cam, prev_inlier_ratios, gamma=5.0):
    """inlier 안정도 계산 (압축 영향 proxy)"""
    r_t = getattr(viewpoint_cam, "inlier_ratio", 1.0)
    if len(prev_inlier_ratios) == 0:
        return 1.0, 0.0
    mean_prev = np.mean(prev_inlier_ratios)
    delta_r = abs(r_t - mean_prev)
    c_artifact = math.exp(-gamma * delta_r)
    return max(0.3, min(1.0, c_artifact)), delta_r


def apply_pose_weight(viewpoint_cam, p_mu=0.1, phase="local"):
    """geometry-aware compression-aware pose gradient scaling"""
    # --- 1. Geometry change 계산 ---
    geom_change_t = compute_geometry_change(viewpoint_cam)
    mean_prev_geom = np.mean(prev_geom_changes) if len(prev_geom_changes) > 0 else geom_change_t
    prev_geom_changes.append(geom_change_t)

    # --- 2. Inlier 안정도 계산 ---
    c_artifact, delta_r = compute_inlier_stability(viewpoint_cam, prev_inlier_ratios)
    r_t = getattr(viewpoint_cam, "inlier_ratio", 1.0)
    prev_inlier_ratios.append(r_t)

    # --- 3. Geometry 변화 판단 ---
    if geom_change_t > mean_prev_geom:
        c_compression = 1.0
        reason = "geom change ↑"
    else:
        c_compression = c_artifact
        reason = "stable → compression-aware"

    # --- 4. Pose weight 계산 ---
    if phase == "local":
        p_mu += 0.2
    w_pose = 1.0 - p_mu * (1.0 - r_t * c_compression)
    w_pose = max(0.0, min(1.0, w_pose))

    # --- 5. Gradient scaling ---
    for p in [getattr(viewpoint_cam, "cam_trans_delta", None),
              getattr(viewpoint_cam, "cam_rot_delta", None)]:
        if p is not None and p.grad is not None:
            p.grad.mul_(w_pose)

    print(f"[{phase.upper()}] uid={getattr(viewpoint_cam,'uid',-1)}, "
          f"inlier={r_t:.3f}, geom={geom_change_t:.4f}/{mean_prev_geom:.4f}, "
          f"Δr={delta_r:.3f}, c_comp={c_compression:.3f}, w_pose={w_pose:.3f}, {reason}")

    return w_pose
