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


def get_dropout_rate(viewpoint_cam, d_mu=0.3, verbose=True):
    """geometry-aware, inlier-stability 기반 dropout 비율 산정"""
    # --- 1. Geometry 변화량 ---
    geom_change_t = compute_geometry_change(viewpoint_cam)
    mean_prev_geom = np.mean(prev_geom_changes) if len(prev_geom_changes) > 0 else geom_change_t
    geom_diff = geom_change_t - mean_prev_geom
    geom_factor = math.exp(-5.0 * abs(geom_diff))
    prev_geom_changes.append(geom_change_t)

    # --- 2. Inlier 안정도 ---
    c_artifact, delta_r = compute_inlier_stability(viewpoint_cam, prev_inlier_ratios)
    r_t = getattr(viewpoint_cam, "inlier_ratio", 1.0)
    prev_inlier_ratios.append(r_t)

    # --- 3. Effective quality (geometry × stability × inlier ratio) ---
    effective_quality = r_t * geom_factor * c_artifact
    drop_rate = min(d_mu * (1.0 - effective_quality), 0.5)

    # --- 4. 디버그 로그 ---
    if verbose:
        uid = getattr(viewpoint_cam, "uid", -1)
        print("=" * 80)
        print(f"[DROPOUT] uid={uid}")
        print(f"  ▸ inlier ratio (r_t):         {r_t:.3f}")
        print(f"  ▸ inlier stability Δr:        {delta_r:.4f} → c_artifact={c_artifact:.3f}")
        print(f"  ▸ geometry change:            {geom_change_t:.5f}")
        print(f"  ▸ mean(prev_geom):            {mean_prev_geom:.5f}")
        print(f"  ▸ Δgeom:                      {geom_diff:+.5f} → geom_factor={geom_factor:.3f}")
        print(f"  ▸ effective_quality:          {effective_quality:.3f}")
        print(f"  ▸ computed drop_rate:         {drop_rate:.3f}")
        print("=" * 80)

    return drop_rate


def get_dropout_rate_simple(viewpoint_cam, d_mu=0.3, verbose=True):
    """단순 inlier 기반 dropout 계산"""
    r_t = getattr(viewpoint_cam, "inlier_ratio", 1.0)
    drop_rate = min(d_mu * (1.0 - r_t), 0.5)

    if verbose:
        uid = getattr(viewpoint_cam, "uid", -1)
        print(f"[DROPOUT_SIMPLE] uid={uid}, inlier={r_t:.3f}, drop_rate={drop_rate:.3f}")
    return drop_rate
