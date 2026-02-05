import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from matplotlib import cm


def load_depth_16(path: Path) -> np.ndarray:
    depth = np.array(Image.open(path), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth map, got shape {depth.shape}")
    return depth


def normalize_depth(
    depth: np.ndarray,
    clip_min: Optional[float],
    clip_max: Optional[float],
) -> np.ndarray:
    mask = depth > 0
    if not np.any(mask):
        raise ValueError("Depth map has no positive values to normalize")

    d_min = clip_min if clip_min is not None else float(depth[mask].min())
    d_max = clip_max if clip_max is not None else float(depth[mask].max())
    if d_min >= d_max:
        raise ValueError(f"Invalid normalization bounds: min={d_min}, max={d_max}")

    depth = np.clip(depth, d_min, d_max, out=np.empty_like(depth))
    norm = (depth - d_min) / (d_max - d_min)
    norm[~mask] = 0.0
    return norm


def colorize_depth(norm_depth: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(norm_depth)
    rgb = (colored[..., :3] * 255.0).astype(np.uint8)
    return rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colorize all ScanNet depth PNGs in a folder and save to depth_color/."
    )
    parser.add_argument(
        "depth_dir",
        type=Path,
        help="Path to folder containing 16-bit grayscale depth PNGs",
    )
    parser.add_argument(
        "--colormap",
        default="turbo",
        help="Matplotlib colormap name (default: turbo)",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Lower bound for normalization (e.g., 500 for ScanNet)",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Upper bound for normalization (e.g., 5000 for ScanNet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    depth_dir = args.depth_dir
    if not depth_dir.is_dir():
        raise ValueError(f"{depth_dir} is not a directory")

    out_dir = depth_dir.parent / "depth_color"
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(depth_dir.glob("*.png"))
    if not pngs:
        raise ValueError(f"No PNGs found in {depth_dir}")

    for p in pngs:
        depth = load_depth_16(p)
        norm = normalize_depth(depth, args.clip_min, args.clip_max)
        colorized = colorize_depth(norm, args.colormap)

        Image.fromarray(colorized).save(out_dir / p.name)

    print(f"[DONE] Colorized depth saved to: {out_dir}")


if __name__ == "__main__":
    main()
