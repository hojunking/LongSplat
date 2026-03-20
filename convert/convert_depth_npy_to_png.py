import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.visualize_utils import vis_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert depth .npy files to colorized PNGs using LongSplat's vis_depth()"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing depth .npy files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: same as input_dir)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir or in_dir

    if not in_dir.is_dir():
        raise ValueError(f"Input dir does not exist: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    npys = sorted(in_dir.glob("*.npy"))
    if not npys:
        raise ValueError(f"No .npy files found in {in_dir}")

    for npy_path in npys:
        depth = np.load(npy_path)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]

        depth_vis = vis_depth(depth)
        out_path = out_dir / f"{npy_path.stem}.png"
        cv2.imwrite(str(out_path), depth_vis)
        print(f"Saved {out_path}")

    print(f"Done. Wrote {len(npys)} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
