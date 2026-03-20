import argparse
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def collect_images(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    images = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=natural_sort_key)


def stem_index(path: Path):
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract frame index from {path}")


def build_prediction_map(test_dir: Path, train_dir: Path):
    prediction_images = collect_images(test_dir) + collect_images(train_dir)
    prediction_map = {}
    for image_path in prediction_images:
        prediction_map[stem_index(image_path)] = image_path
    return prediction_map


def add_label(frame, text: str):
    cv2.putText(
        frame,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def make_diff_panel(gt_frame, pred_frame, diff_gain: float):
    abs_diff = cv2.absdiff(gt_frame, pred_frame)
    diff_gray = abs_diff.mean(axis=2).astype(np.uint8)
    diff_scaled = np.clip(diff_gray.astype(np.float32) * diff_gain, 0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(pred_frame, 0.55, diff_heatmap, 0.45, 0.0)
    return overlay


def build_video(
    gt_dir: Path,
    test_dir: Path,
    train_dir: Path,
    output_path: Path,
    fps: float,
    diff_gain: float,
    label: str,
    crf: int,
    preset: str,
):
    gt_images = collect_images(gt_dir)
    prediction_map = build_prediction_map(test_dir, train_dir)
    matched_pairs = []

    for gt_path in gt_images:
        frame_idx = stem_index(gt_path)
        pred_path = prediction_map.get(frame_idx)
        if pred_path is not None:
            matched_pairs.append((frame_idx, gt_path, pred_path))

    if not matched_pairs:
        raise ValueError("No matching frames found between GT and prediction images.")

    first_gt = cv2.imread(str(matched_pairs[0][1]))
    first_pred = cv2.imread(str(matched_pairs[0][2]))
    if first_gt is None or first_pred is None:
        raise ValueError("Failed to read first frame for initialization.")

    height, width = first_gt.shape[:2]
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=[
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-movflags",
            "+faststart",
        ],
    )

    try:
        for frame_idx, gt_path, pred_path in matched_pairs:
            gt_frame = cv2.imread(str(gt_path))
            pred_frame = cv2.imread(str(pred_path))
            if gt_frame is None or pred_frame is None:
                raise ValueError(f"Failed to read frame pair: {gt_path}, {pred_path}")

            if gt_frame.shape[:2] != (height, width):
                gt_frame = cv2.resize(gt_frame, (width, height), interpolation=cv2.INTER_AREA)
            if pred_frame.shape[:2] != (height, width):
                pred_frame = cv2.resize(pred_frame, (width, height), interpolation=cv2.INTER_AREA)

            diff_overlay = make_diff_panel(gt_frame, pred_frame, diff_gain)
            diff_labeled = add_label(diff_overlay, "Diff Heatmap")
            writer.append_data(cv2.cvtColor(diff_labeled, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()

    print(f"Saved {len(matched_pairs)} frames to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a diff heatmap video from GT and render images."
    )
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--diff-gain", type=float, default=4.0)
    parser.add_argument("--label", type=str, default="Render")
    parser.add_argument("--crf", type=int, default=32)
    parser.add_argument("--preset", type=str, default="slow")
    return parser.parse_args()


def main():
    args = parse_args()
    build_video(
        gt_dir=args.gt_dir,
        test_dir=args.test_dir,
        train_dir=args.train_dir,
        output_path=args.output,
        fps=args.fps,
        diff_gain=args.diff_gain,
        label=args.label,
        crf=args.crf,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
