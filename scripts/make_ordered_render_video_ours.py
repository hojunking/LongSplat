import argparse
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio


DEFAULT_TEST_DIR = Path(
    "/home/knuvi/Desktop/song/LongSplat/outputs/free_qp27_ours/"
    "lab_qp27_ours/test/ours_40000/renders"
)
DEFAULT_TRAIN_DIR = Path(
    "/home/knuvi/Desktop/song/LongSplat/outputs/free_qp27_ours/"
    "lab_qp27_ours/train/ours_40000/renders"
)
DEFAULT_OUTPUT = Path(
    "/home/knuvi/Desktop/song/LongSplat/video_outputs/"
    "lab_qp27_ours_renders.mp4"
)
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


def build_video(image_paths, output_path: Path, fps: float):
    if not image_paths:
        raise ValueError("No images found to encode.")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")
    height, width = first_frame.shape[:2]
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
        output_params=["-movflags", "+faststart"],
    )

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Failed to read image: {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine images from two render directories, sort by filename, and encode an MP4 video."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    ordered_images = collect_images(args.test_dir) + collect_images(args.train_dir)
    ordered_images = sorted(ordered_images, key=natural_sort_key)
    build_video(ordered_images, args.output, args.fps)
    print(f"Saved {len(ordered_images)} frames to {args.output}")


if __name__ == "__main__":
    main()
