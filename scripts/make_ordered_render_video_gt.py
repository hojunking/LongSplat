import argparse
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio


DEFAULT_IMAGE_DIR = Path(
    "/home/knuvi/Desktop/song/LongSplat/data/compress-x/free/lab/images_2"
)
DEFAULT_OUTPUT = Path(
    "/home/knuvi/Desktop/song/LongSplat/video_outputs/lab_gt.mp4"
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
        description="Encode a video from one image directory sorted by filename."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    ordered_images = collect_images(args.image_dir)
    build_video(ordered_images, args.output, args.fps)
    print(f"Saved {len(ordered_images)} frames to {args.output}")


if __name__ == "__main__":
    main()
