#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path


def load_colmap_loader():
    repo_root = Path(__file__).resolve().parents[1]
    loader_path = repo_root / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("colmap_loader_local", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_ranges(numbers):
    if not numbers:
        return []

    ranges = []
    start = prev = numbers[0]
    for num in numbers[1:]:
        if num == prev + 1:
            prev = num
            continue
        ranges.append((start, prev))
        start = prev = num
    ranges.append((start, prev))
    return ranges


def format_range(start, end):
    if start == end:
        return f"{start:06d}"
    return f"{start:06d}-{end:06d}"


def main():
    parser = argparse.ArgumentParser(description="Inspect a COLMAP images.bin file.")
    parser.add_argument("images_bin", help="Path to COLMAP images.bin")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every frame name stored in images.bin",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print image_id, camera_id, qvec, and tvec for every entry",
    )
    args = parser.parse_args()

    images_bin = Path(args.images_bin)
    if not images_bin.exists():
        raise FileNotFoundError(f"images.bin not found: {images_bin}")

    colmap_loader = load_colmap_loader()
    extrinsics = colmap_loader.read_extrinsics_binary(str(images_bin))
    entries = sorted(extrinsics.values(), key=lambda item: item.name)

    frame_names = [Path(entry.name).stem for entry in entries]
    frame_numbers = []
    non_numeric = []
    for name in frame_names:
        if name.isdigit():
            frame_numbers.append(int(name))
        else:
            non_numeric.append(name)

    print(f"images.bin: {images_bin}")
    print(f"entry_count: {len(entries)}")

    if frame_numbers:
        frame_numbers = sorted(frame_numbers)
        print(f"numeric_frame_count: {len(frame_numbers)}")
        print(f"first_frame: {frame_numbers[0]:06d}")
        print(f"last_frame: {frame_numbers[-1]:06d}")

        ranges = summarize_ranges(frame_numbers)
        print(f"continuous_ranges: {len(ranges)}")
        print("ranges:")
        for start, end in ranges:
            print(f"  {format_range(start, end)}")

        missing_ranges = []
        for (prev_start, prev_end), (next_start, next_end) in zip(ranges, ranges[1:]):
            if next_start > prev_end + 1:
                missing_ranges.append((prev_end + 1, next_start - 1))
        print(f"missing_range_count: {len(missing_ranges)}")
        if missing_ranges:
            print("missing_ranges:")
            for start, end in missing_ranges:
                print(f"  {format_range(start, end)}")

    if non_numeric:
        print(f"non_numeric_frame_names: {len(non_numeric)}")
        for name in non_numeric:
            print(f"  {name}")

    print("first_10_names:")
    for name in frame_names[:10]:
        print(f"  {name}")

    print("last_10_names:")
    for name in frame_names[-10:]:
        print(f"  {name}")

    if args.show_all:
        print("all_frame_names:")
        for name in frame_names:
            print(name)

    if args.show_metadata:
        print("all_entries:")
        for entry in entries:
            print(
                f"name={entry.name} image_id={entry.id} camera_id={entry.camera_id} "
                f"qvec={entry.qvec.tolist()} tvec={entry.tvec.tolist()}"
            )


if __name__ == "__main__":
    main()
