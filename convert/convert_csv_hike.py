#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from convert_csv import build_global_frame_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert hike x265 with_tld CSVs to trustmap CSVs."
    )
    parser.add_argument(
        "--input-root",
        default="comp_log/hike_0316",
        help="Directory containing *_images_x265_QP##_with_tld.csv files.",
    )
    parser.add_argument(
        "--output-root",
        default="comp_log",
        help="Directory for generated trustmap CSVs.",
    )
    parser.add_argument(
        "--scene",
        default="forest1",
        help="Scene name to convert. Use --all-scenes to process every matching file.",
    )
    parser.add_argument(
        "--qps",
        nargs="+",
        default=["QP27", "QP47"],
        help="QP labels to convert.",
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Convert every matching scene under --input-root.",
    )
    parser.add_argument(
        "--gop-len",
        type=int,
        default=32,
        help="GOP length for Global Frame ID.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug rows for each CSV.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing input root: {input_root}")

    pattern = re.compile(r"(?P<scene>.+?)_images_x265_(?P<qp>QP\d+)_with_tld\.csv$")
    csv_paths = sorted(input_root.glob("*_images_x265_QP*_with_tld.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No matching with_tld CSVs found under {input_root}")

    target_qps = {qp.upper() for qp in args.qps}

    for qp_csv in csv_paths:
        match = pattern.match(qp_csv.name)
        if not match:
            print(f"Skipping unexpected filename: {qp_csv}")
            continue

        scene = match.group("scene")
        qp = match.group("qp").upper()

        if not args.all_scenes and scene != args.scene:
            continue
        if qp not in target_qps:
            continue

        output_csv = output_root / f"{scene}_{qp.lower()}_trustmap.csv"

        print("\n====================================")
        print(f"📂 Processing: {scene} ({qp})")
        print(f"Input:  {qp_csv}")
        print(f"Output: {output_csv}")
        print("====================================")

        try:
            build_global_frame_table(
                str(qp_csv),
                str(output_csv),
                gop_len=args.gop_len,
                debug=args.debug,
            )
        except Exception as e:
            print(f"❌ Error processing {scene} ({qp}): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
