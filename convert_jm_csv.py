#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from convert_csv import build_global_frame_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert jm_out hmstyle CSVs to trustmap CSVs.")
    parser.add_argument(
        "--jm-root",
        default="data/compress-o/jm_out",
        help="Root directory containing jm_out scene CSVs.",
    )
    parser.add_argument(
        "--output-root",
        default="comp_log/jm",
        help="Output directory for trustmap CSVs.",
    )
    parser.add_argument("--gop-len", type=int, default=32, help="GOP length for Global Frame ID.")
    parser.add_argument("--debug", action="store_true", help="Print debug rows for each CSV.")
    args = parser.parse_args()

    jm_root = Path(args.jm_root)
    output_root = Path(args.output_root)

    if not jm_root.is_dir():
        raise FileNotFoundError(f"Missing jm_out root: {jm_root}")

    pattern = re.compile(r"(?P<scene>.+)_qp(?P<qp>\d+)_hmstyle\.csv$")
    csv_paths = sorted(jm_root.rglob("*_hmstyle.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No hmstyle CSVs found under {jm_root}")

    for qp_csv in csv_paths:
        match = pattern.match(qp_csv.name)
        if not match:
            print(f"Skipping unexpected filename: {qp_csv}")
            continue

        scene = match.group("scene")
        qp = match.group("qp")
        output_csv = output_root / f"{scene}_qp{qp}_trustmap.csv"

        print(f"\n====================================")
        print(f"📂 Processing: {scene} (qp{qp})")
        print(f"Input:  {qp_csv}")
        print(f"Output: {output_csv}")
        print(f"====================================")

        try:
            build_global_frame_table(str(qp_csv), str(output_csv), gop_len=args.gop_len, debug=args.debug)
        except Exception as e:
            print(f"❌ Error processing {scene} (qp{qp}): {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
