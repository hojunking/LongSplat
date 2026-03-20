import argparse
import json
import os
from pathlib import Path

try:
    import gspread
    from gspread_formatting import format_cell_range, get_user_entered_format
except Exception:
    gspread = None
    format_cell_range = None
    get_user_entered_format = None


def copy_format_from_previous_row(sheet, dest_row):
    if format_cell_range is None or get_user_entered_format is None:
        return
    if dest_row <= 2:
        return
    source_row = dest_row - 1
    columns = [chr(i) for i in range(ord('B'), ord('H') + 1)]
    for col in columns:
        source_cell = f"{col}{source_row}"
        dest_cell = f"{col}{dest_row}"
        try:
            fmt = get_user_entered_format(sheet, source_cell)
            if fmt:
                format_cell_range(sheet, dest_cell, fmt)
        except Exception:
            pass


def read_info_json(info_json_path):
    with open(info_json_path, "r") as f:
        data = json.load(f)

    # Expected format from f2-nerf/scripts/eval.py:
    # {"psnr": {"...": v, "mean": v}, "ssim": {...}, "lpips": {...}}
    def mean_of(key):
        if key not in data or not isinstance(data[key], dict):
            return 0.0
        return float(data[key].get("mean", 0.0))

    return {
        "PSNR": f"{mean_of('psnr'):.4f}",
        "SSIM": f"{mean_of('ssim'):.4f}",
        "LPIPS": f"{mean_of('lpips'):.4f}",
    }


def iter_info_files(exp_eval_root, method_dir):
    root = Path(exp_eval_root)
    if not root.is_dir():
        raise FileNotFoundError(f"exp_eval root not found: {exp_eval_root}")

    # scene_dir/method_dir/info.json
    for scene_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        info_path = scene_dir / method_dir / "info.json"
        if info_path.is_file():
            yield scene_dir.name, str(info_path)


def upload_row(sheet, method_name, metrics, pose_defaults=True):
    all_values = sheet.col_values(2)
    row_number = len(all_values) + 1
    copy_format_from_previous_row(sheet, row_number)

    rpe_trans = "0.000" if pose_defaults else ""
    rpe_rot = "0.000" if pose_defaults else ""
    ate = "0.000" if pose_defaults else ""

    updates = [
        {"range": f"B{row_number}", "values": [[method_name]]},
        {"range": f"C{row_number}", "values": [[metrics["PSNR"]]]},
        {"range": f"D{row_number}", "values": [[metrics["SSIM"]]]},
        {"range": f"E{row_number}", "values": [[metrics["LPIPS"]]]},
        {"range": f"F{row_number}", "values": [[rpe_trans]]},
        {"range": f"G{row_number}", "values": [[rpe_rot]]},
        {"range": f"H{row_number}", "values": [[ate]]},
    ]
    sheet.batch_update(updates)
    return row_number


def main():
    parser = argparse.ArgumentParser(description="Batch upload f2-nerf exp_eval metrics to gspread")
    parser.add_argument("exp_eval_root", type=str, help="e.g. /home/.../f2-nerf/exp_eval/longsplat_free_comp")
    parser.add_argument("sheet_name", type=str, help="worksheet name in spreadsheet EX-results")
    parser.add_argument("--method-dir", type=str, default="f2nerf_comp_o", help="method subdir name under each scene dir")
    parser.add_argument("--method-prefix", type=str, default="", help="prefix for method column, e.g. f2-nerf/")
    parser.add_argument("--account-json", type=str, default=os.environ.get("GSPREAD_ACCOUNT_JSON", "/workdir/gspread/account.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = list(iter_info_files(args.exp_eval_root, args.method_dir))
    if not files:
        print("❌ No info.json files found.")
        return

    print(f"Found {len(files)} scenes under {args.exp_eval_root}")

    if args.dry_run:
        for scene_name, info_path in files:
            metrics = read_info_json(info_path)
            method_name = f"{args.method_prefix}{scene_name}_{args.method_dir}"
            print(f"[DRY] {method_name}: PSNR={metrics['PSNR']} SSIM={metrics['SSIM']} LPIPS={metrics['LPIPS']}")
        return

    if gspread is None:
        raise ModuleNotFoundError("gspread not installed. Install with: pip install gspread gspread-formatting")

    gc = gspread.service_account(filename=args.account_json)
    sh = gc.open("EX-results")
    sheet = sh.worksheet(args.sheet_name)

    for scene_name, info_path in files:
        metrics = read_info_json(info_path)
        method_name = f"{args.method_prefix}{scene_name}_{args.method_dir}"
        row = upload_row(sheet, method_name, metrics, pose_defaults=True)
        print(f"✅ Row {row}: {method_name} | PSNR={metrics['PSNR']} SSIM={metrics['SSIM']} LPIPS={metrics['LPIPS']}")

if __name__ == "__main__":
    main()
