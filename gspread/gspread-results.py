import gspread
import json
import sys
from gspread_formatting import get_user_entered_format, format_cell_range

def read_experiment_data(json_file_path):
    """
    results.json 파일을 읽고 PSNR, SSIM, LPIPS 값을 파싱합니다.
    - 두 가지 JSON 구조를 모두 처리합니다.
    - 'ours_30000_vs_OrigGT' 키가 있으면 우선적으로 사용합니다.
    - 값을 소수점 4째자리까지 포맷팅합니다.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 우선적으로 사용할 결과 키
    preferred_key = "ours_30000_vs_OrigGT"

    if preferred_key in data:
        metrics = data[preferred_key]
    # 우선 키가 없으면 JSON 데이터의 첫 번째 값(dict)을 사용
    elif data:
        first_key = next(iter(data))
        metrics = data[first_key]
    else:
        # 데이터가 비어있는 예외적인 경우
        return None

    # PSNR, SSIM, LPIPS 값을 추출하고 포맷팅
    # 값이 없을 경우를 대비하여 .get(key, 0) 사용
    psnr = metrics.get("PSNR", 0)
    ssim = metrics.get("SSIM", 0)
    lpips = metrics.get("LPIPS", 0)

    return {
        "PSNR": f"{psnr:.4f}",
        "SSIM": f"{ssim:.4f}",
        "LPIPS": f"{lpips:.4f}",
    }

def read_pose_txt(pose_file_path):
    """
    pose.txt 파일을 읽고 RPE_trans, RPE_rot, ATE 값을 파싱합니다.
    예시:
      RPE_trans: 0.590, RPE_rot: 1.235, ATE: 0.006
    """
    try:
        with open(pose_file_path, 'r') as f:
            line = f.readline().strip()
        
        # 각 항목을 콜론으로 분리하여 float으로 변환
        parts = line.split(',')
        data = {}
        for part in parts:
            key, val = part.strip().split(':')
            data[key.strip()] = float(val.strip())
        
        return {
            "RPE_trans": f"{data.get('RPE_trans', 0):.3f}",
            "RPE_rot": f"{data.get('RPE_rot', 0):.3f}",
            "ATE": f"{data.get('ATE', 0):.3f}",
        }

    except Exception as e:
        print(f"❌ Error reading pose file: {e}")
        return {"RPE_trans": "0.000", "RPE_rot": "0.000", "ATE": "0.000"}
    

def copy_format_from_previous_row(sheet, dest_row):
    """이전 행의 서식을 새 행에 복사합니다."""
    if dest_row <= 2:
        return
        
    source_row = dest_row - 1
    columns = [chr(i) for i in range(ord('B'), ord('H') + 1)]  # B~H열까지 복사

    for col in columns:
        source_cell = f'{col}{source_row}'
        dest_cell = f'{col}{dest_row}'
        try:
            fmt = get_user_entered_format(sheet, source_cell)
            if fmt:
                format_cell_range(sheet, dest_cell, fmt)
        except Exception:
            pass


def save_gspread(json_path, pose_path, method_name, sheet_name):
    """파싱된 데이터를 Google Sheets에 저장합니다."""
    try:
        gc = gspread.service_account(filename='/workdir/gspread/account.json')
        sh = gc.open("EX-results")
        sheet = sh.worksheet(sheet_name)

        # 데이터 읽기
        result_data = read_experiment_data(json_path)
        pose_data = read_pose_txt(pose_path)
        if result_data is None:
            print("❌ Error: No data found in results.json.")
            return

        # 다음 빈 행 찾기
        all_values = sheet.col_values(2)
        row_number = len(all_values) + 1

        copy_format_from_previous_row(sheet, row_number)
        
        print(f"Uploading to Sheet '{sheet_name}', Row {row_number}...")
        print(f"  Method: {method_name}")
        print(f"  PSNR={result_data['PSNR']}, SSIM={result_data['SSIM']}, LPIPS={result_data['LPIPS']}")
        print(f"  RPE_trans={pose_data['RPE_trans']}, RPE_rot={pose_data['RPE_rot']}, ATE={pose_data['ATE']}")

        # 시트 업데이트
        updates = [
            {'range': f'B{row_number}', 'values': [[method_name]]},
            {'range': f'C{row_number}', 'values': [[result_data["PSNR"]]]},
            {'range': f'D{row_number}', 'values': [[result_data["SSIM"]]]},
            {'range': f'E{row_number}', 'values': [[result_data["LPIPS"]]]},
            {'range': f'F{row_number}', 'values': [[pose_data["RPE_trans"]]]},
            {'range': f'G{row_number}', 'values': [[pose_data["RPE_rot"]]]},
            {'range': f'H{row_number}', 'values': [[pose_data["ATE"]]]},
        ]
        
        sheet.batch_update(updates)
        print("✅ Data uploaded successfully!")

    except FileNotFoundError:
        print("❌ Error: Authentication file not found at '/workdir/gspread/account.json'.")
    except gspread.exceptions.SpreadsheetNotFound:
        print("❌ Error: Spreadsheet 'EX-results' not found. Check the name and permissions.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"❌ Error: Worksheet '{sheet_name}' not found in the spreadsheet.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python upload_results.py <path_to_results.json> <method_name> <sheet_name>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    pose_path = sys.argv[2]
    method_name = sys.argv[3]
    sheet_name = sys.argv[4]
    
    save_gspread(json_path, pose_path, method_name, sheet_name)