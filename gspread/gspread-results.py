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

def copy_format_from_previous_row(sheet, dest_row):
    """이전 행의 서식을 새 행에 복사합니다."""
    if dest_row <= 2: # 2번째 행 이하는 복사할 서식이 없음
        return
        
    source_row = dest_row - 1
    # B열부터 H열까지 서식 복사 (필요에 따라 범위 수정)
    columns = [chr(i) for i in range(ord('B'), ord('I') + 1)]

    for col in columns:
        source_cell = f'{col}{source_row}'
        dest_cell = f'{col}{dest_row}'
        try:
            fmt = get_user_entered_format(sheet, source_cell)
            if fmt:
                format_cell_range(sheet, dest_cell, fmt)
        except Exception as e:
            # 서식 가져오기 실패 시 무시하고 계속 진행
            # print(f"Could not copy format for cell {source_cell}: {e}")
            pass


def save_gspread(json_path, method_name, sheet_name):
    """파싱된 데이터를 Google Sheets에 저장합니다."""
    try:
        # 서비스 계정 인증 (경로는 실제 파일 위치에 맞게 수정)
        gc = gspread.service_account(filename='/workdir/gspread/account.json')
        
        # 워크북 및 워크시트 열기
        sh = gc.open("EX-results") # 스프레드시트 이름
        sheet = sh.worksheet(sheet_name) # 시트 이름

        # JSON 파일에서 데이터 읽기
        result_data = read_experiment_data(json_path)
        if result_data is None:
            print("❌ Error: No data found in results.json.")
            return

        # 데이터를 추가할 다음 빈 행 찾기 (B열 기준)
        all_values = sheet.col_values(2) # B열
        row_number = len(all_values) + 1

        # 이전 행 서식 복사
        copy_format_from_previous_row(sheet, row_number)
        
        print(f"Uploading to Sheet '{sheet_name}', Row {row_number}...")
        print(f"  Method: {method_name}")
        print(f"  Data: PSNR={result_data['PSNR']}, SSIM={result_data['SSIM']}, LPIPS={result_data['LPIPS']}")

        # 시트에 한 번에 업데이트 (API 호출 최소화)
        # 시트의 열 순서에 맞게 'range'를 수정하세요 (B=Method, C=PSNR, D=SSIM, E=LPIPS)
        updates = [
            {'range': f'B{row_number}', 'values': [[method_name]]},
            {'range': f'C{row_number}', 'values': [[result_data["PSNR"]]]},
            {'range': f'D{row_number}', 'values': [[result_data["SSIM"]]]},
            {'range': f'E{row_number}', 'values': [[result_data["LPIPS"]]]},
        ]
        
        sheet.batch_update(updates)
        print("✅ Data uploaded successfully!")

    except FileNotFoundError:
        print(f"❌ Error: Authentication file not found at '/workdir/gspread/account.json'.")
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"❌ Error: Spreadsheet '3dgs-pc' not found. Check the name and permissions.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"❌ Error: Worksheet '{sheet_name}' not found in the spreadsheet.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python upload_results.py <path_to_results.json> <method_name> <sheet_name>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    method_name = sys.argv[2]
    sheet_name = sys.argv[3]
    
    save_gspread(json_path, method_name, sheet_name)