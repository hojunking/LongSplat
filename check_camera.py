import struct
import os
from collections import defaultdict

# COLMAP 카메라 모델 ID와 (모델 이름, 파라미터 개수)를 매핑
# 이 정보는 COLMAP 공식 문서를 기반으로 합니다.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12)
}

def read_colmap_cameras_bin(path_to_cameras_bin):
    """
    COLMAP의 cameras.bin 파일을 읽어 카메라 정보를 파싱하는 함수.
    """
    cameras = {}
    if not os.path.exists(path_to_cameras_bin):
        print(f"오류: 파일을 찾을 수 없습니다 -> {path_to_cameras_bin}")
        return None

    with open(path_to_cameras_bin, 'rb') as fid:
        # 파일 헤더: 카메라 개수 (unsigned long long)
        num_cameras = struct.unpack('<Q', fid.read(8))[0]
        
        for _ in range(num_cameras):
            # 카메라 데이터 읽기
            camera_id, model_id, width, height = struct.unpack('<iiQQ', fid.read(24))
            
            # 모델 정보 가져오기
            model_name, num_params = CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))
            
            # 파라미터 읽기 (double * num_params)
            params = struct.unpack(f'<{num_params}d', fid.read(8 * num_params))
            
            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "num_params": num_params,
                "params": params
            }
    return cameras

def summarize_camera_info(cameras):
    """
    파싱된 카메라 정보 요약
    """
    if not cameras:
        return "카메라 정보 없음"
    
    summary = defaultdict(int)
    # 각 카메라 모델별로 개수를 센다
    for cam in cameras.values():
        summary[cam['model']] += 1
    
    # 예시로 첫 번째 카메라의 상세 정보 추가
    first_cam = next(iter(cameras.values()))
    info_str = f"모델: {first_cam['model']}, 파라미터 개수: {first_cam['num_params']}"
    
    # 요약 결과 문자열 생성
    summary_str = ", ".join([f"{model}({count}개)" for model, count in summary.items()])
    return f"{summary_str} (예시: {info_str})"


# --- 실행 부분 ---
if __name__ == "__main__":
    # ❗️❗️ 여기에 비교할 두 폴더의 경로를 입력하세요.
    base_path = "/home/knuvi/Desktop/song/LongSplat/data/compress-x/tnt"
    church_path = os.path.join(base_path, "Church/sparse/0/cameras.bin")
    barn_path = os.path.join(base_path, "Barn/sparse/0/cameras.bin")

    print("="*50)
    print("COLMAP 카메라 모델 비교 분석")
    print("="*50)
    
    # Barn 씬 정보 분석 및 출력
    print(f"🔍 분석 중: {barn_path}")
    barn_cameras = read_colmap_cameras_bin(barn_path)
    if barn_cameras:
        barn_summary = summarize_camera_info(barn_cameras)
        print(f"✅ Barn 씬 카메라 정보: {barn_summary}\n")
    else:
        print("-> Barn 씬 정보를 읽는 데 실패했습니다.\n")

    # Church 씬 정보 분석 및 출력
    print(f"🔍 분석 중: {church_path}")
    church_cameras = read_colmap_cameras_bin(church_path)
    if church_cameras:
        church_summary = summarize_camera_info(church_cameras)
        print(f"❌ Church 씬 카메라 정보: {church_summary}\n")
    else:
        print("-> Church 씬 정보를 읽는 데 실패했습니다.\n")
        
    print("="*50)
    print("결론:")
    if barn_cameras and church_cameras:
        barn_model = next(iter(barn_cameras.values()))['model']
        church_model = next(iter(church_cameras.values()))['model']
        
        if church_model not in ["PINHOLE", "SIMPLE_PINHOLE"]:
            print(f"Church 씬은 3DGS가 지원하지 않는 '{church_model}' 모델을 사용합니다.")
            print("이 모델은 렌즈 왜곡 정보를 포함하고 있어 에러가 발생합니다.")
            print(f"반면 Barn 씬은 지원되는 '{barn_model}' 모델을 사용합니다.")
            print("해결책으로 'colmap image_undistorter'를 사용하여 Church 씬을 변환해야 합니다.")
        else:
            print("두 씬 모두 지원되는 카메라 모델을 사용하는 것으로 보입니다.")


    
    colmap image_undistorter \
    --image_path /workdir/data/compress-x/tnt/Church/images \
    --input_path /workdir/data/compress-x/tnt/Church/sparse/0 \
    --output_path /workdir/data/compress-x/tnt/Church/colmap_undistorted