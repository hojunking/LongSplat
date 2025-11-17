import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

# 장면 리스트
SCENES = ["grass", "hydrant", "lab", "pillar", "road", "sky", "stair"]

# 색상 매핑 함수
def get_color_by_bits(bits):
    if bits <= 200_000:
        return '#FF4444'  # 빨강
    elif bits <= 1_000_000:
        return '#FFA500'  # 주황
    elif bits <= 4_000_000:
        return "#40FF00"  # 노랑
    else:
        return '#4A90E2'  # 파랑

# 출력 디렉토리
os.makedirs('./plots_bits_rangecolor', exist_ok=True)

for scene in SCENES:
    print(f"\n처리 중: {scene}")

    try:
        csv_path = f'../comp_log/{scene}_qp37_trustmap.csv'
        df = pd.read_csv(csv_path)

        # Bits 기준 오름차순 정렬
        df_sorted = df.sort_values('Bits', ascending=True).reset_index(drop=True)

        # 색상 매핑
        df_sorted['Color'] = df_sorted['Bits'].apply(get_color_by_bits)

        # 그래프 생성
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(12, 6))

        # 막대그래프
        ax.bar(range(len(df_sorted)), df_sorted['Bits'], 
               color=df_sorted['Color'], alpha=0.9, width=0.9)

        # y축 범위 고정 (0 ~ 1,200,000)
        ax.set_ylim(0, 1_200_000)

        # y축 천단위 포맷
        formatter = FuncFormatter(lambda x, p: f'{int(x):,}')
        ax.yaxis.set_major_formatter(formatter)

        # 제목 및 축
        ax.set_xlabel('Frame Index (Sorted by Bits)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bits', fontsize=12, fontweight='bold')
        ax.set_title(f'Bit Distribution by Range - {scene.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        # 범례
        legend_elements = [
            Patch(facecolor='#FF4444', label='≤ 200K bits'),
            Patch(facecolor='#FFA500', label='200K–1M bits'),
            Patch(facecolor="#40FF00", label='1M–4M bits'),
            Patch(facecolor='#4A90E2', label='> 4M bits'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        plt.tight_layout()

        # 저장
        output_path = f'./plots_bits_rangecolor/bit_distribution_{scene}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 콘솔 출력 요약
        print(f"✓ 그래프 생성 완료: {output_path}")
        print(f"  - 총 프레임 수: {len(df_sorted)}")
        print(f"  - Bits 범위: {df_sorted['Bits'].min():,} ~ {df_sorted['Bits'].max():,}")
        print(f"  - 평균 Bits: {df_sorted['Bits'].mean():,.0f}")

    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {csv_path}")
    except Exception as e:
        print(f"✗ 에러 발생: {e}")

print("\n\n=== 모든 그래프 생성 완료 ===")
