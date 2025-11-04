import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import os

# 장면 리스트
SCENES = ["grass", "hydrant", "lab", "pillar", "road", "sky", "stair"]

# 프레임 타입별 색상 정의
FRAME_COLORS = {
    'I-SLICE': '#FF4444',    # 빨강 (I-frame)
    'P-SLICE': '#4444FF',    # 파랑 (P-frame)
    'B-SLICE': '#44FF44',    # 초록 (B-frame, 대문자)
    'b-SLICE': '#FFA500',    # 오렌지 (b-frame, 소문자)
}

# 출력 디렉토리 생성
os.makedirs('./plots_bits', exist_ok=True)

# 각 장면에 대해 그래프 생성
for scene in SCENES:
    print(f"\n처리 중: {scene}")
    
    try:
        # CSV 파일 읽기
        csv_path = f'../comp_log/{scene}_qp37_trustmap.csv'
        df = pd.read_csv(csv_path)
        
        # Global_Frame_ID로 정렬
        df_sorted = df.sort_values('Global_Frame_ID').reset_index(drop=True)
        
        # Type 컬럼 정리 (공백 제거)
        df_sorted['Type'] = df_sorted['Type'].str.strip()
        
        # 프레임 타입별로 색상 매핑
        colors = [FRAME_COLORS.get(frame_type, '#888888') for frame_type in df_sorted['Type']]
        
        # 한글 폰트 설정 (한글이 깨지지 않도록)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 막대그래프 생성
        bars = ax.bar(df_sorted['Global_Frame_ID'], df_sorted['Bits'], 
                      color=colors, width=0.8, edgecolor='none', alpha=0.8)
        
        # y축을 일반 숫자 형식으로 표시 (과학적 표기법 비활성화)
        ax.ticklabel_format(style='plain', axis='y')
        # 천단위 구분자 추가
        formatter = FuncFormatter(lambda x, p: f'{int(x):,}')
        ax.yaxis.set_major_formatter(formatter)
        
        # 범례 생성 (각 프레임 타입별로)
        from matplotlib.patches import Patch
        legend_elements = []
        frame_types_present = df_sorted['Type'].unique()
        
        for frame_type in ['I-SLICE', 'P-SLICE', 'B-SLICE', 'b-SLICE']:
            if frame_type in frame_types_present:
                label = frame_type.replace('-SLICE', '-frame')
                legend_elements.append(Patch(facecolor=FRAME_COLORS[frame_type], 
                                            label=label, alpha=0.8))
        
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        # 그래프 꾸미기
        ax.set_xlabel('Global Frame ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bits', fontsize=12, fontweight='bold')
        ax.set_title(f'Frame ID vs Bits - {scene.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 그래프 저장
        output_path = f'./plots_bits/frame_bits_bar_{scene}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # 메모리 절약을 위해 figure 닫기
        
        print(f"✓ 그래프 생성 완료: {output_path}")
        print(f"  - 총 프레임 수: {len(df_sorted)}")
        print(f"  - Bits 범위: {df_sorted['Bits'].min():,} ~ {df_sorted['Bits'].max():,}")
        
        # 프레임 타입별 통계
        print(f"  - 프레임 타입 분포:")
        for frame_type in df_sorted['Type'].unique():
            count = (df_sorted['Type'] == frame_type).sum()
            avg_bits = df_sorted[df_sorted['Type'] == frame_type]['Bits'].mean()
            print(f"    • {frame_type}: {count}개 (평균 {avg_bits:,.0f} bits)")
        
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {csv_path}")
    except Exception as e:
        print(f"✗ 에러 발생: {e}")

print("\n\n=== 모든 그래프 생성 완료 ===")