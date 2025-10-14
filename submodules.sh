#!/bin/bash
set -e  # 에러 시 중단
echo "=== Installing CUDA Extensions (submodules) ==="

# 환경 변수: GPU 아키텍처 자동 설정 (A6000/3090 = 8.6)
export TORCH_CUDA_ARCH_LIST="8.6"

# 1️⃣ simple-knn
if [ -d "./submodules/simple-knn" ]; then
    echo "[1/3] Installing simple-knn..."
    pip install ./submodules/simple-knn --no-cache-dir --force-reinstall
else
    echo "[1/3] Skipped: ./submodules/simple-knn not found."
fi

# 2️⃣ diff-gaussian-rasterization
if [ -d "./submodules/diff-gaussian-rasterization" ]; then
    echo "[2/3] Installing diff-gaussian-rasterization..."
    pip install ./submodules/diff-gaussian-rasterization --no-cache-dir --force-reinstall
else
    echo "[2/3] Skipped: ./submodules/diff-gaussian-rasterization not found."
fi

# 3️⃣ fused-ssim (CUDA Extension)
if [ -d "./submodules/fused-ssim" ]; then
    echo "[3/3] Installing fused-ssim..."
    rm -rf ./submodules/fused-ssim/build ~/.cache/torch_extensions
    pip install ./submodules/fused-ssim --no-build-isolation --no-cache-dir --force-reinstall -v
else
    echo "[3/3] Skipped: ./submodules/fused-ssim not found."
fi

echo "=== All submodules installed successfully! ==="
