# CUDA 12.1 + nvcc 포함 + Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
WORKDIR /workspace

# 1. 기본 의존성 설치
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    git wget curl ninja-build build-essential cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# python3 → python alias
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 2. pip 최신화
RUN pip install --upgrade pip setuptools wheel

# 3. CUDA 아키텍처 강제 (RTX 4090 대응)
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

# 4. PyTorch (CUDA 12.1 wheel)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. requirements.txt 설치 (코드를 마운트 후 실행 권장)
# COPY requirements.txt /workspace/
# RUN pip install -r requirements.txt

# 6. submodules 빌드 (컨테이너 실행 후 권장)
# RUN pip install ./submodules/simple-knn \
#     && pip install ./submodules/diff-gaussian-rasterization \
#     && pip install ./submodules/fused-ssim --no-build-isolation --no-cache-dir -v
