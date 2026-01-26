FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# 1. Install System Dependencies required for compiling MMCV and Image Processing
# 'ninja-build' speeds up compilation; 'libgl1' is for OpenCV
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Install AWS CLI v2 (Required for S3 interactions)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# 3. Create a clean Conda environment with Python 3.9.7 
# We name it 'internimage'
RUN conda create -n internimage python=3.9.7 -y

# 4. Activate the Conda environment for all subsequent RUN instructions
# This is the standard pattern to make Docker use a specific Conda env
SHELL ["conda", "run", "-n", "internimage", "/bin/bash", "-c"]

# 5. Install PyTorch 2.8.0
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 6. Install Build Dependencies for MMCV
RUN pip install cython packaging opencv-python termcolor yacs pyyaml scipy boto3

# 7. Compile MMCV-Full 1.7.2 from Source
# We set environment variables to force CUDA kernel compilation
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN pip install -U openmim
RUN mim install mmcv-full==1.7.2
RUN mim install mmsegmentation==0.29.1
RUN pip install timm==0.6.11 mmdet==2.28.1



# 9. (Optional) Install 'InternImage' specific dependencies if not included in your repo
# We pin numpy<2.0 because MMCV 1.x can have issues with numpy 2.0
RUN pip install numpy==1.26.4 pydantic==1.10.13

# Set the working directory
WORKDIR /workspace

COPY ops_dcnv3 ops_dcnv3

WORKDIR ops_dcnv3
RUN sh make.sh

WORKDIR /workspace

COPY . .

# 10. Final Configuration
WORKDIR /workspace
RUN chmod +x entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]
