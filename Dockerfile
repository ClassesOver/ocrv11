# 基于 Ubuntu 22.04
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（使用 uv 安装 Python 3.11）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libzbar0 \
        tzdata \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1-mesa-glx \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        wget \
        curl \
        git \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv 到全局位置
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -m 755 /root/.local/bin/uv /usr/local/bin/uv

# 使用 uv 安装 Python 3.11
RUN uv python install 3.11

# 创建应用目录
RUN mkdir -p /app/code /app/logs

# 使用 uv 创建 Python 3.11 虚拟环境
RUN uv venv --python 3.11 /app/.venv

# 将虚拟环境加入全局 PATH，设置 VIRTUAL_ENV
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# 设置工作目录
WORKDIR /app/code

# 复制依赖文件并安装（利用缓存层）
COPY requirements.txt .
RUN uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir -r requirements.txt && \
    uv pip install --no-cache-dir -U ultralytics && \
    uv pip install --no-cache-dir -U gunicorn && \
    uv pip install --no-cache-dir -U uvicorn && \
    uv pip install --no-cache-dir -U onnx && \
    uv pip install --no-cache-dir -U onnxruntime && \
    uv pip install --no-cache-dir -U pip && \
    uv pip install --no-cache-dir -U gevent && \
    uv pip install --no-cache-dir -U transformers && \
    uv pip install --no-cache-dir openvino && \
    uv pip install --no-cache-dir openvino-telemetry && \
    uv pip install --no-cache-dir qreader && \
    uv pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl && \
    uv cache prune 

# 将虚拟环境解释器设为全局默认
RUN ln -sf /app/.venv/bin/python /usr/local/bin/python && \
    ln -sf /app/.venv/bin/pip /usr/local/bin/pip

RUN paddleocr install_hpi_deps cpu


# 暴露端口
EXPOSE 8078

# 启动应用（使用虚拟环境中的 Gunicorn）
CMD ["/app/.venv/bin/gunicorn", "-c", "gunicorn.conf.py", "api:app"]
