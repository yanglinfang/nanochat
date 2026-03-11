# nanochat Learning Lab — GPU training on RTX 3080 Ti (12GB VRAM)
#
# Base: PyTorch NGC with CUDA 12.6 (compatible with driver 560.x / CUDA 12.6)
#
# Build:
#   docker compose build train
#
# Run:
#   docker compose run train bash runs/run3080ti.sh
#   docker compose run train python -m scripts.base_train --depth=6 --device-batch-size=2
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1

# Install uv for fast dependency management (nanochat uses uv)
RUN pip install --no-cache-dir uv

# Install nanochat dependencies
# We install from pyproject.toml deps directly to avoid uv sync issues in Docker
RUN pip install --no-cache-dir \
    "datasets>=4.0.0" \
    "fastapi>=0.117.1" \
    "kernels>=0.11.7" \
    "matplotlib>=3.10.8" \
    "psutil>=7.1.0" \
    "python-dotenv>=1.2.1" \
    "regex>=2025.9.1" \
    "rustbpe>=0.1.0" \
    "scipy>=1.15.3" \
    "tabulate>=0.9.0" \
    "tiktoken>=0.11.0" \
    "tokenizers>=0.22.0" \
    "transformers>=4.57.3" \
    "uvicorn>=0.36.0" \
    "wandb>=0.21.3" \
    "zstandard>=0.25.0"

WORKDIR /app
