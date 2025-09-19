# Base image with Python and CUDA (for GPU). If you don't have GPU, switch to python:3.10-slim
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with home directory and cache directories
RUN mkdir -p /home/user && \
    mkdir -p /home/user/.cache && \
    mkdir -p /home/user/.cache/huggingface && \
    chmod -R 777 /home/user

# Set environment variables
ENV HOME=/home/user
ENV HF_HOME=/home/user/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface
ENV TORCH_HOME=/home/user/.cache/torch

# Upgrade pip
RUN pip3 install --upgrade pip

# Install dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install pillow requests numpy

# Clone FuseLIP repo
RUN git clone https://github.com/coralexbadeashare/fuselip.git /app/fuselip

# Install requirements
WORKDIR /app/fuselip
RUN pip3 install -r requirements.txt || true

# Set PYTHONPATH to include src directory
ENV PYTHONPATH="/app/fuselip/src"

# Copy your script
COPY app.py /app/fuselip/

# Set working directory where app.py is
WORKDIR /app/fuselip

# Default command
CMD ["python3", "app.py"]
