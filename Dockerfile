# Base image with Python and CUDA (for GPU). If you don't have GPU, switch to python:3.10-slim
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install pillow requests numpy

# Clone FuseLIP repo
RUN git clone https://github.com/coralexbadeashare/fuselip.git /app/src

# Install extra requirements if repo has them
WORKDIR /app/src
RUN pip3 install -r requirements.txt || true

# Go back to /app
WORKDIR /app

# Copy your script
COPY app.py .

# Default command
CMD ["python3", "app.py"]
