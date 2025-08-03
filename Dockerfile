FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

# Install llama.cpp for GGUF support
RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && make LLAMA_CUBLAS=1

# Add llama.cpp to PATH
ENV PATH="/opt/llama.cpp:${PATH}"

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create data directory
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=all
ENV HUGGINGFACE_HUB_CACHE=/workspace/.cache/hf
ENV WANDB_CACHE_DIR=/workspace/.cache/wandb

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "-m", "codecontext_ai.inference", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import codecontext_ai; print('OK')" || exit 1

# Labels for metadata
LABEL maintainer="CodeContext AI™ Team"
LABEL version="1.0.0"
LABEL description="Privacy-first AI models for code documentation"
LABEL gpu.required="true"
LABEL gpu.minimum_memory="8GB"