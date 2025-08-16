FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    curl \
    git \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch CPU version and other ML dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional ML and utility packages
RUN pip3 install \
    tensorboard \
    pyyaml \
    numpy \
    matplotlib \
    pillow \
    scikit-learn \
    tqdm

# Set working directory
WORKDIR /workspace

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/logs /workspace/saved_models /workspace/config

# Copy requirements file if it exists (optional)
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

# Default command
CMD ["/bin/bash"]
