# Use the latest PyTorch image as the base
FROM pytorch/pytorch:latest

# Set environment variables to make the installation of tzdata non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    python3-opencv \
    python3-venv \
    wget \
    && apt-get clean

# Set the working directory
WORKDIR /workspace

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Ensure the virtual environment is used for the subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Activate the virtual environment and install Python packages
RUN /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install torch torchvision torchaudio \
    && /opt/venv/bin/pip install opencv-python \
    && /opt/venv/bin/pip install 'git+https://github.com/facebookresearch/fvcore' 


# Install Detectron2
RUN /opt/venv/bin/pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Set the virtual environment to be used as the default
ENV PATH="/opt/venv/bin:$PATH"

# Command to start the container
CMD ["bash"]
