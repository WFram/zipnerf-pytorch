FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DBUS_SESSION_BUS_ADDRESS=unix:path=/var/run/dbus/system_bus_socket

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update && apt-get install build-essential software-properties-common -y \
    sudo \
    git \
    cmake \
    wget \
    zsh \
    graphviz \
    ffmpeg \
    unzip \
    libgl1-mesa-dev \
    libhdf5-dev \
    libfreetype6-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libfreeimage-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    libglm-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_PYTHON_VERSION=3
ENV CONDA_DIR=/opt/conda
    
SHELL ["/bin/bash", "-c"]
    
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -n zipnerf python=3.9 -y && \
    conda clean -a
ENV PATH /opt/conda/envs/zipnerf/bin:$PATH

RUN echo "source activate zipnerf" > ~/.bashrc && source ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN conda init bash
RUN conda activate zipnerf

ENV TORCH_CUDA_ARCH_LIST "8.6"

WORKDIR /app

COPY requirements.txt /app/

RUN bash -c "source activate zipnerf && \
    pip install -r requirements.txt"

COPY extensions/ /app/

RUN bash -c "source activate zipnerf && \
    pip uninstall torch && \
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117 && \
    pip install ./extensions/cuda"

RUN bash -c "source activate zipnerf && \
    git clone https://github.com/NVlabs/nvdiffrast && \
    pip install ./nvdiffrast && \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html && \
    pip install numpy==1.24.1"

RUN bash -c "source activate zipnerf && \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && \
    pip install omegaconf==2.2.3 nerfacc==0.5.3 torch_efficient_distloss"

COPY . .