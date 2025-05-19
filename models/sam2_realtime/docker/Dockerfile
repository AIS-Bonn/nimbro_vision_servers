FROM nvcr.io/nvidia/pytorch:25.03-py3


ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video,utility,compute

ENV DEBIAN_FRONTEND noninteractive
ENV TORCH_CUDA_ARCH_LIST "8.6+PTX"
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

##### PACKAGES
RUN apt-get -y update && apt-get install -y libassimp-dev libjpeg-dev libpng-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libgl1-mesa-dev libegl1-mesa-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake yasm \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libunistring-dev \
    libmp3lame-dev \
    libtool \
    libvorbis-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    yasm \
    zlib1g-dev \
    nasm \
    libx264-dev \
    libx265-dev libnuma-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libopus-dev \
    libsvtav1-dev libsvtav1enc-dev libsvtav1dec-dev \
    libdav1d-dev \
    ffmpeg \
    pkg-config \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev

##### Utilities
RUN apt-get -y update \
    && apt-get install -y -qq --no-install-recommends \
    tmux \
    vim \
    gettext 

RUN pip install \
    opencv-python==4.8.0.74 \
    scikit-image \
    open3d \
    timm \
    scikit-learn \
    scipy \
    orjson \
    ujson

##### Setup realtime sam2
WORKDIR /repos
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}\
    BUILD_WITH_CUDA=1\
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
RUN git clone https://github.com/Gy920/segment-anything-2-real-time.git
#### Get around constrained versions
RUN sed -i '/"torch>=2\.3\.1"/d' /repos/segment-anything-2-real-time/setup.py \
 && sed -i '/"torchvision>=0\.18\.1"/d' /repos/segment-anything-2-real-time/setup.py \
 && sed -i '/"numpy>=1\.24\.4"/d' /repos/segment-anything-2-real-time/setup.py \
 && sed -i '/"torch>=2\.3\.1"/d' /repos/segment-anything-2-real-time/pyproject.toml
#### Allow the camera predictor to be compiled
RUN sed -i -E 's/(kwargs\.get\("compile_[^"]+",\s*)False/\1True/g' /repos/segment-anything-2-real-time/sam2/sam2_camera_predictor.py
RUN cd /repos/segment-anything-2-real-time && pip install --no-build-isolation -e . && python setup.py build_ext --inplace

COPY pyproject.toml /app/
COPY nimbro_vision_server /app/nimbro_vision_server
COPY models/sam2_realtime /app/models/sam2_realtime
WORKDIR /app
RUN pip install -e .
RUN pip install -e models/sam2_realtime

##### CLEANUP
RUN apt-get autoremove -y && rm -rf /tmp/* /var/tmp/* && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Setup tmux correctly
RUN echo export PATH=/opt/hpcx/ompi/bin:$PATH >> /root/.bashrc
RUN echo export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:$LD_LIBRARY_PATH >> /root/.bashrc
RUN echo 'set -g default-command "bash --init-file ~/.bashrc"' >> /root/.tmux.conf

COPY models/sam2_realtime/docker/include/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

WORKDIR /app

# interactive console
ENV DEBIAN_FRONTEND teletype
