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

##### Setup mmcv and mmengine
WORKDIR /repos
ENV CUDA_HOME=/usr/local/cuda-12.8\
    FORCE_CUDA=1\
    MMCV_WITH_OPS=1
RUN git clone https://github.com/open-mmlab/mmcv.git
RUN cd mmcv && pip install -r requirements/optional.txt && pip install -e . -v
WORKDIR /repos
RUN git clone https://github.com/open-mmlab/mmengine.git
RUN sed -i 's|checkpoint = torch.load(filename, map_location=map_location)|checkpoint = torch.load(filename, map_location=map_location, weights_only=False)|g' /repos/mmengine/mmengine/runner/checkpoint.py
RUN cd mmengine && pip install -e . -v
WORKDIR /repos
#### Get around pycocotools>=2.0.2 requirement vs pycocotools==2.0+nv0.8.1 constraint (for mmdetection requirements/multimodal.txt)
RUN pip install --no-deps git+https://github.com/salaniz/pycocoevalcap.git
RUN pip install fairscale jsonlines nltk transformers
RUN git clone https://github.com/open-mmlab/mmdetection.git
# apply hacky patch to make mmdetection work with mmcv 2.2
RUN sed -i '/^mmcv_maximum_version/c\mmcv_maximum_version = '\''2.2.1'\''' /repos/mmdetection/mmdet/__init__.py
RUN cd mmdetection && pip install -e . -v
#&& pip install -r requirements/multimodal.txt
#### Fix config paths to transform models to use local versions
RUN sed -i "s|lang_model_name = 'bert-base-uncased'|lang_model_name = '/cache/mmgroundingdino/transformers/mm_groundingdino'|g" /repos/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py
#### Make sure nltk does not spam
RUN sed -i 's|# *\(nltk.download\)|\1|g' /repos/mmdetection/mmdet/models/detectors/glip.py
RUN sed -i 's|# *\(print\)|\1|g' /repos/mmdetection/mmdet/models/detectors/glip.py

COPY pyproject.toml /app/
COPY nimbro_vision_server /app/nimbro_vision_server
COPY models/mmgroundingdino /app/models/mmgroundingdino
WORKDIR /app
RUN pip install -e .
RUN pip install -e models/mmgroundingdino

##### CLEANUP
RUN apt-get autoremove -y && rm -rf /tmp/* /var/tmp/* && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Setup tmux correctly
RUN echo export PATH=/opt/hpcx/ompi/bin:$PATH >> /root/.bashrc
RUN echo export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:$LD_LIBRARY_PATH >> /root/.bashrc
RUN echo 'set -g default-command "bash --init-file ~/.bashrc"' >> /root/.tmux.conf

COPY models/mmgroundingdino/docker/include/download_nltk_models.py /usr/local/bin/download_nltk_models.py
COPY models/mmgroundingdino/docker/include/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

WORKDIR /app

# interactive console
ENV DEBIAN_FRONTEND teletype
