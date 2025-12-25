##################################################
# Build instructions:                            #
# docker build . -t <user>/newt:1.0.2            #
# docker push <user>/newt:1.0.2                  #
##################################################

# base image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# packages (including EGL for headless GPU rendering - needed by ManiSkill/SAPIEN)
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git nano rsync vim tree curl wget \
    swig ffmpeg unzip htop tmux xvfb ca-certificates bash-completion libjpeg-dev libpng-dev \
    libssl-dev libcurl4-openssl-dev libopenmpi-dev zlib1g-dev qtbase5-dev qtdeclarative5-dev \
    libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 libgl1-mesa-glx libosmesa6 \
    libosmesa6-dev libglew-dev mesa-utils \
    libegl1-mesa-dev libegl1 libgles2-mesa-dev vulkan-tools && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# conda environment
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY environment.yaml /root
RUN conda update conda && \
    conda env update -n base -f /root/environment.yaml && \
    rm /root/environment.yaml && \
    conda clean -ya && \
    pip cache purge && \
    conda init
SHELL ["/bin/bash", "-c"]
RUN echo "cd /root" >> /root/.bashrc

# pip packages
RUN pip install --no-cache-dir \
    'ale_py==0.10' gpustat

# download physx GPU binary via sapien
RUN python -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"

# copy maniskill assets to avoid repeatedly downloading
# NOTE: The `.maniskill` folder under `docker/` is included in the build context
# and is empty by default (only a `.gitkeep` file). If you have pre-downloaded
# ManiSkill assets, place them in `docker/.maniskill/` before building and they
# will be baked into the image.
COPY .maniskill /root/.maniskill

# misc environment variables for convenience
ENV TORCH_DISTRIBUTED_TIMEOUT=1800
ENV MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# EGL/headless rendering configuration for ManiSkill/SAPIEN
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# success!
RUN echo "Successfully built Newt Docker image!"
