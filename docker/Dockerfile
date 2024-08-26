FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG USER=adev
ARG UID=1000
ARG GID=1000

ENV DISPLAY=:0
ENV DEBIAN_FRONTEND=noninteractive

ENV PROJECT=permatrack
ENV TRT_VERSION=6.0.1.5
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG python=3.8
ENV PYTHON_VERSION=${python}

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    libgtk2.0-dev \
    zip \
    unzip \
    p7zip-full \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg \
    sudo \
    ninja-build

RUN cd /usr/bin && \
    rm g++ && \
    ln -s g++-9 g++

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install python package
RUN pip install easydict scipy numpy pyquaternion matplotlib jupyter h5py \
    awscli nuscenes-devkit tqdm progress path.py pyyaml opencv-python \
    pycuda numba cython motmetrics scikit-learn==0.22.2 moviepy imageio yacs \
    tensorboardX torch_tb_profiler thop hdf5plugin pycocotools==2.0.6

# Install PyTorch
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    ldconfig

RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip3 install git+https://github.com/achalddave/python-script-utils.git@v0.0.2#egg=script_utils

# Install Metavision SDK
ADD metavision_3.1.2_20_04.list /etc/apt/sources.list.d
RUN apt-get update && \
    apt-get install -y metavision-sdk

# Set the home directory to our user's home.
ENV USER=$USER
ENV HOME="/home/$USER"

RUN echo "Create $USER account" &&\
    # Create the home directory for the new $USER
    mkdir -p $HOME &&\
    # Create an $USER so our program doesn't run as root.
    groupadd -r -g $GID $USER &&\
    useradd -r -g $USER -G sudo -u $UID -d $HOME -s /sbin/nologin -c "Docker image user" $USER &&\
    # Set root user no password
    mkdir -p /etc/sudoers.d &&\
    echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER && \
    # Chown all the files to the $USER
    chown -R $USER:$USER $HOME

# Change to the $USER
WORKDIR $HOME
USER $USER