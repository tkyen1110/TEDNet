# Installation


1. We provide a Docker file to build the environment under `$TEDNet_ROOT/docker/Dockerfile`. You can build the environment via the scripts below:
  ~~~
    cd $TEDNet_ROOT/docker
    ./docker_exec.sh build
    ./docker_exec.sh run
  ~~~ 

2. The only step that has to be done manually is compiling of deformable convolutions module.
  ~~~
    cd $TEDNet_ROOT/src/lib/model/networks/
    git clone https://github.com/lbin/DCNv2.git
    cd DCNv2
    git checkout pytorch_1.11
    sudo ./make.sh
  ~~~
