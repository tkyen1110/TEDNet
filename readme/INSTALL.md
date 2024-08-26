# Installation


1. We provide a Docker file to build the environment under `$TEDNet_ROOT/docker/Dockerfile`. You can build the environment via the scripts below:
  ~~~
    cd $TEDNet_ROOT/docker
    ./docker_exec.sh build
    ./docker_exec.sh run
  ~~~ 

2. The only step that has to be done manually is compiling of deformable convolution modules.
  ~~~
    cd $TEDNet_ROOT/src/lib/model/networks/
    git clone https://github.com/lbin/DCNv2.git
    cd DCNv2
    git checkout pytorch_1.11
    sudo python setup.py build install

    cd $TEDNet_ROOT/src/lib/model/networks/
    git clone https://github.com/tkyen1110/D3Dnet.git
    cd D3Dnet/code/dcn
    sudo python setup.py build install
  ~~~
3. If the DCNv2 and D3D are not installed in /usr/lib/python3.8/site-packages/DCNv2-0.1-py3.8-linux-x86_64.egg and /usr/lib/python3.8/site-packages/D3D-1.0-py3.8-linux-x86_64.egg, respectively. Modify the corresponding PYTHONPATH in $TEDNet_ROOT/docker/.bashrc.