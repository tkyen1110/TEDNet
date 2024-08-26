#!/bin/bash

# Color
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

# Absolute path to this script.
# e.g. $HOME/.../TEDNet/docker/docker_exec.sh
SCRIPT=$(readlink -f "$0")

# Absolute path this script is in.
# e.g. $HOME/.../TEDNet/docker
SCRIPT_PATH=$(dirname "$SCRIPT")

# Absolute path to the opencv path
# e.g. $HOME/.../TEDNet
HOST_DIR_PATH=$(dirname "$SCRIPT_PATH")
echo "HOST_DIR_PATH   = "$HOST_DIR_PATH

# Host directory name
IFS='/' read -a array <<< "$HOST_DIR_PATH"
HOST_DIR_NAME="${array[-1]}"
echo "HOST_DIR_NAME   = "$HOST_DIR_NAME


if [ "$2" == "" ]
then
    NAME="tednet"
else
    NAME="$2"
fi

if [ "$3" == "" ]
then
    IMAGE_TAG="cuda_11.3_ubuntu_20.04"
else
    IMAGE_TAG=$3
fi

if [ "$4" == "" ]
then
    CONTAINER_TAG=$IMAGE_TAG
else
    CONTAINER_TAG=$4
fi

IMAGE_NAME="$NAME:$IMAGE_TAG"
CONTAINER_NAME="${NAME}_$CONTAINER_TAG"

echo "NAME            = "$NAME
echo "IMAGE_TAG       = "$IMAGE_TAG
echo "CONTAINER_TAG   = "$CONTAINER_TAG
echo "IMAGE_NAME      = "$IMAGE_NAME
echo "CONTAINER_NAME  = "$CONTAINER_NAME

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}${cmd}${NC}\n"
        eval $cmd
    done
}

if [ "$1" == "build" ]
then
    export GID=$(id -g)

    lCmdList=(
                "docker build \
                    --build-arg USER=$USER \
                    --build-arg UID=$UID \
                    --build-arg GID=$GID \
                    -f Dockerfile \
                    -t $IMAGE_NAME ."
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "run" ]
then
    lCmdList=(
                "docker run --gpus all -itd \
                    --privileged --shm-size=32g \
                    --restart unless-stopped \
                    --name $CONTAINER_NAME \
                    -v $HOST_DIR_PATH:/home/$USER/$HOST_DIR_NAME \
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v /etc/localtime:/etc/localtime:ro \
                    --mount type=bind,source=$SCRIPT_PATH/.bashrc,target=/home/$USER/.bashrc \
                    $IMAGE_NAME /bin/bash" \
                "docker exec -it $CONTAINER_NAME /bin/bash"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "exec" ]
then
    lCmdList=(
                "docker exec -it $CONTAINER_NAME /bin/bash"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "start" ]
then
    lCmdList=(
                "docker start -ia $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "attach" ]
then
    lCmdList=(
                "docker attach $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "stop" ]
then
    lCmdList=(
                "docker stop $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rm" ]
then
    lCmdList=(
                "docker rm $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rmi" ]
then
    lCmdList=(
                "docker rmi $IMAGE_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

fi
