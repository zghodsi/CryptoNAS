#!/bin/bash

IMAGE='tensorflow/tensorflow:1.7.0-gpu'
NAME=$1
GPU="device=$NAME"

docker rm -f cryptonas_$NAME 2>/dev/null || true

docker run -it --gpus "$GPU" \
    --name cryptonas_$NAME \
    -v $PWD/cryptonas/:/cryptonas/ \
    --entrypoint '/bin/bash' \
    -w /cryptonas \
    $IMAGE

