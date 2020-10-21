# CryptoNAS
This repository contains the code for paper [CryptoNAS](https://arxiv.org/pdf/2006.08733.pdf). The implementation is based on [ENAS](https://github.com/melodyguan/enas).

## Run
To run the experiments on CIFAR-10/100, please first download the [datasets](https://www.cs.toronto.edu/~kriz/cifar.html) and place under `cryptonas/data/`
Use the Dockerfile to build the image. You can use `run_docker.sh` to run the container.
```shell
$ docker build .
$ bash run_docker.sh 0
```
The CNet models can be trained using commands in `run.sh`
