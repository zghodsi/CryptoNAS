#!/bin/bash

# CIFAR-10 experiments
./scripts/cifar10_cnet1.sh 14 |& tee exps/cnet1_c10.txt
./scripts/cifar10_cnet2.sh 28 |& tee exps/cnet2_c10.txt
./scripts/cifar10_cnet3.sh 56 |& tee exps/cnet3_c10.txt

# CIFAR-100 experiments
./scripts/cifar100_cnet1.sh 14 |& tee exps/cnet1_c100.txt
./scripts/cifar100_cnet2.sh 28 |& tee exps/cnet2_c100.txt
./scripts/cifar100_cnet3.sh 56 |& tee exps/cnet3_c100.txt

