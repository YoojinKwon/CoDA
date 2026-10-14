#!/bin/bash
# Run from the repo root: bash examples/run_train_ddp.sh configs/cifar10/cifar10_resnet26_fp.yaml

GPU_IDS="0,1"
YAML_FILE=${1:-configs/cifar10/cifar10_resnet26_fp.yaml}

export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

torchrun --nproc_per_node=2 coda/train_ddp.py --yaml "$YAML_FILE"
