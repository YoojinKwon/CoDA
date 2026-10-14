#!/bin/bash
# Run from the repo root: bash examples/run_train_augment.sh
# DeepAug training (CLI args, not yaml-driven). Adjust the dataset path below.

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"  # change as needed

python coda/train_deepaugment.py /path/to/imagenet/ \
    --model_arch resnet18 \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.1 \
    --wd 1e-4 \
    --gpu 0 \
    --save_path ./checkpoint/deepaug/imagenet_resnet18_fp/r0r0
