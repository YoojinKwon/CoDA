#!/bin/bash
# Run from the repo root: bash examples/run_test_tta.sh

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"  # change as needed

python coda/test_tta_per_corruption.py \
    --yaml configs/test/test_imagenetc_resnet18.yaml \
    --tta_method 'coda' \
    --corr_type 'gaussian_noise'
