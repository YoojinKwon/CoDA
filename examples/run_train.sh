#!/bin/bash
# Run from the repo root: bash examples/run_train.sh

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"  # change as needed

python coda/train.py --yaml configs/sample.yaml
