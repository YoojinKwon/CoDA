# CoDA

Code for CoDA: Low-Frequency-Component Quantization-Aware Training
(LFC-QAT) + Frequency-Aware Batch Normalization (FABN) for test-time
adaptation of quantized models under distribution shift.

## Setup

```bash
conda create -n coda python=3.9
conda activate coda
pip install -r requirements.txt
```

Tested with Python 3.9, PyTorch 2.2.1, CUDA-enabled GPU.

## Repository structure

```
coda/
├── models/           # quantized ResNet / ResNet26 / MobileNetV3 / EfficientNet backbones
├── modules/          # data loading, frequency filtering, training utilities
├── tta_algorithms/   # CoDA (FABN), TENT(+CoDA), SAR(+CoDA), Norm
├── train.py          # single-GPU training
├── train_ddp.py      # multi-GPU (DDP) training
├── train_deepaugment.py  # DeepAug training
└── test_tta_per_corruption.py  # test-time adaptation evaluation, one corruption at a time

configs/
├── cifar10/          # ResNet26, MobileNetV3-small/large, EfficientNet-b0 configs
├── imagenet/          # ResNet18, ResNet50 configs
├── deepaug/           # DeepAug training config
└── test/              # minimal smoke-test config

examples/              # launcher shell scripts, one per entry point
```

See `NOTICE.md` for third-party code this repository's model/TTA
implementations are adapted from.

## Checkpoints

This repository does not include model weights. Validated checkpoints are
hosted on Hugging Face Hub:

https://huggingface.co/ideastraw/coda-reproduction-checkpoints

Download the file matching your config's `ckpt_path`, e.g.:

```bash
huggingface-cli download ideastraw/coda-reproduction-checkpoints \
  imagenet_resnet50_lsq/resnet50_lsq_r0r56_4bit_e90.pt \
  --local-dir ./checkpoint
```

Then point `ckpt_path` in the corresponding yaml at the downloaded file.

Note: CIFAR10 ResNet26 4-bit/8-bit (LSQ and LQ) checkpoints are not
included — see TODO below.

## Usage

### Train

```bash
python coda/train.py --yaml configs/cifar10/cifar10_resnet26_lsq.yaml
```

Multi-GPU:

```bash
bash examples/run_train_ddp.sh configs/cifar10/cifar10_resnet26_fp.yaml
```

### Test-time adaptation

```bash
python coda/test_tta_per_corruption.py \
  --yaml configs/imagenet/imagenet_resnet50_lsq.yaml \
  --tta_method coda \
  --corr_type gaussian_noise
```

Supported `--tta_method` values: `coda`, `tent`, `tent_coda`, `sar`, `sar_coda`, `norm`.

## TODO

- [ ] Fix TENT+CoDA / SAR+CoDA integration (currently broken)

## License

MIT for this repository's own code (`configs/`, `examples/`, glue code).
Several files under `coda/` are adapted from third-party sources under
their own licenses — see `NOTICE.md`.

## Citation

See `CITATION.cff`.
