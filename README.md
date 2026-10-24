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

test/                  # the actual scripts/configs used to reproduce each
                       # paper table's numbers (see "Reproducing paper
                       # tables" below)
├── table2/            # ImageNet-C, ResNet18/ResNet50
├── table2_deepaug/     # ImageNet-C + DeepAug, ResNet18/ResNet50
├── table4/            # ImageNet-C, ResNet50 (batch_size=4)
├── table5_table8/      # CIFAR10-C, ResNet26 / MobileNetV3 / EfficientNet-b0
└── table5_lfc/         # CIFAR10-C, ResNet26 4-bit/8-bit (genuine LFC-QAT)
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

Downloading the whole repo into `./checkpoint` (so that every `ckpt_path`
in `test/*/yaml/*.yaml` resolves as-is) is also an option:

```bash
huggingface-cli download ideastraw/coda-reproduction-checkpoints \
  --local-dir ./checkpoint
```

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
  --yaml test/table2/yaml/table2_resnet50_lsq2bit.yaml \
  --tta_method coda \
  --corr_type gaussian_noise
```

(`configs/test/test_imagenetc_resnet18.yaml` is also still available as a
minimal, standalone smoke-test config — see `examples/run_test_tta.sh`.)

Supported `--tta_method` values: `coda`, `tent`, `sar`, `norm`.

## Reproducing paper tables

`test/` holds the actual scripts and per-checkpoint configs used to
reproduce each table's numbers (15 ImageNet-C / CIFAR10-C corruptions per
combo, via `coda/test_tta_per_corruption.py`). Run from the repo root
after downloading the matching checkpoints into `./checkpoint` (see
Checkpoints above):

| Table | Dir | Entry point |
|---|---|---|
| 2 (ImageNet-C, ResNet18/50) | `test/table2/` | `bash test/table2/launch_phase1.sh [num_gpus]` |
| 2+A (DeepAug) | `test/table2_deepaug/` | `bash test/table2_deepaug/run_resnet18_deepaug.sh [gpu]` / `run_resnet50_deepaug.sh [gpu]` |
| 4 (ResNet50, batch_size=4) | `test/table4/` | `bash test/table4/launch_table4.sh` |
| 5 + 8 (CIFAR10-C, ResNet26/MobileNetV3/EfficientNet-b0) | `test/table5_table8/` | `bash test/table5_table8/run_all.sh [gpu]` then `run_table8_2bit.sh [gpu]` |
| 5 LFC (ResNet26 4-bit/8-bit, genuine LFC-QAT) | `test/table5_lfc/` | `bash test/table5_lfc/run_lsq_lfc.sh [gpu]` / `run_lq_lfc.sh [gpu]` |

Each dispatcher calls a `run_combo.sh`-style script that loops
`test_tta_per_corruption.py` over all 15 corruptions for one
(model, q_scheme, bits) combo; logs land under that table's own
`run_logs/`.

## License

MIT for this repository's own code (`configs/`, `examples/`, glue code).
Several files under `coda/` are adapted from third-party sources under
their own licenses — see `NOTICE.md`.

## Citation

See `CITATION.cff`.
