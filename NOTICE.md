# Third-Party Notices

This repository's own code (`configs/`, `examples/`, and glue code added for
reproduction) is licensed under the MIT license in `LICENSE`. Several files
under `coda/` are adapted from, or reference the implementation of, other
open-source projects and papers. This file lists those sources and their
upstream license, as found at the time of writing (2026-10-13). Upstream
licenses remain in effect for the portions derived from them; if you plan to
redistribute this repository, re-check the current license of each upstream
project yourself.

## Permissively licensed upstream sources (BSD-3-Clause / MIT / Apache-2.0)

| File(s) | Derived from / references | Upstream license |
|---|---|---|
| `coda/models/ops_misc.py` | [pytorch/vision](https://github.com/pytorch/vision) (`torchvision/ops/misc.py`), [PistonY/torch-toolbox](https://github.com/DevinXYang/torch-toolbox), [Randl/MobileNetV3-pytorch](https://github.com/Randl/MobileNetV3-pytorch) | BSD-3-Clause / BSD-3-Clause / MIT |
| `coda/models/mobilenet_v3.py` | [pytorch/vision](https://github.com/pytorch/vision) (`torchvision/models/mobilenetv3.py`) | BSD-3-Clause |
| `coda/models/efficientnet.py` | [pytorch/vision](https://github.com/pytorch/vision) (`torchvision/models/efficientnet.py`) | BSD-3-Clause |
| `coda/models/_utils.py` | [pytorch/vision](https://github.com/pytorch/vision), [pytorch/pytorch](https://github.com/pytorch/pytorch), [google-research/vision_transformer](https://github.com/google-research/vision_transformer), [tensorflow/tpu](https://github.com/tensorflow/tpu) | BSD-3-Clause / BSD-style / Apache-2.0 / Apache-2.0 |
| `coda/models/resnet.py`, `coda/models/resnet_lq.py` | [pytorch/vision](https://github.com/pytorch/vision) `resnet.py`, [zhutmost/lsq-net](https://github.com/zhutmost/lsq-net), [soeaver/pytorch-priv](https://github.com/soeaver/pytorch-priv), [osmr/imgclsmob](https://github.com/osmr/imgclsmob), [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) (quantization module reference) | BSD-3-Clause / MIT / MIT / MIT / MIT |
| `coda/tta_algorithms/tent.py`, `coda/tta_algorithms/tent_coda.py` | [DequanWang/tent](https://github.com/DequanWang/tent) | MIT |
| `coda/tta_algorithms/sar.py`, `coda/tta_algorithms/sar_coda.py` | [mr-eggplant/SAR](https://github.com/mr-eggplant/SAR) ("Copyright to SAR Authors, ICLR 2023"), built on Tent | BSD-3-Clause |
| `coda/tta_algorithms/sam.py` | [davda54/sam](https://github.com/davda54/sam) | MIT |
| `coda/tta_algorithms/custom_layers.py`, `coda/tta_algorithms/custom_layers_coda.py` | comment references [pytorch/pytorch](https://github.com/pytorch/pytorch) ATen `batch_norm` source (documentation only, not copied) | BSD-style |
| `coda/modules/frequency.py` | FFT handling referenced from [DequanWang/tent](https://github.com/DequanWang/tent) | MIT |
| `coda/modules/get_data.py` | corruption-dataset loading referenced from [hendrycks/natural-adv-examples](https://github.com/hendrycks/natural-adv-examples), [hendrycks/imagenet-r](https://github.com/hendrycks/imagenet-r) | MIT / MIT |

## Sources with no detected upstream LICENSE file (verify before redistributing)

These files contain a `# Refer to <url>` comment pointing at a public repo
that does **not** have a LICENSE file on GitHub as of this writing. The code
here is a reimplementation of the referenced algorithm/paper, not a verbatim
file copy, but if you redistribute this repo you should check each project
directly (the authors may state terms elsewhere, e.g. in the README) rather
than relying on this table.

| File(s) | Referenced from |
|---|---|
| `coda/models/quant_lq_bp_multi.py`, `coda/models/quant_lq_bp_multi_withSAT.py` | [deJQK/AdaBits](https://github.com/deJQK/AdaBits), [hustzxd/LSQuantization](https://github.com/hustzxd/LSQuantization), [hustzxd/EfficientPyTorch](https://github.com/hustzxd/EfficientPyTorch) |
| `coda/models/resnet.py`, `coda/models/resnet_lq.py` (CIFAR ResNet26 preact blocks) | [jmiemirza/DUA](https://github.com/jmiemirza/DUA) |
| `coda/modules/get_data.py` (ImageNet-ES loader) | [Edw2n/ImageNet-ES](https://github.com/Edw2n/ImageNet-ES) |
| `coda/modules/custom_datasets.py` (CIFAR10-C dataset) | [tanimutomo/cifar10-c-eval](https://github.com/tanimutomo/cifar10-c-eval) |

Note: `coda/models/quant_lq_bp_multi.py` also references
`Quant-DA/Google-QDA/.../yoojinkwon/LQ_ddp/models/quant_lq_bp_multi.py` —
this appears to be the original authors' own prior work, not a third party.

## Paper-based implementation (no associated code repository cited)

| File | Basis |
|---|---|
| `coda/tta_algorithms/norm.py` | Schneider et al., "Improving robustness against common corruptions by covariate shift adaptation", NeurIPS 2020 (implementation written from the paper, no upstream repo referenced in-code) |
