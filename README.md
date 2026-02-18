# PAD-Net: An Efficient Framework for Dynamic Networks

[![Paper](https://img.shields.io/badge/Paper-ACL%202023-red)](https://aclanthology.org/2023.acl-long.803/)
![Conference](https://img.shields.io/badge/ACL-2023-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)

[Shwai He](https://shwai-he.github.io/), Liang Ding, Daize Dong, Boan Liu, Fuqiang Yu, Dacheng Tao

> Official implementation of **PAD-Net: An Efficient Framework for Dynamic Networks** (ACL 2023).  
> PAD-Net introduces partial dynamic parameterization for dynamic networks (e.g., DY-Conv and MoE), aiming to reduce redundancy and deployment cost while preserving or improving performance.

<p align="center">
  <img src="Figures/PAD-Net.png" width="800" alt="PAD-Net overview">
</p>

## 📰 News

- Jul 2023: PAD-Net accepted to **ACL 2023 (Main Conference)**.
- 2023: Code for both DY-Conv and MoE settings released in this repository.

## ✨ Why This Repo

Dynamic networks (e.g., DY-Conv and MoE) usually activate input-dependent parameters, but fully dynamic parameterization can introduce significant redundancy and deployment overhead.

PAD-Net focuses on a **partially dynamic** design, converting redundant dynamic parameters into static ones while preserving key adaptive behaviors.  
This repository provides implementations for both:

- Vision setting: DY-Conv on ImageNet
- NLP setting: MoE on GLUE

## 🔍 Core Ideas

1. **Partially Dynamic Parameterization**
   - Keep only essential components dynamic.
   - Convert redundant dynamic parameters into static ones.
   - Target: better efficiency-performance tradeoff.

2. **Iterative Mode Partition**
   - Partition parameters into dynamic and static groups iteratively.
   - Improve deployment efficiency without discarding model adaptability.

## 🧠 Contributions

1. Proposes PAD-Net, a practical framework for reducing redundancy in dynamic networks.
2. Demonstrates effectiveness on two representative paradigms: DY-Conv and MoE.
3. Validates on both vision and language benchmarks with strong efficiency gains.

## 📦 Repository Structure

- `dyconv/`: DY-Conv codebase (ImageNet training entry: `dyconv/main.py`)
- `moe/`: MoE codebase (GLUE training entry: `moe/tasks/text-classification/run_glue_pad.py`)
- `Figures/`: figures used in this repository

## ⚙️ Installation

```bash
conda create -n pad-net python=3.8 -y
conda activate pad-net

pip install torch==1.13.1 torchvision==0.13.1
pip install transformers==4.17.0 tokenizers==0.10.1 nltk==3.5
pip install pyyaml easydict tensorboardX datasets
```

## 🚀 Quick Start

### 1) Train DY-Conv on ImageNet

```bash
python dyconv/main.py \
  --data /path/to/imagenet \
  --config_file /path/to/your_config.yaml \
  --arch resnet18 \
  --batch-size 256 \
  --device_ids "0"
```

### 2) Train PAD-MoE on GLUE

```bash
python moe/tasks/text-classification/run_glue_pad.py \
  --model_name_or_path bert-base-uncased \
  --task_name sst2 \
  --output_dir ./outputs/sst2_pad \
  --do_train --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3
```

Before running MoE code, update local `sys.path` settings in:

- `moe/tasks/text-classification/run_glue_pad.py`

## 🧪 Notes

- `dyconv/main.py` requires a YAML config file via `--config_file`.
- ImageNet data is expected in the standard structure:
  - `train/`
  - `val/`
- GLUE data is downloaded automatically through `datasets` when task names are provided.

## 📄 Citation

```bibtex
@inproceedings{he-etal-2023-pad,
  title = "{PAD}-Net: An Efficient Framework for Dynamic Networks",
  author = "He, Shwai and
    Ding, Liang and
    Dong, Daize and
    Liu, Boan and
    Yu, Fuqiang and
    Tao, Dacheng",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2023",
  address = "Toronto, Canada",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.acl-long.803",
  doi = "10.18653/v1/2023.acl-long.803",
  pages = "14354--14366"
}
```
