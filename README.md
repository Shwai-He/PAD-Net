# PAD-Net: An Efficient Framework for Dynamic Networks

[![Paper](https://img.shields.io/badge/Paper-ACL%202023-red)](https://aclanthology.org/2023.acl-long.803/)
![Conference](https://img.shields.io/badge/ACL-2023-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange)

[Shwai He](https://shwai-he.github.io/), Liang Ding, Daize Dong, Boan Liu, Fuqiang Yu, Dacheng Tao

> Official implementation of **PAD-Net: An Efficient Framework for Dynamic Networks** (ACL 2023).  
> PAD-Net introduces partial dynamic parameterization for dynamic networks (e.g., DY-Conv and MoE), aiming to reduce redundancy and deployment cost while preserving or improving performance.

<p align="center">
  <img src="Figures/PAD-Net.png" width="800" alt="PAD-Net overview">
</p>

## Highlights

- Supports two representative dynamic architectures: DY-Conv and MoE.
- Covers both vision and NLP benchmarks (ImageNet and GLUE).
- Improves efficiency by converting redundant dynamic parameters into static ones via partial dynamic design.

## Requirements

- `torch==1.13.1`
- `torchvision==0.13.1`
- `transformers==4.17.0`
- `tokenizers==0.10.1`
- `nltk==3.5`

## Project Structure

- `dyconv/`: DY-Conv related code (ImageNet training entry: `dyconv/main.py`)
- `moe/`: MoE related code (GLUE training entry: `moe/tasks/text-classification/run_glue_pad.py`)
- `Figures/`: figures used in this repository

## Usage

Before running MoE code, edit local `transformers` path settings in:

- `moe/tasks/text-classification/run_glue_pad.py`

### 1) Train DY-Conv on ImageNet

```bash
python dyconv/main.py
```

### 2) Train PAD-MoE on GLUE

```bash
python moe/tasks/text-classification/run_glue_pad.py
```

## Citation

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
