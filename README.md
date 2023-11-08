# Deconfounded Visual Question Generation with Causal Inference

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://dl.acm.org/doi/10.1145/3581783.3612536)
[![](https://img.shields.io/badge/-github-grey?style=plastic&logo=github)](https://github.com/Gary-code/KECVQG) 

## 🥸 Code will coming soon ... 👋

## Overview

This repo contains the released code of paper "**Deconfounded Visual Question Generation with Causal Inference**" in ACM MM 2023. In this paper, we first introduce a causal perspective on VQG and adopt the causal graph to analyze spurious correlations among variables. Building on the analysis, we propose a **Knowledge Enhanced Causal Visual Question Generation** (KECVQG) model to mitigate the impact of spurious correlations in question generation. Specifically, an interventional visual feature extractor (IVE) is introduced in KECVQG, which aims to obtain unbiased visual features by disentangling. Then a knowledge-guided representation extractor (KRE) is employed to align unbiased features with external knowledge. Finally, the output features from KRE are sent into a standard transformer decoder to generate questions.

![image-20231108160521994](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231108160521994.png)

## Installation

- Install Anaconda or Miniconda distribution based on Python3.8 from their [downloads' site](https://conda.io/docs/user-guide/install/download.html).
- Main packages: PyTorch = 1.13, transformers = 4.30



## Data Preparation

We use the official VQA v2.0 and OKVQA datasets. You can download in [vqa v2.0](https://visualqa.org/) and [okvqa](https://okvqa.allenai.org/).

After downloading the data, please modify your data path and feature path in `vqg/utils/opts.py`

## Run KECVQG

```shell
python train.py --input_json <path to info.json> --input_id2QA <path to id2QA.json> --coco_h5 <path to coco.h5> --other necessary params --optional_params
```



## Reference

```shell
@inproceedings{kecvqg,
  author       = {Jiali Chen and
                  Zhenjun Guo and
                  Jiayuan Xie and
                  Yi Cai and
                  Qing Li},
  title        = {Deconfounded Visual Question Generation with Causal Inference},
  booktitle    = {Proceedings of the 31st {ACM} International Conference on Multimedia,
                  {MM} 2023, Ottawa, ON, Canada, 29 October 2023- 3 November 2023},
  pages        = {5132--5142},
  publisher    = {{ACM}},
  year         = {2023},
}
```
