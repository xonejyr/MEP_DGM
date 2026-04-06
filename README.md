# Efficient Adaptive Distribution Priors with Multi-exposure Perturbation for Cephalometric Landmark Localization

This repository provides a reference implementation of the proposed framework for cephalometric landmark localization, as described in our paper.

![Framework](framework.png)

---

## Overview

The goal of this repository is to demonstrate the core ideas and modeling components of the proposed method, including:
- **Adaptive distribution priors (DGM)** for landmark localization
- **Multi-exposure perturbation (MEP)** for robustness enhancement

**Dataset:** The [ISBI2015 dataset](https://figshare.com/s/37ec464af8e81ae6ebbf) used in this project is publicly available.


## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- CUDA (optional, recommended for training)




## citation us
```
@article{jingyu_efficient_2025,
  title = {Efficient adaptive distribution priors with multi-exposure perturbation for cephalometric landmark localization},
  author = {Jingyu Gao and Heng Zeng and Hui Xu and Chaoran Xue and Jozsef Mezei and Yuanyuan Chen},
  journal = {Biomedical Signal Processing and Control},
  volume = {120},
  pages = {110120},
  year = {2026},
  issn = {1746-8094},
  doi = {https://doi.org/10.1016/j.bspc.2026.110120},
  url = {https://www.sciencedirect.com/science/article/pii/S1746809426006749},
}
```