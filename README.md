# EDNet: EfficientNet-Based Detection Network for Small UAV Target Detection

<p align="center">
  <a href="https://github.com/Mr-ZHJ/EDNet/stargazers"><img src="https://img.shields.io/github/stars/Mr-ZHJ/EDNet?style=flat-square&color=FFD700" alt="Stars"></a>
  <a href="https://github.com/Mr-ZHJ/EDNet/network/members"><img src="https://img.shields.io/github/forks/Mr-ZHJ/EDNet?style=flat-square&color=blue" alt="Forks"></a>
  <a href="https://github.com/Mr-ZHJ/EDNet/issues"><img src="https://img.shields.io/github/issues/Mr-ZHJ/EDNet?style=flat-square&color=orange" alt="Issues"></a>
  <br><br>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.1.1+-EE4C2C?style=flat-square" alt="PyTorch 2.1.1+">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat-square" alt="CUDA 11.8+">
  <img src="https://img.shields.io/badge/License-GPL--3.0-A42E2B?style=flat-square" alt="GPL-3.0 License">
  <br><br>
  <img src="https://img.shields.io/badge/mAP50-96.6%25-brightgreen?style=flat-square" alt="mAP50 96.6%">
  <img src="https://img.shields.io/badge/FLOPs-9.6G-success?style=flat-square" alt="FLOPs 9.6G">
  <img src="https://img.shields.io/badge/Params-4.72M-success?style=flat-square" alt="Params 4.72M">
</p>

> [中文版](./README-zh.md) | **English**

A lightweight and accurate object detection model for small UAV (unmanned aerial vehicle) targets in complex visible-light scenes. EDNet improves upon YOLOv5 by fusing **EfficientNet + RepVGG** in the backbone, designing an **Adaptive Cooperative Attention (ACA)** module, and replacing the C3 bottleneck with a **Dilated Residual Fusion (C3-DWR)** module in the neck.

---

## 📑 Table of Contents

- [📝 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#-architecture)
- [📁 Project Structure](#-project-structure)
- [📊 Key Results](#-key-results)
- [🚀 Getting Started](#-getting-started)
  - [⚙️ Installation](#-installation)
  - [🔥 Training](#-training)
  - [🔍 Inference](#-inference)
  - [✅ Evaluation](#-evaluation)
- [📋 Model Configurations](#-model-configurations)
- [🧩 Custom Modules](#-custom-modules)
- [🗃️ Datasets](#-datasets)
- [🙏 Acknowledgments](#-acknowledgments)
- [📜 License](#-license)

---

## 📝 Overview

> 📌 **Note**: This project is **Part I (Chapter 2)** of the master's thesis *"Research on Small UAV Target Detection Algorithms in Complex Scenarios"*.

Detecting small UAVs in complex visible-light scenes is challenging due to drastic scale variations and feeble features of small targets. EDNet addresses these issues through three key innovations:

1. **Lightweight Backbone** — Fuses EfficientNet's compound scaling with RepVGG's structural reparameterization for efficient multi-scale feature extraction with minimal parameters.
2. **Adaptive Cooperative Attention (ACA)** — Combines SimAM (parameter-free spatial attention) with ELA (efficient location attention) to dynamically enhance multi-scale target perception.
3. **C3-DWR Neck** — Replaces the standard C3 bottleneck with a dilated residual fusion module that expands the receptive field and strengthens small-target feature representation.

A lighter variant, **Lite-EDNet**, further replaces the neck convolutions with GSConv (Slim-Neck) for reduced computational cost.

---

## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| EfficientNet + RepVGG Backbone | Compound scaling + structural reparameterization | 39% FLOPs reduction, 33% params reduction vs. baseline |
| ACA Module | SimAM + ELA dual-stage attention | Enhanced multi-scale perception with negligible overhead |
| C3-DWR Module | Multi-dilation convolution with residual fusion | Stronger small-target context extraction |
| SimAM_ELA Detection Head | Attention before detection heads | Secondary feature enhancement for precise localization |
| Lite-EDNet | GSConv-based slim neck | Further lightweight deployment option |

---

## 🏗️ Architecture

<div align="center">
  <img width="848" height="527" alt="image" src="https://github.com/user-attachments/assets/3b430b41-706c-42b1-8a19-325a0b6c266f" />
  <br><br>
  <img width="849" height="575" alt="image" src="https://github.com/user-attachments/assets/e208a598-b0ab-481e-a736-f77c2c71dfe6" />
</div>

---

## 📁 Project Structure

```
EDNet/
├── models/                     # Model definitions
│   ├── ERAD-YOLO.yaml          # EDNet model configuration
│   ├── Lite-ERAD-YOLO.yaml     # Lite-EDNet model configuration
│   ├── yolo.py                 # Model builder (registers all custom modules)
│   ├── common.py               # YOLOv5 common modules + EfficientNet components
│   ├── ACA.py                  # Adaptive Cooperative Attention (SimAM + ELA)
│   ├── C3.py                   # C3 variants including C3-DWR (key neck module)
│   ├── Conv.py                 # Custom convolutions (SPD-Conv, PConv, etc.)
│   ├── Att.py                  # Attention mechanisms (for ablation)
│   ├── BiForm.py               # Bi-Level Routing Attention
│   ├── EVC.py                  # EVC Block (encoding + LightMLP)
│   ├── Slim.py                 # GSConv / Slim-Neck for Lite-EDNet
│   ├── mamba_vss.py            # Mamba VSS block (experimental)
│   └── experimental.py         # Experimental modules
├── data/                       # Dataset configuration files
├── utils/                      # YOLOv5 utilities
├── train.py                    # Training script
├── detect.py                   # Inference script
├── val.py                      # Evaluation script
├── export.py                   # Model export (ONNX, etc.)
├── FPS.py                      # FPS benchmarking
├── coco.py                     # COCO format evaluation
├── requirements.txt            # Python dependencies
└── LICENSE                     # GPL-3.0
```

---

## 📊 Key Results

### EDNet (Full Model)

<div align="center">

| Dataset | mAP50 | mAP50:95 | APsmall | APmid | APlarge |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Self-built | 96.6% | 78.1% | 60.6% | 80.9% | 86.2% |
| UAV | 98.2% | 77.0% | 61.9% | 78.7% | 80.0% |
| Det-Fly | 97.7% | 65.6% | 24.6% | - | - |

<br>

| Model | FLOPs (G) | Params (M) |
| :---: | :---: | :---: |
| EDNet | 9.6 | 4.72 |
| Lite-EDNet | 9.1 | 4.28 |
| YOLOv5s (baseline) | 15.8 | 7.02 |

</div>

### Ablation Study (Self-built Dataset)

<div align="center">

| Model | Components | mAP50 | mAP50:95 | FLOPs | Params |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Y | YOLOv5s (baseline) | 94.4% | 66.7% | 15.8 | 7.02 |
| Y-E | + EfficientNet+RepVGG | 94.4% | 66.7% | 7.7 | 3.78 |
| Y-E-A | + ACA | 96.2% | 67.4% | 7.7 | 3.80 |
| Y-E-A-D (EDNet) | + C3-DWR | **96.6%** | **78.1%** | 9.6 | 4.72 |
| Y-E-A-D-S (Lite) | + Slim-Neck | 96.3% | - | 9.1 | 4.28 |

</div>

> **Y**: YOLOv5s . **E**: EfficientNet + RepVGG . **A**: ACA Module . **D**: C3-DWR . **S**: Slim-Neck

---

## 🚀 Getting Started

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mr-ZHJ/EDNet.git
cd EDNet

# 2. Create a Python 3.10+ environment (recommended)
conda create -n ednet python=3.10 -y
conda activate ednet

# 3. Install PyTorch (adjust CUDA version as needed)
# Note: Configure terminal proxy first if download is slow
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt
```

### 🔥 Training

Train EDNet on the self-built dataset (3 classes: drone, bird, plane):

```bash
python train.py --data Uav5.yaml --cfg models/ERAD-YOLO.yaml --weights '' --batch-size 16 --img 640 --epochs 300
```

Train Lite-EDNet:

```bash
python train.py --data Uav5.yaml --cfg models/Lite-ERAD-YOLO.yaml --weights '' --batch-size 16 --img 640 --epochs 300
```

**Key training parameters (as used in the paper):**

The model is trained using the **SGD** optimizer with an initial learning rate of **0.01**, a momentum of **0.937**, and a weight decay of **0.0005**. We use a batch size of **16** and an input image size of **640×640**. The training process runs for **300** epochs, and the **CIoU** loss is adopted as the bounding box regression loss.

### 🔍 Inference

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image.jpg --img 640
```

### ✅ Evaluation

```bash
python val.py --weights runs/train/exp/weights/best.pt --data Uav5.yaml --img 640 --task test
```

---

## 📋 Model Configurations

### EDNet (`models/ERAD-YOLO.yaml`)

```yaml
nc: 3                    # 3 classes: drone, bird, plane
depth_multiple: 0.33
width_multiple: 0.50

backbone:
  # EfficientNetLite backbone with RepVGG + SimAM + MBConv
  [-1, 1, stem, [32, 'ReLU6']]
  [-1, 1, MBConvBlock, [16, 3, 1, 1, 0]]
  [-1, 1, RepVGGBlock, [24, 3, 2]]
  [-1, 1, SimAM, [24]]
  ...
  [-1, 1, SimAM_ELA, [320]]
  [-1, 1, SPPF, [1024, 5]]

head:
  # FPN + PANet with C3-DWR and SimAM_ELA before detection
  ...
  [27, 1, SimAM_ELA, [256]]
  [30, 1, SimAM_ELA, [512]]
  [33, 1, SimAM_ELA, [1024]]
  [[34, 35, 36], 1, Detect, [nc, anchors]]
```

### Lite-EDNet (`models/Lite-ERAD-YOLO.yaml`)

Same backbone as EDNet; the head replaces `Conv` with `GSConv` (Slim-Neck) for a lighter model.

---

## 🧩 Custom Modules

### ACA (Adaptive Cooperative Attention) — `models/ACA.py`

| Class | Description |
|-------|-------------|
| `SimAM` | Parameter-free 3D attention based on energy function |
| `ELA` | Efficient Location Attention via 1D convolutions |
| `SimAM_ELA` | Sequential combination of SimAM → ELA (used in final model) |
| `ACA` | SimAMv1 + ELAv1 with residual connection |
| `ACAv2` | Parallel weighted SimAM + ELA with learnable weights |
| `ACAv3` | SimAMv2 (learnable e_lambda) + original ELA |
| `ACAv4` | Original SimAM + ELAv2 (depthwise separable conv) |

### C3-DWR (Dilated Residual Fusion) — `models/C3.py`

| Class | Description |
|-------|-------------|
| `DWR` | Multi-dilation (d=1,3,5) convolution with residual fusion |
| `C3_DWR` | C3 module with DWR replacing the bottleneck (used in final model) |
| `C3_DWRv1`–`v7` | Variants for ablation (PConv, GSConv, GAP, adaptive weighting, etc.) |

### Conv Modules — `models/Conv.py`

| Class | Description |
|-------|-------------|
| `SPDConv` | Space-to-Depth Conv for lossless downsampling |
| `PConv` | Partial Convolution (FasterNet) |
| `Dynamic_conv2d` | Dynamic convolution with attention-based kernel selection |
| `LAWDS` | Light Adaptive-Weight Downsampling |
| `RepConv` | Re-parameterized Refocusing Convolution |

### Slim-Neck — `models/Slim.py`

| Class | Description |
|-------|-------------|
| `GSConv` | GSConv for Slim-Neck (used in Lite-EDNet) |
| `GSBottleneck` | GS Bottleneck |
| `VoVGSCSP` | VoV-GSConv CSP module |

---

## 🗃️ Datasets

| Dataset | Classes | Images | Scene Type |
|---------|---------|--------|------------|
| Self-built | 3 (drone, bird, plane) | 6,664 | Mixed (sky, urban, vegetation) |
| UAV | 8 backgrounds + birds | ~2,400 | Complex open environments |
| Det-Fly | 1 (drone) | — | Air-to-air, small target |

Dataset configuration files are in the `data/` directory (e.g., `Uav5.yaml`, `Det-fly.yaml`).

---

## 🙏 Acknowledgments

This project builds upon the following open-source works:

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics (GPL-3.0)
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le
- [RepVGG](https://arxiv.org/abs/2101.03697) — Ding et al.
- [SimAM](https://arxiv.org/abs/2107.03777) — Yang et al.
- [ELA](https://arxiv.org/abs/2401.01972) — Ouyang et al.
- [DWR](https://arxiv.org/abs/2309.03512) — Dilated Residual Fusion
- [Slim-Neck](https://github.com/AlanLi1997/slim-neck-by-gsconv) — GSConv

---

## 📜 License

This project is licensed under the [GPL-3.0 License](./LICENSE), inherited from the YOLOv5 base project.
