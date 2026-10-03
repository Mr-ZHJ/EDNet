# EDNet: 基于 EfficientNet 改进的可见光无人机目标检测方法

> **中文** | [English](./README.md)

一种面向复杂场景下小型无人机目标的轻量化高精度检测模型。EDNet 在 YOLOv5 基础上，融合 **EfficientNet + RepVGG** 构建骨干网络，设计 **自适应协同注意力（ACA）** 模块，并采用 **可扩张残差融合（C3-DWR）** 模块改进颈部网络，有效解决无人机目标尺度失衡与小目标特征薄弱问题。

---

## 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [网络架构](#网络架构)
- [项目结构](#项目结构)
- [实验结果](#实验结果)
- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [模型训练](#模型训练)
  - [推理检测](#推理检测)
  - [模型评估](#模型评估)
- [模型配置](#模型配置)
- [自定义模块说明](#自定义模块说明)
- [数据集](#数据集)
- [致谢](#致谢)
- [开源协议](#开源协议)

---

## 概述

> **说明**：本项目为硕士毕业论文《复杂场景下小型无人机目标检测算法研究》的**第一部分（第二章）**。

在复杂可见光场景中检测小型无人机面临两大核心挑战：目标尺度变化显著、小目标特征表征薄弱。EDNet 通过三项关键改进解决上述问题：

1. **轻量化骨干网络** — 融合 EfficientNet 复合缩放策略与 RepVGG 结构重参数化，在高效提取多尺度特征的同时大幅降低参数量与计算量。
2. **自适应协同注意力（ACA）** — 结合 SimAM（无参数空间注意力）与 ELA（高效位置注意力），通过双阶段特征增强动态提升多尺度目标感知能力。
3. **C3-DWR 颈部网络** — 采用可扩张残差融合模块替换 C3 瓶颈结构，通过多尺度空洞卷积扩大感受野，增强小目标上下文特征表征。

此外，精简版 **Lite-EDNet** 进一步用 GSConv（Slim-Neck）替换颈部卷积，实现更轻量的部署方案。

---

## 核心特性

| 特性 | 说明 | 优势 |
|------|------|------|
| EfficientNet + RepVGG 骨干 | 复合缩放 + 结构重参数化 | 计算量降低 39%，参数量降低 33% |
| ACA 注意力模块 | SimAM + ELA 双阶段注意力 | 以极小开销增强多尺度目标感知 |
| C3-DWR 模块 | 多膨胀率空洞卷积 + 残差融合 | 强化小目标上下文特征提取 |
| SimAM_ELA 检测头 | 检测头前注意力二次增强 | 提升目标定位精度 |
| Lite-EDNet | GSConv 精简颈部 | 更轻量的部署选择 |

---

## 网络架构

```
输入图像 (640×640)
       │
   ┌───┴───────────────────────────────────────────────────┐
   │              骨干网络 (EfficientNet + RepVGG)            │
   │  Stem → RepVGGBlock → SimAM → MBConv×17 → SimAM_ELA → SPPF │
   └───┬───────────────┬───────────────┬─────────────────────┘
       │ P3/8          │ P4/16         │ P5/32
   ┌───┴───────────────┴───────────────┴─────────────────────┐
   │                  颈部网络 (FPN + PANet)                   │
   │        C3_DWR ←→ C3_DWR ←→ C3_DWR                       │
   └───┬───────────────┬───────────────┬─────────────────────┘
       │ 小目标          │ 中目标        │ 大目标
   ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
   │SimAM_  │       │SimAM_  │       │SimAM_  │
   │  ELA   │       │  ELA   │       │  ELA   │
   └───┬───┘       └───┬───┘       └───┬───┘
       └────────┬────────┴────────────────┘
                │
          ┌─────┴─────┐
          │  Detect    │  (P3, P4, P5)
          └───────────┘
```

---

## 项目结构

```
EDNet/
├── models/                     # 模型定义
│   ├── ERAD-YOLO.yaml          # EDNet 模型配置文件
│   ├── Lite-ERAD-YOLO.yaml     # Lite-EDNet 模型配置文件
│   ├── yolo.py                 # 模型构建器（注册所有自定义模块）
│   ├── common.py               # YOLOv5 公共模块 + EfficientNet 组件
│   ├── ACA.py                  # 自适应协同注意力（SimAM + ELA）
│   ├── C3.py                   # C3 变体模块，含 C3-DWR（核心颈部改进）
│   ├── Conv.py                 # 自定义卷积模块（SPD-Conv、PConv 等）
│   ├── Att.py                  # 注意力机制（用于消融实验）
│   ├── BiForm.py               # 双层路由注意力
│   ├── EVC.py                  # EVC 编码块
│   ├── Slim.py                 # GSConv / Slim-Neck（Lite-EDNet 用）
│   ├── mamba_vss.py            # Mamba VSS 模块（实验性）
│   └── experimental.py         # 实验性模块
├── data/                       # 数据集配置文件
├── utils/                      # YOLOv5 工具函数
├── train.py                    # 训练脚本
├── detect.py                   # 推理脚本
├── val.py                      # 评估脚本
├── export.py                   # 模型导出（ONNX 等）
├── FPS.py                      # FPS 基准测试
├── coco.py                     # COCO 格式评估
├── requirements.txt            # Python 依赖
└── LICENSE                     # GPL-3.0 开源协议
```

---

## 实验结果

### EDNet（完整模型）

| 数据集 | mAP50 | mAP50:95 | APsmall | APmid | APlarge |
|--------|-------|----------|---------|-------|---------|
| 自制数据集 | 96.6% | 78.1% | 60.6% | 80.9% | 86.2% |
| UAV 数据集 | 98.2% | 77.0% | 61.9% | 78.7% | 80.0% |
| Det-Fly 数据集 | 97.7% | 65.6% | 24.6% | — | — |

| 模型 | 计算量 (GFLOPs) | 参数量 (M) |
|------|-----------------|------------|
| EDNet | 9.6 | 4.72 |
| Lite-EDNet | 9.1 | 4.28 |
| YOLOv5s（基线） | 15.8 | 7.02 |

### 消融实验（自制数据集）

| 模型 | 组件 | mAP50 | mAP50:95 | FLOPs | Params |
|------|------|-------|----------|-------|--------|
| Y | YOLOv5s（基线） | 94.4% | 66.7% | 15.8 | 7.02 |
| Y-E | + EfficientNet+RepVGG | 94.4% | 66.7% | 7.7 | 3.78 |
| Y-E-A | + ACA 注意力 | 96.2% | 67.4% | 7.7 | 3.80 |
| Y-E-A-D (EDNet) | + C3-DWR 颈部 | **96.6%** | **78.1%** | 9.6 | 4.72 |
| Y-E-A-D-S (Lite) | + Slim-Neck | 96.3% | — | 9.1 | 4.28 |

> **Y**: YOLOv5s · **E**: EfficientNet + RepVGG · **A**: ACA 注意力 · **D**: C3-DWR · **S**: Slim-Neck

---

## 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-username/EDNet.git
cd EDNet

# 创建 Python 3.10+ 环境（推荐）
conda create -n ednet python=3.10 -y
conda activate ednet

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt
```

### 模型训练

训练 EDNet（3 类：无人机、鸟类、飞机）：

```bash
python train.py --data Uav5.yaml --cfg models/ERAD-YOLO.yaml --weights '' --batch-size 16 --img 640 --epochs 300
```

训练 Lite-EDNet：

```bash
python train.py --data Uav5.yaml --cfg models/Lite-ERAD-YOLO.yaml --weights '' --batch-size 16 --img 640 --epochs 300
```

论文中使用的关键训练参数：

| 参数 | 值 |
|------|-----|
| 优化器 | SGD |
| 学习率 | 0.01 |
| 动量 | 0.937 |
| 权重衰减 | 0.0005 |
| 批次大小 | 16 |
| 输入尺寸 | 640×640 |
| 训练轮次 | 300 |
| 损失函数 | CIoU |

### 推理检测

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image.jpg --img 640
```

### 模型评估

```bash
python val.py --weights runs/train/exp/weights/best.pt --data Uav5.yaml --img 640 --task test
```

---

## 模型配置

### EDNet (`models/ERAD-YOLO.yaml`)

```yaml
nc: 3                    # 3 个类别：无人机、鸟类、飞机
depth_multiple: 0.33
width_multiple: 0.50

backbone:
  # EfficientNetLite 骨干 + RepVGG + SimAM + MBConv
  [-1, 1, stem, [32, 'ReLU6']]
  [-1, 1, MBConvBlock, [16, 3, 1, 1, 0]]
  [-1, 1, RepVGGBlock, [24, 3, 2]]
  [-1, 1, SimAM, [24]]
  ...
  [-1, 1, SimAM_ELA, [320]]
  [-1, 1, SPPF, [1024, 5]]

head:
  # FPN + PANet + C3-DWR + 检测头前 SimAM_ELA
  ...
  [27, 1, SimAM_ELA, [256]]
  [30, 1, SimAM_ELA, [512]]
  [33, 1, SimAM_ELA, [1024]]
  [[34, 35, 36], 1, Detect, [nc, anchors]]
```

### Lite-EDNet (`models/Lite-ERAD-YOLO.yaml`)

骨干网络与 EDNet 相同；颈部网络将 `Conv` 替换为 `GSConv`（Slim-Neck），实现更轻量的模型。

---

## 自定义模块说明

### ACA（自适应协同注意力）— `models/ACA.py`

| 类名 | 说明 |
|------|------|
| `SimAM` | 基于能量函数的无参数三维注意力 |
| `ELA` | 基于一维卷积的高效位置注意力 |
| `SimAM_ELA` | SimAM → ELA 顺序串联（最终模型使用） |
| `ACA` | SimAMv1 + ELAv1 + 残差连接 |
| `ACAv2` | 并行加权 SimAM + ELA（可学习权重） |
| `ACAv3` | SimAMv2（可学习 e_lambda）+ 原始 ELA |
| `ACAv4` | 原始 SimAM + ELAv2（深度可分离卷积） |

### C3-DWR（可扩张残差融合）— `models/C3.py`

| 类名 | 说明 |
|------|------|
| `DWR` | 多膨胀率（d=1,3,5）空洞卷积 + 残差融合 |
| `C3_DWR` | 用 DWR 替换 C3 瓶颈结构（最终模型使用） |
| `C3_DWRv1`–`v7` | 消融实验变体（PConv、GSConv、GAP、自适应加权等） |

### 卷积模块 — `models/Conv.py`

| 类名 | 说明 |
|------|------|
| `SPDConv` | 空间到深度卷积（无损下采样） |
| `PConv` | 部分卷积（FasterNet） |
| `Dynamic_conv2d` | 动态卷积（注意力选择卷积核） |
| `LAWDS` | 轻量自适应权重下采样 |
| `RepConv` | 重参数化重聚焦卷积 |

### Slim-Neck — `models/Slim.py`

| 类名 | 说明 |
|------|------|
| `GSConv` | GSConv 精简颈部卷积（Lite-EDNet 使用） |
| `GSBottleneck` | GS 瓶颈模块 |
| `VoVGSCSP` | VoV-GSConv CSP 模块 |

---

## 数据集

| 数据集 | 类别数 | 图像数 | 场景类型 |
|--------|--------|--------|----------|
| 自制数据集 | 3（无人机、鸟类、飞机） | 6,664 | 多场景（天空、城市、植被） |
| UAV 数据集 | 8 种背景 + 鸟类 | ~2,400 | 复杂开放环境 |
| Det-Fly 数据集 | 1（无人机） | — | 空对空、小目标 |

数据集配置文件位于 `data/` 目录下（如 `Uav5.yaml`、`Det-fly.yaml`）。

---

## 致谢

本项目基于以下开源工作进行二次开发：

- [YOLOv5](https://github.com/ultralytics/yolov5) — Ultralytics（GPL-3.0）
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le
- [RepVGG](https://arxiv.org/abs/2101.03697) — Ding et al.
- [SimAM](https://arxiv.org/abs/2107.03777) — Yang et al.
- [ELA](https://arxiv.org/abs/2401.01972) — Ouyang et al.
- [DWR](https://arxiv.org/abs/2309.03512) — 可扩张残差融合
- [Slim-Neck](https://github.com/AlanLi1997/slim-neck-by-gsconv) — GSConv

---

## 开源协议

本项目继承 YOLOv5 基础项目的 [GPL-3.0 开源协议](./LICENSE)。