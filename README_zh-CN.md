<div align="center">

![轻量级人脸伪造检测](banner.svg)

# 轻量级人脸伪造检测

### 多尺度全局特征 · 自适应通道注意力 · 面向边缘部署

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-模型-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![任务](https://img.shields.io/badge/任务-人脸伪造检测-00C2FF)](#method)
[![论文](https://img.shields.io/badge/论文-Springer-6F42C1)](https://doi.org/10.1007/s11042-026-21154-4)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11042--026--21154--4-2D6CDF)](https://doi.org/10.1007/s11042-026-21154-4)

[English](README.md) · **简体中文**

[论文](#paper) · [网络结构](#architecture) · [快速开始](#quick-start) · [引用](#citation)

</div>

> 面向边缘设备、移动终端和流媒体场景的轻量级、伪影感知人脸伪造检测网络。

## ✨ 项目简介

随着人脸生成与编辑技术快速发展，伪造人脸在视觉上的真实感不断提高，给内容可信度、身份认证和数字媒体安全带来了新的挑战。现有检测方法往往依赖局部视觉伪影或空间异常，但这些线索容易受到图像压缩、分辨率变化和对抗扰动等因素影响，导致模型在复杂场景中的检测性能下降。

为解决这一问题，我们提出了一种融合多尺度全局特征与自适应加权通道自注意力的轻量级人脸伪造检测网络。我们的目标是在保持检测性能和鲁棒性的同时，降低模型的参数量与计算开销，使其更适合部署在边缘设备、移动终端和流媒体检测等资源受限场景中。

本仓库提供该研究相关的 PyTorch 核心网络实现。

<a id="paper"></a>

## 📄 论文信息

<div align="center">

**Deep lightweight face forgery detection network using multi-scale global features and adaptive weighted channel self-attention**

Haojun Xu · Yonghang Fu · **Yudong Wu<sup>✉</sup>** · Fengyong Li

*Multimedia Tools and Applications*, Volume 85, Article 67, 2026

[DOI](https://doi.org/10.1007/s11042-026-21154-4) · <sup>✉</sup> 通讯作者：[w9687777@163.com](mailto:w9687777@163.com)

</div>

## 🚀 项目亮点

| ⚡ 轻量高效 | 🧠 伪影感知 | 📡 面向部署 |
| :---: | :---: | :---: |
| 深度可分离卷积有效控制模型复杂度 | 局部与全局通道交互增强特征表达 | 宽度系数支持不同算力配置 |

<a id="method"></a>

## 🧩 方法概述

### 轻量级特征提取

主干网络采用深度可分离卷积，将标准卷积分解为逐通道卷积和逐点卷积。逐通道卷积负责提取空间特征，逐点卷积负责完成通道间的信息融合。每次卷积后均使用批归一化和 LeakyReLU 激活函数，在控制计算开销的同时完成有效的特征提取。

### 自适应通道特征建模

我们在 SE 通道重标定结构中引入伪影感知交互模块。网络在完成全局平均池化与通道降维后，通过两个互补分支建模局部和全局通道关系：

- 局部分支使用一维卷积捕获相邻通道特征之间的关系；
- 全局分支使用全连接层建立全部通道特征之间的交互；
- 两个分支经过融合后，以固定缩放系数加入残差连接：

```text
Y = X + 0.1 × (Local(X) + Global(X))
```

融合特征随后恢复至原通道维度，并通过 Sigmoid 函数生成自适应通道权重，使网络更加关注与人脸伪造检测相关的有效特征。

### 可调节网络宽度

`ResidualNN` 提供宽度系数 `alpha`，用于统一调整各阶段的通道数量；`make_divisible` 则将通道数对齐到指定整数倍。通过这一设计，可以根据目标设备的算力和内存条件调整模型规模。

<a id="architecture"></a>

## 🏗️ 网络结构

以默认配置 `alpha=1.0` 和 `224 × 224` RGB 图像为例，代码中的数据流如下：

```mermaid
flowchart LR
    A["RGB 输入<br/>B × 3 × 224 × 224"] --> B["3×3 卷积<br/>BN + LeakyReLU"]
    B --> C["伪影感知<br/>SE 模块"]
    C --> D["5 × 深度可分离<br/>卷积模块"]
    D --> E["伪影感知<br/>SE 模块"]
    E --> F["全局平均池化"]
    F --> G["全连接层 + Sigmoid"]
    G --> H["二分类分数<br/>B × 1"]

    classDef input fill:#102a56,stroke:#30e3ff,color:#fff,stroke-width:2px;
    classDef block fill:#1c2351,stroke:#8b6cff,color:#fff,stroke-width:2px;
    classDef output fill:#123c4a,stroke:#46edc8,color:#fff,stroke-width:2px;
    class A input;
    class B,C,D,E,F,G block;
    class H output;
```

默认通道数依次为 `32 → 64 → 128 → 128 → 256 → 256`。

## 📁 仓库结构

```text
Deep-Lightweight-Face-Forgery-Detection/
├── README.md          # 英文项目介绍
├── README_zh-CN.md    # 中文项目介绍
├── banner.svg         # 项目横幅
└── SAE.py             # 网络定义与最小运行示例
```

`SAE.py` 包含以下主要组件：

| 组件 | 功能 |
| --- | --- |
| `ArtifactInteractionBlock` | 建模局部与全局通道关系 |
| `SEBlock` | 生成自适应通道权重 |
| `conv_bn` | 标准卷积特征提取模块 |
| `conv_dw` | 深度可分离卷积模块 |
| `make_divisible` | 根据宽度系数调整通道数 |
| `ResidualNN` | 完整的轻量级二分类网络 |

<a id="quick-start"></a>

## ⚙️ 快速开始

### 环境要求

- Python 3.10 或更高版本
- PyTorch

请根据本机的 CPU 或 CUDA 环境安装合适版本的 PyTorch，然后运行：

```bash
python SAE.py
```

脚本会构建默认模型，并使用形状为 `[4, 3, 224, 224]` 的随机张量完成一次前向传播。预期输出形状为 `[4, 1]`。

也可以在自己的代码中导入模型：

```python
import torch
from SAE import ResidualNN

model = ResidualNN(alpha=1.0)
model.eval()

inputs = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    scores = model(inputs)

print(scores.shape)  # torch.Size([4, 1])
```

当前示例使用随机初始化模型演示代码接口。仓库暂未包含训练权重、数据预处理流程、训练脚本和评估脚本。

<a id="citation"></a>

## 📝 引用

如果本项目对您的研究有所帮助，欢迎引用我们的论文：

```bibtex
@article{xu2026deep,
  title   = {Deep lightweight face forgery detection network using multi-scale global features and adaptive weighted channel self-attention},
  author  = {Xu, Haojun and Fu, Yonghang and Wu, Yudong and Li, Fengyong},
  journal = {Multimedia Tools and Applications},
  volume  = {85},
  pages   = {67},
  year    = {2026},
  doi     = {10.1007/s11042-026-21154-4}
}
```
