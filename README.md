<<<<<<< HEAD
# GUI Grounding - Person5: 一致性与自纠错

## 📋 项目概述

本项目是 ScreenSpot GUI 定位实验的一部分，负责研究预测一致性、自纠错与单步定位。

## 🎯 研究目标

研究模型能否发现不可靠的预测，并通过重复预测或验证提高稳定性。

## 📊 实验结果

### ScreenSpot Web 数据集 (436 samples)

| 策略 | Click Accuracy | 说明 |
|------|----------------|------|
| **Single** | 87.2% | 单次预测基线 |
| **Retry** | 87.6% | 低置信时重新预测 |
| **Aggregate** | 87.4% | 多次采样聚合 |

### 自纠错效果分析

- **Retry 策略**: 纠正错误 3 例，改错 1 例，净收益 +2
- **Aggregate 策略**: 纠正错误 2 例，改错 1 例，净收益 +1

### 按数据类型分析

| 类型 | 准确率 |
|------|--------|
| Text | 91.3% |
| Icon | 82.5% |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.57.0+
- CUDA GPU (推荐 24GB+ 显存)

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate scipy pillow gradio
=======
# gui-grounding
>>>>>>> ec9f52e51758293af79e55608c6c16379c3912dc
