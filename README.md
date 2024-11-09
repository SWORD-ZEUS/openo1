# OpenO1

OpenO1是一个基于大型语言模型的开源项目，旨在通过结合思维链（Chain of Thought, CoT）、强化学习（RL）和蒙特卡洛树搜索（MCTS）来提升AI的推理能力。

## 项目概述

本项目的主要目标是构建一个能够生成高质量思维链的AI系统。我们使用了以下关键技术：

- 监督微调（SFT）：使用CoT数据训练模型生成思维步骤
- 强化学习（RL）：通过self-play优化Generator和Verifier
- 蒙特卡洛树搜索（MCTS）：用于选择最佳的思维链路径

## 主要组件

1. Generator：生成候选思维步骤
2. Verifier：评估思维步骤的质量并提供反馈
3. Reward Model：为每个思维步骤和整个思维链打分

## 安装

1. 克隆仓库:
   ```
   git clone https://github.com/SWORD-ZEUS/openo1.git
   cd openo1
   ```

2. 安装依赖:
   ```
   pip install -r requirements.txt
   ```

3. 下载Llama 3 7B模型并放置在适当的位置。

## 使用方法

1. 准备数据集:
   将PRM800K数据集放在 `dataset/prm800k` 目录下。

2. 配置训练参数:
   编辑 `configs/sft_config.yaml` 文件。

3. 运行SFT训练:
   ```
   python scripts/run_sft.py
   ```

## 项目结构

