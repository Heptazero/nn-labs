# Hopfield Networks is All You Need 实验

## 1. MNIST 联想记忆：从经典 Hopfield 到现代 Hopfield

[`hopfield_image_retrieval_colab.ipynb`](./hopfield_image_retrieval_colab.ipynb)
按 `a 储存输入 -> b 权重/记忆表征 -> c 检索初态 -> d 动力学 -> e 测量 -> f 展示`
组织中文公共组件。经典 Hopfield 实验记录异步检索的能量、翻转步数与二维投影轨迹；
二值 Dense Memory 和连续 Modern Hopfield 分别展示高阶匹配与注意力竞争。
误差图同时显示单条目标数据和条件平均值，不以代表图片代替量化结果。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/hopfield%20is%20all%20you%20need/hopfield_image_retrieval_colab.ipynb)

## 2. 使用 HopfieldPooling 处理 MNIST Bags

本实验基于论文
[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)
配套仓库中的官方 MNIST Bags 示例，并适配当前版本的 Colab、PyTorch 与 NumPy。

核心模型流程如下：

```text
MNIST 图片 -> CNN 特征提取器 -> HopfieldPooling -> sigmoid 二分类器
```

每个样本是一个包含若干 MNIST 图片的变长 bag。bag 中只要出现至少一个目标数字
（默认是 `9`）就是正类，否则是负类。

同一个任务上会分别训练 `Attention`、`GatedAttention` 和 `HopfieldPooling`
三种模型。实验保留官方抽样规则、训练循环、超参数与绘图方式。旧版 ADMIL
数据加载器被等价的当前 API 实现替代；变长 bag 仍彼此独立，并使用
`batch_size=1` 加载，因此不需要 padding，也不会跨 bag 错误调用 `torch.stack`。

第一个代码单元会直接从 GitHub 安装固定版本的 `hopfield-layers`，下载固定版本的
`AttentionDeepMIL`，随后导入 `HopfieldPooling`。ADMIL 在这里仅提供两个基线模型。

在 Colab 中打开 notebook：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/hopfield%20is%20all%20you%20need/mnist_bags_hopfield_pooling_colab.ipynb)

选择 **运行时 -> 重新启动会话并全部运行**。第一个代码单元成功时会在末尾显示
`hflayers 导入成功`。
