# MNIST Bags with HopfieldPooling

This is the official MNIST-bags demonstration from
[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217), copied
from `ml-jku/hopfield-layers` and made directly runnable in Colab.

The model is:

```text
MNIST image -> CNN feature extractor -> HopfieldPooling -> sigmoid classifier
```

Each sample is a bag of MNIST images with a variable length sampled around ten
instances. A positive bag contains at least one target digit (9 by default); a
negative bag contains no target digit.

The original data loader, Attention and GatedAttention baselines,
HopfieldPooling model, training loops, hyperparameters, and plots are retained.
The only additions are a Colab setup cell and compatibility updates for current
NumPy and Python versions. The setup cell downloads both `hopfield-layers` and
the original `AttentionDeepMIL` dependency.

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/hopfield%20is%20all%20you%20need/mnist_bags_hopfield_pooling_colab.ipynb)

Choose **Runtime -> Restart session and run all**. The first code cell installs
the two upstream repositories and switches to the working directory expected
by the official notebook.
