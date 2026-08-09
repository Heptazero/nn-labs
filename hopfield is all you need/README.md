# MNIST Bags with HopfieldPooling

This starts from the official MNIST-bags demonstration from
[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) and is
made directly runnable with current Colab, PyTorch, and NumPy versions.

The model is:

```text
MNIST image -> CNN feature extractor -> HopfieldPooling -> sigmoid classifier
```

Each sample is a bag of MNIST images with a variable length sampled around ten
instances. A positive bag contains at least one target digit (9 by default); a
negative bag contains no target digit.

The Attention and GatedAttention baselines, HopfieldPooling model, sampling
rule, training loops, hyperparameters, and plots are retained. The historical
ADMIL data loader is replaced by an equivalent local implementation using
current APIs. Variable-length bags remain separate and are loaded with
`batch_size=1`, so no padding or invalid cross-bag `torch.stack` is needed.
The first code cell installs a pinned `hopfield-layers` revision directly from
GitHub, downloads a pinned `AttentionDeepMIL` revision, and then imports
`HopfieldPooling`. Installation and import live in the same cell so setup cannot
be skipped accidentally; ADMIL supplies only the two baseline models.

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/hopfield%20is%20all%20you%20need/mnist_bags_hopfield_pooling_colab.ipynb)

Choose **Runtime -> Restart session and run all**. A successful first code cell
ends with `hflayers import OK`.
