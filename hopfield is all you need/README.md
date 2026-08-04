# MNIST Bags with HopfieldPooling

This notebook is a Colab-friendly, self-contained version of the MNIST-bags
demonstration from [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217).

The model is:

```text
MNIST image -> CNN feature extractor -> HopfieldPooling -> sigmoid classifier
```

Each sample is a bag of MNIST images with a variable length sampled around ten
instances. A positive bag contains at least one target digit (9 by default); a
negative bag contains no target digit.

The notebook keeps the central HopfieldPooling experiment and its historical
training scale and hyperparameters, but generates the bags locally. It
therefore does not need the separate `AttentionDeepMIL` repository used by the
original historical notebook in `ml-jku/hopfield-layers`.

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/hopfield%20is%20all%20you%20need/mnist_bags_hopfield_pooling_colab.ipynb)

The first code cell installs `hflayers` directly from its upstream repository.
