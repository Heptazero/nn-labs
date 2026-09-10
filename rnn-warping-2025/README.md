# RNNs Perform Task Computations by Dynamically Warping Neural Representations

This folder keeps the paper authors' code separate from the reproducible Colab entrypoints maintained in `nn-labs`.

## Layout

- `upstream/`: an unchanged snapshot of the authors' supplied code, excluding only `.DS_Store`.
- `reproduction/`: Colab-ready notebook copies. Each copy adds one bootstrap cell and removes cached execution outputs.
- `prepare_colab_notebooks.py`: regenerates the Colab copies from `upstream/`.

The separation is intentional: results from `upstream/` describe the supplied implementation, while fixes, localization, new controls, and later experiments belong in `reproduction/`.

## Run in Colab

No Google Drive mount is required. Open one notebook and choose **Runtime → Restart session and run all**. The first code cell clones `nn-labs`, installs the local `riemannian_dynamics` package, and switches to the correct model directory.

- [Static network](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn-warping-2025/reproduction/static_network_colab.ipynb)
- [Evidence integration](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn-warping-2025/reproduction/evidence_integration_colab.ipynb)
- [Working memory](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn-warping-2025/reproduction/wm_colab.ipynb)

The dependency versions in the supplied `setup.py` are not pinned. Static notebook validation therefore confirms structure and import setup only; the numerical and visual reproduction gate remains a fresh Colab run.
