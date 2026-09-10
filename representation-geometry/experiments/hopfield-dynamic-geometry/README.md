# Hopfield dynamic geometry

This folder contains small, Colab-ready experiments that adapt time-varying
pullback-metric analysis to Hopfield retrieval dynamics.

- `experiment_00_pullback_metric_colab.ipynb`: one-dimensional visual calibration
  with two memories in a continuous Hopfield network.
- `experiment_01_boundary_localization_colab.ipynb`: quantitative boundary
  localization across seeds and A/B correlations. Its frozen raw outputs live in
  `artifacts/experiment_01/`.

Both notebooks are method checks, not reproductions of the RNN dynamic-warping
paper and not evidence of a new Hopfield result. Experiment 1 uses its own JAX
core because `am_bench` currently exposes discrete PyTorch retrieval dynamics, while
this experiment needs continuous RK4 trajectories and differentiation through the
full flow. We will extract a shared interface only when later experiments reveal a
stable boundary across model types.
