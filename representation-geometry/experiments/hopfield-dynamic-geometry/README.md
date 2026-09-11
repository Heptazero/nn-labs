# Hopfield dynamic geometry

This folder contains small, Colab-ready experiments that adapt time-varying
pullback-metric analysis to Hopfield retrieval dynamics.

- `experiment_00_pullback_metric_colab.ipynb`: one-dimensional visual calibration
  with two memories in a continuous Hopfield network.
- `experiment_01_boundary_localization_colab.ipynb`: quantitative boundary
  localization across seeds and A/B correlations. It is sealed as a numerical
  calibration; its frozen raw outputs live in `artifacts/experiment_01/`.
- `experiment_02_moving_boundary_colab.ipynb`: moves the one-dimensional basin
  boundary with unequal memory weights and compares pullback G with overlap and
  energy baselines. Frozen outputs live in `artifacts/experiment_02/`.
- `experiment_03_2d_directional_geometry_colab.ipynb`: tests whether the major
  eigenvector of the 2x2 pullback metric follows the normal of a curved basin
  boundary. Frozen outputs live in `artifacts/experiment_03/`.

These notebooks are method checks, not reproductions of the RNN dynamic-warping
paper and not evidence of a new Hopfield result. Experiment 1 uses its own JAX
core because `am_bench` currently exposes discrete PyTorch retrieval dynamics, while
this experiment needs continuous RK4 trajectories and differentiation through the
full flow. We will extract a shared interface only when later experiments reveal a
stable boundary across model types.
