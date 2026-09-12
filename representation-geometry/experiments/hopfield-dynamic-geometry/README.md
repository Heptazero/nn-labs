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
- `experiment_04_direction_mechanism_controls_colab.ipynb`: reads every
  direction at the same true-boundary point and compares finite-time pullback
  geometry with the overlap gradient, the energy Hessian, initial strain `S0`,
  and time-varying strain `St`. Frozen outputs live in `artifacts/experiment_04/`.
- `experiment_05_multimemory_rank_collapse.py`: development core for matched-IPR
  softmax/sparsemax retrieval with the complete memory bank, exact memory-span
  local and cumulative Jacobian spectra, seven numerical checks, and explicit
  censoring/saturation audits. Development-only outputs live in
  `artifacts/experiment_05/development/`; they are readiness evidence, not the
  confirmatory 05A result.

These notebooks are method checks, not reproductions of the RNN dynamic-warping
paper and not evidence of a new Hopfield result. Experiment 1 uses its own JAX
core because `am_bench` currently exposes discrete PyTorch retrieval dynamics, while
this experiment needs continuous RK4 trajectories and differentiation through the
full flow. We will extract a shared interface only when later experiments reveal a
stable boundary across model types.
