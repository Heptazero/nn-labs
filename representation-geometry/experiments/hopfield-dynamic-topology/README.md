# Hopfield dynamic topology

This directory studies the topology of ensembles of Hopfield retrieval states.
It is a separate research line from `../hopfield-dynamic-geometry/`: the old
pullback/Jacobian experiments remain sealed and are not renumbered or reused as
topology results.

## File layout

- `experiment_01_dynamic_point_cloud_topology_protocol.md`: frozen question,
  controls, development gate, formal budget, outputs, and stop rules.
- `experiment_01_dynamic_point_cloud_topology.py`: implementation to be created
  only after the protocol is read.
- `experiment_01_dynamic_point_cloud_topology_colab.ipynb`: Colab runner to be
  generated after local tests and the development preflight pass.
- `tests/test_experiment_01.py`: numerical and data-contract tests.
- `artifacts/experiment_01/development/`: one-seed readiness evidence.
- `artifacts/experiment_01/formal/`: frozen 10-seed results, created only if the
  development gate passes.

## Reuse boundary

Reuse memory/cue generation, the classical Hebbian model contract, trajectory
observation, and provenance from
`../../../associative-memory/benchmarks/am-bench/src/am_bench/`. Keep persistent
homology, endpoint quotienting, duplicate-state weighting, and topology-specific
plots in this directory until their interfaces are stable.

The first experiment uses only the asynchronous binary classical Hopfield
network. Continuous tanh, iterative softmax, and iterative sparsemax belong to a
later matched-model experiment; they must not be silently added to experiment 01.

## Start here in a new task

1. Read `/Users/heptazero/Documents/project/AGENTS.md`.
2. Read this README and the experiment 01 protocol in full.
3. Read the active handoff `hopfield-dynamic-topology.md`.
4. Inspect the cited shared source before coding; do not copy notebook-only
   implementations when a tested component already exists.
5. Implement and run only the development preflight first. Do not relax a failed
   gate or start the formal budget automatically.

