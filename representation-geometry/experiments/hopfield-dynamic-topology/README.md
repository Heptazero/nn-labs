# Hopfield dynamic topology

This directory studies the topology of ensembles of Hopfield retrieval states.
It is a separate research line from `../hopfield-dynamic-geometry/`: the old
pullback/Jacobian experiments remain sealed and are not renumbered or reused as
topology results.

## File layout

- `experiment_01_dynamic_point_cloud_topology_protocol.md`: frozen question,
  controls, development gate, formal budget, outputs, and stop rules.
- `experiment_01_dynamic_point_cloud_topology_colab.ipynb`: the user-facing
  and self-contained runner for both development and formal numerical execution.
  All experiment code, self-checks, figures, and export logic live in this one
  notebook. It does not download or import an experiment-specific Python file.
- `artifacts/experiment_01/development/`: one-seed readiness evidence.
- `artifacts/experiment_01/formal/`: frozen 10-seed results, created only if the
  development gate passes.

## Reuse boundary

Use the tested behavior in
`../../../associative-memory/benchmarks/am-bench/src/am_bench/` as the reference
for memory/cue generation, the classical Hebbian model contract, trajectory
observation, and provenance. Copy the minimal required implementation into
clearly named notebook sections so the Colab remains self-contained, and verify
equivalence with notebook self-checks. Keep persistent homology, endpoint
quotienting, duplicate-state weighting, and topology-specific plots in the same
notebook.

The first experiment uses only the asynchronous binary classical Hopfield
network. Continuous tanh, iterative softmax, and iterative sparsemax belong to a
later matched-model experiment; they must not be silently added to experiment 01.

## Start here in a new task

1. Read `/Users/heptazero/Documents/project/AGENTS.md`.
2. Read this README and the experiment 01 protocol in full.
3. Read the active handoff `hopfield-dynamic-topology.md`.
4. Inspect the cited shared source before coding, then place all required code in
   the notebook rather than creating an experiment-specific `.py` module.
5. Organize the notebook with numbered Markdown headings, Colab's table of
   contents, and foldable code cells. Put fast formula/data-contract self-checks
   before the development runner.
6. Run only the development preflight in Colab first. Do not treat static
   notebook inspection as numerical evidence, relax a failed gate, or start the
   formal budget automatically.
