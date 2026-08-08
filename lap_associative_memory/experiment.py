"""End-to-end LAP associative-memory experiment.

The important modelling choice is that every mechanism has two branches:

1. a legal branch that only sees the node and its DAG parents;
2. a weak full-state residual branch that can create illegal coupling.

Without the second branch, the LAP mixed partial is identically zero by
construction, so changing ``lambda_lap`` would not test anything. LAP is used to
regularize the residual branch while the legal causal branch remains expressive.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bipolar(x: Tensor) -> Tensor:
    """Return {-1,+1} values, mapping exact zero to +1."""

    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def flip_bits(x: Tensor, probability: float, *, generator: torch.Generator | None = None) -> Tensor:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    mask = torch.rand(x.shape, device=x.device, generator=generator) < probability
    return torch.where(mask, -x, x)


@dataclass(frozen=True)
class SCMData:
    graph: str
    patterns: Tensor
    nodes: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    node_dim: int
    weights: Mapping[str, Tensor]

    @property
    def total_dim(self) -> int:
        return len(self.nodes) * self.node_dim

    @property
    def slices(self) -> dict[str, slice]:
        return {
            node: slice(i * self.node_dim, (i + 1) * self.node_dim)
            for i, node in enumerate(self.nodes)
        }


def _random_weight(dim: int, generator: torch.Generator) -> Tensor:
    # 1/sqrt(d) keeps pre-activations well scaled; signs remain deterministic.
    signs = torch.randint(0, 2, (dim, dim), generator=generator).float().mul_(2).sub_(1)
    return signs / math.sqrt(dim)


def generate_chain(
    n_patterns: int,
    node_dim: int = 10,
    p_flip: float = 0.05,
    seed: int = 0,
) -> SCMData:
    """Generate X -> Y -> Z observational patterns."""

    generator = torch.Generator().manual_seed(seed)
    w_xy = _random_weight(node_dim, generator)
    w_yz = _random_weight(node_dim, generator)
    x = bipolar(torch.randn(n_patterns, node_dim, generator=generator))
    y_clean = bipolar(x @ w_xy.T)
    y = flip_bits(y_clean, p_flip, generator=generator)
    z_clean = bipolar(y @ w_yz.T)
    z = flip_bits(z_clean, p_flip, generator=generator)
    return SCMData(
        graph="chain",
        patterns=torch.cat((x, y, z), dim=1),
        nodes=("X", "Y", "Z"),
        parents={"X": (), "Y": ("X",), "Z": ("Y",)},
        node_dim=node_dim,
        weights={"XY": w_xy, "YZ": w_yz},
    )


def generate_fork(
    n_patterns: int,
    node_dim: int = 10,
    p_flip: float = 0.05,
    seed: int = 0,
) -> SCMData:
    """Generate the confounded fork C -> X and C -> Y."""

    generator = torch.Generator().manual_seed(seed)
    w_cx = _random_weight(node_dim, generator)
    w_cy = _random_weight(node_dim, generator)
    c = bipolar(torch.randn(n_patterns, node_dim, generator=generator))
    x = flip_bits(bipolar(c @ w_cx.T), p_flip, generator=generator)
    y = flip_bits(bipolar(c @ w_cy.T), p_flip, generator=generator)
    return SCMData(
        graph="fork",
        patterns=torch.cat((c, x, y), dim=1),
        nodes=("C", "X", "Y"),
        parents={"C": (), "X": ("C",), "Y": ("C",)},
        node_dim=node_dim,
        weights={"CX": w_cx, "CY": w_cy},
    )


class EnergyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class CausalEnergyMemory(nn.Module):
    """Node-wise energy model with LAP-regularized illegal residual couplings."""

    def __init__(
        self,
        nodes: Sequence[str],
        parents: Mapping[str, Sequence[str]],
        node_dim: int,
        hidden_dim: int = 64,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.nodes = tuple(nodes)
        self.parents = {node: tuple(parents[node]) for node in self.nodes}
        self.node_dim = node_dim
        self.total_dim = len(self.nodes) * node_dim
        self.residual_scale = residual_scale
        self.slices = {
            node: slice(i * node_dim, (i + 1) * node_dim)
            for i, node in enumerate(self.nodes)
        }
        self.legal = nn.ModuleDict()
        self.residual = nn.ModuleDict()
        for node in self.nodes:
            legal_nodes = (node,) + self.parents[node]
            self.legal[node] = EnergyMLP(len(legal_nodes) * node_dim, hidden_dim)
            self.residual[node] = EnergyMLP(self.total_dim, hidden_dim)

    def _legal_input(self, z: Tensor, node: str) -> Tensor:
        ordered = (node,) + self.parents[node]
        return torch.cat([z[..., self.slices[name]] for name in ordered], dim=-1)

    def mechanism_energies(self, z: Tensor) -> dict[str, Tensor]:
        return {
            node: self.legal[node](self._legal_input(z, node))
            + self.residual_scale * self.residual[node](z)
            for node in self.nodes
        }

    def forward(self, z: Tensor) -> Tensor:
        energies = self.mechanism_energies(z)
        return torch.stack(list(energies.values()), dim=0).sum(dim=0)

    def lap_penalty(self, z: Tensor, probes: int = 1) -> Tensor:
        """Hutchinson estimate of illegal mixed-Hessian Frobenius norms.

        For every mechanism i, this estimates
        ||d^2 E_i / (d z_i d z_A)||_F^2 for nodes A that are neither i nor a
        parent of i. Two calls to ``autograd.grad`` compute each mixed partial.
        """

        if probes < 1:
            raise ValueError("probes must be >= 1")
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)

        penalties: list[Tensor] = []
        energies = self.mechanism_energies(z)
        for node, energy in energies.items():
            grad_all = torch.autograd.grad(
                energy.sum(), z, create_graph=True, retain_graph=True
            )[0]
            grad_self = grad_all[..., self.slices[node]]
            illegal_nodes = [
                other
                for other in self.nodes
                if other != node and other not in self.parents[node]
            ]
            for _ in range(probes):
                probe = torch.empty_like(grad_self).bernoulli_(0.5).mul_(2).sub_(1)
                mixed_all = torch.autograd.grad(
                    (grad_self * probe).sum(),
                    z,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                for illegal in illegal_nodes:
                    penalties.append(mixed_all[..., self.slices[illegal]].square().mean())

        if not penalties:
            return z.sum() * 0.0
        return torch.stack(penalties).mean()


class ModernHopfield:
    """Non-parametric log-sum-exp Modern Hopfield baseline."""

    def __init__(self, patterns: Tensor, beta: float = 1.0):
        self.patterns = patterns.detach().clone()
        self.beta = beta

    def energy(self, z: Tensor) -> Tensor:
        similarities = self.beta * (z @ self.patterns.T)
        return -torch.logsumexp(similarities, dim=-1) / self.beta

    def retrieve(
        self,
        initial: Tensor,
        clamp_mask: Tensor | None = None,
        clamp_values: Tensor | None = None,
        steps: int = 50,
        damping: float = 0.8,
        tolerance: float = 1e-5,
    ) -> Tensor:
        z = initial.detach().clone()
        if clamp_mask is None:
            clamp_mask = torch.zeros_like(z, dtype=torch.bool)
        if clamp_values is None:
            clamp_values = z
        for _ in range(steps):
            weights = torch.softmax(self.beta * (z @ self.patterns.T), dim=-1)
            proposal = weights @ self.patterns
            updated = (1 - damping) * z + damping * proposal
            updated = torch.where(clamp_mask, clamp_values, updated)
            if (updated - z).abs().max().item() < tolerance:
                z = updated
                break
            z = updated
        return bipolar(z)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 2e-3
    negative_flip: float = 0.30
    negatives_per_positive: int = 2
    margin: float = 1.0
    hidden_dim: int = 48
    residual_scale: float = 0.25
    lap_probes: int = 1
    log_every: int = 10


@dataclass(frozen=True)
class RetrievalConfig:
    steps: int = 80
    learning_rate: float = 0.12
    tolerance: float = 1e-4


@dataclass(frozen=True)
class ExperimentConfig:
    node_dim: int = 6
    pattern_counts: tuple[int, ...] = (4, 8)
    lambdas: tuple[float, ...] = (0.0, 0.1, 1.0)
    seeds: tuple[int, ...] = (0,)
    p_flip: float = 0.05
    capacity_noise: float = 0.10
    basin_noise: tuple[float, ...] = (0.0, 0.2, 0.4)
    training: TrainingConfig = TrainingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    @classmethod
    def full(cls) -> "ExperimentConfig":
        return cls(
            node_dim=10,
            pattern_counts=(5, 10, 20, 50, 100),
            lambdas=(0.0, 0.01, 0.1, 1.0, 10.0),
            seeds=(0, 1, 2),
            basin_noise=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            training=TrainingConfig(epochs=300, batch_size=32, hidden_dim=64),
            retrieval=RetrievalConfig(steps=150, learning_rate=0.08),
        )


def train_energy_memory(
    data: SCMData,
    lambda_lap: float,
    config: TrainingConfig,
    seed: int,
    device: torch.device,
) -> tuple[CausalEnergyMemory, pd.DataFrame]:
    set_seed(seed)
    model = CausalEnergyMemory(
        data.nodes,
        data.parents,
        data.node_dim,
        hidden_dim=config.hidden_dim,
        residual_scale=config.residual_scale,
    ).to(device)
    patterns = data.patterns.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(config.epochs):
        permutation = torch.randperm(len(patterns), device=device)
        epoch_recon = 0.0
        epoch_lap = 0.0
        batches = 0
        for start in range(0, len(patterns), config.batch_size):
            positive = patterns[permutation[start : start + config.batch_size]]
            positive = positive.detach().requires_grad_(lambda_lap > 0)
            negatives = [flip_bits(positive.detach(), config.negative_flip) for _ in range(config.negatives_per_positive)]
            negative = torch.cat(negatives, dim=0)

            e_pos = model(positive)
            e_neg = model(negative).reshape(config.negatives_per_positive, len(positive)).mean(dim=0)
            recon = F.softplus(config.margin + e_pos - e_neg).mean()
            lap = model.lap_penalty(positive, probes=config.lap_probes) if lambda_lap > 0 else recon.new_zeros(())
            loss = recon + lambda_lap * lap

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_recon += float(recon.detach())
            epoch_lap += float(lap.detach())
            batches += 1

        should_log = epoch == 0 or (epoch + 1) % config.log_every == 0 or epoch + 1 == config.epochs
        if should_log:
            # lambda=0 still needs the real diagnostic value for Figure 4.
            if lambda_lap == 0:
                diagnostic_batch = patterns[: min(config.batch_size, len(patterns))].detach().requires_grad_(True)
                diagnostic_lap = float(model.lap_penalty(diagnostic_batch).detach())
            else:
                diagnostic_lap = epoch_lap / max(batches, 1)
            history.append(
                {
                    "epoch": epoch + 1,
                    "recon_loss": epoch_recon / max(batches, 1),
                    "lap_loss": diagnostic_lap,
                    "total_loss": epoch_recon / max(batches, 1) + lambda_lap * diagnostic_lap,
                }
            )
    return model, pd.DataFrame(history)


def relax_energy(
    model: CausalEnergyMemory,
    initial: Tensor,
    config: RetrievalConfig,
    clamp_mask: Tensor | None = None,
    clamp_values: Tensor | None = None,
) -> Tensor:
    z = initial.detach().clone()
    if clamp_mask is None:
        clamp_mask = torch.zeros_like(z, dtype=torch.bool)
    if clamp_values is None:
        clamp_values = z
    for _ in range(config.steps):
        z = z.detach().requires_grad_(True)
        gradient = torch.autograd.grad(model(z).sum(), z)[0]
        free_gradient = gradient.masked_fill(clamp_mask, 0.0)
        updated = (z - config.learning_rate * free_gradient).clamp(-1.0, 1.0)
        updated = torch.where(clamp_mask, clamp_values, updated).detach()
        if free_gradient.abs().max().item() < config.tolerance:
            z = updated
            break
        z = updated
    return bipolar(z)


Retriever = Callable[[Tensor, Tensor | None, Tensor | None], Tensor]


def _energy_retriever(model: CausalEnergyMemory, config: RetrievalConfig) -> Retriever:
    return lambda initial, mask=None, values=None: relax_energy(model, initial, config, mask, values)


def _hopfield_retriever(model: ModernHopfield) -> Retriever:
    return lambda initial, mask=None, values=None: model.retrieve(initial, mask, values)


def abduct_intervene_predict(
    retrieve: Retriever,
    observation: Tensor,
    observed_mask: Tensor,
    intervention_mask: Tensor,
    intervention_values: Tensor,
    preserve_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run abduction, hard intervention, and counterfactual prediction.

    ``observed_mask`` is clamped during abduction. Prediction starts from the
    abduced state, replaces the intervened coordinates, and only clamps the
    intervention plus an optional factual context (for example the confounder C).
    """

    abduced = retrieve(observation, observed_mask, observation)
    counterfactual_start = torch.where(intervention_mask, intervention_values, abduced)
    prediction_mask = intervention_mask.clone()
    if preserve_mask is not None:
        prediction_mask = prediction_mask | preserve_mask
    predicted = retrieve(counterfactual_start, prediction_mask, counterfactual_start)
    return abduced, predicted


def retrieval_scores(retrieve: Retriever, patterns: Tensor, noise: float) -> dict[str, float]:
    corrupted = flip_bits(patterns, noise)
    recalled = retrieve(corrupted, None, None)
    match = recalled.eq(patterns)
    return {
        "exact_retrieval": float(match.all(dim=-1).float().mean()),
        "bit_accuracy": float(match.float().mean()),
    }


def fork_leakage(retrieve: Retriever, data: SCMData) -> dict[str, float]:
    """Intervene on X while C is fixed; Y must remain unchanged."""

    slices = data.slices
    observation = data.patterns.clone()
    do_values = observation.clone()
    do_values[:, slices["X"]] *= -1
    observed_mask = torch.ones_like(observation, dtype=torch.bool)
    intervention_mask = torch.zeros_like(observation, dtype=torch.bool)
    intervention_mask[:, slices["X"]] = True
    preserve_mask = torch.zeros_like(observation, dtype=torch.bool)
    preserve_mask[:, slices["C"]] = True
    abduced, after = abduct_intervene_predict(
        retrieve,
        observation,
        observed_mask,
        intervention_mask,
        do_values,
        preserve_mask,
    )
    before_y = abduced[:, slices["Y"]]
    after_y = after[:, slices["Y"]]
    # normalized_l2 is in [0,1] for bipolar vectors.
    normalized_l2 = (after_y - before_y).square().mean(dim=-1).sqrt() / 2
    return {
        "leakage": float(normalized_l2.mean()),
        "leakage_bit_fraction": float(after_y.ne(before_y).float().mean()),
    }


def chain_causal_accuracy(retrieve: Retriever, data: SCMData) -> dict[str, float]:
    """do(X=-X), then compare Y and Z with the noise-free SCM prediction."""

    slices = data.slices
    observation = data.patterns.clone()
    do_values = observation.clone()
    do_values[:, slices["X"]] *= -1
    observed_mask = torch.ones_like(observation, dtype=torch.bool)
    intervention_mask = torch.zeros_like(observation, dtype=torch.bool)
    intervention_mask[:, slices["X"]] = True
    _, after = abduct_intervene_predict(
        retrieve,
        observation,
        observed_mask,
        intervention_mask,
        do_values,
    )
    x_do = do_values[:, slices["X"]]
    expected_y = bipolar(x_do @ data.weights["XY"].T)
    expected_z = bipolar(expected_y @ data.weights["YZ"].T)
    y_acc = after[:, slices["Y"]].eq(expected_y).float().mean()
    z_acc = after[:, slices["Z"]].eq(expected_z).float().mean()
    return {"causal_accuracy_y": float(y_acc), "causal_accuracy_z": float(z_acc)}


def _metric_rows(
    values: Mapping[str, float],
    *,
    model: str,
    lambda_lap: float | None,
    n_patterns: int,
    seed: int,
    graph: str,
    noise: float | None = None,
    evaluation: str,
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "lambda_lap": lambda_lap,
            "n_patterns": n_patterns,
            "seed": seed,
            "graph": graph,
            "evaluation": evaluation,
            "noise": noise,
            "metric": metric,
            "value": value,
        }
        for metric, value in values.items()
    ]


def _evaluate_model(
    retrieve: Retriever,
    chain: SCMData,
    fork: SCMData,
    config: ExperimentConfig,
    *,
    model_name: str,
    lambda_lap: float | None,
    seed: int,
) -> list[dict[str, object]]:
    n_patterns = len(chain.patterns)
    rows = _metric_rows(
        retrieval_scores(retrieve, chain.patterns, config.capacity_noise),
        model=model_name,
        lambda_lap=lambda_lap,
        n_patterns=n_patterns,
        seed=seed,
        graph="chain",
        noise=config.capacity_noise,
        evaluation="capacity",
    )
    for noise in config.basin_noise:
        rows.extend(
            _metric_rows(
                retrieval_scores(retrieve, chain.patterns, noise),
                model=model_name,
                lambda_lap=lambda_lap,
                n_patterns=n_patterns,
                seed=seed,
                graph="chain",
                noise=noise,
                evaluation="basin",
            )
        )
    rows.extend(
        _metric_rows(
            chain_causal_accuracy(retrieve, chain),
            model=model_name,
            lambda_lap=lambda_lap,
            n_patterns=n_patterns,
            seed=seed,
            graph="chain",
            evaluation="causal_intervention",
        )
    )
    # Leakage must use a model trained on fork data, so it is appended separately.
    return rows


def run_grid(
    config: ExperimentConfig,
    output_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline and LAP grids, checkpointing CSVs after every configuration."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    metric_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        for n_patterns in config.pattern_counts:
            # The seed depends on N but not lambda: all lambdas see identical data.
            data_seed = seed * 10_000 + n_patterns
            chain = generate_chain(n_patterns, config.node_dim, config.p_flip, data_seed)
            fork = generate_fork(n_patterns, config.node_dim, config.p_flip, data_seed + 1)

            baseline_chain = ModernHopfield(chain.patterns)
            baseline_fork = ModernHopfield(fork.patterns)
            metric_rows.extend(
                _evaluate_model(
                    _hopfield_retriever(baseline_chain),
                    chain,
                    fork,
                    config,
                    model_name="Modern Hopfield",
                    lambda_lap=None,
                    seed=seed,
                )
            )
            metric_rows.extend(
                _metric_rows(
                    fork_leakage(_hopfield_retriever(baseline_fork), fork),
                    model="Modern Hopfield",
                    lambda_lap=None,
                    n_patterns=n_patterns,
                    seed=seed,
                    graph="fork",
                    evaluation="intervention_leakage",
                )
            )

            for lambda_lap in config.lambdas:
                chain_model, chain_history = train_energy_memory(
                    chain, lambda_lap, config.training, data_seed + 101, device
                )
                fork_model, fork_history = train_energy_memory(
                    fork, lambda_lap, config.training, data_seed + 202, device
                )
                chain_model = chain_model.cpu()
                fork_model = fork_model.cpu()
                metric_rows.extend(
                    _evaluate_model(
                        _energy_retriever(chain_model, config.retrieval),
                        chain,
                        fork,
                        config,
                        model_name="LAP E-SCM",
                        lambda_lap=lambda_lap,
                        seed=seed,
                    )
                )
                metric_rows.extend(
                    _metric_rows(
                        fork_leakage(_energy_retriever(fork_model, config.retrieval), fork),
                        model="LAP E-SCM",
                        lambda_lap=lambda_lap,
                        n_patterns=n_patterns,
                        seed=seed,
                        graph="fork",
                        evaluation="intervention_leakage",
                    )
                )
                for graph, frame in (("chain", chain_history), ("fork", fork_history)):
                    records = frame.assign(
                        graph=graph,
                        lambda_lap=lambda_lap,
                        n_patterns=n_patterns,
                        seed=seed,
                    ).to_dict("records")
                    history_rows.extend(records)

                pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
                pd.DataFrame(history_rows).to_csv(output / "training_history.csv", index=False)

    return pd.DataFrame(metric_rows), pd.DataFrame(history_rows)


def _capacity_by_lambda(metrics: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    subset = metrics[
        (metrics.metric == "exact_retrieval")
        & (metrics.evaluation == "capacity")
    ].copy()
    # capacity_noise occurs once plus potentially once in basin_noise; duplicates are harmless after mean.
    curve = (
        subset.groupby(["model", "lambda_lap", "n_patterns"], dropna=False, as_index=False)
        .value.mean()
    )
    rows: list[dict[str, object]] = []
    for keys, group in curve.groupby(["model", "lambda_lap"], dropna=False):
        valid = group[group.value >= threshold]
        rows.append(
            {
                "model": keys[0],
                "lambda_lap": keys[1],
                "capacity": 0 if valid.empty else int(valid.n_patterns.max()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(metrics: pd.DataFrame, history: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create the four figures requested in the experiment specification."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    lap_metrics = metrics[metrics.model == "LAP E-SCM"]
    capacities = _capacity_by_lambda(lap_metrics)
    leakage = (
        lap_metrics[lap_metrics.metric == "leakage"]
        .groupby("lambda_lap", as_index=False)
        .value.mean()
    )
    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(leakage.lambda_lap, leakage.value, "o-", color="#c44e52", label="Leakage")
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_xlabel(r"LAP strength $\lambda$")
    ax1.set_ylabel("Normalized intervention leakage", color="#c44e52")
    ax2 = ax1.twinx()
    ax2.plot(capacities.lambda_lap, capacities.capacity, "s--", color="#4c72b0", label="Capacity")
    ax2.set_ylabel("95% capacity (patterns)", color="#4c72b0")
    fig.suptitle("LAP modularity–capacity trade-off")
    fig.tight_layout()
    path = output / "figure1_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    capacity_curve = metrics[
        (metrics.metric == "exact_retrieval") & (metrics.evaluation == "capacity")
    ].groupby(["model", "lambda_lap", "n_patterns"], dropna=False, as_index=False).value.mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for (model, lam), group in capacity_curve.groupby(["model", "lambda_lap"], dropna=False):
        label = model if model == "Modern Hopfield" else f"LAP λ={lam:g}"
        ax.plot(group.n_patterns, group.value, marker="o", label=label)
    ax.axhline(0.95, color="black", linestyle=":", linewidth=1, label="95% threshold")
    ax.set(xlabel="Stored patterns N", ylabel="Exact retrieval rate", ylim=(-0.03, 1.03), title="Capacity curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "figure2_capacity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    max_n = int(metrics.n_patterns.max())
    basin = metrics[
        (metrics.metric == "exact_retrieval")
        & (metrics.n_patterns == max_n)
        & (metrics.evaluation == "basin")
    ].groupby(["model", "lambda_lap", "noise"], dropna=False, as_index=False).value.mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for (model, lam), group in basin.groupby(["model", "lambda_lap"], dropna=False):
        label = model if model == "Modern Hopfield" else f"LAP λ={lam:g}"
        ax.plot(100 * group.noise, group.value, marker="o", label=label)
    ax.set(xlabel="Initial bit-flip noise (%)", ylabel="Exact retrieval rate", ylim=(-0.03, 1.03), title=f"Basin of attraction (N={max_n})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "figure3_basin.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    diagnostic = history.groupby(["lambda_lap", "epoch"], as_index=False).lap_loss.mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for lam, group in diagnostic.groupby("lambda_lap"):
        ax.plot(group.epoch, group.lap_loss, label=f"λ={lam:g}")
    ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="LAP mixed-partial penalty", title="LAP optimization diagnostic")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "figure4_lap_training.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths
