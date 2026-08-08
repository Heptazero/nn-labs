from __future__ import annotations

import torch

from lap_associative_memory.experiment import (
    abduct_intervene_predict,
    CausalEnergyMemory,
    ModernHopfield,
    RetrievalConfig,
    TrainingConfig,
    fork_leakage,
    generate_chain,
    generate_fork,
    relax_energy,
    train_energy_memory,
)


def test_scm_generation_is_reproducible_and_bipolar() -> None:
    first = generate_chain(7, node_dim=4, seed=12)
    second = generate_chain(7, node_dim=4, seed=12)
    assert first.patterns.shape == (7, 12)
    assert torch.equal(first.patterns, second.patterns)
    assert set(first.patterns.unique().tolist()) <= {-1.0, 1.0}


def test_lap_is_zero_without_illegal_branch_and_trainable_with_it() -> None:
    data = generate_chain(4, node_dim=3, seed=3)
    z = data.patterns.clone().requires_grad_(True)
    legal_only = CausalEnergyMemory(data.nodes, data.parents, 3, hidden_dim=8, residual_scale=0.0)
    assert float(legal_only.lap_penalty(z).detach()) == 0.0

    model = CausalEnergyMemory(data.nodes, data.parents, 3, hidden_dim=8, residual_scale=0.5)
    penalty = model.lap_penalty(z)
    assert float(penalty.detach()) > 0
    penalty.backward()
    residual_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.residual.parameters()
        if parameter.grad is not None
    )
    assert residual_grad > 0


def test_hard_intervention_values_remain_clamped() -> None:
    data = generate_chain(3, node_dim=2, seed=4)
    model = CausalEnergyMemory(data.nodes, data.parents, 2, hidden_dim=8)
    initial = data.patterns.clone()
    initial[:, data.slices["X"]] *= -1
    mask = torch.zeros_like(initial, dtype=torch.bool)
    mask[:, data.slices["X"]] = True
    recalled = relax_energy(
        model,
        initial,
        RetrievalConfig(steps=3, learning_rate=0.05),
        mask,
        initial,
    )
    assert torch.equal(recalled[mask], initial[mask])


def test_three_step_intervention_keeps_do_value() -> None:
    data = generate_chain(3, node_dim=2, seed=5)
    baseline = ModernHopfield(data.patterns)
    observation = data.patterns.clone()
    observed_mask = torch.ones_like(observation, dtype=torch.bool)
    intervention_mask = torch.zeros_like(observation, dtype=torch.bool)
    intervention_mask[:, data.slices["X"]] = True
    do_values = observation.clone()
    do_values[:, data.slices["X"]] *= -1
    _, predicted = abduct_intervene_predict(
        lambda x, m, v: baseline.retrieve(x, m, v),
        observation,
        observed_mask,
        intervention_mask,
        do_values,
    )
    assert torch.equal(predicted[intervention_mask], do_values[intervention_mask])


def test_training_and_fork_metric_smoke() -> None:
    data = generate_fork(4, node_dim=2, seed=8)
    model, history = train_energy_memory(
        data,
        lambda_lap=0.1,
        config=TrainingConfig(epochs=2, batch_size=4, hidden_dim=8, log_every=1),
        seed=9,
        device=torch.device("cpu"),
    )
    assert len(history) == 2
    baseline = ModernHopfield(data.patterns)
    leakage = fork_leakage(lambda x, m, v: baseline.retrieve(x, m, v), data)
    assert 0 <= leakage["leakage"] <= 1
    assert torch.isfinite(torch.tensor(history.lap_loss.to_numpy())).all()
