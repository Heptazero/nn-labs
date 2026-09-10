"""Colab 数值验收；本地只做静态检查，不把待运行断言称为通过。"""
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from .metrics import retrieve_measured
from .models import MODEL_FACTORIES
from .models.base import binary_state
from .models.modern import IterativeModernHopfield
from .tasks import a1_make_independent_binary, c1_make_hamming_cue


def verify_extraction(reference_notebook):
    """只对固定提交的旧 Notebook 使用；在调用入口核验其 Git blob 摘要。"""
    notebook = json.loads(Path(reference_notebook).read_text())
    old = {"torch": torch, "dataclass": dataclass, "field": field,
           "Any": Any, "Callable": Callable}
    for index in (4, 6):
        exec(compile("".join(notebook["cells"][index]["source"]),
                     f"frozen-reference-cell-{index}", "exec"), old)
    checks = 0
    for seed in (7, 19):
        memories = a1_make_independent_binary(16, 4, seed).patterns
        for noise in (0.0, 0.25, 0.5):
            target = memories[0]
            cue = c1_make_hamming_cue(target, noise, seed + 1)
            for name, factory in MODEL_FACTORIES.items():
                original = old["MODEL_FACTORIES"][name]().fit(memories)
                extracted = factory().fit(memories)
                assert "target" not in inspect.signature(extracted.retrieve).parameters
                before = original.retrieve(cue, target=target, update_seed=13, max_sweeps=5)
                after = retrieve_measured(extracted, cue, target=target, update_seed=13, max_sweeps=5)
                if before.final_state.is_floating_point():
                    torch.testing.assert_close(after.final_state, before.final_state, rtol=1e-12, atol=1e-12)
                else:
                    assert torch.equal(after.final_state, before.final_state), name
                for attribute in ("status", "sweeps", "state_updates", "retrieval_flops", "diagnostics"):
                    assert getattr(before, attribute) == getattr(after, attribute), (name, attribute)
                for attribute in ("energy_trace", "error_trace"):
                    torch.testing.assert_close(
                        torch.tensor(getattr(after, attribute), dtype=torch.float64),
                        torch.tensor(getattr(before, attribute), dtype=torch.float64), rtol=1e-12, atol=1e-12)
                assert original.resource_summary() == extracted.resource_summary(), name
                # 外部目标和恶意修改观测副本均不能改变模型终态。
                other_target = retrieve_measured(extracted, cue, target=-target, update_seed=13, max_sweeps=5)
                mutated_observation = extracted.retrieve(
                    cue, update_seed=13, max_sweeps=5,
                    observer=lambda step, state, energy, diag: state.fill_(123))
                assert torch.equal(other_target.final_state, after.final_state), name
                assert torch.equal(mutated_observation.final_state, after.final_state), name
                checks += 1
    try:
        binary_state(torch.tensor([1.5, -1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("fractional binary input accepted")
    return {"extraction_cases_passed": checks, "reference": "e9adf216c5b5a0b329a102c0c25954e6ff48aa12"}


def verify_dynamics():
    """真实迭代、失败分母、固定点延展、配对、部分运行与恢复的一次小验收。"""
    from .dynamics import DynamicsConfig, _trial, run_dynamics, transition_table
    from .models.base import RetrievalResult, observe

    patterns = a1_make_independent_binary(16, 4, 3).patterns
    cue = c1_make_hamming_cue(patterns[0], 0.25, 4)
    model = IterativeModernHopfield(0.1).fit(patterns)
    trajectory = {}
    result = model.retrieve(cue, update_seed=0, max_sweeps=4,
                            observer=lambda step, state, energy, diag: trajectory.update({step: state}))
    manual = cue.double()
    for step in range(1, result.sweeps + 1):
        manual, _ = model.readout(manual)
        torch.testing.assert_close(trajectory[step], manual, rtol=0, atol=0)

    class Fixed:
        def retrieve(self, cue, *, observer, **kwargs):
            observe(observer, 0, cue)
            observe(observer, 1, cue)
            return RetrievalResult(cue, "fixed", 1, 16, 100)

    held = _trial(Fixed(), cue, 0, patterns, 0, 3)
    assert held[2]["held_after_fixed"] and held[3]["available"]
    failed = _trial(None, cue, 0, patterns, 0, 3, "deliberate fit failure")
    assert failed[0]["available"] and all(not r["available"] and not r["top1_correct"] for r in failed[1:])

    config = DynamicsConfig(N_values=(16,), P_values=(2,), seeds=(0, 1),
                            corruption_levels=(0.0, 0.25), targets_per_set=2, max_sweeps=3,
                            models=("classical_hebb", "continuous_modern_iterative"))
    with tempfile.TemporaryDirectory() as directory:
        partial, progress = run_dynamics(config, directory, {"test": "fixed"}, max_new_batches=1)
        assert not progress["complete"] and progress["completed_batches"] == 1
        complete, progress = run_dynamics(config, directory, {"test": "fixed"})
        resumed, again = run_dynamics(config, directory, {"test": "fixed"})
        assert progress["complete"] and again == progress and complete.equals(resumed)
        assert len(complete) == config.retrieval_count * (config.max_sweeps + 1)
        assert complete.iloc[:len(partial)].equals(partial)
        initial = complete[complete.step == 0]
        assert initial.groupby(["pattern_set_id", "target_id", "corruption_level"]).top1_correct.nunique().eq(1).all()
        adjacent = transition_table(complete)
        relative = transition_table(complete, relative_to_first=True)
        assert len(adjacent) and len(relative)
        assert (adjacent.wrong_to_right <= adjacent.previous_wrong).all()
        assert (adjacent.right_to_wrong <= adjacent.previous_right).all()
        # 图必须由相同原始表生成；不在绘图时重跑模型。
        from .dynamics import plot_dynamics_curves
        import matplotlib.pyplot as plt
        figure = plot_dynamics_curves(complete, N=16, P=2, corruption_level=0.25)
        assert len(figure.axes[0].lines) == 2
        plt.close(figure)
    return {"dynamics_contract": "passed", "resume_contract": "passed"}
