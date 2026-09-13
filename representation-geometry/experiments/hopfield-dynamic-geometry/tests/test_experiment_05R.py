"""实验 05R 的逐 cue IPR 求根、变 alpha Jacobian 与停止门。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from experiment_05_multimemory_rank_collapse import (
    make_cues_for_targets,
    make_memories,
    memory_span_basis,
    rank_trajectory,
)
from experiment_05R_per_cue_ipr_matching import (
    PerCuePilotConfig,
    audit_pilot,
    rank_trajectory_per_cue_alpha,
    run_pilot,
    solve_alpha_for_ipr,
    solve_cue_pairs,
    summarize_interaction,
)


class PerCueMatchingTests(unittest.TestCase):
    def setUp(self):
        self.config = PerCuePilotConfig(
            N=32,
            K=16,
            memory_seeds=(0, 1),
            corruption_rates=(0.10,),
            targets_per_seed=4,
            target_iprs=(2.0, 4.0, 8.0),
            jacobian_steps=3,
            bisection_steps=60,
            monotonicity_samples=33,
            bootstrap_samples=1_000,
        )
        self.base = self.config.base_config()
        self.X = make_memories(self.base, 0)
        self.Q, self.C = memory_span_basis(self.X)
        targets = np.arange(self.config.targets_per_seed)
        self.cues, self.metadata = make_cues_for_targets(
            self.X, self.base, 0, targets, mask_count=1
        )

    def test_solver_hits_all_three_targets_for_both_methods(self):
        score = (self.X.T @ self.cues[0]) / self.config.N
        for target in self.config.target_iprs:
            for method in ("softmax", "sparsemax"):
                result = solve_alpha_for_ipr(score, target, method, self.config)
                self.assertEqual(result["status"], "matched", result)
                self.assertLess(result["target_relative_error"], 1e-8)
                self.assertTrue(result["monotonicity_passed"])

    def test_top_score_tie_marks_target_unattainable(self):
        score = torch.tensor([1.0, 1.0, 1.0] + [0.0] * 13)
        result = solve_alpha_for_ipr(score, 2.0, "softmax", self.config)
        self.assertEqual(result["status"], "unattainable_by_tie")
        self.assertEqual(result["top_tie_count"], 3)

    def test_pair_deletion_is_symmetric(self):
        X = torch.eye(16, dtype=torch.float64).repeat(2, 1)
        cue = torch.zeros(32, dtype=torch.float64)
        cue[:3] = 1.0
        metadata = pd.DataFrame([{
            "cue_index": 0, "memory_seed": 0, "target": 0,
            "rho": 0.10, "mask_index": 0,
        }])
        solver, pairs = solve_cue_pairs(
            X, cue.unsqueeze(0), metadata, self.config
        )
        low = pairs[pairs.level == "low"].iloc[0]
        self.assertFalse(bool(low.included))
        self.assertEqual(low.pair_status, "unattainable_by_tie")
        self.assertEqual(len(solver[solver.level == "low"]), 2)

    def test_per_cue_alpha_matches_old_constant_alpha_path(self):
        cues = self.cues[:2]
        alpha = 1.5
        old, old_spectra = rank_trajectory(
            cues, self.Q, self.C, alpha, "softmax", self.base
        )
        new, new_spectra = rank_trajectory_per_cue_alpha(
            cues, torch.full((2,), alpha), self.Q, self.C,
            "softmax", self.config,
        )
        columns = [
            "local_effective_rank", "cumulative_effective_rank",
            "trace_G", "spectral_G", "dimension_survival", "dimension_auc",
            "ipr", "support_size",
        ]
        np.testing.assert_allclose(
            new[columns], old[columns], rtol=1e-12, atol=1e-12,
            equal_nan=True,
        )
        for key in old_spectra:
            np.testing.assert_allclose(new_spectra[key], old_spectra[key], atol=1e-12)

    def test_interaction_summary_uses_seed_as_independent_unit(self):
        rows = []
        deltas = {"low": -0.1, "middle": 0.2, "high": 0.0}
        for seed in (0, 1):
            for level, delta in deltas.items():
                for method, auc in (
                    ("softmax", 0.4), ("sparsemax", 0.4 + delta)
                ):
                    rows.append({
                        "memory_seed": seed, "level": level, "method": method,
                        "time": 1, "dimension_auc": auc,
                    })
        seed_level, interaction = summarize_interaction(
            pd.DataFrame(rows), self.config
        )
        self.assertEqual(len(seed_level), 6)
        np.testing.assert_allclose(interaction.C, 0.25)

    def test_audit_applies_frozen_gate_order(self):
        solver_rows = []
        pair_rows = []
        for index in range(10):
            pair_rows.append({
                "included": index < 9,
                "pair_status": "matched" if index < 9 else "unbracketed",
                "pair_relative_error": 0.0 if index < 9 else np.nan,
            })
            for _ in range(2):
                solver_rows.append({
                    "status": "matched" if index < 9 else "unbracketed",
                    "target_relative_error": 0.0 if index < 9 else np.nan,
                })
        rank = pd.DataFrame(columns=[
            "time", "dimension_survival", "dimension_auc", "trace_G",
            "spectral_G", "cumulative_effective_rank", "local_effective_rank",
        ])
        checks = pd.DataFrame({"passed": [True] * 14})
        interaction = pd.DataFrame({"C": [0.1, 0.1]})
        special = PerCuePilotConfig(
            N=32, K=16, memory_seeds=(0, 1), corruption_rates=(0.10,),
            targets_per_seed=5, target_iprs=(2.0, 4.0, 8.0),
            jacobian_steps=3, bootstrap_samples=1_000,
        )
        # 10 supplied pairs represent only part of the expected 30, so the first
        # frozen gate must stop before any later signal is considered.
        summary = audit_pilot(
            pd.DataFrame(solver_rows), pd.DataFrame(pair_rows), rank,
            checks, interaction, special,
        )
        self.assertEqual(
            summary["stopping_outcome"],
            "stop_pair_attainability_below_95_percent",
        )

    def test_tiny_end_to_end_pilot_writes_auditable_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_pilot(self.config, temporary)
            required = {
                "solver_results.csv.gz", "pair_status.csv",
                "jacobian_results.csv.gz", "jacobian_spectra",
                "self_checks.csv", "seed_level_auc.csv",
                "seed_interaction.csv", "pilot_summary.json",
                "protocol.json", "conclusion.md", "main_figure.png",
            }
            self.assertTrue(required.issubset({path.name for path in Path(temporary).iterdir()}))
            self.assertEqual(
                len(list((Path(temporary) / "jacobian_spectra").glob("seed*.npz"))),
                len(self.config.memory_seeds),
            )
            self.assertEqual(summary["expected_pair_count"], 24)
            self.assertEqual(summary["self_check_count"], 14)
            self.assertTrue(summary["all_seed_self_checks_passed"])


if __name__ == "__main__":
    unittest.main()
