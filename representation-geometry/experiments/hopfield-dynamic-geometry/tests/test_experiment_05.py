"""实验 05 的更新、Jacobian、秩指标与七项自检。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from experiment_05_multimemory_rank_collapse import (
    RankCollapseConfig,
    _autodiff_local,
    add_dimension_survival,
    attention_diagnostics,
    audit_rank_preflight,
    choose_matched_ipr_levels,
    eigenspectrum,
    full_local_jacobian,
    hopfield_step,
    local_jacobian,
    make_development_cues,
    make_memories,
    memory_span_basis,
    rank_trajectory,
    run_numerical_self_checks,
    spectrum_metrics,
)


class RankCollapseTests(unittest.TestCase):
    def setUp(self):
        self.config = RankCollapseConfig(
            N=32,
            K=16,
            memory_seeds=(0,),
            corruption_rates=(0.10,),
            alpha_grid=(0.5, 1.0, 2.0),
            jacobian_steps=3,
            endpoint_steps=10,
            development_targets=4,
            development_masks=1,
        )
        self.X = make_memories(self.config, 0)
        self.Q, self.C = memory_span_basis(self.X)
        self.cues, self.metadata = make_development_cues(
            self.X, self.config, 0
        )

    def test_step_is_convex_readout_over_all_memories(self):
        for method in ("softmax", "sparsemax"):
            output, weights, scores = hopfield_step(
                self.cues, self.X, 1.5, method
            )
            self.assertEqual(weights.shape, (len(self.cues), self.config.K))
            np.testing.assert_allclose(weights.sum(dim=-1), 1.0, atol=1e-12)
            self.assertTrue(bool((weights >= -1e-14).all()))
            np.testing.assert_allclose(output, weights @ self.X.T, atol=1e-12)
            diagnostics = attention_diagnostics(
                weights, scores, self.config.support_tolerance
            )
            self.assertTrue(bool((diagnostics["ipr"] >= 1).all()))

    def test_analytic_local_jacobian_matches_autodiff(self):
        cue = self.cues[0]
        for method in ("softmax", "sparsemax"):
            _, weights, _ = hopfield_step(
                cue.unsqueeze(0), self.X, 1.5, method
            )
            analytic = full_local_jacobian(
                weights, self.X, 1.5, method, self.config.support_tolerance
            )[0]
            autodiff = _autodiff_local(cue, self.X, 1.5, method)
            np.testing.assert_allclose(analytic, autodiff, rtol=1e-10, atol=1e-10)

    def test_reduced_jacobian_is_exact_memory_span_restriction(self):
        cue = self.cues[0]
        _, weights, _ = hopfield_step(
            cue.unsqueeze(0), self.X, 1.5, "softmax"
        )
        full = full_local_jacobian(
            weights, self.X, 1.5, "softmax", self.config.support_tolerance
        )[0]
        reduced = local_jacobian(
            weights, self.C, 1.5, self.config.N,
            "softmax", self.config.support_tolerance,
        )[0]
        np.testing.assert_allclose(
            reduced, self.Q.T @ full @ self.Q, rtol=1e-10, atol=1e-10
        )

    def test_spectrum_metrics_have_expected_values(self):
        matrix = torch.diag(torch.tensor([3.0, 2.0, 0.0]))
        eigenvalues = eigenspectrum(matrix.unsqueeze(0))
        metrics = spectrum_metrics(eigenvalues, 1e-10)
        self.assertAlmostEqual(float(metrics["trace"][0]), 13.0)
        self.assertAlmostEqual(float(metrics["spectral"][0]), 9.0)
        self.assertAlmostEqual(float(metrics["stable_rank"][0]), 13 / 9)
        self.assertEqual(int(metrics["numeric_rank"][0]), 2)

    def test_dimension_survival_uses_t1_and_sustained_thresholds(self):
        frame = pd.DataFrame({
            "batch_index": [0] * 5,
            "time": [0, 1, 2, 3, 4],
            "cumulative_effective_rank": [4.0, 3.0, 2.5, 1.8, 1.2],
            "trace_G": [4.0, 3.0, 2.0, 1.0, 0.5],
        })
        result = add_dimension_survival(frame, self.config)
        survival = result.dimension_survival.to_numpy()
        self.assertTrue(np.isnan(survival[0]))
        np.testing.assert_allclose(survival[1:], [1.0, 0.75, 0.4, 0.1])
        self.assertAlmostEqual(float(result.dimension_auc.iloc[0]), 0.5625)
        self.assertEqual(float(result.t50.iloc[0]), 3.0)
        self.assertEqual(float(result.t10.iloc[0]), 4.0)

    def test_rank_trajectory_records_local_and_cumulative_spectra(self):
        frame, spectra = rank_trajectory(
            self.cues[:2], self.Q, self.C, 1.5, "softmax", self.config
        )
        self.assertEqual(len(frame), 2 * (self.config.jacobian_steps + 1))
        self.assertEqual(
            spectra["cumulative_eigenvalues"].shape,
            (2, self.config.jacobian_steps + 1, self.config.K),
        )
        at_zero = frame[frame.time == 0]
        np.testing.assert_allclose(
            at_zero.cumulative_effective_rank, self.config.K, atol=1e-10
        )
        at_one = frame[frame.time == 1]
        np.testing.assert_allclose(at_one.dimension_survival, 1.0, atol=1e-12)

    def test_matched_ipr_levels_follow_log_spacing_not_pair_index(self):
        ipr_values = [
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.7, 1.8, 2.0, 8.0, 64.0,
        ]
        rows = []
        for method in ("softmax", "sparsemax"):
            for alpha, ipr in enumerate(ipr_values, start=1):
                rows.append({
                    "rho": 0.10,
                    "method": method,
                    "alpha": float(alpha),
                    "median_first_ipr": ipr,
                })
        levels, ranges = choose_matched_ipr_levels(
            pd.DataFrame(rows), self.config
        )
        middle = levels[
            (levels.level == "middle") & (levels.method == "softmax")
        ]
        self.assertTrue(bool(ranges.three_levels_available.all()))
        self.assertLess(
            abs(float(middle.iloc[0].target_ipr) - 8.0) / 8.0, 0.05
        )
        self.assertAlmostEqual(
            float(middle.iloc[0].requested_log_ipr), 8.0, places=2
        )

    def test_rank_audit_accepts_only_expected_censoring(self):
        base, _ = rank_trajectory(
            self.cues[:2], self.Q, self.C, 1.5, "softmax", self.config
        )
        pieces = []
        for level in ("low", "middle", "high"):
            for method in ("softmax", "sparsemax"):
                frame = base.copy()
                frame["rho"] = 0.10
                frame["level"] = level
                frame["method"] = method
                frame["target"] = 0
                frame["mask_index"] = frame.batch_index
                pieces.append(frame)
        rank_results = pd.concat(pieces, ignore_index=True)
        audit_config = RankCollapseConfig(
            N=32, K=16, memory_seeds=(0,), corruption_rates=(0.10,),
            alpha_grid=(0.5, 1.0, 2.0), jacobian_steps=3,
            endpoint_steps=10, development_targets=1, development_masks=2,
        )
        audit, summary = audit_rank_preflight(rank_results, audit_config)
        self.assertEqual(summary["unexpected_nonfinite_count"], 0)
        self.assertEqual(summary["unexpected_dimension_survival_nan_count"], 0)
        self.assertEqual(summary["actual_rank_rows"], summary["expected_rank_rows"])
        self.assertEqual(len(audit), 6)

    def test_all_seven_numerical_checks_pass(self):
        checks = run_numerical_self_checks(
            self.X, self.Q, self.C, self.cues[0], self.config
        )
        self.assertEqual(len(checks), 7)
        self.assertTrue(bool(checks.passed.all()), checks.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
