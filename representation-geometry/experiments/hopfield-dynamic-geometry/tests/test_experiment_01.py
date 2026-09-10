"""实验 1 的模式构造、边界、度量峰和产物契约。"""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from experiment_01_boundary_localization import (
    BoundaryExperimentConfig,
    compiled_dynamics,
    estimate_metric_peak,
    gate_summary,
    hebbian_two_memory_weights,
    locate_boundary,
    make_binary_pair,
    run_experiment,
    write_artifacts,
)


class BoundaryLocalizationTests(unittest.TestCase):
    def test_pair_has_requested_discrete_overlap(self):
        A, B, achieved = make_binary_pair(60, 7, 0.5)
        self.assertEqual(achieved, 0.5)
        self.assertEqual(float(A @ B / 60), achieved)
        self.assertTrue(np.isin(A, (-1, 1)).all())
        self.assertTrue(np.isin(B, (-1, 1)).all())

    def test_weight_matrix_is_symmetric_without_self_connections(self):
        A, B, _ = make_binary_pair(20, 2, 0.0)
        W = np.asarray(hebbian_two_memory_weights(jnp.asarray(A), jnp.asarray(B)))
        np.testing.assert_allclose(W, W.T, rtol=0, atol=0)
        np.testing.assert_allclose(np.diag(W), 0, rtol=0, atol=0)

    def test_overlap_boundary_is_half_and_G0_is_constant(self):
        config = BoundaryExperimentConfig(
            N=20,
            seeds=(0,),
            requested_overlaps=(0.0,),
            dt=0.03,
            metric_steps=40,
            boundary_steps=160,
            kappa_points=41,
        )
        A_np, B_np, _ = make_binary_pair(config.N, 0, 0.0)
        A, B = jnp.asarray(A_np), jnp.asarray(B_np)
        W = hebbian_two_memory_weights(A, B)
        maps = compiled_dynamics(config)
        boundary = locate_boundary(maps["final_margin"], A, B, W, config)
        self.assertAlmostEqual(boundary["boundary_kappa"], 0.5, places=10)
        kappas = jnp.linspace(0, 1, config.kappa_points)
        G = np.asarray(maps["metric_grid"](kappas, A, B, W))
        expected = np.sum((B_np - A_np) ** 2)
        np.testing.assert_allclose(G[:, 0], expected, rtol=1e-12, atol=1e-12)

    def test_flat_metric_has_no_peak(self):
        kappas = np.linspace(0, 1, 11)
        result = estimate_metric_peak(
            kappas, np.ones(11), flat_relative_tolerance=1e-10
        )
        self.assertTrue(np.isnan(result["metric_peak_kappa"]))
        self.assertEqual(result["metric_peak_contrast"], 1)

    def test_gate_rejects_endpoints_that_have_not_separated(self):
        config = BoundaryExperimentConfig(
            seeds=(0,), requested_overlaps=(0.0,), evaluation_start_time=0.0
        )
        conditions = pd.DataFrame([{
            "left_endpoint_margin": 1e-8,
            "right_endpoint_margin": -1e-8,
            "left_endpoint_speed": 1e-10,
            "right_endpoint_speed": 1e-10,
            "boundary_kappa": 0.5,
            "boundary_bracket_width": 0.0,
        }])
        time_results = pd.DataFrame([{
            "time": 0.0,
            "peak_identifiable": True,
            "localization_error": 0.0,
        }])
        summary = gate_summary(conditions, time_results, config)
        self.assertFalse(summary["endpoint_separated"])
        self.assertFalse(summary["endpoint_valid"])
        self.assertFalse(summary["passed"])

    def test_small_run_writes_reviewable_artifacts(self):
        config = BoundaryExperimentConfig(
            N=20,
            seeds=(0, 1),
            requested_overlaps=(0.0, 0.5),
            dt=0.03,
            metric_steps=30,
            boundary_steps=120,
            kappa_points=41,
            evaluation_start_time=0.3,
            representative_overlap=0.5,
        )
        conditions, time_results, grids = run_experiment(config)
        self.assertEqual(len(conditions), 4)
        self.assertEqual(
            len(time_results), len(conditions) * (config.metric_steps + 1)
        )
        self.assertEqual(
            grids["G"].shape,
            (len(conditions), config.kappa_points, config.metric_steps + 1),
        )
        self.assertTrue(time_results[time_results.time == 0].metric_peak_kappa.isna().all())
        with tempfile.TemporaryDirectory() as temporary:
            summary, conclusion = write_artifacts(
                temporary, conditions, time_results, grids, config
            )
            for name in (
                "conditions.csv",
                "time_results.csv",
                "overlap_summary.csv",
                "metric_grid.npz",
                "config.json",
                "summary.json",
                "main_figure.png",
                "main_figure.pdf",
                "conclusion_draft.md",
            ):
                self.assertTrue((Path(temporary) / name).is_file(), name)
            self.assertEqual(conclusion, Path(temporary) / "conclusion_draft.md")
            self.assertEqual(summary["conditions"], 4)


if __name__ == "__main__":
    unittest.main()
