"""实验 2 的偏置权重、边界方向、能量和产物契约。"""
from pathlib import Path
import sys
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from experiment_01_boundary_localization import (
    compiled_dynamics,
    hebbian_two_memory_weights,
    locate_boundary,
    make_binary_pair,
)
from experiment_02_moving_boundary import (
    MovingBoundaryConfig,
    biased_two_memory_weights,
    continuous_hopfield_energy,
    count_sign_transitions,
    estimate_zero_crossing,
    run_experiment,
    write_artifacts,
)


class MovingBoundaryTests(unittest.TestCase):
    def test_delta_zero_matches_experiment_one_weights(self):
        A_np, B_np, _ = make_binary_pair(30, 3, 0.0)
        A, B = jnp.asarray(A_np), jnp.asarray(B_np)
        np.testing.assert_allclose(
            np.asarray(biased_two_memory_weights(A, B, 0.0)),
            np.asarray(hebbian_two_memory_weights(A, B)),
            rtol=0,
            atol=0,
        )

    def test_positive_delta_moves_boundary_toward_B(self):
        config = MovingBoundaryConfig(
            N=60,
            seeds=(0,),
            deltas=(-0.3, 0.0, 0.3),
            boundary_steps=1200,
            metric_steps=10,
            kappa_points=21,
            topology_points=21,
        )
        A_np, B_np, _ = make_binary_pair(config.N, 0, 0.0)
        A, B = jnp.asarray(A_np), jnp.asarray(B_np)
        maps = compiled_dynamics(config)
        boundaries = []
        for delta in config.deltas:
            W = biased_two_memory_weights(A, B, delta)
            result = locate_boundary(maps["final_margin"], A, B, W, config)
            boundaries.append(result["boundary_kappa"])
        self.assertLess(boundaries[0], 0.5)
        self.assertAlmostEqual(boundaries[1], 0.5, places=10)
        self.assertGreater(boundaries[2], 0.5)

    def test_energy_matches_dense_weight_formula(self):
        generator = np.random.default_rng(2)
        A_np, B_np, _ = make_binary_pair(20, 2, 0.0)
        states = generator.normal(size=(3, 4, 20))
        delta = 0.2
        gain = 3.0
        W = np.asarray(biased_two_memory_weights(
            jnp.asarray(A_np), jnp.asarray(B_np), delta
        ))
        activations = np.tanh(gain * states)
        clipped = np.clip(
            activations,
            -1 + 10 * np.finfo(float).eps,
            1 - 10 * np.finfo(float).eps,
        )
        dense = -0.5 * np.einsum(
            "...i,ij,...j->...", activations, W, activations
        ) + np.sum(
            (
                clipped * np.arctanh(clipped)
                + 0.5 * np.log1p(-np.square(clipped))
            ) / gain,
            axis=-1,
        )
        efficient = continuous_hopfield_energy(
            states, A_np, B_np, delta, gain
        )
        np.testing.assert_allclose(efficient, dense, rtol=1e-12, atol=1e-12)

    def test_zero_crossing_requires_one_boundary(self):
        kappas = np.linspace(0, 1, 5)
        one = estimate_zero_crossing(kappas, 0.63 - kappas)
        self.assertEqual(one["zero_count"], 1)
        self.assertAlmostEqual(one["zero_kappa"], 0.63)
        multiple = estimate_zero_crossing(
            kappas, np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        )
        self.assertGreater(multiple["zero_count"], 1)
        self.assertTrue(np.isnan(multiple["zero_kappa"]))
        self.assertEqual(
            count_sign_transitions(np.array([1, 0, -1]), tolerance=1e-12), 1
        )

    def test_small_run_writes_reviewable_artifacts(self):
        config = MovingBoundaryConfig(
            N=60,
            seeds=(0,),
            deltas=(-0.15, 0.0, 0.15),
            boundary_steps=1200,
            metric_steps=20,
            kappa_points=21,
            topology_points=21,
            evaluation_start_time=0.2,
            representative_delta=0.15,
        )
        conditions, time_results, grids = run_experiment(config)
        self.assertEqual(len(conditions), 3)
        self.assertTrue((conditions.topology_status == "valid").all())
        self.assertEqual(
            grids["G"].shape,
            (len(conditions), config.kappa_points, config.metric_steps + 1),
        )
        self.assertLessEqual(float(np.diff(grids["energy"], axis=2).max()), 1e-10)
        with tempfile.TemporaryDirectory() as temporary:
            summary, conclusion = write_artifacts(
                temporary, conditions, time_results, grids, config
            )
            for name in (
                "conditions.csv",
                "time_results.csv",
                "estimator_summary.csv",
                "observable_grids.npz",
                "config.json",
                "summary.json",
                "main_figure.png",
                "main_figure.pdf",
                "conclusion_draft.md",
            ):
                self.assertTrue((Path(temporary) / name).is_file(), name)
            self.assertIn("passed_boundary_tracking", summary)
            self.assertEqual(conclusion, Path(temporary) / "conclusion_draft.md")


if __name__ == "__main__":
    unittest.main()
