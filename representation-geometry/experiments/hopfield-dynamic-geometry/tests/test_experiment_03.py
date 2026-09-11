"""实验 3 的二维坐标、方向度量、拓扑门和产物契约。"""
from pathlib import Path
import sys
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from experiment_01_boundary_localization import make_binary_pair
from experiment_02_moving_boundary import biased_two_memory_weights
from experiment_03_2d_directional_geometry import (
    DirectionalGeometryConfig,
    acute_angle_deg,
    compiled_2d_dynamics,
    make_perpendicular_direction,
    run_experiment,
    write_artifacts,
)


class DirectionalGeometryTests(unittest.TestCase):
    def test_perpendicular_direction_has_matched_scale(self):
        A, B, _ = make_binary_pair(60, 4, 0.0)
        eta = make_perpendicular_direction(A, B, 4)
        self.assertAlmostEqual(float(eta @ A), 0.0, places=11)
        self.assertAlmostEqual(float(eta @ B), 0.0, places=11)
        self.assertAlmostEqual(
            float(np.linalg.norm(eta)), float(np.linalg.norm(B - A)), places=11
        )

    def test_initial_metric_is_isotropic_in_balanced_coordinates(self):
        config = DirectionalGeometryConfig(
            seeds=(0,), deltas=(0.3,), metric_steps=2,
            boundary_steps=20, u_points=11, v_points=5,
            topology_u_points=11, representative_delta=0.3,
            representative_time=0.02, direction_evaluation_start_time=0.02,
        )
        A_np, B_np, _ = make_binary_pair(config.N, 0, 0.0)
        eta_np = make_perpendicular_direction(A_np, B_np, 0)
        A, B, eta = map(jnp.asarray, (A_np, B_np, eta_np))
        W = biased_two_memory_weights(A, B, 0.3)
        maps = compiled_2d_dynamics(config)
        metric = np.asarray(maps["metric_grid"](
            jnp.asarray([[0.5, 0.0]]), A, B, eta, W
        ))[0, 0]
        expected = float(np.sum(np.square(B_np - A_np)))
        np.testing.assert_allclose(
            metric, expected * np.eye(2), rtol=1e-12, atol=1e-12
        )

    def test_acute_angle_ignores_eigenvector_sign(self):
        first = np.array([1.0, 0.0])
        second = np.array([-1.0, 0.0])
        self.assertAlmostEqual(float(acute_angle_deg(first, second)), 0.0)

    def test_small_run_has_curved_boundary_and_writes_artifacts(self):
        config = DirectionalGeometryConfig(
            seeds=(0,),
            deltas=(0.3,),
            metric_steps=50,
            boundary_steps=1200,
            u_points=31,
            v_points=7,
            topology_u_points=21,
            direction_evaluation_start_time=0.5,
            representative_delta=0.3,
            representative_time=1.0,
        )
        conditions, boundaries, directions, raw = run_experiment(config)
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions.iloc[0].topology_status, "valid")
        self.assertGreater(conditions.iloc[0].boundary_u_range, 0.01)
        at_one = directions[np.isclose(directions.time, 1.0)]
        self.assertLess(float(at_one.G_angle_deg.median()), 5.0)
        with tempfile.TemporaryDirectory() as temporary:
            summary, conclusion = write_artifacts(
                temporary, conditions, boundaries, directions, raw, config
            )
            for name in (
                "conditions.csv",
                "boundary_points.csv",
                "direction_results.csv.gz",
                "estimator_summary.csv",
                "directional_arrays.npz",
                "config.json",
                "summary.json",
                "main_figure.png",
                "main_figure.pdf",
                "conclusion_draft.md",
            ):
                self.assertTrue((Path(temporary) / name).is_file(), name)
            self.assertTrue(summary["passed_directional_geometry"])
            self.assertEqual(conclusion, Path(temporary) / "conclusion_draft.md")


if __name__ == "__main__":
    unittest.main()
