"""实验 4 的解析控制、短时机制与产物契约。"""
from pathlib import Path
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from experiment_01_boundary_localization import make_binary_pair
from experiment_02_moving_boundary import biased_two_memory_weights
from experiment_03_2d_directional_geometry import make_perpendicular_direction
from experiment_04_direction_mechanism_controls import (
    MechanismControlConfig,
    compiled_mechanism_controls,
    continuous_hopfield_energy_jax,
    run_experiment,
    vector_field_jacobian,
    write_artifacts,
)


class DirectionMechanismControlTests(unittest.TestCase):
    def setUp(self):
        self.config = MechanismControlConfig(
            seeds=(0,), deltas=(0.3,), metric_steps=2,
            boundary_steps=20, u_points=11, v_points=5,
            topology_u_points=11, representative_delta=0.3,
            representative_time=0.02, direction_evaluation_start_time=0.02,
        )
        A_np, B_np, _ = make_binary_pair(self.config.N, 0, 0.0)
        eta_np = make_perpendicular_direction(A_np, B_np, 0)
        self.A, self.B, self.eta = map(jnp.asarray, (A_np, B_np, eta_np))
        self.W = biased_two_memory_weights(self.A, self.B, 0.3)

    def test_analytic_vector_field_jacobian_matches_autodiff(self):
        state = 0.2 * self.A - 0.1 * self.B + 0.05 * self.eta

        def field(x):
            return -x + self.W @ jnp.tanh(self.config.gain * x)

        expected = jax.jacfwd(field)(state)
        actual = vector_field_jacobian(state, self.W, self.config.gain)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_energy_decreases_along_vector_field(self):
        state = 0.2 * self.A - 0.1 * self.B + 0.05 * self.eta
        energy_gradient = jax.grad(continuous_hopfield_energy_jax)(
            state, self.W, self.config.gain
        )
        field = -state + self.W @ jnp.tanh(self.config.gain * state)
        self.assertLess(float(energy_gradient @ field), 0.0)

    def test_initial_strain_is_metric_time_derivative(self):
        controls = compiled_mechanism_controls(self.config)
        coordinates = jnp.asarray([[0.56, 0.1]])
        _, _, metric, _, _, strain = [
            np.asarray(value)
            for value in controls(coordinates, self.A, self.B, self.eta, self.W)
        ]
        finite_difference = (metric[0, 1] - metric[0, 0]) / self.config.dt
        relative_error = np.linalg.norm(finite_difference - strain[0, 0]) / np.linalg.norm(
            strain[0, 0]
        )
        self.assertLess(relative_error, 0.15)

    def test_overlap_gradient_and_energy_hessian_match_autodiff(self):
        controls = compiled_mechanism_controls(self.config)
        coordinates = jnp.asarray([[0.56, 0.1]])
        _, _, _, overlap_gradient, energy_hessian, _ = controls(
            coordinates, self.A, self.B, self.eta, self.W
        )

        from experiment_03_2d_directional_geometry import _trajectory_2d

        def margin_at_first_step(q):
            state = _trajectory_2d(
                q, self.A, self.B, self.eta, self.W,
                gain=self.config.gain, dt=self.config.dt, steps=self.config.metric_steps,
            )[1]
            return state @ (self.A - self.B) / self.config.N

        def energy_at_first_step(q):
            state = _trajectory_2d(
                q, self.A, self.B, self.eta, self.W,
                gain=self.config.gain, dt=self.config.dt, steps=self.config.metric_steps,
            )[1]
            return continuous_hopfield_energy_jax(
                state, self.W, self.config.gain
            )

        np.testing.assert_allclose(
            overlap_gradient[0, 1], jax.grad(margin_at_first_step)(coordinates[0]),
            rtol=1e-11, atol=1e-11,
        )
        np.testing.assert_allclose(
            energy_hessian[0, 1], jax.hessian(energy_at_first_step)(coordinates[0]),
            rtol=1e-10, atol=1e-10,
        )

    def test_small_run_writes_mechanism_artifacts(self):
        config = MechanismControlConfig(
            seeds=(0,), deltas=(0.3,), metric_steps=50,
            boundary_steps=1200, u_points=31, v_points=7,
            topology_u_points=21, direction_evaluation_start_time=0.5,
            representative_delta=0.3, representative_time=1.0,
        )
        conditions, boundaries, directions, raw = run_experiment(config)
        self.assertEqual(conditions.iloc[0].topology_status, "valid")
        early = directions[np.isclose(directions.time, config.early_mechanism_time)]
        self.assertTrue(early.G_vs_S0_angle_deg.notna().all())
        with tempfile.TemporaryDirectory() as temporary:
            summary, conclusion = write_artifacts(
                temporary, conditions, boundaries, directions, raw, config
            )
            for name in (
                "conditions.csv",
                "boundary_points.csv",
                "direction_controls.csv.gz",
                "estimator_summary.csv",
                "mechanism_arrays.npz",
                "config.json",
                "summary.json",
                "main_figure.png",
                "main_figure.pdf",
                "conclusion_draft.md",
            ):
                self.assertTrue((Path(temporary) / name).is_file(), name)
            self.assertIn("mechanism_outcome", summary)
            self.assertEqual(conclusion, Path(temporary) / "conclusion_draft.md")


if __name__ == "__main__":
    unittest.main()
