"""球面状态空间、球面 DAM 和两类轨迹运行器的数值契约。"""
import math
import unittest

import torch

from am_bench.models.spherical import SphericalPolynomialDAM
from am_bench.state_spaces import (
    project_to_tangent,
    sample_tangent_directions,
    sample_uniform_sphere,
    spherical_geodesic,
    spherical_retraction,
)
from am_bench.trajectory import converged_rollout, fixed_horizon_rollout


class SphericalComponentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_uniform_sphere_sampling_is_reproducible(self):
        first = sample_uniform_sphere(
            7, 5, generator=torch.Generator().manual_seed(11)
        )
        second = sample_uniform_sphere(
            7, 5, generator=torch.Generator().manual_seed(11)
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(
            first.norm(dim=-1), torch.full((7,), math.sqrt(5), dtype=torch.float64)
        )

    def test_tangent_geodesic_and_retraction_preserve_geometry(self):
        points = sample_uniform_sphere(
            3, 6, generator=torch.Generator().manual_seed(12)
        )
        directions = sample_tangent_directions(
            points, 4, generator=torch.Generator().manual_seed(13)
        )
        torch.testing.assert_close(
            (directions * points[:, None, :]).sum(-1),
            torch.zeros((3, 4), dtype=torch.float64),
            rtol=0,
            atol=1e-12,
        )
        moved = spherical_geodesic(points, directions, 0.2)
        torch.testing.assert_close(
            moved.norm(dim=-1), points.norm(dim=-1)[:, None].expand(-1, 4)
        )
        cosine = (moved * points[:, None, :]).sum(-1) / points.norm(dim=-1)[:, None].square()
        torch.testing.assert_close(cosine, torch.full_like(cosine, math.cos(0.2)))
        retracted = spherical_retraction(points, 0.3 * directions)
        torch.testing.assert_close(
            retracted.norm(dim=-1), points.norm(dim=-1)[:, None].expand(-1, 4)
        )

    def test_projection_removes_only_radial_component(self):
        point = torch.tensor([2.0, 0.0], dtype=torch.float64)
        vectors = torch.tensor([[3.0, 4.0], [-1.0, 2.0]], dtype=torch.float64)
        tangent = project_to_tangent(point, vectors)
        torch.testing.assert_close(
            tangent, torch.tensor([[0.0, 4.0], [0.0, 2.0]], dtype=torch.float64)
        )

    def test_fixed_horizon_rollout_retains_autograd(self):
        initial = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
        states = fixed_horizon_rollout(lambda state: 2 * state, initial, 3)
        self.assertEqual(tuple(states.shape), (4, 2))
        states[-1].sum().backward()
        torch.testing.assert_close(initial.grad, torch.full_like(initial, 8.0))

    def test_converged_and_censored_rollouts_are_distinct(self):
        converged = converged_rollout(
            lambda state: state / 2,
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            max_steps=20,
            tolerance=1e-3,
        )
        self.assertEqual(converged.status, "converged")
        self.assertLess(float(converged.residuals[-1]), 1e-3)
        censored = converged_rollout(
            lambda state: state + 1,
            torch.zeros(2, dtype=torch.float64),
            max_steps=3,
            tolerance=1e-3,
        )
        self.assertEqual(censored.status, "max_steps")
        self.assertEqual(censored.steps, 3)

    def test_spherical_dam_step_energy_and_retrieval(self):
        patterns = torch.tensor(
            [[math.sqrt(2), 0.0], [0.0, math.sqrt(2)]], dtype=torch.float64
        )
        model = SphericalPolynomialDAM(degree=3).fit(patterns)
        self.assertAlmostEqual(float(model.energy(patterns[0])), -1 / 3)
        torch.testing.assert_close(model.step(patterns[0]), patterns[0])

        states = sample_uniform_sphere(
            5, 2, generator=torch.Generator().manual_seed(14)
        )
        direction = sample_tangent_directions(
            states[0], 1, generator=torch.Generator().manual_seed(15)
        )[0]
        trajectory, tangent = torch.func.jvp(
            lambda state: fixed_horizon_rollout(model.step, state, 2),
            (states[0],),
            (direction,),
        )
        self.assertEqual(tuple(trajectory.shape), (3, 2))
        self.assertTrue(bool(torch.isfinite(tangent).all()))
        torch.testing.assert_close(
            model.step(states).norm(dim=-1),
            torch.full((5,), math.sqrt(2), dtype=torch.float64),
        )

        observations = []
        result = model.retrieve(
            patterns[0],
            update_seed=0,
            max_sweeps=3,
            observer=lambda step, state, energy, diagnostics: observations.append(step),
        )
        self.assertEqual(result.status, "fixed")
        self.assertEqual(result.sweeps, 1)
        self.assertEqual(observations, [0, 1])


if __name__ == "__main__":
    unittest.main()
