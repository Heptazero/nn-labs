import unittest

import numpy as np

from am_bench.curvature_gate import (
    cross_validated_scalar_gate,
    fit_exponential_rates,
    hebbian_weights,
    majority_mixture,
    matched_directions,
)


class CurvatureGateTests(unittest.TestCase):
    def test_three_pattern_mixture_has_no_ties(self):
        patterns = np.array([
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [-1, 1, 1, -1],
        ], dtype=float)
        np.testing.assert_array_equal(majority_mixture(patterns), [1, 1, 1, -1])
        with self.assertRaises(ValueError):
            majority_mixture(patterns[:2])

    def test_hebbian_matrix_is_symmetric_and_has_zero_diagonal(self):
        patterns = np.array([[1, -1, 1], [-1, -1, 1], [1, 1, -1]], dtype=float)
        weights = hebbian_weights(patterns)
        np.testing.assert_allclose(weights, weights.T)
        np.testing.assert_array_equal(np.diag(weights), np.zeros(3))

    def test_fixed_window_recovers_known_exponential_rates(self):
        times = np.linspace(0, 4, 401)
        expected = np.array([0.25, 0.8, 1.4])
        speeds = 3.0 * np.exp(-times[:, None] * expected[None, :])
        rates, r2 = fit_exponential_rates(times, speeds, (0.25, 1.25))
        np.testing.assert_allclose(rates, expected, atol=1e-12)
        np.testing.assert_allclose(r2, np.ones(3), atol=1e-12)

    def test_local_directions_are_reproducible_unit_vectors(self):
        first = matched_directions(123, 8, 20)
        second = matched_directions(123, 8, 20)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(8))

    def test_scalar_gate_generalizes_across_held_out_banks(self):
        banks = {seed: {"pure": 1.0 + seed / 1000, "mixture": 0.5 + seed / 1000}
                 for seed in range(20)}
        result = cross_validated_scalar_gate(banks)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
