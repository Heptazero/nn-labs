import unittest

import numpy as np

from am_bench.joint_gate import (
    JointGateConfig,
    anti_hebbian_direction,
    classify_state,
    outcome_flags,
    paired_bootstrap_difference,
)


class JointGateTests(unittest.TestCase):
    def test_anti_hebbian_direction_is_symmetric_and_zero_diagonal(self):
        states = np.array([[1.0, -1.0, 0.5], [-0.5, 0.2, 1.0]])
        direction = anti_hebbian_direction(states)
        np.testing.assert_allclose(direction, np.swapaxes(direction, 1, 2))
        np.testing.assert_array_equal(np.diagonal(direction, axis1=1, axis2=2), np.zeros((2, 3)))

    def test_state_classifier_separates_pure_mixture_and_other(self):
        patterns = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=float)
        mixture = np.array([1, 1, 1], dtype=float)
        self.assertEqual(classify_state(patterns[1], patterns, mixture), ("pure", 1))
        self.assertEqual(classify_state(mixture, patterns, mixture), ("mixture", -1))
        self.assertEqual(classify_state(-mixture, patterns, mixture), ("other", -1))

    def test_paired_bootstrap_uses_within_bank_difference(self):
        first = np.ones(20)
        second = np.zeros(20)
        mean, interval = paired_bootstrap_difference(first, second, seed=1, repetitions=100)
        self.assertEqual(mean, 1.0)
        self.assertEqual(interval, [1.0, 1.0])

    def test_budget_must_align_with_time_step(self):
        with self.assertRaises(ValueError):
            JointGateConfig(plasticity_budget=1.005).validate()

    def test_nonconverged_pure_identity_is_not_scored_as_success(self):
        self.assertEqual(outcome_flags("mixture", 3, "pure", 1, False), (False, False))
        self.assertEqual(outcome_flags("mixture", 3, "pure", 1, True), (False, True))


if __name__ == "__main__":
    unittest.main()
