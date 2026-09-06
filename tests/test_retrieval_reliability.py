"""验证实验边界：原公式复用、干预几何、无标签决策和并列统计。"""
import unittest
import numpy as np
import torch

from am_bench.models.modern import IterativeModernHopfield
from am_bench.reliability import (
    a1_make_memories, b1_store_memories, c1_make_queries, c2_make_directions,
    c3_spherical_probes, d1_retrieve, d2_majority_repair, e1_selective_risk,
)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.patterns = a1_make_memories(32, 16, "close_pairs", 55)
        self.model = b1_store_memories(self.patterns)
        self.cues, _, _ = c1_make_queries(self.patterns, 4, ("gaussian_1.0",), 56)

    def test_existing_scalar_and_iterative_parity(self):
        ids, states, _ = d1_retrieve(self.model, self.cues, 12)
        iterative = IterativeModernHopfield(beta=self.model.beta).fit(self.patterns)
        for i, query in enumerate(self.cues):
            reference = query.clone()
            for _ in range(12):
                reference, _ = self.model.readout(reference)
            torch.testing.assert_close(states[i], reference, rtol=1e-12, atol=1e-12)
            result = iterative.retrieve(query, update_seed=0, max_sweeps=12)
            torch.testing.assert_close(states[i], result.final_state, rtol=1e-12, atol=1e-12)
            self.assertEqual(int(ids[i]), int((self.patterns @ reference).argmax()))

    def test_spherical_probe_geometry_all_methods(self):
        base, _, _ = d1_retrieve(self.model, self.cues, 12)
        for method in ("guided", "shuffled", "random"):
            axes, _ = c2_make_directions(self.model, self.cues, base, method, 4, 57)
            torch.testing.assert_close((axes * self.cues[:, None]).sum(-1), torch.zeros(4, 4).double(), atol=1e-12, rtol=0)
            for angle in (0.1, 0.25):
                probes = c3_spherical_probes(self.cues, axes, angle)
                norm = self.cues.norm(dim=-1)
                torch.testing.assert_close(probes.norm(dim=-1), norm[:, None].expand(-1, 8))
                cosine = (probes * self.cues[:, None]).sum(-1) / norm[:, None].square()
                torch.testing.assert_close(cosine, torch.full_like(cosine, np.cos(angle)))

    def test_clean_direction_fallback_is_finite(self):
        axes, degenerate = c2_make_directions(self.model, self.patterns[:2], torch.tensor([0, 1]), "guided", 4, 9)
        self.assertTrue(bool((degenerate >= 1).all()))
        self.assertTrue(bool(torch.isfinite(axes).all()))
        torch.testing.assert_close(axes.norm(dim=-1), torch.ones(2, 4).double())

    def test_strict_majority_and_ties(self):
        base = torch.tensor([0, 0, 2])
        votes = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 0], [1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2, 0, 0]])
        self.assertEqual(d2_majority_repair(base, votes, 3).tolist(), [1, 0, 2])

    def test_tied_risk_is_permutation_invariant(self):
        errors = np.array([0, 1, 1, 0, 1])
        scores = np.array([0, 0, 1, 1, 1])
        expected = (1 + 2 * (2 / 3)) / 4
        self.assertAlmostEqual(e1_selective_risk(errors, scores), expected)
        order = [4, 3, 2, 1, 0]
        self.assertAlmostEqual(e1_selective_risk(errors[order], scores[order]), expected)
        self.assertAlmostEqual(e1_selective_risk(errors, np.zeros(5)), errors.mean())

    def test_close_pairs_and_target_sampling(self):
        self.assertTrue(bool(((self.patterns[::2] != self.patterns[1::2]).sum(-1) == 4).all()))
        cues, labels, _ = c1_make_queries(self.patterns, 4, ("clean", "block"), 22)
        self.assertEqual(labels[:4].tolist(), labels[4:].tolist())
        torch.testing.assert_close(cues[:4], self.patterns[labels[:4]])


if __name__ == "__main__":
    unittest.main()
