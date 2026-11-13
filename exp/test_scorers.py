"""正式 FP／EE 公式和共享秩统计的确定性回归；不以攻击效果作为通过条件。"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.stats import spearmanr
import c1_recent_scorers as c1
import c2_deployed_family1 as c2
from tools.rank_metrics import average_ranks, spearman, correlation_summary


class RankTests(unittest.TestCase):
    def test_average_ties(self):
        np.testing.assert_array_equal(average_ranks([4, 2, 2, 9]), [2, .5, .5, 3])

    def test_scipy_oracle(self):
        rng = np.random.default_rng(32)
        for n in (2, 3, 8, 30):
            for _ in range(40):
                a, b = rng.integers(-2, 3, (2, n))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    reference = float(spearmanr(a, b).statistic)
                got = spearman(a, b)
                if np.isnan(reference):
                    self.assertTrue(np.isnan(got))
                else:
                    self.assertAlmostEqual(got, reference, places=14)

    def test_constants_and_short_inputs(self):
        for a, b in (([], []), ([1], [2]), ([1, 1, 1], [1, 2, 3]),
                     ([1, 2, 3], [4, 4, 4])):
            self.assertTrue(np.isnan(spearman(a, b)))

    def test_small_nonconstant_values_are_ranked(self):
        self.assertAlmostEqual(spearman([1e-20, 2e-20, 3e-20], [1, 2, 3]), 1)

    def test_permutation_and_sign(self):
        a, b = np.array([1, 1, 4, 2]), np.array([4, 2, 1, 1])
        perm = [3, 0, 2, 1]
        self.assertAlmostEqual(spearman(a, b), spearman(a[perm], b[perm]))
        self.assertAlmostEqual(spearman(a, -b), -spearman(a, b))

    def test_invalid_inputs(self):
        for a, b in (([1], [1, 2]), ([[1, 2]], [[1, 2]]), ([np.nan, 1], [1, 2]),
                     ([np.inf, 1], [1, 2])):
            with self.assertRaises(ValueError):
                spearman(a, b)

    def test_null_and_partial_seed_summary(self):
        for values in ([np.nan, np.nan], [.5, np.nan]):
            summary = correlation_summary(values)
            self.assertIsNone(summary["mean"])
            json.dumps(summary, allow_nan=False)
        self.assertEqual(correlation_summary([.2, .4])["defined"], 2)

    def test_shared_metric_wiring(self):
        self.assertIs(c1.spearman, spearman)
        self.assertIs(c2.spearman, spearman)
        root = Path(__file__).resolve().parent
        for name in ("a20_impossibility_scope.py", "a24_weight_cap.py", "a26_real_confound.py"):
            tree = ast.parse((root / name).read_text())
            self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == "tools.rank_metrics"
                                for n in ast.walk(tree)), name)


class ScorerTests(unittest.TestCase):
    def test_fp_known_fractional_share(self):
        got = c1.fp_ee_weights(.2, .8, [.3, .5, .1], [.7, .6, .9], "fp")
        np.testing.assert_allclose(got, [2/7, 5/7, 0])

    def test_ee_paper_additive_example(self):
        got = c1.fp_ee_weights(0, 3, [0, 1, 2], [3, 2, 1], "ee")
        np.testing.assert_allclose(got, [.25, 1/3, 5/12])

    def test_ee_mixed_channels_independent_oracle(self):
        p, q = np.array([.1, .9, .7]), np.array([.1, .2, .9])
        beta = np.array([sum(.6-p[j] for j in range(3) if j != i)/4 for i in range(3)])
        gamma = np.array([sum(q[j]-.5 for j in range(3) if j != i)/4 for i in range(3)])
        score = (np.maximum(beta, 0) + np.maximum(gamma, 0))/2
        old = np.maximum((beta+gamma)/2, 0)
        events = {}
        got = c1.fp_ee_weights(.5, .6, p, q, "ee", events=events)
        np.testing.assert_allclose(got, score/score.sum())
        self.assertGreater(np.max(np.abs(got-old/old.sum())), .1)
        self.assertEqual(events["ee_mixed_sign_rounds"], 1)

    def test_fp_loo_and_ioi_fallback(self):
        for p, q, branch in (([-2, -3], [-1, -2], "fp_loo_fallback"),
                             ([1, 2], [2, 3], "fp_ioi_fallback")):
            events = {}
            got = c1.fp_ee_weights(0, 0, p, q, "fp", events=events)
            np.testing.assert_allclose(got, [1/3, 2/3])
            self.assertEqual(events[branch], 1)

    def test_all_zero_local_fallback(self):
        for kind in ("fp", "ee"):
            events = {}
            got = c1.fp_ee_weights(0, 0, [0, 0], [0, 0], kind, events=events)
            np.testing.assert_array_equal(got, [.5, .5])
            self.assertEqual(events["uniform_local_fallback"], 1)

    def test_negative_budget_does_not_reverse_shares(self):
        p, q = np.array([.3, .5, .1]), np.array([.7, .6, .9])
        for kind in ("fp", "ee"):
            a = c1.fp_ee_weights(.2, .8, p, q, kind)
            b = c1.fp_ee_weights(-9.8, -9.2, p-10, q-10, kind)
            np.testing.assert_allclose(a, b, atol=1e-14)

    def test_minshift_is_separate_policy(self):
        args = (0, 4, [1, 2, 3], [3, 2, 1], "fp")
        np.testing.assert_allclose(c1.fp_ee_weights(*args), [1/6, 2/6, 3/6])
        np.testing.assert_allclose(c1.fp_ee_weights(*args, policy="minshift"), [0, 1/3, 2/3])

    def test_symmetry_and_report_path(self):
        p, q = np.array([.2, .4, .7]), np.array([.3, .6, .9])
        perm = [2, 0, 1]
        for kind, scorer in (("fp", c1.w_fp), ("ee", c1.w_ee)):
            expected = c1.fp_ee_weights(.1, 1, p, q, kind)
            np.testing.assert_allclose(c1.fp_ee_weights(.1, 1, p[perm], q[perm], kind), expected[perm])
            with patch.object(c1, "marginals", return_value=(.1, 1, p, q)):
                got = scorer(np.zeros((3, 2)), {"misreport": [0]}, {})
            pp, qq = c1.apply_misreport(p, q, {"misreport": [0]})
            np.testing.assert_allclose(got, c1.fp_ee_weights(.1, 1, pp, qq, kind))

    def test_nonfinite_is_not_silently_uniform(self):
        with self.assertRaises(ValueError):
            c1.fp_ee_weights(0, 1, [np.nan, .2], [.2, .3], "fp")

    def test_short_simulations(self):
        cfg = dict(c1.CFG, N=3, d=2, n_local=10, n_test=8, rounds=4)
        for name in ("FP (2026)", "EE (2026)"):
            for attack in ("none", "tsp-var", "misreport"):
                result = c1.sim(attack, name, cfg, 7)
                self.assertEqual(result["events"]["evaluations"], 4)
                self.assertTrue(np.all(np.isfinite(result["mse"])))
                self.assertAlmostEqual(result["w_last"].sum(), 1)

    def test_import_is_quiet(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import c1_recent_scorers; import c2_deployed_family1"],
                                cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
