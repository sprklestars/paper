"""FIVA 正式式 (1)--(8) 的确定性回归测试；不依赖旧攻击数字。"""
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import c2_deployed_family1 as c2


class FivaTests(unittest.TestCase):
    def test_known_weights_and_two_round_variance(self):
        est = np.array([[1.0], [4.0]])
        sizes = np.array([0.5, 0.5])
        weights, variance = c2.fiva_weights(est, sizes, np.ones(1))
        np.testing.assert_allclose(weights[:, 0], [0.8, 0.2], rtol=1e-14)
        np.testing.assert_allclose(variance, [1 / 1.575], rtol=1e-14)
        second, variance2 = c2.fiva_weights(est, sizes, variance)
        np.testing.assert_allclose(second, weights, rtol=1e-14)
        np.testing.assert_allclose(variance2, [1 / 2.12125], rtol=1e-14)
        # 旧实现混入历史项后给出 0.61904762，必须排除。
        self.assertGreater(abs(weights[0, 0] - 0.619047619), 0.1)

    def test_history_does_not_enter_current_mean(self):
        est = np.array([[1.0, 4.0], [4.0, 1.0]])
        sizes = np.array([0.25, 0.75])
        first, _ = c2.fiva_weights(est, sizes, np.array([1.0, 2.0]))
        changed, _ = c2.fiva_weights(est, sizes, np.array([0.001, 100.0]), lam=0)
        np.testing.assert_allclose(first, changed)
        np.testing.assert_allclose(first.sum(axis=0), 1)
        np.testing.assert_allclose(first[:, 0], [4 / 7, 3 / 7])
        self.assertTrue(np.all(first >= 0))

    def test_zero_lambda(self):
        weights, variance = c2.fiva_weights(np.array([[1.0], [4.0]]),
                                           np.array([0.5, 0.5]), np.array([9.0]), lam=0)
        np.testing.assert_allclose(weights[:, 0], [0.8, 0.2])
        np.testing.assert_allclose(variance, [1 / 0.625])

    def test_welford_and_broadcast_gradient_variance(self):
        grads = np.array([[1.0, 2.0], [3.0, -2.0], [5.0, 0.0]])
        np.testing.assert_allclose(c2.welford(grads), np.var(grads, axis=0))
        prior = np.array([0.2, 0.8])
        got = c2.fiva_gradient_variance(grads, 0.3, prior)
        np.testing.assert_allclose(got, prior + 3 * 0.3 ** 2 * grads.var(axis=0))
        np.testing.assert_array_equal(prior, [0.2, 0.8])

    def test_zero_variance_stabilizers(self):
        est = np.zeros((2, 2))
        sizes = np.array([0.25, 0.75])
        for numerics in ("formula", "official-epsilon"):
            w, v = c2.fiva_weights(est, sizes, np.ones(2), numerics=numerics)
            np.testing.assert_allclose(w, [[0.25, 0.25], [0.75, 0.75]])
            self.assertTrue(np.all(np.isfinite(v)))
            self.assertTrue(np.all(v > 0))
            if numerics == "official-epsilon":
                np.testing.assert_allclose(v, 1e-5)
        with self.assertRaises(ValueError):
            c2.fiva_weights(est, sizes, np.ones(2), floor=0)

    def test_official_epsilon_arithmetic(self):
        est = np.array([[1e-6, 2.0], [1e-3, 3.0]])
        sizes = np.array([0.5, 0.5])
        prior = np.array([0.01, 0.4])
        q = sizes[:, None] / (est + 1e-5)
        w, v = c2.fiva_weights(est, sizes, prior, numerics="official-epsilon")
        np.testing.assert_allclose(w, q / q.sum(axis=0))
        np.testing.assert_allclose(v, np.maximum(1 / (0.95 / prior + q.sum(axis=0)), 1e-5))

    def test_invalid_inputs(self):
        est, sizes, prior = np.ones((2, 1)), np.full(2, 0.5), np.ones(1)
        for options in ({"lam": -0.1}, {"floor": -1}, {"lam": np.nan},
                        {"numerics": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                c2.fiva_weights(est, sizes, prior, **options)
        for bad in (np.array([[-1.0], [1.0]]), np.array([[np.nan], [1.0]])):
            with self.assertRaises(ValueError):
                c2.fiva_weights(bad, sizes, prior)

    def test_three_round_simulation_against_independent_recurrence(self):
        cfg = dict(c2.CFG, N=3, k_adv=1, d=2, n_local=10, n_test=6, rounds=3)
        for scorer in ("fiva-g", "fiva-p"):
            calls = []

            def local(rng, w, tgt, sigma, n_loc, d, lr):
                i = len(calls) % cfg["N"]
                grads = np.array([[1 + i, 0.5], [0.2, 2 + i]])
                iters = w - lr * np.cumsum(grads, axis=0)
                calls.append((grads.copy(), iters.copy(), tgt.copy(), w.copy()))
                return iters[-1] - w, grads, iters

            with patch.object(c2, "local_work", side_effect=local):
                run = c2.sim("none", scorer, cfg, 7, var0=0.4, trace=True)
            variance = np.full(2, 0.4)
            model = np.zeros(2)
            for t, row in enumerate(run["trace"]):
                local_calls = calls[t * cfg["N"]:(t + 1) * cfg["N"]]
                estimates = np.array([
                    variance + 2 * cfg["lr"] ** 2 * np.var(g, axis=0)
                    if scorer == "fiva-g" else np.var(p, axis=0)
                    for g, p, _, _ in local_calls])
                precision = (1 / cfg["N"]) / np.maximum(estimates, c2.VAR_FLOOR)
                weights = precision / precision.sum(axis=0)
                np.testing.assert_allclose(row["start_var"], variance)
                np.testing.assert_allclose(row["local_var"], estimates)
                np.testing.assert_allclose(row["weights"], weights)
                variance = 1 / (c2.FIVA_LAM / variance + precision.sum(axis=0))
                model += np.sum(weights * np.array([p[-1] - w for _, p, _, w in local_calls]), axis=0)
                np.testing.assert_allclose(row["global_var"], variance)
                np.testing.assert_allclose(row["model"], model)
            for _, _, target, _ in calls:
                np.testing.assert_array_equal(target, calls[0][2])

    def test_only_tsp_has_biased_local_target(self):
        cfg = dict(c2.CFG, N=3, k_adv=1, d=2, n_local=10, n_test=6, rounds=1)
        original = c2.local_work
        for attack in ("none", "fang-trim", "shej-minmax", "tsp-var"):
            targets = []

            def record(rng, w, target, *args):
                targets.append(target.copy())
                return original(rng, w, target, *args)

            with patch.object(c2, "local_work", side_effect=record):
                c2.sim(attack, "fiva-g", cfg, 1)
            difference = np.linalg.norm(targets[0] - targets[1])
            self.assertAlmostEqual(difference, c2.TSP_DELTA if attack == "tsp-var" else 0)

    def test_fiva_p_weights_independent_of_lambda(self):
        cfg = dict(c2.CFG, N=3, k_adv=1, d=2, n_local=10, n_test=6, rounds=3)
        first = c2.sim("tsp-var", "fiva-p", cfg, 1, lam=0)
        second = c2.sim("tsp-var", "fiva-p", cfg, 1, lam=0.95)
        np.testing.assert_array_equal(first["final_C"], second["final_C"])
        np.testing.assert_array_equal(first["mse"], second["mse"])
        self.assertNotEqual(first["global_var"][-1], second["global_var"][-1])

    def test_import_does_not_run_experiments(self):
        here = Path(__file__).resolve().parent
        command = [sys.executable, "-B", "-c",
                   "import c2_deployed_family1 as c; assert not hasattr(c, 'out')"]
        result = subprocess.run(command, cwd=here, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


def run_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FivaTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("FIVA 公式回归测试失败")
    return result.testsRun


if __name__ == "__main__":
    run_checks()
