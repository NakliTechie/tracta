"""Check tracta's standard-library statistics against SciPy, and the questions against their invariants."""
import math
import unittest

from tracta.compare import evaluate
from tracta.questions import boundary_items, norm
from tracta.stats import budget_spent, check_alpha, clopper_pearson, jsd, sign_test_greater

try:
    from scipy.stats import binomtest
except ImportError:  # SciPy is a test-only cross-check
    binomtest = None


class Stats(unittest.TestCase):
    @unittest.skipIf(binomtest is None, "scipy not installed")
    def test_sign_test_matches_scipy(self):
        for k, n in [(20, 20), (6, 14), (112, 112), (0, 5), (30, 50)]:
            want = binomtest(k, n, 0.5, alternative="greater").pvalue
            self.assertAlmostEqual(sign_test_greater(k, n), want, places=10)
        self.assertEqual(sign_test_greater(0, 0), 1.0)

    @unittest.skipIf(binomtest is None, "scipy not installed")
    def test_clopper_pearson_matches_scipy(self):
        for k, n in [(0, 436), (20, 436), (436, 436), (3, 10)]:
            ci = binomtest(k, n).proportion_ci(method="exact")
            lo, hi = clopper_pearson(k, n)
            self.assertAlmostEqual(lo, ci.low, places=6)
            self.assertAlmostEqual(hi, ci.high, places=6)

    def test_budget_never_exceeds_alpha(self):
        self.assertLess(budget_spent(100_000, 0.05), 0.05)
        self.assertAlmostEqual(check_alpha(1, 0.05), 0.05 * 6 / math.pi ** 2)

    def test_jsd_bounds(self):
        same = {"a": math.log(0.5), "b": math.log(0.5)}
        self.assertAlmostEqual(jsd(same, same), 0.0)
        self.assertAlmostEqual(jsd({"a": 0.0}, {"b": 0.0}), 1.0)


class Questions(unittest.TestCase):
    def test_seeded_and_complete(self):
        a, b = boundary_items(1), boundary_items(2)
        self.assertEqual(len(a), 436)
        self.assertEqual({i["id"] for i in a}, {i["id"] for i in b})
        self.assertNotEqual([i["prompt"] for i in a], [i["prompt"] for i in b])
        self.assertEqual(a, boundary_items(1))

    def test_norm(self):
        self.assertEqual(norm("The atomic number of Hydrogen is 1."), "1")
        self.assertEqual(norm("**Hydrogen**"), "hydrogen")
        self.assertEqual(norm("12,345\n"), "12345")


class Decision(unittest.TestCase):
    def run_of(self, answers):
        return {"answers": answers, "prompts": {k: k for k in answers}}

    def test_verdicts(self):
        ids = [f"q{i}" for i in range(400)]
        base = self.run_of({i: "1" for i in ids})
        noisy = self.run_of({i: ("2" if n < 8 else "1") for n, i in enumerate(ids)})
        same = self.run_of({i: ("2" if 100 <= n < 106 else "1") for n, i in enumerate(ids)})
        moved = self.run_of({i: ("2" if n < 120 else "1") for n, i in enumerate(ids)})
        self.assertEqual(evaluate([base], moved, 1)["verdict"], "no verdict")
        self.assertEqual(evaluate([base, noisy], same, 1)["verdict"], "no change detected")
        self.assertEqual(evaluate([base, noisy], moved, 1)["verdict"], "changed")

    def test_budget_counts_tested_checks_only(self):
        ids = [f"q{i}" for i in range(50)]
        base = self.run_of({i: "1" for i in ids})
        untested = evaluate([base], base, 1)
        self.assertEqual(untested["alpha_this_check"], 0.0)
        self.assertEqual(untested["alpha_spent"], 0.0)
        tested = evaluate([base, base], base, 2, tested_before=0)
        self.assertAlmostEqual(tested["alpha_this_check"], check_alpha(1))


if __name__ == "__main__":
    unittest.main()
