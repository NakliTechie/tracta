"""Exact tests and false-alarm accounting, standard library only."""
import math


def sign_test_greater(k, n):
    """Exact one-sided sign test: P(X >= k) for X ~ Binomial(n, 1/2). 1.0 when there is nothing to compare."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, level=0.95):
    """Exact two-sided interval for a binomial rate, by bisection on the binomial tail."""
    if n == 0:
        return 0.0, 1.0
    a = (1 - level) / 2
    low = 0.0 if k == 0 else _lower(k, n, a)
    high = 1.0 if k == n else _upper(k, n, a)
    return low, high


def _lower(k, n, a):
    # Smallest p with P(X >= k | p) >= a.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - binom_cdf(k - 1, n, mid) < a:
            lo = mid
        else:
            hi = mid
    return hi


def _upper(k, n, a):
    # Largest p with P(X <= k | p) >= a.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) >= a:
            lo = mid
        else:
            hi = mid
    return lo


def check_alpha(k, alpha=0.05):
    """False-alarm budget for check number k (1-based): alpha * 6 / (pi^2 k^2).

    The shares sum to alpha over unlimited checks, so the chance of ever raising a false alarm on an
    unchanged provider stays at or below alpha however often you check."""
    return alpha * 6 / (math.pi ** 2 * k ** 2)


def budget_spent(k, alpha=0.05):
    return sum(check_alpha(i, alpha) for i in range(1, k + 1))


def jsd(p_logs, q_logs):
    """Jensen-Shannon divergence (bits) between two top-k log-probability maps, tail mass as one bucket."""
    def dist(top):
        d = {t: math.exp(v) for t, v in top.items()}
        d["<other>"] = max(0.0, 1.0 - sum(d.values()))
        return d
    p, q = dist(p_logs), dist(q_logs)
    keys = set(p) | set(q)
    out = 0.0
    for t in keys:
        a, b = p.get(t, 0.0), q.get(t, 0.0)
        m = (a + b) / 2
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out
