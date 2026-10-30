package core

import (
	"math"
	"math/big"
)

// SignTestGreater is the exact one-sided sign test: P(X >= k) for X ~ Binomial(n, 1/2).
// It is 1 when there is nothing to compare. Computed with exact integers.
func SignTestGreater(k, n int) float64 {
	if n == 0 {
		return 1
	}
	sum := new(big.Int)
	c := new(big.Int)
	for i := k; i <= n; i++ {
		sum.Add(sum, c.Binomial(int64(n), int64(i)))
	}
	p, _ := new(big.Rat).SetFrac(sum, new(big.Int).Lsh(big.NewInt(1), uint(n))).Float64()
	return p
}

func logPmf(i, n int, p float64) float64 {
	a, _ := math.Lgamma(float64(n + 1))
	b, _ := math.Lgamma(float64(i + 1))
	c, _ := math.Lgamma(float64(n - i + 1))
	return a - b - c + float64(i)*math.Log(p) + float64(n-i)*math.Log1p(-p)
}

// binomCDF is P(X <= k) for X ~ Binomial(n, p), 0 < p < 1.
func binomCDF(k, n int, p float64) float64 {
	s := 0.0
	for i := 0; i <= k; i++ {
		s += math.Exp(logPmf(i, n, p))
	}
	return math.Min(1, s)
}

// ClopperPearson is the exact two-sided 95% interval for a binomial rate, by bisection.
func ClopperPearson(k, n int) (float64, float64) {
	if n == 0 {
		return 0, 1
	}
	a := 0.025
	low, high := 0.0, 1.0
	if k > 0 { // smallest p with P(X >= k | p) >= a
		lo, hi := 0.0, 1.0
		for range 100 {
			mid := (lo + hi) / 2
			if 1-binomCDF(k-1, n, mid) < a {
				lo = mid
			} else {
				hi = mid
			}
		}
		low = hi
	}
	if k < n { // largest p with P(X <= k | p) >= a
		lo, hi := 0.0, 1.0
		for range 100 {
			mid := (lo + hi) / 2
			if binomCDF(k, n, mid) >= a {
				lo = mid
			} else {
				hi = mid
			}
		}
		high = lo
	}
	return low, high
}

// CheckAlpha is the false-alarm budget for tested check number k (1-based): alpha*6/(pi^2 k^2).
// The shares sum to alpha over unlimited checks.
func CheckAlpha(k int, alpha float64) float64 {
	return alpha * 6 / (math.Pi * math.Pi * float64(k) * float64(k))
}

// BudgetSpent is the budget used by the first k tested checks.
func BudgetSpent(k int, alpha float64) float64 {
	s := 0.0
	for i := 1; i <= k; i++ {
		s += CheckAlpha(i, alpha)
	}
	return s
}

// JSD is the Jensen-Shannon divergence in bits between two top-k log-probability maps, with the
// tail mass of each as one bucket.
func JSD(pLogs, qLogs map[string]float64) float64 {
	dist := func(top map[string]float64) map[string]float64 {
		d := map[string]float64{}
		sum := 0.0
		for t, v := range top {
			d[t] = math.Exp(v)
			sum += d[t]
		}
		d["<other>"] = math.Max(0, 1-sum)
		return d
	}
	p, q := dist(pLogs), dist(qLogs)
	keys := map[string]bool{}
	for t := range p {
		keys[t] = true
	}
	for t := range q {
		keys[t] = true
	}
	out := 0.0
	for t := range keys {
		a, b := p[t], q[t]
		m := (a + b) / 2
		if a > 0 {
			out += 0.5 * a * math.Log2(a/m)
		}
		if b > 0 {
			out += 0.5 * b * math.Log2(b/m)
		}
	}
	return out
}
