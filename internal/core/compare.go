package core

import (
	"sort"
)

// Run is one baseline or check run as stored on disk.
type Run struct {
	Kind    string                        `json:"kind"`
	Number  int                           `json:"number"`
	Time    string                        `json:"time"`
	Model   string                        `json:"model,omitempty"`
	Answers map[string]string             `json:"answers"`
	Prompts map[string]string             `json:"prompts"`
	Tokens  map[string]map[string]float64 `json:"tokens"`
}

// Example is one question whose answer moved.
type Example struct {
	ID       string `json:"id"`
	Prompt   string `json:"prompt"`
	Baseline string `json:"baseline"`
	Now      string `json:"now"`
}

// Fingerprint summarises first-token divergence; supporting evidence only.
type Fingerprint struct {
	Probes         int     `json:"probes"`
	MedianJSDCheck float64 `json:"median_jsd_check"`
	MedianJSDNoise float64 `json:"median_jsd_noise"`
}

// Result is one check's outcome. Field names match the Python package's JSON report.
type Result struct {
	CheckNumber    int          `json:"check_number"`
	TestNumber     *int         `json:"test_number"`
	Items          int          `json:"items"`
	Changed        int          `json:"changed"`
	ChangedRate    float64      `json:"changed_rate"`
	ChangedCI      [2]float64   `json:"changed_ci"`
	AlphaTotal     float64      `json:"alpha_total"`
	AlphaThisCheck float64      `json:"alpha_this_check"`
	AlphaSpent     float64      `json:"alpha_spent"`
	BaselineRuns   int          `json:"baseline_runs"`
	Examples       []Example    `json:"examples"`
	Verdict        string       `json:"verdict"`
	Reason         string       `json:"reason,omitempty"`
	NullChanged    int          `json:"null_changed"`
	NullItems      int          `json:"null_items"`
	NullRate       float64      `json:"null_rate"`
	NullCI         [2]float64   `json:"null_ci"`
	NullRun        int          `json:"null_run"`
	OnlyCheck      int          `json:"only_check"`
	OnlyNoise      int          `json:"only_noise"`
	PValue         *float64     `json:"p_value"`
	Fingerprint    *Fingerprint `json:"fingerprint"`
}

func changedIDs(a, b Run, ids []string) []string {
	var out []string
	for _, i := range ids {
		if Norm(a.Answers[i]) != Norm(b.Answers[i]) {
			out = append(out, i)
		}
	}
	return out
}

func commonIDs(runs []Run) []string {
	var ids []string
	for id := range runs[0].Answers {
		ok := true
		for _, r := range runs[1:] {
			if _, has := r.Answers[id]; !has {
				ok = false
				break
			}
		}
		if ok {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

// Evaluate compares a check run with the baseline runs. testedBefore counts earlier checks that ran
// a test; it sets this check's budget share. A check without a verdict spends no budget.
func Evaluate(baselines []Run, check Run, number int, alpha float64, testedBefore int) Result {
	ids := commonIDs(append(append([]Run{}, baselines...), check))
	n := len(ids)
	moved := changedIDs(baselines[0], check, ids)
	lo, hi := ClopperPearson(len(moved), n)
	r := Result{CheckNumber: number, Items: n, Changed: len(moved), ChangedRate: float64(len(moved)) / float64(n),
		ChangedCI: [2]float64{lo, hi}, AlphaTotal: alpha, BaselineRuns: len(baselines), Examples: []Example{}}
	for _, i := range moved[:min(8, len(moved))] {
		r.Examples = append(r.Examples, Example{i, baselines[0].Prompts[i], baselines[0].Answers[i], check.Answers[i]})
	}
	if len(baselines) < 2 {
		r.Verdict = "no verdict"
		r.Reason = "one baseline run cannot measure the provider's own noise; run `tracta baseline` again, ideally on another day"
		r.AlphaSpent = BudgetSpent(testedBefore, alpha)
		return r
	}
	// Paired exact test. Anchor on baseline 1; compare the check run and the latest other baseline
	// run with it, question by question. If the provider has not changed, those two runs are
	// exchangeable, so on each question where exactly one of them disagrees with the anchor, either
	// is equally likely to be the one. The count of "check disagrees, baseline agrees" is then
	// Binomial(discordant, 1/2), whatever each question's own noise rate is.
	other := baselines[len(baselines)-1]
	checkMoved := map[string]bool{}
	for _, i := range moved {
		checkMoved[i] = true
	}
	noiseMoved := changedIDs(baselines[0], other, ids)
	noiseSet := map[string]bool{}
	for _, i := range noiseMoved {
		noiseSet[i] = true
		if !checkMoved[i] {
			r.OnlyNoise++
		}
	}
	for _, i := range moved {
		if !noiseSet[i] {
			r.OnlyCheck++
		}
	}
	p := SignTestGreater(r.OnlyCheck, r.OnlyCheck+r.OnlyNoise)
	k := testedBefore + 1
	r.TestNumber = &k
	r.AlphaThisCheck = CheckAlpha(k, alpha)
	r.AlphaSpent = BudgetSpent(k, alpha)
	r.NullChanged, r.NullItems, r.NullRate = len(noiseMoved), n, float64(len(noiseMoved))/float64(n)
	nlo, nhi := ClopperPearson(len(noiseMoved), n)
	r.NullCI = [2]float64{nlo, nhi}
	r.NullRun = other.Number
	if r.NullRun == 0 {
		r.NullRun = len(baselines)
	}
	r.PValue = &p
	if p < r.AlphaThisCheck {
		r.Verdict = "changed"
	} else {
		r.Verdict = "no change detected"
	}
	r.Fingerprint = fingerprint(baselines, check)
	return r
}

func median(v []float64) float64 {
	sort.Float64s(v)
	m := len(v) / 2
	if len(v)%2 == 1 {
		return v[m]
	}
	return (v[m-1] + v[m]) / 2
}

func fingerprint(baselines []Run, check Run) *Fingerprint {
	b1, b2, now := baselines[0].Tokens, baselines[1].Tokens, check.Tokens
	if len(b1) == 0 || len(b2) == 0 || len(now) == 0 {
		return nil
	}
	var probes []string
	for p := range b1 {
		if _, ok := b2[p]; ok {
			if _, ok := now[p]; ok {
				probes = append(probes, p)
			}
		}
	}
	if len(probes) == 0 {
		return nil
	}
	var c, z []float64
	for _, p := range probes {
		c = append(c, JSD(now[p], b1[p]))
		z = append(z, JSD(b2[p], b1[p]))
	}
	return &Fingerprint{len(probes), median(c), median(z)}
}
