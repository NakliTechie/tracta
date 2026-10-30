package core

import (
	"fmt"
	"os"
	"strings"
)

const evidence = "In tracta's experiments on ten published models from 0.35B to 4B parameters, serving the model's " +
	"published Q4_K_M instead of its Q8_0 changed 4.6–75% of these answers, and serving a different model " +
	"changed 45–97%. Serving the same file again changed 0–0.5%, and serving it through a second inference " +
	"engine changed 10–15% (measured for four of the ten). A check therefore has less power when your provider's " +
	"own noise is high, and a large model loses fewer of these answers to 4-bit quantisation than a small one."

var limits = []string{
	"A change means the answers moved more than the provider's own noise. It does not show intent: an " +
		"announced model update, a new default system prompt or new sampling settings also count.",
	"A provider that recognises these requests can serve the real model to them. Your question list comes " +
		"from a private seed, which makes that harder, not impossible.",
	"A change that leaves short factual answers alone is invisible to this check.",
}

func pct(v float64) string { return fmt.Sprintf("%.1f%%", 100*v) }

func cell(s string) string {
	s = strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(s), "|", `\|`), "\n", " ")
	if r := []rune(s); len(r) > 60 {
		s = string(r[:60])
	}
	return s
}

// Markdown renders a check's report for people.
func Markdown(c Config, r Result, checkModel string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# tracta check %d: %s\n\n**Result: %s.**\n\n", r.CheckNumber, c.Name, r.Verdict)
	if r.Verdict == "no verdict" {
		fmt.Fprintf(&b, "%s.\n\n", r.Reason)
	}
	model := "`" + c.Model + "`"
	if checkModel != "" && checkModel != c.Model {
		model = fmt.Sprintf("`%s` (this check called `%s`)", c.Model, checkModel)
	}
	fmt.Fprintf(&b, "| Measure | Value |\n|---|---|\n| Endpoint | `%s` |\n| Model | %s |\n| Questions compared | %d |\n", c.Endpoint, model, r.Items)
	fmt.Fprintf(&b, "| Answers that changed against baseline 1 | %d (%s, 95%% CI %s–%s) |\n", r.Changed, pct(r.ChangedRate), pct(r.ChangedCI[0]), pct(r.ChangedCI[1]))
	if r.PValue != nil {
		fmt.Fprintf(&b, "| Provider's own noise (baseline %d against baseline 1) | %d of %d (%s, 95%% CI %s–%s) |\n",
			r.NullRun, r.NullChanged, r.NullItems, pct(r.NullRate), pct(r.NullCI[0]), pct(r.NullCI[1]))
		fmt.Fprintf(&b, "| Questions where only today's answer moved / only baseline %d's moved | %d / %d |\n", r.NullRun, r.OnlyCheck, r.OnlyNoise)
		fmt.Fprintf(&b, "| One-sided exact sign test p | %.3g |\n| False-alarm budget for this check | %.4f |\n", *r.PValue, r.AlphaThisCheck)
	}
	fmt.Fprintf(&b, "| False-alarm budget spent so far | %.4f of %g |\n\n", r.AlphaSpent, r.AlphaTotal)
	if f := r.Fingerprint; f != nil {
		fmt.Fprintf(&b, "Fingerprint (supporting evidence, does not decide): median first-token divergence %.4f bits now, against %.4f bits between baseline runs, over %d probes.\n\n",
			f.MedianJSDCheck, f.MedianJSDNoise, f.Probes)
	}
	if len(r.Examples) > 0 {
		b.WriteString("Answers that changed (first 8):\n\n| Question | Baseline 1 | Now |\n|---|---|---|\n")
		for _, e := range r.Examples {
			fmt.Fprintf(&b, "| %s | %s | %s |\n", cell(e.Prompt), cell(e.Baseline), cell(e.Now))
		}
		b.WriteString("\n")
	}
	b.WriteString("## How to read this\n\n" + evidence + "\n\n")
	for _, l := range limits {
		b.WriteString("- " + l + "\n")
	}
	return b.String()
}

// WriteReport writes the Markdown and JSON reports and returns the Markdown path.
func WriteReport(s Store, c Config, r Result, checkModel string) (string, error) {
	md := s.ReportPath(r.CheckNumber, "md")
	if err := os.WriteFile(md, []byte(Markdown(c, r, checkModel)), 0o644); err != nil {
		return "", err
	}
	return md, writeJSON(s.ReportPath(r.CheckNumber, "json"), r)
}
