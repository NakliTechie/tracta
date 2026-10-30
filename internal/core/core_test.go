package core

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func load(t *testing.T, name string) Run {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var r Run
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func near(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if want == 0 && got == 0 {
		return
	}
	if math.Abs(got-want) > tol*math.Max(1, math.Abs(want)) && math.Abs(got-want)/math.Abs(want) > tol {
		t.Errorf("%s: got %.17g, want %.17g", label, got, want)
	}
}

// TestGoldenAgainstPython replays stored runs from the end-to-end test (Gemma 3 1B on llama-server:
// two Q8_0 baselines; checks on one baseline, the unchanged Q8_0, the published Q4_K_M and a swap)
// and requires the Python package's results, written to testdata/expected.json by tracta/compare.py.
func TestGoldenAgainstPython(t *testing.T) {
	data, err := os.ReadFile("testdata/expected.json")
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]struct {
		Verdict        string     `json:"verdict"`
		Changed        int        `json:"changed"`
		Items          int        `json:"items"`
		ChangedCI      [2]float64 `json:"changed_ci"`
		AlphaThisCheck float64    `json:"alpha_this_check"`
		AlphaSpent     float64    `json:"alpha_spent"`
		OnlyCheck      int        `json:"only_check"`
		OnlyNoise      int        `json:"only_noise"`
		PValue         *float64   `json:"p_value"`
		Fingerprint    *struct {
			MedianJSDCheck float64 `json:"median_jsd_check"`
			MedianJSDNoise float64 `json:"median_jsd_noise"`
		} `json:"fingerprint"`
	}
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	b := []Run{load(t, "baseline-1"), load(t, "baseline-2")}
	got := map[string]Result{
		"one-baseline": Evaluate(b[:1], load(t, "check-1"), 1, 0.05, 0),
		"check-2":      Evaluate(b, load(t, "check-2"), 2, 0.05, 0),
		"check-3":      Evaluate(b, load(t, "check-3"), 3, 0.05, 1),
		"check-4":      Evaluate(b, load(t, "check-4"), 4, 0.05, 2),
	}
	for name, w := range want {
		g := got[name]
		if g.Verdict != w.Verdict || g.Changed != w.Changed || g.Items != w.Items || g.OnlyCheck != w.OnlyCheck || g.OnlyNoise != w.OnlyNoise {
			t.Errorf("%s: got %s/%d/%d/%d/%d, want %s/%d/%d/%d/%d", name, g.Verdict, g.Changed, g.Items, g.OnlyCheck, g.OnlyNoise,
				w.Verdict, w.Changed, w.Items, w.OnlyCheck, w.OnlyNoise)
		}
		near(t, name+" changed CI low", g.ChangedCI[0], w.ChangedCI[0], 1e-6)
		near(t, name+" changed CI high", g.ChangedCI[1], w.ChangedCI[1], 1e-6)
		near(t, name+" alpha this check", g.AlphaThisCheck, w.AlphaThisCheck, 1e-12)
		near(t, name+" alpha spent", g.AlphaSpent, w.AlphaSpent, 1e-12)
		if (g.PValue == nil) != (w.PValue == nil) {
			t.Errorf("%s: p-value presence differs", name)
		} else if w.PValue != nil {
			near(t, name+" p", *g.PValue, *w.PValue, 1e-9)
		}
		if (g.Fingerprint == nil) != (w.Fingerprint == nil) {
			t.Errorf("%s: fingerprint presence differs", name)
		} else if w.Fingerprint != nil {
			near(t, name+" jsd check", g.Fingerprint.MedianJSDCheck, w.Fingerprint.MedianJSDCheck, 1e-9)
			near(t, name+" jsd noise", g.Fingerprint.MedianJSDNoise, w.Fingerprint.MedianJSDNoise, 1e-9)
		}
	}
}

func TestSignTest(t *testing.T) {
	near(t, "6 of 14", SignTestGreater(6, 14), 0.7880249023437500, 1e-12) // sum_{i>=6} C(14,i)/2^14
	near(t, "20 of 20", SignTestGreater(20, 20), math.Pow(2, -20), 1e-12)
	if SignTestGreater(0, 0) != 1 {
		t.Error("empty comparison must give p = 1")
	}
}

func TestBudgetNeverExceedsAlpha(t *testing.T) {
	if s := BudgetSpent(100000, 0.05); s >= 0.05 {
		t.Errorf("budget after 100000 checks = %g, want < 0.05", s)
	}
}

func TestQuestions(t *testing.T) {
	a, b := BoundaryItems(1), BoundaryItems(2)
	if len(a) != 436 {
		t.Fatalf("got %d items, want 436", len(a))
	}
	ids := map[string]bool{}
	for _, it := range a {
		ids[it.ID] = true
	}
	if len(ids) != 436 {
		t.Error("item ids are not unique")
	}
	same := true
	for i := range a {
		if a[i] != BoundaryItems(1)[i] {
			t.Fatal("same seed gave a different list")
		}
		if a[i].Prompt != b[i].Prompt {
			same = false
		}
	}
	if same {
		t.Error("different seeds gave the same list")
	}
	for in, want := range map[string]string{"The atomic number of Hydrogen is 1.": "1", "**Hydrogen**": "hydrogen", "12,345\n": "12345"} {
		if got := Norm(in); got != want {
			t.Errorf("Norm(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestOwnQuestions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "q.jsonl")
	os.WriteFile(path, []byte("{\"id\":\"a\",\"prompt\":\"What is 2+2?\",\"answer\":\"4\"}\n\n{\"id\":\"b\",\"prompt\":\"Capital of Peru?\"}\n"), 0o644)
	items, err := ReadQuestions(path)
	if err != nil || len(items) != 2 || items[1].ID != "b" {
		t.Fatalf("ReadQuestions = %v, %v", items, err)
	}
	s := NewStore(dir, "w")
	c, err := s.Create("openai", "http://x/v1", "m", "", nil, items)
	if err != nil || c.Generator != QuestionsFile {
		t.Fatalf("Create = %+v, %v", c, err)
	}
	if c, err = s.Config(); err != nil {
		t.Fatal(err)
	}
	got, err := s.Items(c)
	if err != nil || len(got) != 2 || got[0] != items[0] {
		t.Fatalf("Items = %v, %v", got, err)
	}
	for _, bad := range []string{"{\"id\":\"a\",\"prompt\":\"x\"}\n{\"id\":\"a\",\"prompt\":\"y\"}\n", "{\"prompt\":\"x\"}\n", "not json\n", "\n"} {
		os.WriteFile(path, []byte(bad), 0o644)
		if _, err := ReadQuestions(path); err == nil {
			t.Errorf("ReadQuestions accepted %q", bad)
		}
	}
}
