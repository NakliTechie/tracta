// tracta checks whether a model provider still serves the model it promised, from its answers alone.
//
//	tracta baseline --name N --endpoint URL --model M [--api openai|anthropic] [--key-env VAR]
//	tracta baseline --name N                  # later baseline runs, ideally on other days
//	tracta check --name N                     # writes a report; exit 0 no change, 2 changed, 3 no verdict
//	tracta status --name N
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/NakliTechie/tracta/internal/core"
	"github.com/NakliTechie/tracta/internal/provider"
)

var version = "0.2.0"

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "tracta: "+format+"\n", a...)
	os.Exit(1)
}

func usage() {
	fmt.Fprint(os.Stderr, `tracta: check whether a model provider still serves the model it promised.

  tracta baseline --name N --endpoint URL --model M [--api openai|anthropic] [--key-env VAR] [--extra-body JSON]
  tracta baseline --name N          record another baseline, ideally on another day
  tracta check --name N             compare with the baselines and write a report
  tracta status --name N            show a watch
  tracta version

Every command takes --root DIR (default ~/.tracta). Run "tracta <command> -h" for all flags.
`)
	os.Exit(2)
}

func collect(o provider.Options, items []core.Item, logprobs bool) (core.Run, error) {
	client, err := provider.New(o)
	if err != nil {
		return core.Run{}, err
	}
	ctx := context.Background()
	run := core.Run{Time: core.Now(), Model: o.Model, Answers: map[string]string{}, Prompts: map[string]string{}}
	for k, it := range items {
		text, _, err := client.Chat(ctx, it.Prompt, 16, 0)
		if err != nil {
			return run, err
		}
		run.Answers[it.ID], run.Prompts[it.ID] = text, it.Prompt
		if (k+1)%50 == 0 {
			fmt.Fprintf(os.Stderr, "  %d/%d questions\n", k+1, len(items))
		}
	}
	if logprobs {
		run.Tokens = map[string]map[string]float64{}
		for i, p := range core.TokenProbes {
			_, top, err := client.Chat(ctx, p, 1, 20)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  fingerprint probes skipped: %v\n", err)
				run.Tokens = nil
				break
			}
			if top != nil {
				run.Tokens[fmt.Sprintf("tok-%02d", i)] = top
			}
		}
		if len(run.Tokens) == 0 {
			run.Tokens = nil
		}
	}
	return run, nil
}

func parseExtra(s string) map[string]any {
	if s == "" {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		fail("--extra-body is not a JSON object: %v", err)
	}
	return m
}

func options(c core.Config, rps float64) provider.Options {
	return provider.Options{API: c.API, Endpoint: c.Endpoint, Model: c.Model, KeyEnv: c.KeyEnv, ExtraBody: c.ExtraBody, RPS: rps}
}

func baseline(args []string) {
	fs := flag.NewFlagSet("baseline", flag.ExitOnError)
	root := fs.String("root", "~/.tracta", "folder for watches")
	name := fs.String("name", "", "watch name (required)")
	endpoint := fs.String("endpoint", "", "base URL, e.g. https://api.example.com/v1 (first run)")
	model := fs.String("model", "", "model name the provider expects (first run)")
	api := fs.String("api", "", "openai (any OpenAI-compatible endpoint) or anthropic (default: anthropic for api.anthropic.com, else openai)")
	keyEnv := fs.String("key-env", "", "environment variable holding the API key (first run)")
	extra := fs.String("extra-body", "", "JSON object merged into every request (first run)")
	items := fs.Int("items", 0, "use only the first N questions (default all 436)")
	logprobs := fs.Bool("logprobs", false, "also record the fingerprint probes (needs top_logprobs)")
	rps := fs.Float64("rps", 2, "requests per second (0 for no limit)")
	fs.Parse(args)
	if *name == "" {
		fail("--name is required")
	}
	s := core.NewStore(*root, *name)
	var c core.Config
	var err error
	if !s.Exists() {
		if *endpoint == "" || *model == "" {
			fail("no watch named %q yet: give --endpoint and --model to create it", *name)
		}
		kind := *api
		if kind == "" {
			kind = "openai"
			if strings.Contains(*endpoint, "anthropic.com") {
				kind = "anthropic"
			}
		}
		if c, err = s.Create(kind, *endpoint, *model, *keyEnv, parseExtra(*extra)); err != nil {
			fail("%v", err)
		}
		fmt.Printf("created watch %q in %s\n", *name, s.Dir)
	} else if c, err = s.Config(); err != nil {
		fail("%v", err)
	}
	qs := core.BoundaryItems(c.Seed)
	if *items > 0 && *items < len(qs) {
		qs = qs[:*items]
	}
	prev, _ := s.Runs("baseline")
	fmt.Fprintf(os.Stderr, "baseline %d for %s at %s: %d questions\n", len(prev)+1, c.Model, c.Endpoint, len(qs))
	run, err := collect(options(c, *rps), qs, *logprobs)
	if err != nil {
		fail("%v", err)
	}
	if run, err = s.SaveRun("baseline", run); err != nil {
		fail("%v", err)
	}
	fmt.Printf("saved baseline %d\n", run.Number)
	if run.Number == 1 {
		fmt.Println("record a second baseline (ideally on another day) before checking, so tracta can measure the provider's own noise")
	}
}

func check(args []string) {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	root := fs.String("root", "~/.tracta", "folder for watches")
	name := fs.String("name", "", "watch name (required)")
	alpha := fs.Float64("alpha", 0.05, "total false-alarm budget over all checks")
	rps := fs.Float64("rps", 2, "requests per second (0 for no limit)")
	model := fs.String("model", "", "call this model instead of the watch's (to test the check with a deliberate swap)")
	extra := fs.String("extra-body", "", "JSON object merged over the watch's extra body for this check")
	fs.Parse(args)
	if *name == "" {
		fail("--name is required")
	}
	s := core.NewStore(*root, *name)
	if !s.Exists() {
		fail("no watch named %q: run `tracta baseline` first", *name)
	}
	c, err := s.Config()
	if err != nil {
		fail("%v", err)
	}
	baselines, err := s.Runs("baseline")
	if err != nil || len(baselines) == 0 {
		fail("no baseline runs yet: run `tracta baseline` first")
	}
	var qs []core.Item
	for _, it := range core.BoundaryItems(c.Seed) {
		if _, ok := baselines[0].Answers[it.ID]; ok {
			qs = append(qs, it)
		}
	}
	o := options(c, *rps)
	if *model != "" {
		o.Model = *model
	}
	if e := parseExtra(*extra); e != nil {
		merged := map[string]any{}
		for k, v := range c.ExtraBody {
			merged[k] = v
		}
		for k, v := range e {
			merged[k] = v
		}
		o.ExtraBody = merged
	}
	prev, _ := s.Runs("check")
	fmt.Fprintf(os.Stderr, "check %d: %d questions\n", len(prev)+1, len(qs))
	run, err := collect(o, qs, len(baselines[0].Tokens) > 0)
	if err != nil {
		fail("%v", err)
	}
	if run, err = s.SaveRun("check", run); err != nil {
		fail("%v", err)
	}
	r := core.Evaluate(baselines, run, run.Number, *alpha, s.TestedChecks())
	path, err := core.WriteReport(s, c, r, o.Model)
	if err != nil {
		fail("%v", err)
	}
	if r.PValue != nil {
		fmt.Printf("%s: %d of %d answers changed, provider noise %.1f%%, p = %.3g\n", r.Verdict, r.Changed, r.Items, 100*r.NullRate, *r.PValue)
	} else {
		fmt.Printf("%s: %d of %d answers changed\n", r.Verdict, r.Changed, r.Items)
	}
	fmt.Printf("report: %s\n", path)
	switch r.Verdict {
	case "changed":
		os.Exit(2)
	case "no verdict":
		os.Exit(3)
	}
}

func status(args []string) {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	root := fs.String("root", "~/.tracta", "folder for watches")
	name := fs.String("name", "", "watch name (required)")
	fs.Parse(args)
	s := core.NewStore(*root, *name)
	c, err := s.Config()
	if err != nil {
		fail("%v", err)
	}
	b, _ := s.Runs("baseline")
	k, _ := s.Runs("check")
	out, _ := json.MarshalIndent(map[string]any{"name": c.Name, "api": c.API, "endpoint": c.Endpoint, "model": c.Model,
		"baselines": len(b), "checks": len(k), "folder": s.Dir}, "", " ")
	fmt.Println(string(out))
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "baseline":
		baseline(os.Args[2:])
	case "check":
		check(os.Args[2:])
	case "status":
		status(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Println("tracta", version)
	default:
		usage()
	}
}
