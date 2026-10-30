package core

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Config describes one watched provider.
type Config struct {
	Name      string         `json:"name"`
	API       string         `json:"api"` // "openai" or "anthropic"
	Endpoint  string         `json:"endpoint"`
	Model     string         `json:"model"`
	KeyEnv    string         `json:"key_env,omitempty"`
	ExtraBody map[string]any `json:"extra_body"`
	Seed      uint64         `json:"seed"`
	Generator string         `json:"generator"`
	Created   string         `json:"created"`
}

// Store is one folder per watched provider:
//
//	<root>/<name>/config.json, runs/baseline-N.json, runs/check-N.json, reports/check-N.{md,json}
type Store struct {
	Dir, Name string
}

func NewStore(root, name string) Store {
	if strings.HasPrefix(root, "~/") {
		home, _ := os.UserHomeDir()
		root = filepath.Join(home, root[2:])
	}
	return Store{filepath.Join(root, name), name}
}

func (s Store) configPath() string { return filepath.Join(s.Dir, "config.json") }

func (s Store) Exists() bool {
	_, err := os.Stat(s.configPath())
	return err == nil
}

func now() string { return time.Now().Format("2006-01-02T15:04:05-0700") }

func (s Store) Create(api, endpoint, model, keyEnv string, extra map[string]any) (Config, error) {
	for _, d := range []string{"runs", "reports"} {
		if err := os.MkdirAll(filepath.Join(s.Dir, d), 0o755); err != nil {
			return Config{}, err
		}
	}
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return Config{}, err
	}
	if extra == nil {
		extra = map[string]any{}
	}
	c := Config{Name: s.Name, API: api, Endpoint: endpoint, Model: model, KeyEnv: keyEnv, ExtraBody: extra,
		Seed: binary.LittleEndian.Uint64(b[:]), Generator: Generator, Created: now()}
	return c, writeJSON(s.configPath(), c)
}

func (s Store) Config() (Config, error) {
	var c Config
	data, err := os.ReadFile(s.configPath())
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(data, &c); err != nil {
		return c, err
	}
	if c.Generator != Generator {
		return c, fmt.Errorf("watch %q was made by another question generator (%q); start a new watch", s.Name, c.Generator)
	}
	return c, nil
}

func (s Store) Runs(kind string) ([]Run, error) {
	files, _ := filepath.Glob(filepath.Join(s.Dir, "runs", kind+"-*.json"))
	num := func(p string) int {
		n, _ := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(filepath.Base(p), kind+"-"), ".json"))
		return n
	}
	sort.Slice(files, func(i, j int) bool { return num(files[i]) < num(files[j]) })
	var runs []Run
	for _, f := range files {
		var r Run
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &r); err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		runs = append(runs, r)
	}
	return runs, nil
}

func (s Store) SaveRun(kind string, r Run) (Run, error) {
	existing, err := s.Runs(kind)
	if err != nil {
		return r, err
	}
	r.Kind, r.Number = kind, len(existing)+1
	return r, writeJSON(filepath.Join(s.Dir, "runs", fmt.Sprintf("%s-%d.json", kind, r.Number)), r)
}

// TestedChecks counts earlier checks whose report ran a test.
func (s Store) TestedChecks() int {
	files, _ := filepath.Glob(filepath.Join(s.Dir, "reports", "check-*.json"))
	n := 0
	for _, f := range files {
		var r struct {
			PValue *float64 `json:"p_value"`
		}
		if data, err := os.ReadFile(f); err == nil && json.Unmarshal(data, &r) == nil && r.PValue != nil {
			n++
		}
	}
	return n
}

func (s Store) ReportPath(number int, suffix string) string {
	return filepath.Join(s.Dir, "reports", fmt.Sprintf("check-%d.%s", number, suffix))
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// WriteJSON is exported for the report writer.
func WriteJSON(path string, v any) error { return writeJSON(path, v) }

// Now is the timestamp format used in runs.
func Now() string { return now() }
