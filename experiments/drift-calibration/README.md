# Drift and fingerprint checks (B1, B2, F1)

Measures whether three statistics can tell a served model from a planted change (its published Q4_K_M, or a swap to another model) better than the model differs from itself. Hypotheses: `plan/research/2026-10-27-detection-hypotheses.md` §6–7. Latest results: `results/2026-10-30/` (ten published families) and `plan/research/2026-10-30-drift-families/results.md`.

## Files

- `probes.py`: the fixed probe sets: 200 open prompts, 436 boundary items (atomic numbers both ways, three-digit by two-digit products), 40 single-token probes.
- `models.py`, `models.json`, `model-repos.json`: every model file pinned by repo, commit and SHA-256; `fetch` downloads and checks them, `verify` checks a local copy.
- `collect.py`: starts `llama-server` for one arm (CPU by default, `TRACTA_DEVICE=MTL0` for Metal, `TRACTA_PORT` for the port), refuses a port served by another model, waits at critical memory pressure, and appends every answer with its timings to one JSONL file.
- `collect_hf.py`: the second engine. The same GGUF run by transformers on CPU in float32, with prompt token IDs from llama-server and llama.cpp's end-of-generation tokens.
- `analyse.py`: B1 over fresh disjoint audits, B2 with exact intervals, F1 equal error rate per change. Fixed seeds per statistic; scores cached by scorer and content.
- `campaign_family.sh`: one family (baseline bank, unchanged audits, downgrade, swap, boundary and token probes). `families-small.sh` and `families-4b.sh` list the ten families tested.
- `report.py`, `plot_families.py`: the per-family table and the README chart.
- `campaign.sh`, `campaign_engine2.sh`, `campaign_audits.sh`, `plot.py`: the first Qwen3-0.6B calibration (results under `results/2026-10-29/`, superseded).

## Run one family

```bash
python models.py fetch /path/to/models
sh campaign_family.sh /path/to/models work/gemma-3-1b gemma-3-1b/gemma-3-1b-it-Q8_0.gguf gemma-3-1b/gemma-3-1b-it-Q4_K_M.gguf gemma-3-270m/gemma-3-270m-it-Q8_0.gguf
python collect_hf.py --model-dir /path/to/models/gemma-3-1b --gguf gemma-3-1b-it-Q8_0.gguf --tag q8-hfday1 --sets boundary,tokens --out work/gemma-3-1b/runs-engine2.jsonl
python analyse.py --runs work/gemma-3-1b/runs.jsonl work/gemma-3-1b/runs-engine2.jsonl --scorer /path/to/Qwen3-0.6B --out work/gemma-3-1b/results.json
```

The scorer is the Hugging Face `Qwen/Qwen3-0.6B` folder. Sampling is fixed in `collect.py`: temperature 0.7 with top-k, top-p and min-p off for open prompts, greedy for boundary items, top-20 log-probabilities for token probes. llama-server's own defaults (top-k 40, top-p 0.95, min-p 0.05) would otherwise apply.

## Arms of one family

| Tag | Model | Role |
|---|---|---|
| `reference` | base Q8_0 | 10 baseline answers per open prompt, the B1 reference bank |
| `q8-audit` | base Q8_0 | 10 fresh answers per prompt: 80 unchanged audits of 25 prompts |
| `q4-audit`, `swap-audit` | Q4_K_M, swap | 5 fresh answers per prompt: 40 audits each |
| `q8-day1`, `q8-day2` | base Q8_0 | boundary and token probes; day 2 with other threads and batching |
| `q8-hfday1`, `q4-hfday1` | Q8_0, Q4_K_M on transformers | the second engine, where transformers can load the architecture |
| `q4-day1`, `swap-day1` | Q4_K_M, swap | boundary and token probes |
