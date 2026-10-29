# Drift and fingerprint calibration (B1, B2, F1)

First run of the drift checks in `plan/research/2026-10-27-detection-hypotheses.md` (§6–7), on local ground truth. A llama.cpp server on CPU plays the provider. It serves Qwen3-0.6B at Q8_0 (the baseline model) and then at Q4_K_M (the planted downgrade), and each statistic is scored on whether it separates the two better than the baseline separates from itself.

## Files

- `probes.py`: the three fixed probe sets: 200 open prompts, 436 boundary items (atomic numbers both ways, 3-digit by 2-digit products), 40 single-token probes.
- `collect.py`: starts `llama-server` on CPU (`--device none`, no op offload) for one arm and appends every answer to one JSONL file.
- `analyse.py`: scores the campaign and writes `results.json`.
  - B1: randomised rank of each served answer's log-probability among 20 reference samples per prompt, Cramér-von Mises test against uniform, rejection rate by prompt count and by dilution share. The scorer is the published bf16 weights through transformers on CPU.
  - B2: greedy answers; share of answers that change against the first unchanged day, overall and on items that day got wrong.
  - F1: Jensen-Shannon divergence between first-token distributions (top 20 log-probabilities plus a remainder bucket), and the number of probes needed to separate models.
- `campaign.sh`: one full campaign of 9 arms.

## Run

```bash
sh experiments/drift-calibration/campaign.sh /Volumes/Models2/Models/tracta /path/to/work-dir
```

The model folder holds `qwen3-0.6b-{Q8_0,Q4_K_M,bf16}.gguf` and `qwen3-0.6b-hf/` (the Hugging Face weights for the scorer). Sampling is fixed in `collect.py`: temperature 0.7 with top-k, top-p and min-p off for open prompts, greedy for boundary items. llama-server's own defaults (top-k 40, top-p 0.95, min-p 0.05) would otherwise apply, which is the configuration confound in charter limit 4.

## Arms

| Tag | Model | Serving | Role |
|---|---|---|---|
| `reference` | Q8_0 | 8 threads, 1 slot | 20 samples per open prompt, the B1 reference |
| `q8-day1..3` | Q8_0 | 8 threads, 1 slot | unchanged model, three seed sets |
| `q8-batched` | Q8_0 | 4 threads, 4 slots, 4 concurrent requests | unchanged model, benign serving change |
| `q4-day1..3` | Q4_K_M | 8 threads, 1 slot | planted downgrade |
| `bf16` | bf16 | 8 threads, 1 slot | same model, higher precision |

Results go to `plan/research/2026-10-29-drift-calibration/`.
