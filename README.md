<h1 align="center">tracta</h1>

<p align="center"><b>Checks you run yourself to see whether a model provider still serves the model it promised, and research on reading which parts of a model carry a behaviour.</b></p>

<p align="center">Runs on your own machine. No account, no server, no telemetry.</p>

<p align="center">
  <img alt="status: research" src="https://img.shields.io/badge/status-research-52514e?style=flat-square">
  <img alt="models tested: 10" src="https://img.shields.io/badge/models%20tested-10-52514e?style=flat-square">
  <img alt="account: none" src="https://img.shields.io/badge/account-none-52514e?style=flat-square">
</p>

## Install

| Need | Command |
|---|---|
| llama.cpp (macOS, Linux) | `brew install llama.cpp` |
| Python 3.12 packages | `pip install numpy scipy torch transformers matplotlib "gguf>=0.19"` |
| The pinned model files | `python experiments/drift-calibration/models.py fetch /path/to/models` |

`models.py fetch` downloads the 24 published GGUF files listed in `models.json` at their pinned commits and checks every SHA-256. Then run one family: the provider serves a published Q8_0, and the script plants its published Q4_K_M and a swap.

```bash
sh experiments/drift-calibration/campaign_family.sh /path/to/models work/gemma-3-1b gemma-3-1b/gemma-3-1b-it-Q8_0.gguf gemma-3-1b/gemma-3-1b-it-Q4_K_M.gguf gemma-3-270m/gemma-3-270m-it-Q8_0.gguf
```

It needs no GPU and no API key, because the "provider" is a local server. `TRACTA_DEVICE=MTL0` serves on Metal instead of CPU.

## Why

You pay a provider for model X and see only its answers. On a busy day it can serve a 4-bit copy, a smaller sibling, or another model, and the bill stays the same. A weight hash cannot help, because you never see the weights.

tracta asks the provider fixed questions on days you trust, keeps the answers as your baseline, and later checks whether new answers still match. It reports a detection with a measured false-alarm rate, never proof. A provider that recognises probe traffic can serve the real model to the probes, so a released tool must make probes look like ordinary use.

## Results

![Per family, the share of 436 greedy answers that changed: unchanged reruns 0–0.5%, a second engine 10–15%, the published Q4_K_M 4.6–75%, a swapped model 45–97%](experiments/drift-calibration/results/2026-10-30/b2-families.png)

Across ten published models from 0.35B to 4B, a swap to the family's Q4_K_M changed 4.6–75% of 436 short greedy answers, and a swap to another model changed 45–97%. The same file served again changed 0–0.5%, and served by a second engine 10–15%. One case is unresolved: Qwen3.5-4B's Q4_K_M moved only 4.6%, and its second-engine noise was not measured. Reproduce with `python experiments/drift-calibration/plot_families.py experiments/drift-calibration/results/2026-10-30 out.png`. Raw answers: [`runs-families.jsonl.gz`](experiments/drift-calibration/results/2026-10-30/runs-families.jsonl.gz).

## The three checks

- **Knowledge-boundary answers (B2).** Greedy answers to short facts and sums near the edge of what the model knows. It is the strongest check here: it separated every planted change from its null.
- **Rank uniformity (B1).** Each new answer is ranked among your baseline answers to the same prompt, under one fixed small scorer model. Its false-alarm rate held at 0–8.8% across ten families. With a scorer from another family it catches cross-family swaps (85–100% at 25 prompts) but not quantisation or smaller siblings. With the served model as scorer it caught Q4_K_M 98.8–100% of the time.
- **Single-token fingerprint (F1).** First-token distributions on 40 one-word prompts. It needs an API that returns log-probabilities, and a second engine in its null to mean anything.

Per-family tables with exact intervals: [`report-small.md`](experiments/drift-calibration/results/2026-10-30/report-small.md) and [`report-4b.md`](experiments/drift-calibration/results/2026-10-30/report-4b.md).

## Also in this repo

The wider question is which component of a model (a neuron, a direction, an expert) carries a behaviour. Two tracks are on paper and not yet run:
- tamper checks for runtime steering and abliteration;
- expert-path attribution on existing open mixture-of-experts models, labelled by topic.

## Commands

```bash
python experiments/drift-calibration/models.py fetch DIR                   # pinned published GGUFs, SHA-256 checked
sh experiments/drift-calibration/campaign_family.sh DIR WORK BASE DOWN SWAP   # one family on llama-server
python experiments/drift-calibration/collect_hf.py --model-dir D --gguf F --tag q8-hfday1 --sets boundary,tokens --out R   # second engine
python experiments/drift-calibration/analyse.py --runs R... --scorer HF_DIR --out results.json   # B1, B2, F1
python experiments/drift-calibration/report.py OUT.md WORK/*/results.json     # per-family table
```

## Verify it yourself

```bash
python experiments/drift-calibration/models.py verify DIR
python -c "import json; r=json.load(open('experiments/drift-calibration/results/2026-10-30/families/gemma-3-1b/results.json')); print({t: round(v['changed_all']['rate'], 3) for t, v in r['b2']['arms'].items()})"
```

`verify` fails unless every model file matches the hash in `models.json`, which is the published Hugging Face hash. The second command prints Gemma 3 1B's B2 shares: 0.0 for the same-engine reruns, 0.154 for the second engine, 0.555 for Q4_K_M and 0.968 for the swap. Analysis uses fixed seeds, so a rerun of `analyse.py` on the same answers gives the same numbers. Every result here is one campaign per family; no repeat campaigns ran.

## License

No license file yet. Code under `experiments/`, data under `experiments/*/results/`. Plans and the research charter live in a private plans store, not in this repo.
