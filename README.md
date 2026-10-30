<h1 align="center">tracta</h1>

<p align="center"><b>Check whether your model provider still serves the model it promised, from its answers alone.</b></p>

<p align="center">One file. Python 3.10+, no other dependencies. No account, no server, no telemetry.</p>

<p align="center">
  <img alt="status: research release" src="https://img.shields.io/badge/status-research%20release-52514e?style=flat-square">
  <img alt="dependencies: none" src="https://img.shields.io/badge/dependencies-none-52514e?style=flat-square">
  <img alt="account: none" src="https://img.shields.io/badge/account-none-52514e?style=flat-square">
</p>

## Install

| Way | Command |
|---|---|
| Single file | `sh scripts/build-pyz.sh`, then run `./dist/tracta.pyz` |
| pipx | `pipx install git+https://github.com/NakliTechie/tracta` |
| From a clone | `python3 -m tracta` |

Watch a provider. Record a baseline on two days you trust, then check on any later day. Each check writes a report to `~/.tracta/<name>/reports/`:

```bash
tracta baseline --name myprovider --endpoint https://api.example.com/v1 --model some-model --key-env EXAMPLE_API_KEY
tracta baseline --name myprovider          # a second baseline, ideally on another day
tracta check --name myprovider             # exit 0 no change, 2 changed, 3 no verdict
```

An example report, from a local server switched to a 4-bit copy, is in [`docs/example-report.md`](docs/example-report.md). Any OpenAI-compatible chat endpoint works. A run is 436 short questions at temperature 0 and up to 16 tokens each. The API key stays in your environment, and nothing but the questions goes to the endpoint you name.

## Why

You pay a provider for model X and see only its answers. On a busy day it can serve a 4-bit copy, a smaller sibling or another model, and the bill stays the same. A weight hash cannot help, because you never see the weights.

tracta asks short factual questions near the edge of what a model knows. A swapped or quantised model gives different answers to many of them. tracta compares today's answers with your baselines and reports a change only when it exceeds the provider's own day-to-day noise, with a stated false-alarm rate.

## Results

![Per family, the share of 436 greedy answers that changed: unchanged reruns 0–0.5%, a second engine 10–15% where measured, the published Q4_K_M 4.6–75%, a swapped model 45–97%](experiments/drift-calibration/results/2026-10-30/b2-families.png)

Across ten published models from 0.35B to 4B, serving the family's published Q4_K_M changed 4.6–75% of these answers, and serving another model changed 45–97%. Serving the same file again changed 0–0.5%. Serving it through a second inference engine changed 10–15%, measured for four of the ten families. One case is unresolved: Qwen3.5-4B's Q4_K_M moved only 4.6%, and its second-engine noise was not measured. These are one campaign per family on local servers, not hosted APIs. Reproduce with `python experiments/drift-calibration/plot_families.py experiments/drift-calibration/results/2026-10-30 out.png`; raw answers in [`runs-families.jsonl.gz`](experiments/drift-calibration/results/2026-10-30/runs-families.jsonl.gz).

## How a check decides

tracta anchors on your first baseline and compares two runs with it, question by question: today's run and your latest other baseline. If the provider has not changed, those two runs are interchangeable. So on questions where exactly one of them moved, each is equally likely to be the one. An exact sign test on those questions gives the p-value, with no assumption that all questions are equally noisy.

Check number k may spend α·6/(π²k²) of a total false-alarm budget α, 0.05 by default. Those shares add up to α over any number of checks, so checking daily does not raise the chance of a false alarm above α. The report shows the budget spent.

## What it cannot tell you

- **Why the answers changed.** An announced model update, a new default system prompt or new sampling settings also count as a change.
- **What a probe-aware provider serves.** A provider that recognises tracta's requests can serve the real model to them. Each user's questions come from a private seed, which makes that harder, not impossible.
- **Changes that spare short factual answers.** They are invisible to this check.
- **Changes before your first baseline.** tracta compares only with your own baselines.

## Research behind it

`experiments/drift-calibration/` holds the experiments: three statistics (B2, used here; a rank test, B1; a token fingerprint, F1), ten model families, a second inference engine, and every model file pinned in `models.json`. Two further tracks are on paper and not yet run: tamper checks for runtime steering and abliteration, and expert-path attribution in mixture-of-experts models.

## Commands

```bash
tracta baseline --name N --endpoint URL --model M [--key-env VAR] [--logprobs] [--rps 2]   # record a baseline
tracta check --name N [--alpha 0.05]                                                       # test and write a report
tracta status --name N                                                                     # show a watch
sh scripts/build-pyz.sh                                                                    # build dist/tracta.pyz
```

`--logprobs` also records 40 one-word probes, so the report can show the token fingerprint as supporting evidence; only the sign test decides. `--extra-body JSON` passes provider-specific fields.

## Verify it yourself

```bash
python3 -m unittest discover -s tests
sh tests/e2e_local.sh /path/to/models /tmp/tracta-e2e
```

The unit tests check the sign test and exact intervals against SciPy, the false-alarm budget, and the verdicts. The end-to-end script drives the real CLI against local llama-server providers built from the pinned files. It expects no verdict from one baseline, no change on the unchanged Q8_0, and "changed" for the published Q4_K_M and for a swap to Gemma 3 270M.

## License

No license file yet. Plans and the research charter live in a private plans store, not in this repo.
