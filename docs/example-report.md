<!-- Example output: tests/e2e_local.sh on 2026-10-30, a local llama-server switched from Gemma 3 1B Q8_0 to its published Q4_K_M. -->
# tracta check 3: e2e

**Result: changed.**

| Measure | Value |
|---|---|
| Endpoint | `http://127.0.0.1:18791/v1` |
| Model | `base` |
| Questions compared | 436 |
| Answers that changed against baseline 1 | 243 (55.7%, 95% CI 50.9%–60.5%) |
| Provider's own noise (baseline 2 against baseline 1) | 0 of 436 (0.0%, 95% CI 0.0%–0.8%) |
| Questions where only today's answer moved / only baseline 2's moved | 243 / 0 |
| One-sided exact sign test p | 7.07e-74 |
| False-alarm budget for this check | 0.0076 |
| False-alarm budget spent so far | 0.0380 of 0.05 |

Fingerprint (supporting evidence, does not decide): median first-token divergence 0.0137 bits now, against 0.0000 bits between baseline runs, over 40 probes.

Answers that changed (first 8):

| Question | Baseline 1 | Now |
|---|---|---|
| Which chemical element has atomic number 6? Answer with the  | Oxygen | Helium |
| Which chemical element has atomic number 9? Answer with the  | Helium | Hydrogen |
| Which chemical element has atomic number 10? Answer with the | Carbon | Hydrogen |
| Which chemical element has atomic number 13? Answer with the | Carbon | Hydrogen |
| Which chemical element has atomic number 14? Answer with the | Carbon | Oganesson |
| Which chemical element has atomic number 15? Answer with the | Chromium | Oganesson |
| Which chemical element has atomic number 18? Answer with the | Oganesson | Hydrogen |
| Which chemical element has atomic number 21? Answer with the | Neon | Sodium |

## How to read this

In tracta's experiments on ten published models from 0.35B to 4B parameters, serving the model's published Q4_K_M instead of its Q8_0 changed 4.6–75% of these answers, and serving a different model changed 45–97%. Serving the same file again changed 0–0.5%, and serving it through a second inference engine changed 10–15%. A check therefore has less power when your provider's own noise is high, and a large model loses fewer of these answers to 4-bit quantisation than a small one.

- A change means the answers moved more than the provider's own noise. It does not show intent: an announced model update, a new default system prompt or new sampling settings also count.
- A provider that recognises these requests can serve the real model to them. Your question list comes from a private seed, which makes that harder, not impossible.
- A change that leaves short factual answers alone is invisible to this check.
