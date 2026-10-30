# tracta check 1: claude

**Result: no change detected.**

| Measure | Value |
|---|---|
| Endpoint | `https://api.anthropic.com` |
| Model | `claude-haiku-5-5` |
| Questions compared | 436 |
| Answers that changed against baseline 1 | 0 (0.0%, 95% CI 0.0%–0.8%) |
| Provider's own noise (baseline 2 against baseline 1) | 0 of 436 (0.0%, 95% CI 0.0%–0.8%) |
| Questions where only today's answer moved / only baseline 2's moved | 0 / 0 |
| One-sided exact sign test p | 1 |
| False-alarm budget for this check | 0.0304 |
| False-alarm budget spent so far | 0.0304 of 0.05 |

## How to read this

In tracta's experiments on ten published models from 0.35B to 4B parameters, serving the model's published Q4_K_M instead of its Q8_0 changed 4.6–75% of these answers, and serving a different model changed 45–97%. Serving the same file again changed 0–0.5%, and serving it through a second inference engine changed 10–15% (measured for four of the ten). A check therefore has less power when your provider's own noise is high, and a large model loses fewer of these answers to 4-bit quantisation than a small one.

- A change means the answers moved more than the provider's own noise. It does not show intent: an announced model update, a new default system prompt or new sampling settings also count.
- A provider that recognises these requests can serve the real model to them. Your question list comes from a private seed, which makes that harder, not impossible.
- A change that leaves short factual answers alone is invisible to this check.
