# tracta check 2: claude

**Result: changed.**

| Measure | Value |
|---|---|
| Endpoint | `https://api.anthropic.com` |
| Model | `claude-haiku-5-5` (this check called `claude-sonnet-5-5`) |
| Questions compared | 436 |
| Answers that changed against baseline 1 | 10 (2.3%, 95% CI 1.1%–4.2%) |
| Provider's own noise (baseline 2 against baseline 1) | 0 of 436 (0.0%, 95% CI 0.0%–0.8%) |
| Questions where only today's answer moved / only baseline 2's moved | 10 / 0 |
| One-sided exact sign test p | 0.000977 |
| False-alarm budget for this check | 0.0076 |
| False-alarm budget spent so far | 0.0380 of 0.05 |

Answers that changed (first 8):

| Question | Baseline 1 | Now |
|---|---|---|
| What is 687 × 73? Answer with the number only. | 50151 | 687 × 73 = 687×70 + 687× |
| What is 569 × 58? Answer with the number only. | 32002 | 33002 |
| What is 791 × 36? Answer with the number only. | 28476 | 791 × 36 = 791 × 30 + |
| What is 892 × 73? Answer with the number only. | 65116 | 892 × 73 = 892×70 + 892× |
| What is 853 × 87? Answer with the number only. | 74211 | 853 × 87 = 853×80 + 853× |
| What is 835 × 43? Answer with the number only. | 35905 | 835 × 43 = 835×40 + 835× |
| What is 716 × 49? Answer with the number only. | 35084 | 716 × 49 = 716 × 50 − |
| What is 886 × 79? Answer with the number only. | 69994 | 886 × 79 = 886 × 80 − |

## How to read this

In tracta's experiments on ten published models from 0.35B to 4B parameters, serving the model's published Q4_K_M instead of its Q8_0 changed 4.6–75% of these answers, and serving a different model changed 45–97%. Serving the same file again changed 0–0.5%, and serving it through a second inference engine changed 10–15% (measured for four of the ten). A check therefore has less power when your provider's own noise is high, and a large model loses fewer of these answers to 4-bit quantisation than a small one.

- A change means the answers moved more than the provider's own noise. It does not show intent: an announced model update, a new default system prompt or new sampling settings also count.
- A provider that recognises these requests can serve the real model to them. Your question list comes from a private seed, which makes that harder, not impossible.
- A change that leaves short factual answers alone is invisible to this check.
