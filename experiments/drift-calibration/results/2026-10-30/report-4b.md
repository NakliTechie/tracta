# Drift checks: per-family summary

Provider serves the published Q8_0. B1 uses the provider's own baseline answers as the reference bank, scored by one fixed Qwen3-0.6B scorer. Rates are over fresh 25-prompt audits at α = 0.05, with exact 95% intervals.

| Family | Swap target | B1 false alarm (Q8 unchanged) | B1 detects Q4_K_M | B1 detects swap | B2 answers changed: Q8 rerun / Q8 other engine / Q4 / swap | F1 EER at 3 probes: Q4 / swap |
|---|---|---|---|---|---|---|
| llama-3.2-3b | Llama-3.2-1B-Instruct-Q8_0 | 3.8% (0.8–10.6, n=80) | 0.0% (0.0–8.8, n=40) | 10.0% (2.8–23.7, n=40) | 0.5% / n/a / 23.4% / 83.9% | 0.0% / 0.0% |
| phi-4-mini | Qwen3.5-4B-Q8_0 | 3.8% (0.8–10.6, n=80) | 2.5% (0.1–13.2, n=40) | 97.5% (86.8–99.9, n=40) | 0.2% / n/a / 37.6% / 45.2% | 0.0% / 0.0% |
| qwen3.5-4b | Qwen3.5-0.8B-Q8_0 | 5.0% (1.4–12.3, n=80) | 2.5% (0.1–13.2, n=40) | 25.0% (12.7–41.2, n=40) | 0.0% / n/a / 4.6% / 84.2% | 0.0% / 0.0% |
| smollm3-3b | SmolLM2-1.7B-Instruct-Q8_0 | 0.0% (0.0–4.5, n=80) | 5.0% (0.6–16.9, n=40) | 100.0% (91.2–100.0, n=40) | 0.0% / n/a / 26.8% / 88.8% | 0.1% / 0.0% |

"Q8 other engine" is the same file run by transformers on CPU (prompt token IDs from llama-server). Where it reads n/a, transformers cannot load that architecture from GGUF, so the B2 and F1 nulls vary only threads and batching on one engine. F1's equal error rate counts every same-model pair, both engines included, as the null.
