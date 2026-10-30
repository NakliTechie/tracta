# Drift checks: per-family summary

Provider serves the published Q8_0. B1 uses the provider's own baseline answers as the reference bank, scored by one fixed Qwen3-0.6B scorer. Rates are over fresh 25-prompt audits at α = 0.05, with exact 95% intervals.

| Family | Swap target | B1 false alarm (Q8 unchanged) | B1 detects Q4_K_M | B1 detects swap | B2 answers changed: Q8 rerun / Q8 other engine / Q4 / swap | F1 EER at 3 probes: Q4 / swap |
|---|---|---|---|---|---|---|
| gemma-3-1b | gemma-3-270m-it-Q8_0 | 5.0% (1.4–12.3, n=80) | 32.5% (18.6–49.1, n=40) | 85.0% (70.2–94.3, n=40) | 0.0% / 15.4% / 55.5% / 96.8% | 14.5% / 0.3% |
| lfm2.5-350m | SmolLM2-360M-Instruct-Q8_0 | 8.8% (3.6–17.2, n=80) | 7.5% (1.6–20.4, n=40) | 100.0% (91.2–100.0, n=40) | 0.0% / n/a / 47.5% / 96.8% | 0.0% / 0.0% |
| llama-3.2-1b | Qwen3-0.6B-Q8_0 | 2.5% (0.3–8.7, n=80) | 7.5% (1.6–20.4, n=40) | 100.0% (91.2–100.0, n=40) | 0.0% / 10.3% / 55.5% / 89.2% | 34.4% / 4.3% |
| qwen3-0.6b | LFM2.5-350M-Q8_0 | 3.8% (0.8–10.6, n=80) | 50.0% (33.8–66.2, n=40) | 100.0% (91.2–100.0, n=40) | 0.0% / 9.9% / 38.5% / 96.8% | 1.1% / 0.0% |
| qwen3.5-0.8b | Qwen3-0.6B-Q8_0 | 6.2% (2.1–14.0, n=80) | 30.0% (16.6–46.5, n=40) | 100.0% (91.2–100.0, n=40) | 0.0% / n/a / 57.8% / 67.9% | 0.0% / 0.0% |
| smollm2-1.7b | SmolLM2-360M-Instruct-Q8_0 | 7.5% (2.8–15.6, n=80) | 35.0% (20.6–51.7, n=40) | 25.0% (12.7–41.2, n=40) | 0.0% / 12.4% / 75.0% / 94.0% | 9.4% / 0.0% |

"Q8 other engine" is the same file run by transformers on CPU (prompt token IDs from llama-server). Where it reads n/a, transformers cannot load that architecture from GGUF, so the B2 and F1 nulls vary only threads and batching on one engine. F1's equal error rate counts every same-model pair, both engines included, as the null.
