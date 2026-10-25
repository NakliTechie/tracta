# tracta

Read and act on internal paths in neural networks: which neuron, direction, or expert carries a behaviour, and what happens when you touch it.

## Install

TODO: nothing to install yet. This is a research project; the first output is a charter in `plan/research/`.

## Why

One primitive, path and component attribution, serves two tracks that keep showing up in the vault:

1. **Detection.** Runtime steering and abliteration change a model's behaviour without changing the weight file. Catching that means reading the activation path: a constant offset along a known direction across depth (white-box), or a capability drop that only appears when a refusal gate is tripped (black-box). Hypotheses H1–H5 are drafted in the knowledge vault.
2. **Routing.** The same "which component carries this" question is the routing question in a mixture-of-experts model. The motivating build is a JEE solver as an MoE: per-topic experts (physics, chemistry, maths subtopics) and a problem-solving router that selects a path through them. Understanding and steering that path is the detection machinery pointed inward.

Research first: a charter that states the shared primitive and the two tracks, then bounded experiments. The detection track's H1–H5 are the first experiment set, paper-only until a run is approved.

## Context

Scaffolded 2026-10-25 from an inline handoff. Tier: **Tool (research)**. Local hardware only (M4 Pro 24 GB now, M5 Ultra from December); no paid GPU.

Prior art and seeds live in the knowledge vault (`~/Code/knowledge`):
- `notes/detecting-runtime-abliteration.md` — the detection synthesis (where the edit lives decides what catches it).
- `notes/crazy-qwen-neuron-steering-recipe.md` — a fixed-address neuron-steering recipe, logged for discussion.
- `plan/research/2026-10-25-abliteration-detection/experiment-design.md` — H1–H5.
- `sources/2026-10-21-heterodoxin-abliteratedouthebox-repo.md` — AbliteratedOutTheBox (the tool).
- `sources/2026-05-21-hikari-gemma4-31b-repe-uncensored-article.md` — Hikari's RepE method.
- `sources/2026-09-22-matryoshka-attribution-mattr-paper.md` — MAttr, component attribution at every sparsity (the routing/attribution link).
- `sources/2026-05-28-zhou-mccoy-frank-continuous-variable-steering-interventions-paper.md` — set a graded trait to a value.

## Contributor notes

Verification budget (SUBSTANCE §3 items 9–11): check once per batch, not between steps; run only the checks covering the changed batch; generated test data to one temp dir (5,000 files / 500 MB cap), deleted after; one diagnostic batch per failure with a written hypothesis first; no new process tooling until a feature gates it; log anything over 50 MB or 1,000 files to the session ledger.
