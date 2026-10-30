# Changelog

All notable changes to tracta. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `tracta baseline --questions FILE` creates a watch that asks your own questions from a JSON Lines file (`{"id", "prompt"}` per line; see `docs/questions-example.jsonl`) instead of the built-in 436. The watch keeps a copy in `questions.json`, `status` shows the question count and source, and the report says the built-in evidence figures do not apply. Go binary only.

### Changed
- The Windows amd64 binary of 0.2.0 passed the four end-to-end cases on a Windows laptop, against a llama-server on another machine: one baseline gives no verdict, an unchanged model gives exit 0, a published Q4_K_M quant and a swap to gemma-3-270m give exit 2. The Linux and Intel macOS builds remain untested.

## [0.2.0] — 2026-10-30

First public release.

### Added
- `tracta`, a command-line check of whether a model provider still serves the model it promised: `baseline`, `check` and `status`, with a Markdown and JSON report per check.
- Static binaries for macOS (arm64, amd64), Linux (amd64, arm64) and Windows (amd64), built by `scripts/build-binaries.sh` with SHA-256 sums.
- Providers: any OpenAI-compatible `/chat/completions` endpoint, and the Claude Messages API through Anthropic's Go SDK.
- Decision rule: an exact paired sign test of today's run against a second baseline, anchored on the first, with no assumption that every question is equally noisy.
- False-alarm control over repeated checks: check k spends α·6/(π²k²) of a total budget α (0.05 by default); checks without a verdict spend nothing.
- 436 knowledge-boundary questions per run; the multiplications and the order come from a private per-watch seed.
- The Python reference version (`python3 -m tracta`), and golden tests that require the Go binary to reproduce its verdicts, counts and p-values.
- Experiments behind the method (`experiments/drift-calibration/`): ten published model families from 0.35B to 4B, a second inference engine, pinned model files (`models.json`), and a test on the Claude API.

### Known limits
- Against two strong models that both know the answers, the check has little margin: a Claude Haiku 5.5 → Sonnet 5.5 swap was detected through only 10 of 436 answers.
- Only the macOS arm64 binary has been run end to end; the other platform builds are untested.

## [0.1.0] — 2026-10-30

Python-only version (not tagged): the same check with a Fisher test, replaced in 0.2.0 by the paired sign test after an independent review.
