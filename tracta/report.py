"""Write a check's report as Markdown (for people) and JSON (for scripts)."""
import json

EVIDENCE = (
    "In tracta's experiments on ten published models from 0.35B to 4B parameters, serving the model's "
    "published Q4_K_M instead of its Q8_0 changed 4.6–75% of these answers, and serving a different model "
    "changed 45–97%. Serving the same file again changed 0–0.5%, and serving it through a second inference "
    "engine changed 10–15%. A check therefore has less power when your provider's own noise is high, and a "
    "large model loses fewer of these answers to 4-bit quantisation than a small one."
)

LIMITS = [
    "A change means the answers moved more than the provider's own noise. It does not show intent: an "
    "announced model update, a new default system prompt or new sampling settings also count.",
    "A provider that recognises these requests can serve the real model to them. Your question list comes "
    "from a private seed, which makes that harder, not impossible.",
    "A change that leaves short factual answers alone is invisible to this check.",
]


def pct(v):
    return f"{100 * v:.1f}%"


def markdown(config, r):
    lines = [f"# tracta check {r['check_number']}: {config['name']}", "",
             f"**Result: {r['verdict']}.**", ""]
    if r["verdict"] == "no verdict":
        lines += [r["reason"] + ".", ""]
    lo, hi = r["changed_ci"]
    lines += ["| Measure | Value |", "|---|---|",
              f"| Endpoint | `{config['endpoint']}` |", f"| Model | `{config['model']}` |",
              f"| Questions compared | {r['items']} |",
              f"| Answers that changed against baseline 1 | {r['changed']} ({pct(r['changed_rate'])}, 95% CI {pct(lo)}–{pct(hi)}) |"]
    if "p_value" in r:
        nlo, nhi = r["null_ci"]
        lines += [f"| Provider's own noise (baseline {r['null_run']} against baseline 1) | {r['null_changed']} of {r['null_items']} "
                  f"({pct(r['null_rate'])}, 95% CI {pct(nlo)}–{pct(nhi)}) |",
                  f"| Questions where only today's answer moved / only baseline {r['null_run']}'s moved | {r['only_check']} / {r['only_noise']} |",
                  f"| One-sided exact sign test p | {r['p_value']:.3g} |",
                  f"| False-alarm budget for this check | {r['alpha_this_check']:.4f} |"]
    lines += [f"| False-alarm budget spent so far | {r['alpha_spent']:.4f} of {r['alpha_total']} |", ""]
    if r.get("fingerprint"):
        f = r["fingerprint"]
        lines += [f"Fingerprint (supporting evidence, does not decide): median first-token divergence "
                  f"{f['median_jsd_check']:.4f} bits now, against {f['median_jsd_noise']:.4f} bits between "
                  f"baseline runs, over {f['probes']} probes.", ""]
    if r["examples"]:
        lines += ["Answers that changed (first 8):", "", "| Question | Baseline 1 | Now |", "|---|---|---|"]
        cell = lambda s: s.strip().replace("|", "\\|").replace("\n", " ")[:60]
        lines += [f"| {cell(e['prompt'])} | {cell(e['baseline'])} | {cell(e['now'])} |" for e in r["examples"]]
        lines.append("")
    lines += ["## How to read this", "", EVIDENCE, ""]
    lines += [f"- {x}" for x in LIMITS]
    return "\n".join(lines) + "\n"


def write(store, config, r):
    md = store.report_path(r["check_number"], "md")
    md.write_text(markdown(config, r))
    store.report_path(r["check_number"], "json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    return md
