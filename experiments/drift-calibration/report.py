"""Write a Markdown report from per-family results.json files.

    python report.py OUT.md WORK_ROOT/*/results.json

One row per family: the false-alarm rate of B1 on the unchanged model, and its detection rate on the
published Q4_K_M downgrade and on the swap, each over fresh 25-prompt audits with exact 95% intervals;
B2's changed-answer share for the unchanged model rerun on the same engine and on a second engine,
for the downgrade and for the swap; F1's equal error rate at 3 probes.
"""
import json
import pathlib
import sys


def pct(r):
    if not r:
        return "n/a"
    return f"{100 * r['rate']:.1f}% ({100 * r['lo']:.1f}–{100 * r['hi']:.1f}, n={r['n']})"


def row(path):
    p = pathlib.Path(path)
    res = json.loads(p.read_text())
    fam = json.loads((p.parent / "family.json").read_text())
    arms = res["b1"]["arms"]
    b2 = res["b2"]["arms"]
    pairs = res["b2"]["pairs"]
    eer = res["f1"]["eer_by_change_and_probes"]
    f1 = lambda change: f"{100 * eer[change]['3']:.1f}%" if change in eer else "n/a"
    b2c = lambda tag: f"{100 * b2[tag]['changed_all']['rate']:.1f}%" if tag in b2 else "n/a"
    engine = pairs.get("q8-day1|q8-hfday1")
    engine = f"{100 * engine['changed_all']['rate']:.1f}%" if engine else "n/a"
    rej = lambda tag: pct(arms.get(tag, {}).get("reject", {}).get("25"))
    return (f"| {p.parent.name} | {pathlib.Path(fam['swap']).stem} | {rej('q8-audit')} | {rej('q4-audit')} | "
            f"{rej('swap-audit')} | {b2c('q8-day2')} / {engine} / {b2c('q4-day1')} / {b2c('swap-day1')} | "
            f"{f1('q4|q8')} / {f1('q8|swap')} |")


def main(out, paths):
    lines = [
        "# Drift checks: per-family summary",
        "",
        "Provider serves the published Q8_0. B1 uses the provider's own baseline answers as the reference bank, "
        "scored by one fixed Qwen3-0.6B scorer. Rates are over fresh 25-prompt audits at α = 0.05, with exact 95% intervals.",
        "",
        "| Family | Swap target | B1 false alarm (Q8 unchanged) | B1 detects Q4_K_M | B1 detects swap | "
        "B2 answers changed: Q8 rerun / Q8 other engine / Q4 / swap | F1 EER at 3 probes: Q4 / swap |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [row(p) for p in sorted(paths)]
    lines += ["", "\"Q8 other engine\" is the same file run by transformers on CPU (prompt token IDs from llama-server). "
              "Where it reads n/a, transformers cannot load that architecture from GGUF, so the B2 and F1 nulls vary only "
              "threads and batching on one engine. F1's equal error rate counts every same-model pair, both engines included, "
              "as the null."]
    pathlib.Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
