"""Draw the README's Results chart: B2 changed-answer share per family, for each kind of change.

    python plot_families.py results/2026-10-30 results/2026-10-30/b2-families.png

One row per family. Markers: the unchanged model rerun on the same engine and on a second engine
(the noise a check must sit above), the published Q4_K_M downgrade, and the swap.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Ordered by family size, smallest first.
ORDER = ["lfm2.5-350m", "qwen3-0.6b", "qwen3.5-0.8b", "gemma-3-1b", "llama-3.2-1b", "smollm2-1.7b",
         "llama-3.2-3b", "smollm3-3b", "phi-4-mini", "qwen3.5-4b"]
SERIES = [  # (label, colour, marker, filled)
    ("Unchanged, same engine", MUTED, "o", False),
    ("Unchanged, second engine", MUTED, "o", True),
    ("Published Q4_K_M", "#2a78d6", "s", True),
    ("Swapped model", "#eb6834", "D", True),
]


def values(res):
    arms, pairs = res["b2"]["arms"], res["b2"]["pairs"]
    get = lambda tag: 100 * arms[tag]["changed_all"]["rate"] if tag in arms else None
    engine = pairs.get("q8-day1|q8-hfday1")
    return [get("q8-day2"), 100 * engine["changed_all"]["rate"] if engine else None, get("q4-day1"), get("swap-day1")]


def main(src, dst):
    root = pathlib.Path(src) / "families"
    fams = [f for f in ORDER if (root / f / "results.json").exists()]
    data = {f: values(json.loads((root / f / "results.json").read_text())) for f in fams}
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for y, f in enumerate(fams):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    for i, (label, colour, marker, filled) in enumerate(SERIES):
        pts = [(data[f][i], y) for y, f in enumerate(fams) if data[f][i] is not None]
        ax.scatter([x for x, _ in pts], [y for _, y in pts], s=46, marker=marker, label=label, zorder=3,
                   facecolor=colour if filled else SURFACE, edgecolor=colour if not filled else SURFACE, linewidth=1.6)
    ax.set_yticks(range(len(fams)), fams)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of 436 greedy boundary answers that changed against the unchanged Q8_0", color=MUTED, fontsize=9)
    ax.set_title("Knowledge-boundary check (B2): answers each change moved, per family",
                 color=INK, fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, -0.24), ncol=4, frameon=False, fontsize=8.5, labelcolor=INK,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(dst, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:3])
