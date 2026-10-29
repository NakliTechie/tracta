"""Draw the README's Results chart from one campaign's results.json.

    python plot.py results/2026-10-29/results.json results/2026-10-29/b1-detection.png

B1 detection rate against the number of calls: the unchanged model (mean of the three Q8_0 days)
and the downgrade at 30%, 50% and 100% of traffic. N = 200 is left out, because a 200-of-200
subset is the same set every draw.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {1.0: "#2a78d6", 0.5: "#eb6834", 0.3: "#1baf7a"}
NUDGE = {1.0: 7, 0.5: -7, 0.3: 0}  # label offsets in points, so the two top lines stay readable


def main(src, dst):
    b1 = json.load(open(src))["b1"]
    ns = [10, 25, 50, 100]
    nulls = [b1["arms"][f"q8-day{d}"]["reject_rate_by_n"] for d in (1, 2, 3)]
    null = [sum(x[str(n)] for x in nulls) / 3 for n in ns]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for p, colour in SERIES.items():
        ys = [b1["dilution_reject_rate"][str(p)][str(n)] for n in ns]
        ax.plot(ns, ys, color=colour, lw=2, marker="o", ms=5, mec=SURFACE, mew=2)
        label = "Downgrade on all traffic" if p == 1.0 else f"Downgrade on {int(p * 100)}% of traffic"
        ax.annotate(label, (ns[-1], ys[-1]), xytext=(8, NUDGE[p]), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.plot(ns, null, color=MUTED, lw=2, ls="--", marker="o", ms=5, mec=SURFACE, mew=2)
    ax.annotate("Unchanged model", (ns[-1], null[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlim(8, 100 * 4.2)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("API calls (one per prompt)", color=MUTED, fontsize=9)
    ax.set_ylabel("Times the drift test fired", color=MUTED, fontsize=9)
    ax.set_title("Qwen3-0.6B served at Q8_0, swapped for Q4_K_M: rank-uniformity test, α = 0.05",
                 color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    fig.tight_layout()
    fig.savefig(dst, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:3])
