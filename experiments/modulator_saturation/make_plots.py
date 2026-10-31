"""Plot Securing per step, before and after the recommended fix.

    python experiments/modulator_saturation/make_plots.py

Writes securing_before_after.png next to this file: one panel per scenario,
50 steps each, for the baseline, F1 alone and F1+F4a (see run_fixes.py).
The same numbers are in results.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import SCENARIOS, patched, run_bridge  # noqa: E402
from run_fixes import FIXES, leak_hook  # noqa: E402

OUT = Path(__file__).resolve().parent / "securing_before_after.png"
STEPS = 50
F1_SWAPS = FIXES[0][3]
SERIES = [  # label, swaps, state hook, colour, line style
    ("baseline", [], None, "#2a78d6", "-"),
    ("F1 (MetaMo)", F1_SWAPS, None, "#eb6834", "--"),
    ("F1+F4a (MetaMo + bridge)", F1_SWAPS, leak_hook(0.20), "#1baf7a", "-"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def securing_trace(scenario, swaps, hook):
    with patched(*swaps):
        rows = run_bridge(scenario, STEPS, state_hook=hook)
    return [r["t"] for r in rows[1:]], [r["securing"] for r in rows[1:]]


def main() -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 3.4), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        for label, swaps, hook, colour, style in SERIES:
            t, s = securing_trace(sc, swaps, hook)
            ax.plot(t, s, color=colour, linestyle=style, linewidth=2, label=label)
        ax.axhline(0.95, color=MUTED, linewidth=1, linestyle=":")
        ax.axhline(0.5, color=GRID, linewidth=1)
        if sc.threat_steps and sc.threat_steps < STEPS:
            ax.axvspan(-0.5, sc.threat_steps - 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.set_title(sc.name, fontsize=10, color=INK)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("step", fontsize=9, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Securing", fontsize=9, color=MUTED)
    axes[0].text(STEPS - 1, 0.93, "saturated (0.95)", fontsize=7, color=MUTED, ha="right", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT, dpi=130, facecolor="#fcfcfb")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
