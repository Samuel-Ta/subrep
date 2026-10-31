"""Test the proposed fixes on every scenario and print the tables.

    python experiments/modulator_saturation/run_fixes.py

Each fix is a monkeypatch (nothing under bridge/ or external/metamo/ is
edited) and is tested alone first; the combination is tested only after each
part has been. Every run: 20 and 50 steps, all five scenarios, seed 7.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import (  # noqa: E402
    SCENARIOS, SUMMARY_HEADER, MetaMoPseudoBimonad, MotivationalState, bimonad_mod, budget_mod, config_mod,
    fmt_first, patched, run_bridge, stability_mod, summary, summary_line, trace_line)

LENGTHS = (20, 50)
NEUTRAL_POINT = budget_mod.MODULATOR_NEUTRAL
ORIGINAL_PROJECT = stability_mod.project_to_safe_region


# -- F1 (MetaMo): raise caution only when the projection actually moved the state ----

def project_conditional(state):
    """project_to_safe_region (dynamics/stability.py:124-141), with the caution
    increment of lines 139-140 applied only when the goal projection changed
    something, i.e. the state was outside the safe region."""
    projected = ORIGINAL_PROJECT(state)
    moved = not np.allclose(projected.G, state.G)
    if moved:
        return projected
    out = projected.copy()
    out.M = state.M.copy()  # undo the unconditional increment for a state already inside
    return out


# -- F2 (MetaMo): the fallback shrinks the step without re-adding caution ------------------

def fallback_without_caution_increment(self, current_state, next_state):
    """_apply_conservative_fallback (category/bimonad.py:143-151): the same
    average and goal projection, but the modulators are the plain average."""
    averaged = MotivationalState(G=current_state.G * 0.5 + next_state.G * 0.5,
                                 M=current_state.M * 0.5 + next_state.M * 0.5)
    projected = ORIGINAL_PROJECT(averaged)
    return MotivationalState(G=projected.G, M=averaged.M)


# -- F4 (bridge): decay every modulator toward neutral after each step --------------------

def leak_hook(rate: float) -> Callable[[int, Any, Any], Any]:
    """Bridge-side, dimension-generic: M <- 0.5 + (1 - rate) (M - 0.5) for every
    modulator, whatever their number; goals untouched."""
    def hook(t, prev, new):
        M = NEUTRAL_POINT + (1.0 - rate) * (np.asarray(new.M) - NEUTRAL_POINT)
        return MotivationalState(G=np.asarray(new.G).copy(), M=M)
    return hook


Fix = Tuple[str, str, str, List[Tuple[Any, str, Any]], Optional[Callable]]
FIXES: List[Fix] = [
    ("F1", "MetaMo", "projection raises caution only when it projects",
     [(bimonad_mod, "project_to_safe_region", project_conditional),
      (stability_mod, "project_to_safe_region", project_conditional)], None),
    ("F2", "MetaMo", "fallback does not re-add caution",
     [(MetaMoPseudoBimonad, "_apply_conservative_fallback", fallback_without_caution_increment)], None),
    ("F3", "MetaMo", "lax-distributive tolerance 1e-3 -> 0.05", [(bimonad_mod, "LAX_DISTRIBUTIVE_DELTA", 0.05)], None),
    ("F4a", "bridge", "modulators decay 20% toward neutral each step", [], leak_hook(0.20)),
    ("F4b", "bridge", "modulators decay 50% toward neutral each step", [], leak_hook(0.50)),
]


def run_all(swaps, hook):
    out = {}
    with patched(*swaps):
        for sc in SCENARIOS:
            for n in LENGTHS:
                rows = run_bridge(sc, n, state_hook=hook)
                out[(sc.name, n)] = {"rows": rows, "summary": summary(rows, sc)}
    return out


def verdict(res) -> Dict[str, Any]:
    """The acceptance checks for a fix, all on 50-step runs."""
    s = {name: res[(name, 50)]["summary"] for name in (sc.name for sc in SCENARIOS)}
    calm = ("Zero", "Neutral outcome", "Good outcomes")
    threat_rows = res[("Constant threat", 50)]["rows"][1:]
    rec = s["Threat then recovery"]["recovery"]
    return {
        "no saturation without threat": all(s[c]["first_saturated"] is None for c in calm),
        "caution rises under threat (Securing >= 0.80 within 3 steps)": any(r["securing"] >= 0.80 for r in threat_rows[:3]),
        "recovers within 15 steps": bool(rec and rec["recovers"]),
        "calm scenarios end within 0.50 +- 0.15": all(abs(s[c]["final_securing"] - 0.5) <= 0.15 for c in calm),
        "budgets still move (eps range over all runs)": (min(v["eps_range"][0] for v in s.values()),
                                                         max(v["eps_range"][1] for v in s.values())),
        "lax-distributive failures (50-step runs)": sum(v["lax_fail"] for v in s.values()),
        "contractivity failures (50-step runs)": sum(v["contractive_fail"] for v in s.values()),
        "fallbacks (50-step runs)": sum(v["fallbacks"] for v in s.values()),
    }


def print_fix(fid, side, title, res, base) -> Dict[str, Any]:
    print(f"\n### {fid} ({side}): {title}\n")
    print(SUMMARY_HEADER)
    for sc in SCENARIOS:
        for n in LENGTHS:
            print(summary_line(sc.name, n, res[(sc.name, n)]["summary"]))
    print("\nBefore -> after, 50 steps (final Securing; first step >= 0.95):")
    for sc in SCENARIOS:
        b, a = base[(sc.name, 50)]["summary"], res[(sc.name, 50)]["summary"]
        print(f"- {sc.name}: {b['final_securing']:.3f} -> {a['final_securing']:.3f}; "
              f"{fmt_first(b['first_saturated'])} -> {fmt_first(a['first_saturated'])}")
    v = verdict(res)
    print("\nAcceptance checks:")
    for k, val in v.items():
        print(f"- {k}: {val}")
    print(f"- Threat then recovery, Securing per step (50): {trace_line(res[('Threat then recovery', 50)]['rows'])}")
    return v


def main() -> None:
    print("# Fixes: modulator saturation\n")
    print("Generated by `python experiments/modulator_saturation/run_fixes.py`. Seed 7; same scenarios and "
          "outcomes as run_hypotheses.py; bridge scales 1.0.")
    base = run_all([], None)
    print("\n### Baseline (no fix)\n")
    print(SUMMARY_HEADER)
    for sc in SCENARIOS:
        for n in LENGTHS:
            print(summary_line(sc.name, n, base[(sc.name, n)]["summary"]))
    verdicts = {"baseline": verdict(base)}

    # each part alone
    for fid, side, title, swaps, hook in FIXES:
        verdicts[fid] = print_fix(fid, side, title, run_all(swaps, hook), base)

    # combinations, only after every part has been tested alone
    f1 = FIXES[0][3]
    combos = [
        ("F1+F4a", "MetaMo + bridge", "conditional projection, and 20% decay toward neutral", f1, leak_hook(0.20)),
        ("F1+F4b", "MetaMo + bridge", "conditional projection, and 50% decay toward neutral", f1, leak_hook(0.50)),
    ]
    for fid, side, title, swaps, hook in combos:
        verdicts[fid] = print_fix(fid, side, title, run_all(swaps, hook), base)

    print("\n### Summary of acceptance checks (50-step runs)\n")
    keys = list(verdicts["baseline"].keys())
    print("| Check | " + " | ".join(verdicts) + " |")
    print("|---|" + "---|" * len(verdicts))
    for k in keys:
        cells = []
        for v in verdicts.values():
            x = v[k]
            cells.append(f"{x[0]:.3f}-{x[1]:.3f}" if isinstance(x, tuple) else str(x))
        print(f"| {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
