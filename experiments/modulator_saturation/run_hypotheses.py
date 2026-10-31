"""Run every hypothesis in HYPOTHESES.md on every scenario and print the tables.

    python experiments/modulator_saturation/run_hypotheses.py

Each experiment changes exactly one component (swapped in with `patched`,
restored afterwards) and is compared with the unchanged baseline. Deterministic:
seeded, and MetaMo's step has no randomness of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import (  # noqa: E402
    GOOD, I_APP, I_SEC, I_THR, I_VAL, M_NAMES, NEUTRAL, SCENARIOS, SUMMARY_HEADER, THREAT, Scenario,
    MetaMoPseudoBimonad, MotivationalState, OpenPsiAppraisal, appraisal_mod, bimonad_mod, coherence_mod,
    config_mod, emitted_budgets, fmt_first, initial_state, patched, run_bridge, run_metamo, stability_mod,
    stimulus_for, summary, summary_line, trace_line)

LENGTHS = (20, 50)
G_IND, G_TRANS = config_mod.G_IND, config_mod.G_TRANS
M_AROUSAL, M_APPROACH, M_RESOLUTION, M_VALENCE = (config_mod.M_AROUSAL, config_mod.M_APPROACH,
                                                  config_mod.M_RESOLUTION, config_mod.M_VALENCE)


# -- a faithful copy of OpenPsiAppraisal.appraise (openpsi/appraisal.py:32-99) with switches ------

def make_appraise(ind_scale_one: bool = False, clip_squash: bool = False):
    sigmoid = appraisal_mod.sigmoid

    def appraise(self, state, stimulus):
        M_prime = state.M.copy()
        g_ind = state.G[G_IND]
        g_trans = state.G[G_TRANS]
        arousal_feedback = sigmoid((state.M[M_AROUSAL] - 0.5) * 5.0)
        trans_scale = np.exp(g_trans - 0.5)
        ind_scale = 1.0 if ind_scale_one else np.exp(g_ind - 0.5)
        benign_novelty = stimulus.novelty * (1.0 - stimulus.risk)
        demanding_context = (stimulus.effort + stimulus.risk) / 2.0
        delta_valence = (0.75 * stimulus.conduciveness + 0.25 * benign_novelty
                         - 0.55 * stimulus.risk - 0.15 * stimulus.effort)
        delta_arousal = (stimulus.novelty * (1.0 + 0.5 * arousal_feedback)
                         + 0.15 * stimulus.risk - 0.35 * stimulus.effort)
        delta_approach = 0.65 * benign_novelty + 0.35 * stimulus.conduciveness - 0.75 * stimulus.risk
        delta_resolution = 0.55 * stimulus.conduciveness + 0.35 * stimulus.effort + 0.20 * stimulus.risk
        delta_threshold = 0.70 * stimulus.risk + 0.25 * demanding_context - 0.15 * stimulus.conduciveness
        delta_securing = (0.80 * stimulus.risk + 0.20 * stimulus.effort
                          - 0.30 * benign_novelty - 0.10 * stimulus.conduciveness)
        M_prime[M_AROUSAL] += delta_arousal * trans_scale
        M_prime[M_APPROACH] += delta_approach * trans_scale
        M_prime[config_mod.M_THRESHOLD] += delta_threshold * ind_scale
        M_prime[config_mod.M_SECURING] += delta_securing * ind_scale
        M_prime[M_VALENCE] += delta_valence
        M_prime[M_RESOLUTION] += delta_resolution
        if clip_squash:
            M_prime = np.clip(M_prime, 0.0, 1.0)
        else:
            M_prime = 1.0 / (1.0 + np.exp(-4.0 * (M_prime - 0.5)))
        return MotivationalState(G=state.G.copy(), M=M_prime)

    return appraise


# -- one-change replacements ------------------------------------------------------------

def project_without_modulator_increment(state):
    """project_to_safe_region (dynamics/stability.py:124-141) without lines 139-140."""
    next_state = state.copy()
    next_state.G[G_IND] = max(next_state.G[G_IND], config_mod.THETA_SAFE)
    other_idx = [i for i in range(next_state.G.shape[0]) if i != G_IND]
    other = next_state.G[other_idx]
    other_norm = np.linalg.norm(other)
    max_other = np.sqrt(max(0.0, config_mod.G_MAX ** 2 - next_state.G[G_IND] ** 2))
    if other_norm > max_other and other_norm > 0.0:
        next_state.G[other_idx] = other * (max_other / other_norm)
    return next_state


def fallback_average_only(self, current_state, next_state):
    """_apply_conservative_fallback (category/bimonad.py:143-151) without the projection."""
    return MotivationalState(G=current_state.G * 0.5 + next_state.G * 0.5,
                             M=current_state.M * 0.5 + next_state.M * 0.5)


EXPERIMENTS: List[Tuple[str, str, List[Tuple[Any, str, Any]], Dict[str, Any]]] = [
    ("H1a", "stimulus scales x10", [], {"payoff_scale": 10.0, "motive_scale": 10.0}),
    ("H1b", "stimulus scales x100", [], {"payoff_scale": 100.0, "motive_scale": 100.0}),
    ("H3", "overgoal scaling fixed at 1", [(OpenPsiAppraisal, "appraise", make_appraise(ind_scale_one=True))], {}),
    ("H4", "sigmoid squash replaced by clip", [(OpenPsiAppraisal, "appraise", make_appraise(clip_squash=True))], {}),
    ("H5", "projection without the +0.1 caution increment",
     [(bimonad_mod, "project_to_safe_region", project_without_modulator_increment),
      (stability_mod, "project_to_safe_region", project_without_modulator_increment)], {}),
    ("H6", "boundary caution disabled", [(bimonad_mod, "raise_boundary_caution", lambda s: s)], {}),
    ("H7", "homeostatic damping disabled", [(bimonad_mod, "apply_homeostatic_damping", lambda s, d: d)], {}),
    ("H8", "lax-distributive check always passes",
     [(MetaMoPseudoBimonad, "check_lax_distributive_law", lambda self, *a, **k: True)], {}),
    ("H9", "fallback averages only (no projection)",
     [(MetaMoPseudoBimonad, "_apply_conservative_fallback", fallback_average_only)], {}),
    ("H10", "contractivity check always passes",
     [(bimonad_mod, "check_contractive_update_law", lambda *a, **k: True)], {}),
]


def all_scenarios(swaps, kwargs, hook=None) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    with patched(*swaps):
        for sc in SCENARIOS:
            for n in LENGTHS:
                rows = run_bridge(sc, n, state_hook=hook, **kwargs)
                out[(sc.name, n)] = {"rows": rows, "summary": summary(rows, sc)}
    return out


def print_table(title: str, results) -> None:
    print(f"\n### {title}\n")
    print(SUMMARY_HEADER)
    for sc in SCENARIOS:
        for n in LENGTHS:
            print(summary_line(sc.name, n, results[(sc.name, n)]["summary"]))


def first_shift(base, exp) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    return {k: (base[k]["summary"]["first_saturated"], exp[k]["summary"]["first_saturated"]) for k in base}


def shift_line(shifts) -> str:
    return "; ".join(f"{name} @{n}: {fmt_first(a)} -> {fmt_first(b)}" for (name, n), (a, b) in shifts.items())


def main() -> None:
    print("# Results: modulator saturation\n")
    print("Generated by `python experiments/modulator_saturation/run_hypotheses.py`. Seed 7. "
          "Outcomes: threat = delta_r -0.5, delta_n -0.5 each, effort 0.5; good = delta_r +0.5, "
          "delta_n +0.2 each, effort 0.2; neutral = all 0. Bridge scales 1.0 unless stated. "
          "Threat then recovery: 5 threatening steps, then good ones.")

    # faithful-copy check: the copied appraisal must reproduce the original exactly
    base = all_scenarios([], {})
    copy = all_scenarios([(OpenPsiAppraisal, "appraise", make_appraise())], {})
    same = all(np.array_equal(base[k]["rows"][i]["M"], copy[k]["rows"][i]["M"])
               for k in base for i in range(len(base[k]["rows"])))
    print(f"\nCopied appraisal reproduces the original on every step of every run: **{same}**")

    print_table("Baseline: the loop as the bridge runs it (MetaMoGovernor)", base)
    for sc in SCENARIOS:
        print(f"- {sc.name}, Securing per step (20): {trace_line(base[(sc.name, 20)]['rows'])}")
        print(f"  Threshold: {trace_line(base[(sc.name, 20)]['rows'], 'threshold')}")
    rows = base[("Neutral outcome", 20)]["rows"][1:]
    print("- Runtime checks per step, Neutral outcome (lax / contractive / safe; F = fails): "
          + " ".join(f"{'P' if r['lax_ok'] else 'F'}{'P' if r['contractive_ok'] else 'F'}{'P' if r['safe_ok'] else 'F'}"
                     for r in rows))

    print("\n### Baseline: MetaMo alone (MetaMoPseudoBimonad stepped directly)\n")
    print(SUMMARY_HEADER)
    for sc in SCENARIOS:
        for n in LENGTHS:
            rs = run_metamo(sc, n)
            print(summary_line(sc.name, n, summary(rs, sc)))

    # H1 at the demo's own calibrated scales, for context
    demo = all_scenarios([], {"payoff_scale": 0.010, "motive_scale": 0.036})
    print_table("H1 context: the demo's calibrated scales (payoff 0.010, motive 0.036)", demo)

    results = {}
    for hid, title, swaps, kwargs in EXPERIMENTS:
        res = all_scenarios(swaps, kwargs)
        results[hid] = res
        print_table(f"{hid}: {title}", res)
        print(f"\nFirst saturated step, baseline -> {hid}: {shift_line(first_shift(base, res))}")
        if hid in ("H6",):
            print(f"- Max boundary pressure over all baseline runs: "
                  f"{max(base[k]['summary']['max_pressure'] for k in base):.3f}")
        if hid == "H7":
            diff = max(abs(a["securing"] - b["securing"]) for k in base
                       for a, b in zip(base[k]["rows"], res[k]["rows"]))
            print(f"- Largest Securing difference from baseline on any step: {diff:.6f}")
        if hid == "H8":
            steps = sum(len(base[k]["rows"]) - 1 for k in base)
            fails = sum(base[k]["summary"]["lax_fail"] for k in base)
            print(f"- Lax-distributive check in the baseline: fails on {fails} of {steps} steps "
                  f"({100 * fails / steps:.0f}%)")
        if hid == "H9":
            print("- Mean per-step Securing increase on steps where a check failed, below 0.95 "
                  "(baseline vs H9):")
            for sc in SCENARIOS:
                k = (sc.name, 50)
                def inc(rs):
                    d = [b["securing"] - a["securing"] for a, b in zip(rs, rs[1:])
                         if b["fallbacks"] > 0 and a["securing"] < 0.95]
                    return np.mean(d) if d else float("nan")
                print(f"  - {sc.name}: {inc(base[k]['rows']):+.4f} vs {inc(res[k]['rows']):+.4f}")
        if hid == "H10":
            print(f"- Contractivity check failures over all baseline runs: "
                  f"{sum(base[k]['summary']['contractive_fail'] for k in base)}")

    # H2: the appraisal alone, applied repeatedly from raised caution
    print("\n### H2: the appraisal alone, applied repeatedly (no decision, no stability machinery)\n")
    print("| Start | Stimulus | Securing after 1, 5, 10, 20 steps | Below 0.60 by step |")
    print("|---|---|---|---|")
    appraisal = OpenPsiAppraisal()
    for start in (0.85, 1.0):
        for label, stim in (("zero", stimulus_for(None)), ("neutral outcome", stimulus_for(NEUTRAL)),
                            ("good outcome", stimulus_for(GOOD))):
            st = initial_state()
            st.M[I_SEC] = start
            st.M[I_THR] = start
            traj = []
            for _ in range(20):
                st = appraisal.appraise(st, stim)
                traj.append(float(st.M[I_SEC]))
            below = next((i + 1 for i, v in enumerate(traj) if v < 0.60), None)
            print(f"| {start:.2f} | {label} | {traj[0]:.3f}, {traj[4]:.3f}, {traj[9]:.3f}, {traj[19]:.3f} | "
                  f"{fmt_first(below)} |")

    # H11: blending (as MetaMo's own example does) and periodic resets
    blend = all_scenarios([], {}, hook=lambda t, prev, new: coherence_mod.blend_states(prev, new))
    print_table("H11a: bridge loop with blend_states toward each step's output", blend)
    print(f"\nFirst saturated step, baseline -> H11a: {shift_line(first_shift(base, blend))}")
    reset = all_scenarios([], {}, hook=lambda t, prev, new: initial_state() if (t + 1) % 10 == 0 else new)
    print_table("H11b: bridge loop with the state reset every 10 steps", reset)

    # H12: MetaMo's own grid-world loop
    gridworld_section()

    # H13: what the bridge reads
    print("\n### H13: all six modulators and the emitted budgets, Threat then recovery (50 steps, baseline)\n")
    rs = base[("Threat then recovery", 50)]["rows"]
    print("| Step | " + " | ".join(M_NAMES) + " | eps | alpha |")
    print("|---|" + "---|" * (len(M_NAMES) + 2))
    for r in rs[1:]:
        if r["t"] in (0, 1, 2, 4, 5, 6, 10, 20, 30, 49):
            print(f"| {r['t']} | " + " | ".join(f"{v:.3f}" for v in r["M"]) + f" | {r['eps']:.4f} | {r['alpha']:.4f} |")
    sat = [r for r in rs[1:] if r["securing"] >= 0.95]
    if sat:
        span = {name: max(r["M"][i] for r in sat) - min(r["M"][i] for r in sat) for i, name in enumerate(M_NAMES)}
        print("\n- While Securing >= 0.95, range of each modulator: "
              + ", ".join(f"{k} {v:.3f}" for k, v in span.items())
              + f"; eps range {min(r['eps'] for r in sat):.4f}-{max(r['eps'] for r in sat):.4f}, "
                f"alpha range {min(r['alpha'] for r in sat):.4f}-{max(r['alpha'] for r in sat):.4f}")


def gridworld_section() -> None:
    usecase = Path(__file__).resolve().parents[2] / "external" / "metamo" / "usecase"
    sys.path.insert(0, str(usecase))
    import agents.metamo_agent as gw_agent  # noqa: E402
    import metamo.core as gw_core  # noqa: E402
    from agents.metamo_agent import MetaMoAgent  # noqa: E402
    from environment.gridworld import GridWorld  # noqa: E402

    print("\n### H12: MetaMo's own grid-world example\n")
    print("Its own loop (`usecase/agents/metamo_agent.py`): `consensus_transition`, then "
          "`project_to_safe_region`, then `blend_states`, with the motivational state reset every episode. "
          "Run (a) with its own environment stimuli and (b) with each scenario's stimulus swapped in for "
          "`build_stimulus`; both with and without the per-episode reset (episodes of 10 steps when resetting).\n")
    print("| Stimuli | Resets | Steps | Securing at 10, 20, 30, 40, 50 | First >= 0.95 | Steps >= 0.95 |")
    print("|---|---|---|---|---|---|")

    def run(stim_fn, resets: bool, steps: int = 50):
        from common import seed_all
        seed_all()
        agent = MetaMoAgent(seed=7)
        env = GridWorld(seed=7, max_steps=50)
        state = env.reset()
        agent.reset_episode()
        secs = []
        # the agent calls the build_stimulus it imported into its own module
        # (usecase/agents/metamo_agent.py:11), so swap it there and in metamo.core
        fake = (lambda env_state, mot=None: stim_fn())
        swaps = [] if stim_fn is None else [(gw_agent, "build_stimulus", fake), (gw_core, "build_stimulus", fake)]
        with patched(*swaps):
            for t in range(steps):
                action, alpha = agent.select_action(state)
                nxt, reward, done, info = env.step(action)
                agent.update(state, action, reward, nxt, done, info.get("event"), alpha)
                state = nxt
                secs.append(float(agent.mot.M[I_SEC]))
                if done:
                    state = env.reset()
                if resets and (t + 1) % 10 == 0:
                    agent.reset_episode()
        return secs

    def scenario_stimulus(sc: Scenario):
        counter = [0]  # fresh per run: step t of the scenario
        return lambda: stimulus_for(sc.outcome(_bump(counter)))

    cases = [("own environment", None)] + [(sc.name, sc) for sc in SCENARIOS]
    for label, sc in cases:
        for resets in (False, True):
            secs = run(None if sc is None else scenario_stimulus(sc), resets)
            first = next((i for i, v in enumerate(secs) if v >= 0.95), None)
            marks = ", ".join(f"{secs[i - 1]:.3f}" for i in (10, 20, 30, 40, 50))
            print(f"| {label} | {'every 10' if resets else 'none'} | 50 | {marks} | {fmt_first(first)} | "
                  f"{sum(1 for v in secs if v >= 0.95)} |")


def _bump(counter: List[int]) -> int:
    t = counter[0]
    counter[0] += 1
    return t


if __name__ == "__main__":
    main()
