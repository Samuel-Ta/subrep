"""Shared harness for the modulator-saturation experiments.

Nothing under bridge/ or external/metamo/ is edited. A component is changed
for one run by swapping it in with `patched(...)` and restoring it afterwards.
The runtime checks and the conservative fallback are observed by read-only
wrappers that call whatever implementation is currently in place.

Two drivers:
  run_bridge  -- the loop as the bridge runs it: MetaMoGovernor.step(outcome)
  run_metamo  -- MetaMo alone: MetaMoPseudoBimonad(OpenPsiAppraisal(),
                 MagusDecision()).step(state, stimulus, candidates)
"""

from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from bridge import budget as budget_mod  # noqa: E402
from bridge import governor as governor_mod  # noqa: E402
from bridge._loader import ensure_metamo_on_path  # noqa: E402
from bridge.governor import MetaMoGovernor, default_goal_vector  # noqa: E402
from bridge.protocol import SkillOutcome  # noqa: E402
from bridge.stimulus import StimulusValues, build_stimulus_values  # noqa: E402

ensure_metamo_on_path()

import category.bimonad as bimonad_mod  # noqa: E402
import core.config as config_mod  # noqa: E402
import dynamics.coherence as coherence_mod  # noqa: E402
import dynamics.stability as stability_mod  # noqa: E402
import openpsi.appraisal as appraisal_mod  # noqa: E402
from category.bimonad import MetaMoPseudoBimonad  # noqa: E402
from core.state import MotivationalState, Stimulus  # noqa: E402
from magus.decision import MagusDecision  # noqa: E402
from openpsi.appraisal import OpenPsiAppraisal  # noqa: E402

SEED = 7
M_NAMES = ("Valence", "Arousal", "Approach", "Resolution", "Threshold", "Securing")
I_THR, I_SEC, I_APP, I_VAL = config_mod.M_THRESHOLD, config_mod.M_SECURING, config_mod.M_APPROACH, config_mod.M_VALENCE
SATURATED = 0.95
RECOVERED = 0.60
RECOVERY_WINDOW = 15
THREAT_STEPS = 5
NUM_OBJECTIVES = 6  # only the scenario outcomes need a length; the harness is generic


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -- scenarios -----------------------------------------------------------------

NEUTRAL = SkillOutcome(delta_r=0.0, delta_n=np.zeros(NUM_OBJECTIVES), admitted=True, effort=0.0)
THREAT = SkillOutcome(delta_r=-0.5, delta_n=np.full(NUM_OBJECTIVES, -0.5), admitted=True, effort=0.5)
GOOD = SkillOutcome(delta_r=0.5, delta_n=np.full(NUM_OBJECTIVES, 0.2), admitted=True, effort=0.2)


@dataclass(frozen=True)
class Scenario:
    name: str
    outcome: Callable[[int], Optional[SkillOutcome]]  # None: the all-zero stimulus
    threat_steps: int = 0

    def is_threat(self, t: int) -> bool:
        return t < self.threat_steps


SCENARIOS = (
    Scenario("Zero", lambda t: None),
    Scenario("Neutral outcome", lambda t: NEUTRAL),
    Scenario("Constant threat", lambda t: THREAT, threat_steps=10 ** 6),
    Scenario("Good outcomes", lambda t: GOOD),
    Scenario("Threat then recovery", lambda t: THREAT if t < THREAT_STEPS else GOOD, threat_steps=THREAT_STEPS),
)
ZERO_VALUES = StimulusValues(novelty=0.0, conduciveness=0.0, risk=0.0, effort=0.0)


# -- swapping a component in and out --------------------------------------------------


@contextmanager
def patched(*swaps: Tuple[Any, str, Any]):
    """Swap (owner, attribute, replacement) in, restore on exit."""
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in swaps]
    try:
        for owner, name, new in swaps:
            setattr(owner, name, new)
        yield
    finally:
        for owner, name, old in reversed(saved):
            setattr(owner, name, old)


class Recorder:
    """Per-step outcomes of the runtime checks and fallback calls."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.lax: List[bool] = []
        self.contractive: List[bool] = []
        self.safe: List[bool] = []
        self.fallbacks = 0

    def take(self) -> Dict[str, Any]:
        out = {"lax_ok": all(self.lax) if self.lax else None,
               "contractive_ok": all(self.contractive) if self.contractive else None,
               "safe_ok": all(self.safe) if self.safe else None,
               "fallbacks": self.fallbacks}
        self.reset()
        return out


@contextmanager
def instrumented():
    """Observe the three runtime checks and the fallback in MetaMoPseudoBimonad.step,
    calling whatever implementation is currently in place."""
    rec = Recorder()
    lax_impl = MetaMoPseudoBimonad.check_lax_distributive_law
    con_impl = bimonad_mod.check_contractive_update_law
    safe_impl = bimonad_mod.is_in_safe_region
    fb_impl = MetaMoPseudoBimonad._apply_conservative_fallback

    def lax(self, *a, **k):
        ok = lax_impl(self, *a, **k)
        rec.lax.append(bool(ok))
        return ok

    def con(*a, **k):
        ok = con_impl(*a, **k)
        rec.contractive.append(bool(ok))
        return ok

    def safe(state):
        ok = safe_impl(state)
        rec.safe.append(bool(ok))
        return ok

    def fb(self, *a, **k):
        rec.fallbacks += 1
        return fb_impl(self, *a, **k)

    with patched((MetaMoPseudoBimonad, "check_lax_distributive_law", lax),
                 (bimonad_mod, "check_contractive_update_law", con),
                 (bimonad_mod, "is_in_safe_region", safe),
                 (MetaMoPseudoBimonad, "_apply_conservative_fallback", fb)):
        yield rec


# -- drivers ------------------------------------------------------------------------


def emitted_budgets(M: np.ndarray, coefficients=None) -> Tuple[float, float]:
    """The epsilon and alpha the bridge would emit for these modulators."""
    coefficients = coefficients or budget_mod.BudgetCoefficients()
    eps, alpha = budget_mod.compute_budgets(securing=float(M[I_SEC]), threshold=float(M[I_THR]),
                                           approach=float(M[I_APP]), coefficients=coefficients)
    return float(eps), float(alpha)


def _row(t: int, state: Any, rec: Dict[str, Any]) -> Dict[str, Any]:
    M = np.asarray(state.M, dtype=float)
    eps, alpha = emitted_budgets(M)
    return {"t": t, "M": M.copy(), "G": np.asarray(state.G, dtype=float).copy(), "securing": float(M[I_SEC]),
            "threshold": float(M[I_THR]), "eps": eps, "alpha": alpha,
            "pressure": float(stability_mod.boundary_pressure(state)), **rec}


StepWrapper = Callable[[Any, Any, Callable[[], Any]], Any]


def run_bridge(scenario: Scenario, steps: int, payoff_scale: float = 1.0, motive_scale: float = 1.0,
               state_hook: Optional[Callable[[int, Any, Any], Any]] = None) -> List[Dict[str, Any]]:
    """The loop as the bridge runs it. state_hook(t, previous, stepped) may
    replace the state after each step (used for blending and resets)."""
    seed_all()
    gov = MetaMoGovernor(payoff_scale=payoff_scale, motive_scale=motive_scale)
    rows = [_row(-1, gov.state, {"lax_ok": None, "contractive_ok": None, "safe_ok": None, "fallbacks": 0})]
    swaps = []
    if scenario.outcome(0) is None:
        swaps.append((governor_mod, "build_stimulus_values", lambda *a, **k: ZERO_VALUES))
    with patched(*swaps), instrumented() as rec:
        for t in range(steps):
            outcome = scenario.outcome(t) or NEUTRAL  # ignored when the zero stimulus is swapped in
            before = gov.state
            gov.step(outcome)
            if state_hook is not None:
                gov._state = state_hook(t, before, gov.state)
            rows.append(_row(t, gov.state, rec.take()))
    return rows


def initial_state():
    return MotivationalState(G=default_goal_vector(config_mod.NUM_GOALS),
                             M=np.full(config_mod.NUM_MODULATORS, budget_mod.MODULATOR_NEUTRAL))


def stimulus_for(outcome: Optional[SkillOutcome], payoff_scale: float = 1.0, motive_scale: float = 1.0) -> Stimulus:
    v = ZERO_VALUES if outcome is None else build_stimulus_values(outcome, payoff_scale=payoff_scale,
                                                                  motive_scale=motive_scale)
    return Stimulus(novelty=v.novelty, conduciveness=v.conduciveness, risk=v.risk, effort=v.effort)


def run_metamo(scenario: Scenario, steps: int, payoff_scale: float = 1.0, motive_scale: float = 1.0
               ) -> List[Dict[str, Any]]:
    """MetaMo alone, stepped directly with the stimulus the bridge would build."""
    seed_all()
    bimonad = MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())
    candidates = MetaMoGovernor()._default_candidates(config_mod.NUM_GOALS)
    state = initial_state()
    rows = [_row(-1, state, {"lax_ok": None, "contractive_ok": None, "safe_ok": None, "fallbacks": 0})]
    with instrumented() as rec:
        for t in range(steps):
            _, state = bimonad.step(state, stimulus_for(scenario.outcome(t), payoff_scale, motive_scale), candidates)
            rows.append(_row(t, state, rec.take()))
    return rows


# -- metrics --------------------------------------------------------------------------


def first_saturated(rows: Sequence[Dict[str, Any]]) -> Optional[int]:
    for r in rows[1:]:
        if r["securing"] >= SATURATED:
            return r["t"]
    return None


def summary(rows: Sequence[Dict[str, Any]], scenario: Scenario) -> Dict[str, Any]:
    steps = rows[1:]
    out = {
        "final_securing": steps[-1]["securing"], "final_threshold": steps[-1]["threshold"],
        "first_saturated": first_saturated(rows),
        "steps_saturated": sum(1 for r in steps if r["securing"] >= SATURATED),
        "final_eps": steps[-1]["eps"], "final_alpha": steps[-1]["alpha"],
        "eps_range": (min(r["eps"] for r in steps), max(r["eps"] for r in steps)),
        "alpha_range": (min(r["alpha"] for r in steps), max(r["alpha"] for r in steps)),
        "lax_fail": sum(1 for r in steps if r["lax_ok"] is False),
        "contractive_fail": sum(1 for r in steps if r["contractive_ok"] is False),
        "safe_fail": sum(1 for r in steps if r["safe_ok"] is False),
        "fallbacks": sum(r["fallbacks"] for r in steps),
        "max_pressure": max(r["pressure"] for r in steps),
        "recovery": None,
    }
    if 0 < scenario.threat_steps < len(steps):
        last_threat = scenario.threat_steps - 1
        after = [r for r in steps if r["t"] > last_threat]
        hit = next((r["t"] - last_threat for r in after if r["securing"] < RECOVERED), None)
        out["recovery"] = {"steps_to_below_0.60": hit, "recovers": hit is not None and hit <= RECOVERY_WINDOW,
                           "securing_at_end_of_threat": steps[last_threat]["securing"]}
    return out


def fmt_first(v: Optional[int]) -> str:
    return "never" if v is None else str(v)


def fmt_recovery(rec: Optional[Dict[str, Any]]) -> str:
    if rec is None:
        return "-"
    return f"{'yes' if rec['recovers'] else 'no'} ({fmt_first(rec['steps_to_below_0.60'])})"


SUMMARY_HEADER = ("| Scenario | Steps | Final Securing | Final Threshold | First >= 0.95 | Steps >= 0.95 | "
                  "Recovers (steps) | Final eps | Final alpha | Lax fails | Contr. fails | Safe fails | Fallbacks |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def summary_line(name: str, steps: int, s: Dict[str, Any]) -> str:
    return (f"| {name} | {steps} | {s['final_securing']:.3f} | {s['final_threshold']:.3f} | "
            f"{fmt_first(s['first_saturated'])} | {s['steps_saturated']} | {fmt_recovery(s['recovery'])} | "
            f"{s['final_eps']:.4f} | {s['final_alpha']:.4f} | {s['lax_fail']} | {s['contractive_fail']} | "
            f"{s['safe_fail']} | {s['fallbacks']} |")


def trace_line(rows: Sequence[Dict[str, Any]], key: str = "securing") -> str:
    return " ".join(f"{r[key]:.2f}" for r in rows[1:])
