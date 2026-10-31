# Hypotheses: modulator saturation in the MetaMo ↔ SubRep loop

Written and committed before any experiment code. Verdicts will come only from
measurements against the predictions below; if a result is surprising, a new
hypothesis is added rather than this file edited.

## The problem

In `demo/run_metamo_pipeline.py --steps 16 --seed 7`, Securing goes
0.500 → 0.850 → 1.000 by step 2 and Threshold follows; from then on the bridge
emits epsilon = 0 and alpha = 0.05 (its floor) whatever the outcome, including
steps where the threat has returned to 0.

These hypotheses come from reading the code listed in the task and that demo
output. One preliminary observation was made before this branch: driving
`MetaMoGovernor` with a constant neutral stimulus (risk 0, conduciveness 0.5,
novelty 0, effort 0) also took Securing to 1.0 within about five steps. It is
treated here as a lead, not as evidence; it is re-measured below.

## Shared definitions (fixed now)

- **Baseline:** the unchanged loop. `MetaMoGovernor` from `bridge/` for the
  bridge path; `MetaMoPseudoBimonad(OpenPsiAppraisal(), MagusDecision())`
  stepped directly for the MetaMo-only path. Same candidates, initial state
  and seeds.
- **Scenarios:** Zero (all four stimulus values 0); Neutral outcome
  (`SkillOutcome(delta_r=0, delta_n=0, admitted=True, effort=0)` through the
  bridge); Constant threat (a bad outcome every step); Good outcomes (a
  better-than-baseline outcome every step); Threat then recovery (5
  threatening steps, then 20 good ones).
- **Run lengths:** 20 steps for every run; 50 steps as well wherever slow
  drift is possible.
- **Saturated:** Securing ≥ 0.95. "First saturated step" is the first step
  where that holds, or "never".
- **Recovers:** in Threat then recovery, Securing falls below 0.60 within 15
  steps of the last threatening step.
- **One change at a time:** each experiment changes exactly one component,
  monkeypatched in and restored afterwards; everything else is the baseline.

## Family 1: the stimulus

**H1. The bridge's stimulus is too strong at the demo's scales.**
- Statement: with small `payoff_scale` / `motive_scale`, ordinary outcomes
  squash to near-maximal risk, and that drives Securing up.
- Prediction: True if, with the scales set 10x larger, Constant threat
  saturates at least 3 steps later and Good outcomes never saturate in 50
  steps. False if the Zero and Neutral scenarios still saturate within 20
  steps at both scale settings: then saturation does not need any stimulus.
- Experiment: rerun all scenarios with the scales x10 (and x100); nothing
  else changed.

## Family 2: the appraisal

**H2. The appraisal has no pull back toward neutral.**
- Statement: `OpenPsiAppraisal.appraise` only adds stimulus-driven deltas; with
  no restoring term, caution carried in from one step persists.
- Prediction: True if, applying only `appraise` repeatedly with a zero
  stimulus from Securing = 0.85, Securing stays above 0.75 after 20 steps.
  False if it returns below 0.60 within 20 steps.
- Experiment: drive `appraise` alone (no decision, no stability machinery) and
  record Securing.

**H3. The overgoal scaling amplifies caution.**
- Statement: `ind_scale = exp(g_ind - 0.5)` multiplies the Securing and
  Threshold deltas; if individuation rises, caution feeds itself.
- Prediction: True if setting `ind_scale = 1` moves the first saturated step
  in Constant threat by 2 or more steps, or removes saturation in any
  scenario. False if no scenario's first saturated step moves by more than 1.
- Experiment: monkeypatch the appraisal with `ind_scale` fixed at 1.

**H4. The sigmoid squash pushes modulators toward the bounds.**
- Statement: the final `1 / (1 + exp(-4 (M' - 0.5)))` squash makes saturation
  easy to reach and hard to leave.
- Prediction: True if replacing the squash with `clip(M', 0, 1)` removes
  saturation or delays it by 3 or more steps in any scenario. False if it
  moves no first saturated step by more than 1.
- Experiment: monkeypatch the appraisal with the squash replaced by a clip.

## Family 3: the stability machinery

**H5. The safe-region projection raises caution on every call.**
- Statement: `project_to_safe_region` adds +0.1 to Securing and Threshold
  unconditionally, even for a state already inside the safe region, and it
  runs on every step.
- Prediction: True if, with only that +0.1 removed (the goal projection kept),
  Zero, Neutral and Good outcomes never saturate in 50 steps and end with
  Securing within 0.50 ± 0.15. False if any of them still saturates within
  20 steps.
- Experiment: monkeypatch `project_to_safe_region` without the modulator
  increment, everywhere it is used.

**H6. Boundary caution raises caution.**
- Statement: `raise_boundary_caution` adds up to +0.25 near the safety
  boundary.
- Prediction: True if disabling it moves any first saturated step by 2 or
  more. False if boundary pressure is 0 on every step of every scenario (the
  state is never near the boundary) and nothing changes.
- Experiment: record boundary pressure per step; monkeypatch
  `raise_boundary_caution` to the identity.

**H7. Homeostatic damping is not involved.**
- Statement: `apply_homeostatic_damping` scales only the goal update, so it
  cannot affect the modulators.
- Prediction: True if disabling it changes no Securing value by more than
  0.01. False if Securing changes by more than 0.05 on any step.
- Experiment: monkeypatch `apply_homeostatic_damping` to return the update
  unchanged.

## Family 4: the runtime checks

**H8. The lax-distributive check fails routinely and triggers the fallback.**
- Statement: its tolerance (`LAX_DISTRIBUTIVE_DELTA = 1e-3`) is so tight that
  the check fails on most steps, and each failure runs the conservative
  fallback.
- Prediction: True if the check fails on at least 50% of baseline steps and
  making it pass moves the first saturated step by 1 or more in any scenario.
  False if it fails on under 10% of steps.
- Experiment: record pass/fail per step; monkeypatch the check to always pass.

**H9. The conservative fallback adds caution instead of only shrinking the step.**
- Statement: `_apply_conservative_fallback` averages the current and next
  states, then calls `project_to_safe_region`, which adds +0.1 again; each
  failed check therefore raises Securing further.
- Prediction: True if, with the fallback replaced by the plain average (no
  projection), the per-step Securing increase falls by at least 0.05 on steps
  where a check fails. False if it changes by under 0.01.
- Experiment: monkeypatch the fallback to return the average only.

**H10. The contractivity check is not involved.**
- Statement: `check_contractive_update_law` returns True away from the
  boundary band, so it never triggers the fallback here.
- Prediction: True if it passes on every step of every scenario. False if it
  fails on any step.
- Experiment: record pass/fail per step; monkeypatch it to always pass.

## Family 5: the loop structure

**H11. The bridge carries state over with no decay and no blending.**
- Statement: MetaMo's own grid-world example moves only part of the way to the
  target (`blend_states`) and resets every episode; the bridge applies each
  full step and never resets, so increments accumulate.
- Prediction: Partial expected. True if wrapping the bridge loop in
  `blend_states(current, step_output)` delays the first saturated step by 5
  or more in every scenario that saturates. False if it delays none by more
  than 1. Partial if it delays saturation past 20 steps but it still occurs
  by 50.
- Experiment: monkeypatch the governor's step to blend toward the step
  output; separately, reset the state every 10 steps.

**H12. MetaMo's grid-world example does not saturate because of how it drives MetaMo.**
- Statement: it uses `consensus_transition` (no runtime checks) plus blending
  and per-episode resets.
- Prediction: True if its own loop, run without resets, keeps Securing below
  0.95 for 20 steps but drifts upward over 50. False if it saturates within 20
  steps like the bridge.
- Experiment: run the grid-world transition path on the five scenarios, with
  and without episode resets.

## Family 6: what the bridge reads

**H13. Reading only three modulators makes the budgets blind once caution saturates.**
- Statement: epsilon and alpha depend only on Securing, Threshold and
  Approach; once the first two saturate, the other modulators can still
  respond to good outcomes, but the budgets cannot.
- Prediction: True if, after saturation in Threat then recovery, Valence or
  Approach moves by 0.10 or more while epsilon stays at 0 and alpha at its
  floor. False if Valence and Approach also stay fixed.
- Experiment: record all six modulators and the emitted epsilon and alpha
  per step.

## Added after the first results

Two results were not anticipated by H1-H13. Following the method, they become
new hypotheses, committed before their experiments; nothing above is edited.

**H14. With the projection's increment removed, a slower drift comes from the safety boundary.** (family 3)
- Observation that prompted it: in the H5 configuration, Zero at 50 steps
  still ends at Securing 0.732, and lax-distributive and contractivity
  failures appear late in the run, though none occur in the first 20 steps.
- Statement: the decision step keeps growing the goals, so the goal norm
  approaches `G_MAX`; inside the boundary band, `raise_boundary_caution` adds
  caution and the contractivity check becomes active.
- Prediction: True if, in the H5 configuration on Zero (50 steps), boundary
  pressure becomes greater than 0 at or before the first step where Securing
  exceeds 0.60, **and** additionally disabling `raise_boundary_caution` (one
  change relative to the H5 configuration) keeps Zero, Neutral and Good
  outcomes within 0.50 ± 0.15 at 50 steps. False if boundary pressure stays 0
  for the whole run.
- Experiment: record the goal norm and boundary pressure per step in the H5
  configuration; then add the identity `raise_boundary_caution`.

**H15. At small bridge scales, the novelty of good outcomes holds Securing down.** (family 1)
- Observation that prompted it: at the demo's calibrated scales (payoff 0.010,
  motive 0.036), Good outcomes did not saturate in 20 steps and Threat then
  recovery recovered, unlike every other configuration of the baseline.
- Statement: the appraisal's `-0.30 * benign_novelty` term is the only one
  strong enough to offset the per-step caution increment, and small scales
  make good outcomes highly novel.
- Prediction: True if, at the demo's scales with novelty forced to 0 in the
  stimulus, Good outcomes saturate within 20 steps. False if they still never
  saturate in 20 steps.
- Experiment: at the demo's scales, wrap the bridge's stimulus builder to set
  novelty to 0; nothing else changed.
