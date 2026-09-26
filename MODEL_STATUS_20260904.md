# Whole-model status — 2026-09-05

## Bottom line

The engineering foundation is approximately 75% complete, while a genuinely deliverable
Wet-to-Clean product is approximately 48% complete. These percentages are planning estimates, not
promotion metrics: the formal product status remains `usable_model: null` until physical-chain,
listening and locked-final gates all pass.

## Component status

| Area | Planning completion | Current evidence | Remaining blocker |
| --- | ---: | --- | --- |
| Rights, source registry and task-specific license gates | 90% | Fail-closed use-specific authorization and audited product sources | Final redistribution bundle audit |
| Family presence and Clean gate | 85% | The accepted nonlinear expert plus any-effect gate is now a separate hash-verified Drive-only package; external random-position software-chain Drive recall is 98.68%; one bounded hard-example pass fixed low-Drive/count-1 recall but was rejected | The rejected candidate retained only 38.10% weak-residual recall and exceeded the false-positive gate at 5.84%; a separate impact audit found 0/42 missed Drive stages safe to bypass, and a real multi-effect physical-hardware veto remains |
| Order-independent graph/expert contract | 95% | Four packaged experts pass all 24 known-control synthetic chain orders; Drive passes separately in every position and experts receive no order or neighbors | End-to-end physical chain validation and runtime headroom |
| Drive/nonlinear | 75% | Product-safe synthetic candidate passed automatic gates, bounded A/B listening, independent package loading and the four-effect/24-order chain gate; exact-control replay, Top-K Clean recoverability and weak-residual presence attempts were rejected | Explicit controls or a materially different observation/source-prior model, general physical Drive coverage, physical veto and locked-final |
| Amp/cab | Frozen (69% historical) | Generic EGDB v21 is 38.89%, Marshall v24 is 42.22%, and fixed EVH/Mesa/SM57 v30 reaches 78.65% internally and 75.80% on pair-disjoint reserved-v2; v31/v32 failed | Explicitly outside the active path: retain evidence but do not train, search/download Amp corpora or open Amp locked-final until the user reopens Amp |
| Dynamics, Echo and generic EQ | 95% | Three separate hash-verified development packages load independently and pass alongside Drive in all 24 chain orders | Physical-chain, listening where audible, and final-product validation |
| Reverb | Frozen (58% historical) | v3 and its full candidate hull are rejected; v4 capacity evidence does not pass observable deployment gates | Explicitly outside the active path: no training, Rochester download or locked-final opening until the user reopens Reverb |
| Modulation | 35% | Tremolo and Flanger safe subdomains pass | Broader Chorus/Phaser/Vibrato state/profile coverage; currently lower priority |
| Human and final acceptance | 18% | Drive v3 bounded listening accepted and its development package passes the 24-order gate; Reverb and Amp are frozen | Generic Drive physical listening, real multi-effect veto, untouched locked-final and package seal |

## Critical path

Items 1-3 below are retained frozen blocker records, not active work orders. With Amp and Reverb
paused, only item 4's generic Drive and non-Amp/non-Reverb chain acceptance remains active.

1. Keep Marshall v6 and the EGDB-PG v5/v7/v8b/v15/v21 evidence frozen. Fresh validation v1 rejected the
   first complex candidate at 15.28% complete passes, including low-gain spectral/high-band and
   aggregate dynamics failures. Consuming v1 as fit data raised old-development performance to
   23.61% for v7 and 29.17% for the bounded v8b dynamics refiner, but v8b did not improve the frozen
   calibration objective enough to qualify. Fresh validation v2 is reserved, absent locally and
   unopened. Category routing, deeper TCNs, selectors/shapers and compact U-Nets are closed. An
   8.27M-parameter, 3.09-second-receptive-field joint Demucs v13 also improves long calibration by
   only 4.13%, so capacity scaling alone is closed. Wet-parameter-estimated gray-box v14 reaches
   22.22%; allowing non-symmetric LTI phase in v15 raises this to 34.72%, while a frozen-base
   dynamics refiner selects epoch zero and is closed. A nine-tap research-only renderer then proved
   that deep stage supervision is useful without contributing product weights. The independently
   reinitialized v21 reaches 38.89% with all four aggregate metrics positive, but still fails the
   per-example and every-category/profile gates; v22 does not improve it. A pinned Apache-2.0
   rusty-amp renderer then supplied eight no-cabinet stages under a fail-closed product gate. Its
   v24 Marshall transfer reaches 42.22%, but fourfold renderer scaling regresses to 40.00%, while a
   single equal-budget EGDB-PG transfer v26 falls to 30.56% despite better calibration loss. Freeze
   v24 and v21; synthetic-stage scaling and cross-software transfer are closed. Require an
   independently licensed physical source/domain or self-verifying Wet parameter estimator.
   Inference must remain Wet-only and order-independent.
   EG-IPT now supplies 8,717 audited physical EVH 5150 III/Mesa V30/SM57 pairs for fit only. The
   first combined v27 learns that fixed source but transfers negatively: 36.11% on EGDB and 6.67%
   on Marshall. A v6/v24/v27 Clean-oracle reaches only 49.63%, so selector training is closed. Keep
   the source and prune v27. A profile-baked v30 then reaches 78.65% internally and 75.80% on
   pair-disjoint reserved-v2; retain it while seeking independent same/near-chain validation.
   The v31 transient residual regresses per-example coverage and is closed. A latest-work check then
   adapted EG-VAE's same-content/cross-tone idea to the existing eight-stage gray box without using
   its commercial-plugin data or large codec. The bounded v32 smoke improves Clean calibration loss
   by only 8.22%, reaches 16.67% unseen-profile passes and collapses its tone embedding; all replay
   identifiability gates fail and its checkpoint is pruned. Do not run a formal v32 or open fresh-v2.
   The bounded search found no product-safe independent EVH/Mesa/SM57 paired source, so v30 is
   explicitly data-blocked pending genuinely new evidence.
2. Freeze and reject Product4 Reverb frequency-profile v3 and its entire eight-candidate family.
   The new calibration-only per-example gate is frozen; v3 has 21/102 effective measurable rows,
   while the true per-bin convex-hull Clean oracle reaches the same 20.59%, so selector tuning cannot
   repair it. Runtime remains reusable engineering evidence; revealed rows may not be retuned.
3. Freeze the v4 analytic inverse as positive capacity evidence but reject its observable and learned
   deployment gates. Exact physical replay is non-identifying, and the new MPS dry-audio prior
   releases 12/91 measurable calibration rows while making five unsafe releases. Development stays
   sealed. A next Reverb attempt requires a different observation model; keep Rochester unopened.
4. The bounded known-control synthetic chain is now closed: Drive, Dynamics, Echo and generic EQ
   pass all 24 permutations under the unchanged gate, and four separate package manifests
   materialize and load with hash verification. Drive passes separately in every chain position.
   Do not mistake this for a blind or physical chain.
   Validate a real combined chain, then the locked-final split and final package. Only then may
   `usable_model` become non-null.

The one permitted hard-example Drive-presence pass is also closed. Its four MPS epochs selected
epoch 4 using calibration only. Development count-1 and low-Drive recall reached 78.26% and 85.71%,
but weak-residual recall stayed at 38.10% and false positives reached 5.84%. No rejected checkpoint
was retained. Do not turn this into an epoch or weighting sweep; the next Drive evidence must change
the observation/evidence domain rather than reuse the revealed synthetic split.

The alternative hypothesis that missed Drive stages are harmless is closed too. On the unchanged
packaged detector, 42 synthetic nonlinear development positives were missed and 0/42 passed the
frozen no-op safety gate. Their median gain shift was 8.73 dB and median spectral/high-band/transient
errors were 0.1913/1.7522/0.5147. Keep `negative_may_bypass: false`.

The current Drive Graybox route is now closed for a stronger reason than knob mismatch. The frozen
Top-K Clean audit kept the 324-candidate grid and ranked only by Wet replay error; truth and Clean
were scoring-only. True controls pass on 8/12, the all-grid Clean oracle also passes only 8/12,
Top-12 covers 5/12 and replay Top-1 covers 1/12. Because no candidate restores the remaining 4/12,
training a selector would optimize ranking without fixing inverse capacity. This does not invalidate
the accepted explicit-control expert or any of the 24 order-independent chain results.

The project is therefore beyond prototype architecture, but not yet at a complete usable model.
With Reverb and Amp frozen, active remaining work is concentrated in generic Drive physical
generalization and real-chain/final acceptance—not in adding more effect families, training a
selector over the rejected Graybox bank, or repeating rejected model sweeps.

## Complete-model count

- Complete end-to-end usable Wet-to-Clean model: **0**. The formal value remains
  `usable_model: null`.
- Strong independent mechanisms with current development evidence: **4** separately packaged
  experts (Drive, Dynamics, Echo and generic EQ). Drive remains limited to the synthetic
  exact-control domain despite passing listening and all 24 synthetic chain orders.
- Reverb and Amp experts meeting the whole product gate: **0**. Reverb v3 and its candidate family
  are rejected; Amp remains diagnostic. Neither is a completed product model.
