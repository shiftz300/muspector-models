# Handoff

## Goal

Build a maintainable, multi-family Wet-to-Clean restoration foundation. It is
not RAT-specific. The client eventually loads independently maintainable family
experts and device adapters, reverses arbitrary effect graphs, supports repeated
families and unusual orders, and preserves explicit Clean profiles.

The current phase is model/data work only. Do not connect UI and do not access
physical audio input/output devices.

## Current verified state (2026-09-02)

### Active family priority

The user explicitly narrowed near-term work on 2026-09-02: prioritize
Drive/Amp and Reverb; do not spend substantial training or resource-search
budget on modulation for now.  The accepted Tremolo/Flanger safe subdomains
remain frozen evidence, and the rejected Chorus/Phaser/Vibrato/Pitch screens
must not be restarted unless the user changes this priority or materially new
licensed aligned data/state appears.

Drive and Amp are not aliases.  Drive v3 remains the accepted generic
nonlinear safe subdomain described below.  Amp is a separate independent
single-effect expert whose internal stages may include preamp nonlinearity,
tone stack, power amp and an explicit cabinet/profile stage, but it receives no
graph order or neighbouring-effect input.

The CC BY 4.0 Marshall JVM410H OD1 archive (`marshall-jvm410h`, Zenodo
7970723) was downloaded and verified at 4,369,237,770 bytes, MD5
`4d08dac89b44f618f2d6c2a69739e104`, then ZIP64/CRC checked before extraction.
The actual archive contains 37 aligned input/preamp/speaker-output settings,
not only the paper's 21 seen B/M/T settings: 25 settings are fit/calibration,
Gain 1 and 8 are development-only, nine author-unseen B/M/T settings remain
locked-final and unopened, and the directory labelled Gain 6 is quarantined
because every contained filename says Gain 4.  The source can establish only
one named speaker-output Amp subdomain; it contains no cabinet or microphone.

Three MPS-trained Marshall Amp mechanisms are retained as rejected diagnostic
evidence, with no demo and no promotion.  Their development pass fractions are
18.52% (FIR + local TCN, epoch 9, CPU RTF 0.02875, SHA-256
`2f5c44366a54d1f67bf1abb495a401441da18e5f2732c45b7ce39ee891617aad`),
27.41% (multiplicative dynamics, epoch 16, CPU RTF 0.06167, SHA-256
`2e45cfef82cdb7b7d6ce0497d11c8fbe0850090fe32d78d56fd6d51ba0e3e85f`),
and 32.59% (frame dynamics, epoch 20, CPU RTF 0.00827, SHA-256
`7b39ea4df6a6da31716a8562466cb2a622eaaac616663539fc42574a4d406543`).
All substantially recover spectrum/high-band energy but fail the frozen
per-example transient/dynamics and coverage gates.  A single calibration
setting in v2 passed individually but reached only 60% on development, below
the 80% selective-profile minimum.  Locked-final was not opened and
`usable_model` remains null.

Guitar-TECHS (Zenodo 14963133) is an additional CC BY 4.0 fixed Amp+cab+mic
source.  P1 and P2 passed the corrected pair audit with fit-derived lags of 35
and 37 frames and are eligible for a future dedicated named-profile model.
P3 had already been reserved as locked-final in the local registry, but its
audio was mistakenly decoded during the first alignment audit.  No model,
threshold or selection used P3; it is permanently marked contaminated, its
local audio and archive were deleted, and it must never be used as final
evidence.  P1/P2 are audited but not yet trained.

The deterministic Reverb follow-up removed the rejected v3 neural residual and
screened bounded frequency-domain profile inverses.  On development, a 4x/6x
gain ceiling could individually pass 22 of the 51 examples rejected by the
exact causal path, but no runtime-observable safety selector generalized.  A
calibration-perfect 25/25 selector fell to 7/17 on disjoint development RIRs
and added reverb on two accepted examples.  It is rejected and must not be
retuned on development.  Evidence is retained at
`runs/foundation/product3-ambience/ambience/deterministic-base-screen.json` and
`deterministic-safety-selector.json`; the existing exact stable-profile
boundary remains the promoted Reverb path.

`runs/foundation/product1` is invalid diagnostic evidence, not a baseline. Its
source realization, Delay supervision, context claim, oracle, metric version
and aggregate-only acceptance were re-audited and found unsound.

The replacement evidence is `runs/foundation/product2`, but it is deliberately
not promoted as an end-to-end usable model:

- `runs/foundation/product3-safe/nonlinear` is the newer automatically accepted
  nonlinear candidate.  It reverses the repository renderer's internal stages
  as bounded level removal, a 257-tap regularized low-pass inverse, analytic
  tanh/atan/cubic inversion and a small local TCN.  It receives no graph order
  or neighbouring-effect input.  Product-only calibration selected epoch 6;
  all three shape strata and both development sources pass, 106/120 examples
  pass individually, and median spectrum/high-band/transient/dynamics
  reductions are 91.44%/99.37%/99.54%/96.87%.  Ordinary-CPU inference is
  0.0288 RTF.  Checkpoint SHA-256 is
  `cc0f9a320345e870b0e2ec56ef331419becfa296b642b620b65df3593c4df75b`.
  The first unbounded `product3` diagnostic is retained because Atan/EGFx
  failed the no-new-clipping gate; it was not promoted.  The user accepted the
  two compact v3 Wet-to-Restored A/B files on 2026-09-02 for the current
  synthetic, exact-control development scope.  This does not establish blind
  controls, named/physical pedal fidelity or multi-effect physical-chain
  acceptance.  Physical-chain veto and locked-final have not passed, so
  `usable_model` remains null.
- Human acceptance artifacts now live only under `demos/`.  The current v3
  full audit package is `demos/drive-product3-safe-demo-20260902-v1`; it contains
  aligned Wet/v2/v3/Clean files, exact controls, attribution and checksums.  The
  compact human package is `demos/drive-product3-safe-ab-demo-20260902-v1`, and
  its bounded acceptance record is
  `demos/drive-product3-safe-human-acceptance-20260902.json`.

- Nonlinear is a trained single-effect expert: an analytic tanh/atan/cubic
  inverse feeds a local TCN residual and uncertainty head. Product-only model
  selection chose epoch 9. Aggregate, all three shape strata and both held-out
  product sources pass development gates. Checkpoint SHA-256 is
  `feb43c7e611f5051ef5c89f3ea6906f4e922a1f3d80ff70b5fcfbb57554d9e62`.
- Dynamics is an exact stateful compressor inverse. It carries only that
  effect's envelope state and passed 48/48 development examples.
- Echo is an exact causal feedback-delay inverse and passed 48/48 development
  examples.
- The measured ordinary-CPU sum for one active nonlinear + dynamics + echo
  graph is 0.441 RTF. This passes 0.5 but has little headroom; repeated effects
  and future ambience still need graph-level budgeting and likely a compiled
  dynamics implementation.
- Known-profile ambience has an exact causal inverse on the stable 54.17%
  subset and explicitly abstains on unstable RIR profiles; blind ambience is
  rejected. No locked-final, research/physical veto, human listening or Demo
  has run, so `usable_model` remains null.
- `runs/foundation/product3-ambience/ambience` is a retained MPS-trained v3
  diagnostic, not a promoted extension.  It selected epoch 8 and reached
  80.83% development coverage by combining exact profiles with a learned
  fallback, but the fallback regressed transient/dynamics quality and added
  reverb on 11.1% of its accepted examples.  A post-hoc risk classifier could
  make the fallback precise only by reducing total coverage to roughly the old
  exact-only 54% boundary.  Do not continue seed or capacity sweeps from this
  checkpoint.  SHA-256 is
  `c7890b248725a303fedfd4762369fe340f608c95d52b9c1109087720f97721bf`.
- `runs/foundation/product3-spectral/spectral` is an automatically accepted
  generic three-band EQ inverse, not a named-device clone.  It uses an explicit
  12 dB inverse-gain ceiling, receives no graph order or neighbouring-effect
  input, and passed every calibration/development example across both sources
  and all low/mid/high-dominant strata.  MPS training tested an optional local
  residual, but calibration selected epoch 0: the bounded analytic inverse was
  better than every learned correction.  Ordinary-CPU RTF is 0.00699 and
  checkpoint SHA-256 is
  `debbe0425bc63b97aa2abbf8c8dbcee70e1ff0fdb204a290e693cfeed4dd3575`.
  Per user instruction, no listening demo was generated; proceed directly to
  the next family after automatic quality acceptance.
- `runs/foundation/product3-modulation-safe/tremolo` is an accepted safe
  subdomain for generic sine/triangle Tremolo.  It estimates the hidden LFO
  phase from Wet harmonics using only this effect's rate/depth/waveform controls
  and abstains by passing Wet through when rate is below 3.2 Hz or the
  calibration-selected harmonic-residual ratio exceeds 0.16.  Development
  coverage is 36/72 (50%); every admitted example passes, both sources and both
  waveform strata pass, median trajectory correlation is 0.99955, and median
  spectrum/high-band/transient/dynamics reductions are
  97.04%/97.02%/96.73%/96.35%.  Ordinary-CPU RTF is 0.000416.  The zero-parameter
  checkpoint SHA-256 is
  `99881d2f462a4945db18751362a5b821949227de163236c8dc239b72b7e40ac4`.
  The 12-epoch learned candidate under `product3-modulation` remains rejected:
  DAFx continuous phrases and slow rates failed trajectory coverage.
- `runs/foundation/product3-flanger-safe/flanger` is an accepted conservative
  generic Flanger subdomain.  A Wet-only real-cepstrum moving-comb search
  estimates the hidden LFO phase, then a causal fractional-delay recurrence
  performs the inverse.  It receives only this effect's controls and no graph
  order, neighbour, Clean or hidden phase.  The final admitted domain is
  `1.0 <= rate < 2.3 Hz` and `0.25 <= mix <= 0.38`; everything else abstains and
  passes Wet through.  A fresh 120-example development-control challenge on
  the same held-out source groups but new controls/crops admitted 34 examples
  (28.33% coverage), and all 34 passed across both sources.  Median
  spectrum/high-band/transient/dynamics reductions are
  77.32%/91.43%/95.85%/93.23%; ordinary-CPU reference RTF is 0.0216.  The
  zero-parameter checkpoint SHA-256 is
  `b299432d3e47f3e315be4aa234a9c5af6ce28eda18ee8639ea9b50a13e1e417a`.
  The initial development control draw exposed a fast-rate interaction and was
  not reused as the final challenge; this adaptation is recorded in metrics.
- Phaser remains rejected after a second, materially different product-safe
  mechanism screen.  `remix/phaser3.py` implements the zero-feedback subset of
  a Stone-style four-stage all-pass topology from the pinned CC0/BSL upstream
  commit `f057ea5c1e368d6cd937f4e359a9514ac72fe902`; the project admits its CC0
  grant and records it as `stone-phaser-cc0`.  Wet-only analysis-by-synthesis
  reduced the old median hidden-phase error from roughly 1.6 radians to about
  0.1 radians and the optimized Python reference met the CPU budget, but phase
  confidence did not guarantee audio safety on continuous phrases.  The first
  margin-only 120-example challenge reached 53.33% coverage at 0.396 RTF but
  still regressed individual spectrum/high-band quality and clipping.  Its
  evidence is retained as
  `runs/foundation/product3-phaser-safe/phaser/margin-only-diagnostic.json`.
  A cross-frequency phase-consensus plus output-clipping gate was then selected
  only on product calibration and tested on a fresh seed.  It admitted 24/120
  examples (20% coverage), below the frozen 25% minimum, and individual
  regressions remained despite admitted median/p90 phase errors of 0.075/0.143
  radians.  Final evidence is
  `runs/foundation/product3-phaser-safe/phaser/metrics.json`; the checkpoint is
  diagnostic only.  Stop Phaser neural/capacity sweeps from this Wet-only cue.
- Chorus is not promoted.  Both the first Wet-STFT trajectory TCN and its
  known-rate-clock variant failed calibration; the latter selected epoch 0
  after five MPS screening epochs.  Only the compact failed metrics are kept at
  `runs/foundation/product3-chorus-screen/chorus/metrics.json`; temporary
  checkpoints were deleted.  A subsequent real-cepstrum moving-comb phase
  search produced deceptively strong development trajectory correlation
  (median 0.9926, 72/72 above 0.75), but its small phase errors became
  multi-sample fractional-delay errors and regressed spectrum after recursive
  inversion.  A fit-trained ridge bias correction still passed only 36.1% of
  development examples.  Even an oracle phase-error gate had only 12.5%
  coverage at 100% individual audio precision; at 18.1% coverage it already
  failed.  Full evidence is
  `runs/foundation/product3-chorus-screen/chorus/physics-screen.json`.  Stop all
  Wet-only Chorus trajectory/capacity/risk-model sweeps.  The next Chorus path
  requires explicit per-effect phase/state from an adapter or materially new
  licensed aligned device data.
- The remaining modulation/pitch screen is frozen at
  `runs/foundation/product3-modulation-next-screen.json`.  A repository-owned
  one-pole all-pass Phaser pilot could not identify hidden sweep phase from
  Wet-only STFT movement: median absolute phase error was 1.71 radians on
  calibration and 1.62 on development.  Do not start a neural Phaser sweep
  from this cue.  Mono Vibrato is even less identifiable because pure
  time-varying delay preserves magnitude; it needs explicit oscillator state or
  a reference.  Pitch/octave remains blocked by missing product-safe aligned
  physical coverage and the required ambiguity contract.  Current legal data
  therefore supports only the accepted Tremolo and Flanger safe subdomains for
  modulation; broader coverage needs new licensed pairs or explicit adapter
  state.
- The current online restoration-resource audit is
  `runs/foundation/product3-restoration-resource-search.json`.  No new aligned
  single-effect audio entered product gradients.  ToneTwist Chorus is
  CC-BY-NC-4.0, IDMT is CC-BY-NC-ND-4.0, EGFx normalized Wet remains
  family/parameter/veto-only, Semantic Timbre has composite semantic targets,
  and all 47 NCSOFT Designed Vocalizations presets contain at least two DSP
  modules.  MIT `freefx` is only a software-renderer reference.  The sole new
  product-compatible mechanism source was the CC0 Stone topology above, whose
  inverse candidate failed promotion.  Do not download the 37.1 GB NCSOFT or
  199 GB Semantic Timbre audio for the independent experts.
- Blind presence now has a development-accepted, external-software-veto-passed
  independent-family stack under
  `runs/foundation/product2/blind2-independent-family-stack`. It combines the
  compact shared Wet-only backbone for echo/ambience, independently fine-tuned
  nonlinear and unknown family experts, and an independently trained any-effect
  Clean gate. None receives order, controls or neighboring-effect input.
  Development Micro/Macro-F1 is 0.8814/0.8817, Clean FPR is 0.79%, and all four
  recalls pass at 0.8019/0.8357/0.9403/0.8395. The complete 1,716-example,
  52-group CC BY 4.0 random-position software-chain development veto also
  passes: Micro/Macro-F1 0.9020/0.9022, Clean FPR 0%, multi-effect Macro-F1
  0.9172 and every family recall at least 0.9159.
- The new gate is `family -> independent family heads -> any-effect Clean veto`;
  the external file gate averages all windows so one local Clean spike cannot
  fabricate an effect. The gate threshold was selected from internal and
  external calibration groups only with a Wilson 95% Clean-FPR upper-bound
  requirement. Tele locked-final remains sealed. This is still not a usable
  restoration product: the real multi-effect physical-hardware veto and exact
  Wet/Candidate/Original listening pass have not run, so `usable_model` remains
  null.
- The Blind data nuisance path no longer clips audio: clipping is nonlinear and
  had created false-negative labels. The admitted DAFx25 random-position corpus
  contributes presence bits only; its author order JSON remains isolated in the
  separate order package. Failed Blind weights and duplicate calibration copies
  were deleted while all metrics/diagnostic JSON was retained.
- Wet-only absolute point-knob regression is also rejected as unidentifiable;
  the failed single-effect family/knob artifacts were removed while metrics
  were retained.

Every inverse expert is order-independent: it maps one effect instance to its
immediate predecessor and may receive only that effect's Wet, controls, state
and optional explicit adapter/profile. Chain order and neighboring effects are
forbidden expert inputs. The separate graph package alone chooses a graph and
invokes these experts in reverse topology. This supersedes the old
`upstream-effect context` wording in `cycles/restore.json`.

MPS was tested outside the sandbox on this Apple Silicon host. For the old
product2 batch shapes, CPU was faster: nonlinear 0.381 s versus 1.399 s and
dynamics 0.366 s versus 3.207 s per representative training batch.  The new
product3 nonlinear architecture reverses that result: a representative v3
training batch measured 0.0613 s on CPU versus 0.0173 s on MPS, so v3 training
uses MPS while its deployment runtime remains measured on ordinary CPU.  Never
carry an accelerator decision across architectures without re-benchmarking.

The 2D Blind ResNet is the counterexample: its complete log-Mel training batch
measured 0.0496 s on MPS versus 0.1538 s on CPU, so Blind training uses MPS.

## Critical correction

Training was stopped before it began because research permission had been mixed
with product-weight permission. The repository now uses a conservative
three-tier boundary:

1. isolated research/evaluation;
2. gradients for redistributable product weights;
3. source or rendered audio redistribution.

Missing or ambiguous permission is denied. Research-only sources may veto a
candidate through external metrics, but may not contribute gradients,
calibration thresholds, model selection, distillation, or a shared learned
encoder to product weights.

Verified current policy:

- ASRNN RAT/DFZ/CS-3: CC BY-NC 4.0; isolated noncommercial benchmark only.
- Apple AU captures: internal research only until written output/weight clearance.
- Spotify Pedalboard: GPL-3.0 tool; keep research-only until render-output and
  weight implications are reviewed.
- ToneTwist, RemFX, pOD-set: blocked or research-only because rights are mixed,
  noncommercial, or missing.
- Does it Chug?: three-player DI source performances are technically relevant,
  but the Hugging Face dataset has no license tag and only generated schema
  metadata in its README, the
  GitHub repository has no LICENSE, and third-party track permission is not a
  product-training grant. Keep metadata-only until the audio license is explicit.
- GADA: the project page confirms three guitarists and electric-guitar DI but
  publishes no audio license. A Range-only inspection of the public
  1,380,776,731-byte ZIP parsed all 4,011 central-directory entries and found no
  LICENSE, LICENCE, COPYING, RIGHTS, TERMS, README or CITATION file. The public
  GitHub issue list also contains no license clarification; keep it blocked.
- GuitarJam: CC0 product Clean source.
- Guitar-TECHS: CC BY 4.0 electric-guitar direct-input plus fixed Amp/cab/mic
  pairs. P1/P2 are development-visible product candidates. P3 was accidentally
  decoded by an alignment audit before training, is permanently contaminated,
  and has been removed locally; it is forbidden from model selection and final
  evidence.
- EG-IPT: CC BY 4.0 raw-DI subset only. Its 8,720 verified mono PCM24 files are
  4.730397 hours from one player, one Gibson SG and one recording session, so
  the complete domain is fit-only and down-weighted.
- Longitudinal Guitar String Ageing: CC BY 4.0 raw electric-guitar DI with no
  amplifier, effect or microphone in the signal chain. The 56 verified mono
  WAV files total 10.746655 hours across two performers, two guitars and two
  interfaces. Both complete performer domains are fit-only and down-weighted:
  each player repeats the same fixed routine on 28 days, performer 2 has a
  documented intermittent high-frequency artefact, and the released CSV files
  omit the segment boundaries claimed by the paper.
- FreePats FSBS Direct and Karoryfer Emilyguitar: CC0 direct sampled instruments,
  fit-only and strongly down-weighted. Their 0.636 and 0.253 hours improve
  pitch/velocity coverage but do not count as independent-player performance
  hours.
- Multimodal Electric Guitar Data: CC BY 4.0. The verified 3.6 GB part2 archive
  contributes 332 mono PCM16 WAV files totaling 17.588686 hours; five clipped
  or near-silent files are excluded, leaving 327 sampled files. The paper
  reports 31 complete players. Because the audio is a fixed Roland amplifier
  line output with undocumented settings, it is never raw DI or a restoration
  target. It is fit-only, low-weight presence augmentation: repository effects
  are added in random order and only their positive labels receive gradients;
  every absent baseline label is masked unknown.
- DAFx25 "Guitar improvisations with chains of five effects": CC BY 4.0. Its
  random-position Wet files are product-eligible for family-presence training
  and external software-chain vetoes, its Clean subset is product-eligible, and
  author order JSON remains isolated in the separate order package. Archived
  Wet is still excluded from restoration gradients.
- EGFxSet: CC BY 4.0; family/parameter and Clean subset are product-eligible
  with attribution, but normalized Wet is excluded from sample-faithful
  restoration gradients.
- Aachen chapel RIR: CC BY 4.0 product temporal training with attribution,
  paired ephemerally with product-eligible Clean programs.
- Repository-owned DSP: Apache-2.0 product renderer.

This is an engineering policy, not legal advice. Do not loosen it without
written permission or a deliberate legal review.

## Implemented

- `remix/RESTORE.md`: family-neutral architecture, Clean profiles, arbitrary
  graph order, runtime and license boundaries.
- `remix/restore.py`: generic `Profile`, `Stage`, and `Graph`; repeated family
  slots and reverse execution.
- `cycles/foundation.json`: license-gated multi-family foundation plan.
- `remix/data_sources.json`: per-source research/weight/audio scopes.
- `remix/license_gate.py`: fail-closed executable source authorization and
  attribution generation.
- `remix/foundation_data.py`: read-only ASRNN/Apple/Aachen audit. Its 4,862
  materialized pairs are explicitly research-only; product training count is 0.
- `remix/product_data.py`: product-only in-memory pairs from CC0/CC BY Clean
  subsets, repository-owned nonlinear/dynamics/temporal DSP, and Aachen RIR.
  Research source IDs are structurally rejected.
- `remix/foundation_model.py`: equal-parameter independent experts with
  mechanism-specific fixed views and disclosed input-context budgets.
- `remix/train_foundation_product.py`: product-only baseline trainer; syntax
  tested and executed only as a non-promoted quick diagnostic.
- `remix/audit_product_coverage.py`: product-only in-memory renderer, oracle,
  identity, metric-stability and history-context audit.
- Tests added: `test_license_gate.py`, `test_foundation_data.py`,
  `test_product_data.py`, `test_foundation_model.py`,
  `test_train_foundation_product.py`, `test_restoration_quality.py`.
- `remix/physical_chain_data.py` and
  `remix/evaluate_physical_chain_presence.py`: fail-closed, MPS-capable frozen
  physical-chain veto. The manifest accepts unordered family sets only and
  rejects order/position metadata, single-effect Wet rows, path escape and
  missing product-evaluation authorization.
- `remix/PHYSICAL_CHAIN_VETO.md`: rights-cleared self-capture protocol and
  minimum evidence contract. The veto cannot fit, calibrate, select or change
  the frozen stack.

Latest completed full test run after the independent Blind family stack:

```text
382 remix tests passed; 1 skipped
80 external tests passed
```

The live license report is `runs/foundation/license/audit.json` and passes. The
corrected research inventory is `runs/foundation/data/audit.json`. No model
training has been formally promoted and no demo was generated. The product-only
quick diagnostic is retained under `runs/foundation/product1`; its coverage
audit confirms the oracle contract but rejects all three candidate experts and
blocks dynamics/temporal training on insufficient history context.

## Rejected legacy models

Do not treat any previous restoration checkpoint, package, metric summary or
Demo as a usable starting model. The current product state is deliberately
`usable_model: null` in `cycles/restore.json`.

The user's listening veto is authoritative and must remain attached to the old
results:

- restoration was audibly incomplete;
- reconstructed audio remained muddy and still sounded driven;
- the old Drive character did not resemble RAT and was generically blurred;
- some supposed restoration increased audible reverb;
- the demonstrations sounded nearly unchanged at first, then closer inspection
  exposed incorrect residual effect and added artifacts.

One old Demo was mislabeled: its “Drive” source was a generic DAFx overdrive,
not a RAT capture. It must never be described as RAT restoration or used as a
named-device demonstration.

The rejected automatic evidence is also explicit:

- old temporal `model20` added reverb on 55.4% of eligible examples and failed
  every aggregate temporal gate; its Demo audio was removed while audit evidence
  was retained;
- retained historical RAT `model11` achieved only 0.51% median transient
  improvement and 5.94% per-example pass coverage;
- four later RAT diagnostics were all rejected:
  - model1: -6.06% transient improvement, 0% pass coverage;
  - model2: +1.74% transient improvement, 8.91% pass coverage;
  - model3: +1.96% transient improvement, 18.81% pass coverage;
  - model4: +29.0% spectrum, +47.9% high-band, +33.3% dynamics, but only +5.40%
    transients and 26.73% pass coverage.

Model4 proves why average subgroup gains are insufficient: it improved several
aggregates while still failing the audible transient/per-example contract. The
compact deterministic temporal-residual candidate family hit the stop rule and
is unsuitable as the product inverse architecture. Do not continue its seed or
capacity sweep, distill it, rename it, package it, or use it to initialize the
new product experts.

DAFx Wet remains excluded from restoration gradients. Its random-position Wet
is admitted only to the separate presence package and software-chain veto; its
author order metadata remains isolated in the order package. ASRNN/Apple and
the legacy checkpoints may be used only as isolated post-freeze evidence under
their license scopes. They may not contribute gradients, calibration, model
selection, shared encoders or marketing demonstrations.

Any future Demo must be a licensed musical phrase with an honest
Wet/Candidate/Original triple, exact effect/device labeling, and automatic plus
human gates already passed. Never create a listening Demo merely because a
training run completed.

## Immediate work

1. Freeze the accepted-development Blind stack and its calibration. Do not tune
   thresholds or heads against development again. The exact component hashes,
   decision flow and remaining blockers are in
   `runs/foundation/product2/blind2-independent-family-stack/stack.json`.
2. Keep restoration Clean separate from family-presence baselines. Independent
   raw-DI Clean remains about 25.25 hours across about seven players, while the
   Wet-only presence model additionally has 17.200143 admitted hours from 31
   published complete amplifier-line players. This closes the presence-player
   diversity gate without pretending the amplifier output is raw DI.
   EG-IPT and Longitudinal Guitar String Ageing raise real-performance Clean
   from about 9.77 to about 25.25 hours, so the 15-hour minimum is closed, but
   only about seven clearly independent real-performance players are now
   represented and the longitudinal hours repeat one fixed routine. The two
   CC0 sampled instruments add 0.889 hours of note coverage without satisfying
   the real-performance or player-diversity gate. Guitar-TECHS P1/P2 remain
   development-visible only; P3 is contaminated and permanently excluded.
   Multimodal Electric Guitar Data is downloaded and positive-only admitted as
   described above; it remains forbidden from restoration targets and all-zero
   Clean supervision.
   Does it Chug? and GADA would each add three DI players, but both remain
   blocked because no product-compatible audio license is published.
3. Add a real multi-effect hardware veto domain. The software-chain veto now
   passes, but EGFx hardware anchors are still single-effect and cannot prove
   physical overlap robustness. A physical veto may reject the frozen stack but
   may not update its weights or calibration. The public-resource search found
   no compatible released physical-chain corpus; rejected candidates and their
   rights/hardware reasons are recorded in
   `runs/foundation/product2/resource_search.json`. Follow
   `remix/PHYSICAL_CHAIN_VETO.md` for a new rights-cleared capture.
4. After the physical veto passes, run exact licensed Wet/Candidate/Original
   listening. Human rejection remains authoritative. Only then may Tele
   locked-final be opened once and the stack considered for promotion.
5. Keep knob point estimates abstained. Introduce intervals/top-k candidates
   only after a forward-replay identifiability audit proves they constrain the
   inverse rather than fabricate precision.
6. Only after all prerequisite automatic gates pass may isolated research veto
   and human Wet/Candidate/Original listening run; neither may update weights or
   calibration, and human rejection still blocks promotion.

## Whole pipeline

Keep the complete workflow visible even while working on one phase. The product
must not collapse it into one opaque model.

### Build path

1. `rights`: verify each source independently for research, product gradients,
   and audio redistribution; deny unknown rights.
2. `clean`: admit licensed Clean performances and assign whole performance,
   player/device/session groups to non-overlapping splits.
3. `pair`: retain the exact immediate predecessor for every effect stage;
   generate product Wet pairs in memory with repository-owned DSP or explicitly
   approved captures. Never write a convenience Wet corpus by default.
4. `audit`: verify geometry, sample rate, finite samples, alignment, latency,
   controls, group leakage, preprocessing and attribution before training.
5. `family`: train the effect-family recognizer separately from restoration.
6. `order`: train forward-order recognition separately; keep top-k graphs and
   abstain on small margins. Genre never hard-codes order.
7. `knobs`: estimate controls/device semantics in a separate package. A named
   pedal is an adapter/validation domain, not the whole architecture.
8. `forward`: maintain effect-family/device forward clones for replay and
   identifiability checks. A forward clone passing does not prove inverse quality.
9. `restore`: train equal-budget independent inverse experts per mechanism first:
   nonlinear, dynamics, spectral, modulation, echo, ambience and pitch. Do not
   force a shared learned encoder until an explicit no-negative-transfer test.
10. `calibrate`: select thresholds using product-eligible calibration data only.
    Research-only evidence cannot select models or thresholds.
11. `gate`: evaluate each stage against its immediate predecessor, then the full
    chain. Require spectrum, high band, transients, dynamics, phase, pitch/onset,
    clipping, mechanism-specific residuals, worst-tail coverage and CPU runtime.
12. `veto`: freeze the product candidate, then run isolated physical/research
    benchmarks. They may reject it but cannot update or choose it.
13. `listen`: only after automatic gates pass, compare a licensed musical phrase
    as exact Wet/Candidate/Original. Human rejection overrides metric averages.
14. `pack`: package family expert, optional device adapter, controls, uncertainty,
    license provenance, attribution, hashes, runtime limits and model card.
15. `catalog`: keep a small common-pedal `base`; optional families/devices live in
    the downloadable `models` directory and remain independently replaceable.

### Client path

1. Decode the user's Wet file offline without normalization, limiting, lossy
   conversion or physical-device access.
2. Estimate family/stage presence, device/control candidates and top-k forward
   orders using their separate packages.
3. Choose the forward graph by confidence plus forward replay. Abstain when the
   graph or controls are not identifiable.
4. Walk the chosen graph in reverse. Each inverse expert recovers the immediate
   predecessor; repeated same-family slots and `reverb -> drive` shoegaze chains
   remain valid.
5. Preserve the original Clean character by default. Apply `reference` or
   `preset` Clean profile only when explicitly selected; never use a profile to
   hide restoration error.
6. Validate every intermediate output and the final output. Stop/abstain rather
   than inventing unrecoverable content or adding blur, Drive, reverb or clipping.
7. Run as bounded background CPU work on an ordinary computer; target whole
   active-graph RTF <= 0.5. Nothing runs in the realtime audio callback.
8. Export audio only after geometry, quality, provenance and uncertainty checks.

### Package boundaries

- `family`, `order`, `knobs`, `forward`, `restore`, `profile`, and `graph` are
  separate maintainable packages/models.
- They may share schemas and deterministic features. Learned sharing is optional
  and must prove no family loses more than the frozen tolerance while reducing
  aggregate memory enough to justify the coupling.
- Training completion, build success, classifier accuracy, forward-clone quality
  and one good aggregate never imply Wet-to-Clean completion.

Suggested checks:

```sh
/Users/shiftz/dev/muspector/.venv/bin/python3.10 -m unittest \
  remix.test_license_gate remix.test_product_data \
  remix.test_foundation_data remix.test_foundation_model -v

/Users/shiftz/dev/muspector/.venv/bin/python3.10 -m remix.license_gate \
  --registry remix/data_sources.json \
  --output runs/foundation/license/audit.json

/Users/shiftz/dev/muspector/.venv/bin/python3.10 \
  -m remix.train_foundation_product --quick \
  --output runs/foundation/product1
```

## Worktree and safety

- Repository: `/Users/shiftz/dev/muspector-models`
- Branch: `main`
- The first audited source snapshot is intended for `main`. Generated runs,
  weights, source audio and archives remain ignored; their decisive summaries
  and hashes are recorded in this handoff. Do not reset or overwrite unrelated
  work.
- `/Users/shiftz/dev/muspector` is also dirty and owns the shared `data` folder;
  treat all source audio as read-only.
- Do not use physical audio devices. Offline file reads and in-memory DSP only.
- Do not connect UI yet.
- Download more data only for a named coverage gap and only after the proposed
  source passes the three-tier license gate; unknown, noncommercial or
  composition-unclear rights remain blocked.
- Avoid garbage: keep audit JSON and one diagnostic checkpoint per mechanism;
  do not retain generated Wet audio or abandoned candidate forests.
