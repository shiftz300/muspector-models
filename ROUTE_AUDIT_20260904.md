# Restoration Route Audit — 2026-09-04

## Verdict

The product architecture has not drifted from the order-independent inverse-expert route.
Each expert maps one current effect Wet signal to that effect's immediate predecessor. The
graph alone owns chain order and invokes experts in reverse topology. Current product status
remains `product2-partial-development-evidence-not-promoted` and `usable_model` is null.

The audit did find execution-policy and algorithm-scheduling drift. In particular,
order-independent had started to be treated as if it meant profile-independent; it does not.
The current effect's explicit profile/device adapter is an allowed expert input. These drifts are
now either closed in code and tests or fail-closed below.

## 2026-09-05 active-family boundary

Reverb and Amp are frozen by explicit user decision and removed from the active critical path. No
training, family-specific resource search/download or locked-final opening is authorized for either
until that family is explicitly reopened. Existing Reverb and Amp sections below remain historical
negative/diagnostic evidence, not a work order. Amp v30 and rejected v31/v32 evidence are retained
unchanged. Current work continues with generic Drive and non-Reverb/non-Amp chain validation.

## 2026-09-05 packaged foundation-chain closure

The package layer previously allowed audio-rendering contracts only for forward models, while all
inverse packages were forced to claim analysis-only behavior. That was correct for knob recovery
but structurally wrong for Wet-to-Clean experts. The schema now admits a separate
`loss_preserving_restore` inverse contract with immutable input, fixed geometry/sample-rate and no
normalization, limiting, dither or lossy encoding. It still rejects a forward render contract when
the package kind is inverse.

Dynamics, Echo and generic EQ are materialized as separate Apache-2.0 development packages from
`models/catalog/foundation/sources.json`. The runtime owns only one effect's Wet, explicit controls
and, for Dynamics, its own envelope state. A distinct executor owns graph reversal. The deterministic
three-effect audit covers all six forward permutations, places each expert in every position, and
passes with maximum absolute error `2.3526e-6` and minimum RMS-error reduction `99.9985%`.
This is valid integration evidence for repository-owned DSP with known controls, but not evidence
for blind graph discovery or a real physical chain; promotion status remains unchanged.

Drive v3 is now a fourth independently loadable development package. The extended audit at
`runs/foundation/product6-drive-packaging/chain-acceptance.json` exercises all 24 permutations of
Drive, Dynamics, Echo and EQ. Every row and each of Drive's four possible positions pass the frozen
quality2 contract; aggregate median spectrum/high-band/transient/dynamics error reductions are
36.68%/95.35%/99.15%/82.36%. No expert receives order or neighbour inputs. This strengthens the
known-control integration result but does not add physical-device, blind-control or final evidence.

The bounded resource-search extension also checked the most relevant RAT-like and parameterized
overdrive ToneTwist deposits using the official Zenodo API. Both are CC-BY-NC-4.0; the Rodent is a
single physical pedal and the 808-Scream is a software emulation. Neither was downloaded or admitted
for product weights, and neither satisfies the real multi-effect physical-chain requirement.

A 2026-09-06 refresh corrected pOD-set from `NOASSERTION` to the official Zenodo API value
`CC-BY-NC-4.0`. No audio was downloaded. The consolidated non-Amp/non-Reverb audit at
`runs/foundation/product9-nonamp-nonreverb-resource-audit/audit.json` also rejects DAFx as software
presence/order-only, EGFxSet as normalized single-effect recognition-only, GUITAR-FX-DIST for
software/single-effect and upstream-rights conflicts, and the checked ToneTwist Drive records for
noncommercial terms. No public product-safe physical multi-effect restoration pair was found.

A frozen 324-candidate inverse-forward replay audit then tested Wet-only Drive control
identifiability on 12 repository-owned examples using MPS. Top-12 exact-tuple coverage was 8.33%,
shape top-1 accuracy 33.33%, cutoff interval coverage 33.33%, and median boundary margin 1.25%; all
five gates failed. This rejects replay self-consistency as a generic knob estimator. The inverse
expert remains valid only when its controls are supplied explicitly, and the audit may not be
retuned into a pass.

The follow-up Graybox audit deliberately stopped requiring the true knobs and instead asked whether
the frozen replay Top-K contains any quality2-passing Clean. On the same 324 candidates and 12
repository-owned examples, true controls pass 66.67%, the all-grid Clean oracle passes 66.67%,
Top-12 passes 41.67%, and replay Top-1 passes 8.33%. The four all-grid failures are not selector
errors: no candidate passes, with failures concentrated in spectrum/high-band recovery and mostly
the 2.2-kHz forward-cutoff cases. The current finite-grid inverse-forward Graybox route is therefore
closed at capacity before selector training. This does not change the order-independent contract:
each candidate still supplies only the current Wet and that effect's controls, with no graph order
or neighbouring stages. It also does not reject the explicit-control Drive expert.

The already accepted Wet-only nonlinear presence expert and any-effect Clean gate are now packaged
separately as `foundation-drive-presence`. The runtime exposes only generic Drive presence and has
no Amp/Reverb, order or control output. It preserves the 80.19% development nonlinear recall and
98.68% external random-position software-chain recall evidence. The 34.78% synthetic single-effect
count-1 recall remains a hard limitation: negative output cannot authorize automatic bypass.

One frozen four-epoch MPS hard-example attempt then tested whether this was merely an optimization
problem. Calibration selected epoch 4 without consulting development. On the single development
opening, count-1 recall reached 78.26% and low-Drive recall 85.71%, but weak-residual recall remained
38.10% and the negative false-positive rate was 5.84%. The attempt is rejected and wrote no model
checkpoint. This rules out more epochs or post-hoc weighting on the same revealed split; it does not
change the currently packaged Drive-presence expert.

A separate frozen impact audit then tested the only safe non-training escape hatch: bypassing missed
Drive as perceptually negligible. The unchanged packaged detector missed 42 synthetic nonlinear
development stages; 0/42 passed the bypass contract. Median gain shift was 8.73 dB, spectral error
0.1913, high-band error 1.7522 and transient error 0.5147. A negative Drive result therefore cannot
authorize no-op restoration. No thresholds or weights were changed and locked-final stayed sealed.

## Drift found and disposition

1. The license gate checked whether a source could influence redistributable weights, but not
   whether it was approved for the requested task. EGFxSet therefore had a theoretical path to
   Wet-to-Clean gradients despite the registry explicitly excluding its normalized Wet files.
   Product training now requires both weight eligibility and an exact registered use.
2. The historical `remix.egfx` and `remix.mixed` trainers were still executable and labelled
   their outputs like development candidates. Both now fail before creating an output directory:
   EGFx Wet is not allowed for restoration gradients, and ASRNN is CC-BY-NC research-only.
3. Guitar-TECHS P1/P2 Amp+cab gray-box experiments had a generic weight-level authorization.
   The source is registered only as a Clean/family source, not as product Amp restoration data.
   The gray-box loader now requests `train-amp` and is therefore intentionally blocked. Existing
   metrics remain diagnostic evidence; P3 remains absent and sealed from this line of work.
4. The existing Aachen-only RIR helper partitions individual files from one room. It is adequate
   for reproducing historical development experiments but cannot claim room-independent product
   validation. Product4 Reverb therefore uses BUT ReverbDB with whole-room splits; no microphone
   or speaker position may cross a split.
5. Historical `cycles/clean*.json` files with `planned` status are records of old experiments,
   not current work orders. Current use-specific gates take precedence over those stale statuses.
6. The first Product4 sampler used the same index modulus for Clean sources and rooms. In the
   two-source/two-room development split this made each source synonymous with one room; in fit,
   each source repeatedly saw only one of four rooms. Product4 now rotates rooms after a complete
   source cycle. A 64-example fit prefix covers all 32 source-room pairs twice, and tests enforce
   the Cartesian crossing.
7. Historical Ambience checkpoint selection used mean composite loss. Quiet target windows could
   make spectral and attack normalization dominate, while lower loss selected worse tail quality.
   Product4 now uses audible-floor/log spectral convergence, Wet-aware quiet normalization, and
   selects among frozen epochs by calibration-only per-source/per-room tail evidence.
8. Product4 originally retained the historical 16,384-frame target: only 0.34 seconds against a
   measured-RIR domain extending to about one second. The target is now at least 65,536 frames
   (1.37 seconds), and Clean crops must contain a measurable active-to-quiet release event.
9. Short-note Clean sources made the tail objective mostly abstain. Product4 gradients now use the
   five licensed longer-program sources only; this is a Reverb-task filter and does not remove
   those sources from other families.
10. Even after room/source deconfounding, sequential RIR selection left 83 of 96 development rows
    in the medium-tail stratum, with only three measurable short tails. Product4 now balances the
    Cartesian product of Clean source and available room-by-decay cells. The development contract
    has 24 examples per room-by-decay cell before source alternation; tests enforce the crossing.
11. The first attempted Product4-specific "foreground" gate merely reinterpreted whole-clip
    universal metrics. It is no longer mislabeled: the executable contract is explicitly
    `tail-removal-with-global-nonregression`, which fails closed. An active-frame-only foreground
    metric remains future work and cannot be used retroactively to promote a checkpoint.
12. The existing Wet-observable profile-fallback safety score returned zero when no measurable Wet
    tail existed. That inverted uncertainty into apparent maximum safety. It now returns a finite
    reject sentinel, so missing evidence always abstains. A regression test enforces this behavior.
13. Product4 work over-prioritized blind universal dereverberation even though the architecture
    explicitly permits the current Reverb instance's RIR/profile and controls. Known-profile
    inversion is again the primary route; blind inference is only an optional unknown-profile
    fallback and must never weaken abstention.

## 2026-09-05 route recheck and frequency-profile result

The route remains order-independent. The new Reverb expert is a per-effect gray box: it uses Wet,
the current effect's RIR/profile and current mix/room-gain controls, while graph order, chain order,
neighbour effects and Clean are absent at inference. Clean remains training-only evidence.

Frequency-bin Schroeder decay estimation plus an eight-candidate bounded shortening bank passes its
96-example Clean-oracle gate. A BUT-only selector then failed a one-shot OpenSLR 26 external check
in 9 of 26 room groups, so it was not promoted. OpenSLR 26 was separately admitted under
Apache-2.0 for product Reverb training with Room001-120/121-150/151-180 assigned to
fit/calibration/development, Room191-200 permanently external-only, and no room overlap.

The mixed-source MPS v3 checkpoint (16,120 parameters, selected epoch 4) passes the complete frozen
192-example development contract. Aggregate eligible-tail pass is 83.13%, median tail-excess
reduction 59.97%, median tail-envelope improvement 59.22%, and added-reverb fraction 0%. Both
sources, every reported room/category group, and short/medium/long decay strata pass. This closes
the mechanism-development step but not product promotion. The subsequent finite-overlap-add audit
includes all eight physical candidates plus the selector: it matches full finite convolution within
9.32e-9 and reaches ordinary-CPU RTF 0.0623. Runtime is now sealed, but listening has not run, a
fresh physical-room veto is absent, and locked-final remains unopened. `usable_model` therefore
stays null. A two-case compact A/B package is now frozen under
`demos/reverb-frequency-profile-v3-ab-20260905-v1.zip`; the cases are fixed by source/domain and
shaping mode rather than by quality score.

The next physical veto is reserved without consuming it: `rochester-rir-fresh-v2` points to the
University of Rochester's 14-room measured 48 kHz mono RIR record under CC-BY-4.0. The registry
allows validation only and blocks gradients, calibration, threshold tuning and selection. Nothing
has been downloaded; file-level integrity and decay-domain admission remain prerequisites.

Human listening then rejected v3. The fixed BUT measured-room example sounded wrong and the fixed
OpenSLR simulated-room example had only a slight effect. Both rows were outside the measurable-tail
subset, yet the soft selector still altered them; each also failed its whole-example gate. This
invalidates product promotion despite the aggregate development and runtime passes. The audit gap
is now explicit: eligible-tail medians do not constrain every input, and the Demo builder did not
require either per-example evidence or exact bypass. v3 is frozen and rejected, Rochester remains
unopened, and the calibration-only measurable-effect-or-hard-bypass contract now covers active
foreground in `remix/reverb_quality2.py`.

That stricter audit rejects v3 itself: 21/102 measurable calibration rows are effective, zero rows
hard-bypass, and only 10.94% of all rows are safe. Two training-only Clean upper bounds isolate the
cause. Selecting one whole candidate or exact Wet reaches 21/102; allowing the stronger per-bin
optimal convex interpolation between Wet and every one of the eight physical candidates still
reaches exactly 21/102 (20.59%). Exact Wet fallback makes every row safe but cannot meet the frozen
50% effective-coverage floor, and every source, room and decay group fails. The decisive report is
`runs/foundation/product4-reverb-frequency-profile-acceptance2-tf-oracle-calibration-v1.json`.
This closes the physical candidate family, not only the learned selector. No further v3 training or
threshold tuning is allowed.

A different v4 gray box then predicted a direct complex residual from Wet plus the current transfer
profile and trained with explicit physical reconvolution-to-Wet consistency. The MPS screen learned
on fit but produced no new effective non-exact calibration rows, so deconvolution-from-features is
under-conditioned and rejected. Supplying the bounded analytic exact/regularized inverse as the
physical core reveals a stronger but narrower result: 86 candidate outputs are effective among 91
measurable rows in the full 192-row calibration set. A Clean capacity oracle can therefore pass,
but it is not deployable.

The fail-closed observable gate releases only 33/91 effective rows (36.26%). A 545-parameter
Wet/profile-only MLP and a corrected shared-source/early-stop repeat release only 1/91 and 9/91,
respectively, under the zero-unsafe-processing requirement. No Reverb development rows were opened.
This closes further gate-capacity and threshold sweeps on the same evidence; the remaining Reverb
research problem is runtime-observable abstention, not analytic inverse capacity. Rochester and
locked-final remain sealed.

Exact re-convolution was then tested as a genuine gray-box self-verifier. It is not identifying:
effective and ineffective analytic inverses both include near-zero candidate-to-Wet replay error,
so convolution consistency cannot select the hidden Clean from finite single-channel audio. A
larger time-frequency audio-prior gate was trained on MPS using 960 fit examples, with its release
threshold frozen on a 192-example fit holdout. One-shot calibration releases 12/91 measurable
examples and mis-releases five unsafe examples (97.40% safe), below both safety and coverage gates.
Its rejected weight was deleted. Development, Rochester fresh and locked-final remain unopened.
This closes exact-replay and learned dry-prior gating under the current single-channel observation
contract; additional observation or a materially different inverse formulation is required.

## Resource cleanup

- 44 GiB of blocked ToneTwist downloads and extracted research copies were moved together to
  `/Users/shiftz/.Trash/muspector-tonetwist-blocked-20260904`; no product-admitted source was moved.
- The 32,969,803,616-byte DAFx Zip64 archive matched the registered MD5, passed CRC for all
  38,801 files, and every member was present in the extracted corpus. The redundant archive was
  then permanently deleted; the extracted CC-BY-4.0 corpus remains intact.
- The official BUT RIR-only archive and nine temporary range parts were deleted only after the
  9,308,593,693-byte archive matched SHA-256
  `d0d14cdd626b4b05401027c5a774b4f24369f4ea38f57480c6fc6e04e9f3f5b5`, all 19,279 tar
  entries were readable, and 2,325 RIR WAVs plus required metadata were extracted. This reclaimed
  about 17.3 GiB; the admitted extracted RIR corpus is about 80 MiB and contains no silence tree.
- The 541,020,037-byte OK5 archive was deleted after its official MD5 matched, ZIP CRC passed, all
  25 SOFA files were extracted and the data gate admitted 27 measurements from 12 rooms. Five
  OpenAIR archives totaling 34,342,703 bytes were likewise deleted after SHA-256/CRC checks and
  extraction. Their 33 measured RIRs yielded 31 in-domain responses from four independent rooms.
  Extracted read-only measurements and audit records remain; the archives are recoverable from the
  registered official URLs.

## Executable invariants

- `remix/test_route_contract.py` instantiates the current inverse experts and rejects any graph,
  topology, order or neighbouring-effect input.
- Product pair loaders use source-by-use authorization. Weight-only authorization is forbidden
  outside the gate implementation.
- `cycles/foundation.json` and `cycles/restore.json` must explicitly retain
  `usable_model: null` until every product gate passes.
- The locked-final split is not opened during fitting, calibration, development iteration or demo
  selection.

## Current programme state

- Graph/family/device/profile boundary: defined and test-covered.
- Forward Clone/system-identification: the requested structural-OOD benchmark, BLA,
  Parallel-Hammerstein/Wiener-Hammerstein/dynamic-gray-box baselines and tiny-network comparisons
  now exist under `clone/` with an experiment registry. They established the structural limits
  and closed the promised gray-box investigation; they are diagnostic tooling and must not become
  the Wet-to-Clean product boundary.
- Family presence: development and external software-chain evidence exists, but the real physical
  multi-effect veto is still missing.
- Drive/nonlinear: useful listening progress exists, but no general physical Drive/Amp product
  model has passed physical veto, listening and locked-final together.
- Amp: the Marshall stage-supervised gray box is the strongest current diagnostic, but the whole
  family remains unpromoted; Guitar-TECHS training is still blocked by its use-specific gate.
- Reverb: Product4 now uses 2,325 measured BUT RIRs under CC-BY-4.0 with whole-room splits:
  four fit rooms, one calibration room, two development rooms and two unopened locked-final
  rooms. Targets are long enough for the admitted decay range, crops are release-event-aware, and
  source, room and decay sampling are crossed. No graph, order or neighbouring-effect input exists.
- Reverb evidence through v12 used the old 16,384-frame target and is retained only as superseded
  short-window diagnostic evidence. Its apparently passing Clean oracle did not survive the valid
  long-window contract and cannot justify the physical family.
- The valid event-aware long-window single-WPE result is
  `runs/foundation/product4-reverb-event-aware-graybox-v15`: epoch 2, 2,962 parameters, 19.67%
  aggregate tail pass, 10.29% median tail-excess reduction, 13.02% median envelope improvement and
  6.56% added reverb. A corrected Clean oracle over that same single candidate reached only 42.62%
  tail pass, so both the estimator and physical candidate were rejected.
- A four-geometry WPE Clean-oracle bank still failed EGFx and ConferenceRoom2. Increasing the mask
  estimator to 75,634 parameters had already regressed quality, so both the WPE candidate family
  and blind capacity expansion are closed.
- A new Wet-only gray-box family uses causal multiband exponential peak-decay states and six
  bounded attenuation candidates, with Wet identity in the convex hull. On the initially
  imbalanced development draw its Clean-oracle passed aggregate, both sources and both rooms, but
  this was not valid evidence. Under the corrected room-by-decay-balanced 96-example contract, its
  oracle reaches 54.84% aggregate tail pass, 84.85% median tail-excess reduction and 26.22% median
  envelope improvement with no added reverb. It passes both rooms, medium and long tails, but fails
  EGFx (17.39% pass) and short tails (45.45% pass; 7.29% median envelope improvement). Therefore
  this family is promising diagnostic evidence but not expressive enough for product training.
- One MPS selector run, `product4-reverb-decay-bank-graybox-v16`, was completed before the decay
  imbalance was discovered. It selected epoch 4 and runs at CPU RTF 0.0566, but is superseded and
  rejected: 45.90% aggregate tail pass, EGFx 19.05%, and material whole-clip nonregression failure.
  It produced no Demo and is not a candidate for retuning.
- A balanced union oracle combining all WPE and exponential-decay directions fixes short-tail
  coverage (63.64% pass) but leaves EGFx unchanged at 17.39%. The hybrid blind family is therefore
  closed as well; it does not justify another selector run.
- The Product4 known-profile analytic path is exact on its stable 22/96 subset but covers only
  22.92%. Its regularized fallback nearly passes without selection, missing only the added-reverb
  gate. A calibration-only Wet-observable selector initially appeared to reach 62.5% development
  coverage, but that evidence was invalidated by the no-tail-is-safe bug above.
- After the fail-closed fix, `product4-reverb-known-profile-selector-v2-fail-closed` passes quality
  on every accepted axis: aggregate, fallback, both Clean sources, both rooms and all three decay
  strata. Coverage is only 45.83% on calibration and 40.63% on development, below the frozen 50%
  minimum. It is a real safe selective subdomain, but remains rejected and is not a usable whole
  Reverb expert. No Demo or locked-final evaluation is allowed.
- A fixed shallow ExtraTrees risk gate over Wet, candidate, profile conditioning and forward-replay
  features was screened without creating a checkpoint. Calibration appeared to admit 93.75% with
  no observed unsafe fallback, but development admitted 84.38% while accepting six unsafe
  fallbacks; EGFx, E112 and medium-tail gates failed. This is distribution shift, so learned risk
  classification is closed rather than retuned on development.
- Fixed regularized-profile gains 2x through 6x were selected on calibration. Both 5x and 6x passed
  every calibration axis; the conservative 5x candidate then added reverb on 8.06% of development
  tails and failed every source and room group. A Wet-envelope ceiling removed added reverb
  entirely, but failed high-band nonregression on both calibration and development. Gain-ceiling
  and envelope-limiter sweeps are closed.
- A finite-latency profile Wiener inverse screened 4,096, 16,384 and 32,768-tap kernels with 4x/6x
  gain ceilings on calibration. Only 32,768/6x passed every calibration axis. On its single fresh
  development evaluation, global nonregression still passed but added-reverb fraction rose from
  1.64% calibration to 17.74%; both sources, both rooms, long and medium tails failed. This exposes
  room-distribution instability rather than insufficient FFT length. Finite-kernel/gain sweeps are
  closed without a checkpoint or Demo.
- Primary literature confirms that mixed-phase acoustic responses generally lack a stable causal
  single-channel exact inverse and that preserving a short direct/early response is more robust
  than full equalization. A clean-room finite-lookahead response-shortening prototype therefore
  replaced the delta target with an early-preserving, exponentially shortened target plus bounded
  partial-band application. No third-party implementation was copied.
- The wider-band v1 passes aggregate, both development rooms and its shaping fallback at 100%
  coverage with zero added reverb, but calibration EGFx/high-band and medium-tail nonregression
  fail. It is rejected; its already-opened development result cannot be used for tuning.
- Calibration-only v2 narrows the transition to 2–3 kHz and selects the lowest strength satisfying
  explicit margin gates. `product4-reverb-profile-shaping-calibration-v2` passes aggregate,
  fallback, both sources, the calibration room and all three decay strata at 100% coverage. Its 76
  fallback cases show 81.48% tail pass, 31.75% median tail reduction and 0% added reverb. It remains
  non-promotable. Its frozen OK5 evaluation covered 108 examples from 12 new rooms but failed the
  shaping-fallback and per-room gates: fallback tail pass was 53.33%, dynamics nonregression was
  86.21%, and five rooms failed. This closes fixed global response-shortening strength as a product
  path; OK5 must not be used to retune it.
- A genuinely gray-box follow-up, `product4-reverb-profile-bank-graybox-v17`, uses the current Wet,
  current RIR and current controls to construct four bounded response-shortening candidates and a
  33,176-parameter MPS-trained convex time-frequency selector. Stable exact cases are immutable,
  Wet identity remains reachable, and no order or neighbouring effect input exists. The internal
  48-example development screen passed all source, room and decay groups with 71.88% tail pass and
  35.03% median tail reduction, but those rooms were already repeatedly used and are not promotion
  evidence.
- The frozen v17 checkpoint then failed its one-shot OpenAIR gate on 124 examples from four fresh
  rooms. Aggregate dynamics nonregression was 89.52%; its 85 shaping cases reached only 53.70% tail
  pass and 26.24% median tail reduction. Both Clean sources, short-tail coverage and two rooms
  failed. The checkpoint is rejected and retained only as evidence that this selector overfits room
  response structure. OpenAIR must not be used to tune it.
- Modulation and spectral work remain lower priority and cannot substitute for Drive/Amp and
  Reverb completion.

### Marshall stage-supervised Amp follow-up

- The licensed Marshall archive includes aligned `input`, `preamp` and `speakerout` files for every
  setting. The prior endpoint-only models ignored that measured internal boundary. A new audit,
  `runs/foundation/product3-amp-marshall/data-audit-stage-v2.json`, hashes all three non-locked
  signals; the nine locked-final settings remain undecoded.
- The clean-room v4 structural cascade follows output profile -> power inverse -> tone-stack
  inverse -> preamp inverse. It has 6,811 parameters and no chain-order or neighbouring-effect
  inputs. A first MPS run reached 33.33% development pass fraction; a calibration-selected
  low-rate continuation reached 34.07%. Aggregate transient improvement crossed 10% only after
  continuation, while per-example and control-stratum gates still failed.
- Applying the intermediate loss to every stage simultaneously regressed to 3.70%, demonstrating
  that endpoint modules were still compensating for one another. The replacement v6 trains
  speakerout -> preamp and preamp -> input separately, freezes each identified section, then uses
  a short joint refinement. Its 10,666-parameter checkpoint is
  `runs/foundation/product3-amp-marshall/amp-stagewise-graybox-v6/model.pt`, SHA-256
  `e70c0c85e0491f0208e9b95eab187212e831fb34d2497ef87787c5e5bc64a75a`.
- v6 is the strongest single Amp result: 40.00% development pass fraction, with median reductions
  of 84.35% spectrum, 97.41% high band, 11.78% transient and 30.78% dynamics at CPU RTF 0.02294.
  It is still rejected because aggregate pass fraction is below 50%, the product minimum is 80%,
  and Bass/Mid/center/Gain strata plus individual settings do not all pass.
- A non-promotable Clean-oracle over six frozen candidates reaches 57.78%, proving candidate
  complementarity. A ridge selector fitted only on calibration Clean and using only Wet features
  plus B/M/T/Gain at inference reaches just 36.30% on development. It cannot recover the oracle
  choices, so selector retuning on revealed development is closed. No Demo was generated.
- The DAFx23 paper supports separating nonlinear preamp/power stages from an explicit linear tone
  stack, and DDSP Guitar Amp independently supports interpretable stage decomposition. The
  supplementary `grey-box-amp` repository exposes no clear code license, so no implementation was
  copied. Only the paper-level architecture and the CC-BY-4.0 dataset contract informed this
  clean-room inverse.
- A follow-up official-source search found four small ToneTwist physical Amp datasets (Blackstar
  HT1, Blackstar HT5 Metal, Engl Retro Tube 50 and Fender Blues Jr). Their official Zenodo APIs all
  report CC-BY-NC-4.0. They are registered as metadata-only/product-blocked and were not
  downloaded; the MIT license on the ToneTwist index code does not override each audio record's
  noncommercial license.
- The remaining eight physical Amp/Preamp records in the same official ToneTwist index were checked
  in one parallel metadata-only pass: Ibanez TSA15 Crunch; MesaBoogie 550 Clean, Crunch and Burn;
  MesaBoogie Mark V Clean, Crunch and Extreme; and UA 6176/610B. Every official Zenodo API also
  reports CC-BY-NC-4.0. Together they would be 573,755,884 bytes of archives, but none was
  downloaded and all are product-blocked in the registry.
- Amp-Space/RockingFace is structurally interesting because its card exposes paired `direct_input`
  and `output` fields plus device/control metadata. It is not usable: access is gated, the YAML
  metadata is absent and the displayed License field is still an unfilled template. The current
  card also says only two of its 52 devices are amplifiers.
- The original Automated-GuitarAmpModelling repository contains real HT1 and Big Muff split pairs
  under a GPL-3.0 repository, including 79,567,284 bytes for the HT1 pair set. Product gradients are
  still fail-closed because dry-performance provenance and trained-weight scope are not separately
  stated, and a later ToneTwist HT1 deposit is explicitly CC-BY-NC-4.0. Open-Amp is likewise
  blocked: its repository has no license, the crowd-sourced ToneLibrary captures have no
  per-capture rights statement, and its documented IDMT clean source is not product-compatible.
- Two source renderer candidates were audited without admitting gradients. Swanky Amp is a
  GPL-3.0 tube-amplifier simulator with no official GitHub binary release. BYOD is a GPL-3.0 modular
  analog-modelled distortion renderer, but the downloaded official macOS 1.3.0 package and arm64
  VST3 both failed `codesign --verify --deep --strict`; the 89,127,908-byte DMG and extracted files
  were deleted without installation or loading. A pinned source build remains possible, but any
  generated-output and trained-weight policy must pass the same use-specific gate first. Synthetic
  renderers may diversify training; they cannot replace fresh physical-device validation.
- A broader official Zenodo search found one new product-compatible source: EGDB-PG v2, released
  by the original EGDB author under CC-BY-4.0. It contains 240 matched clean DI tracks and 256
  Amp+cab preset renders (96 low-gain, 64 crunch and 96 high-gain), about 514 rendered hours. The
  135,406,622,188-byte ZIP is stored without per-file compression and supports HTTP byte ranges, so
  the full archive must never be downloaded. Its 9,470,404-byte central directory was read directly
  and confirms 61,512 FLAC renders plus 240 clean WAV files. One clean/three-profile probe was
  extracted by exact ranges: all four files are 44.1-kHz mono, sample-aligned at 800,471 frames and
  match their ZIP CRC32 values. The profiles have fixed latencies of roughly 40-67 frames and may
  invert polarity, which the future loader must correct from fit data only.
- `remix/egdb_pg_subset_v1.json` freezes a small profile-disjoint contract before training: three
  profiles (one per gain class) for fit/calibration, three unseen profiles for development, three
  unopened profiles for locked-final, and disjoint content IDs. The large archive remains remote;
  only exact byte ranges for the declared fit/calibration/development subset may be materialized.
  This is a useful new commercial-grade software Amp+cab domain, not physical-device evidence and
  not a Marshall speaker-output target. A separate 1.0-GB EGDB/BIAS-FX2 distortion-recovery record
  and GUITAR-FX-DIST remain blocked because their upstream audio rights are unresolved or conflict
  with the current IDMT NC-ND license.
- The declared EGDB-PG subset is now materialized through 8-way `curl` ranges: 336/336 files and
  769,446,382 bytes pass ZIP CRC32, 44.1-kHz mono format and exact clean/wet frame-pair checks. It
  contains 0.641038 unique clean hours and 1.923415 rendered wet hours. No archive was downloaded,
  and all ZIP-directory/probe temporary files were deleted. Fit-only coarse alignment shows profile
  medians near 39 frames (crunch), 57 frames (low gain) and 62-63 frames (high gain), with most
  correlations polarity-inverted; content-dependent correlation outliers mean the loader must fit
  one robust profile transform from fit rows rather than align each development example.
- That three-profile pilot was intentionally superseded after the official EGDB-PG paper's ablation
  showed that few tones overfit and that larger tone diversity improves unseen-amplifier behavior.
  The same frozen track/development/locked-final boundaries now contain 30 fit/calibration profiles
  (ten per gain class) and three disjoint development profiles. Only exact ZIP byte ranges were
  fetched. An initial 8-way request hit Zenodo HTTP 429 and was stopped; missing-only planning then
  resumed at four-way parallelism with retry backoff. The final audit verifies 1,956/1,956 files,
  4,529,111,989 valid bytes, 33 Wet profiles, exact paired frame counts and no locked-final download.
- The old Guitar-TECHS models could not be reused because they require a known profile one-hot at
  inference. `remix/egdb_pg_amp_data.py` and the v1-v3 models instead expose only Wet audio; category
  and profile identifiers are sampling/reporting labels and never forward inputs. All manifests
  explicitly reject graph order, neighboring effects, profile IDs and Clean/oracle inputs.
- The first waveform TCN runs were rejected immediately: both the old loss and its low-learning-rate
  replay selected epoch zero, and the old normalized loss favored near-silence on quiet DI windows.
  A bounded per-example multiresolution spectral/envelope/correlation loss fixed that objective
  pathology. A Wet-only STFT-mask v2 then trained stably on MPS.
- The 9-profile v2 checkpoint reached 11.11% complete unseen-profile passes while improving median
  spectrum/high-band errors by 60.00%/92.17%; dynamics regressed 30.40% and transient improvement was
  only 3.80%. Its two-stage spectral-plus-temporal v3 raised complete passes to 12.50% and reduced the
  dynamics regression to 23.25%, but remained rejected.
- With 30 profiles, v2 reached 15.28% complete unseen-profile passes, 60.67% spectrum and 92.89%
  high-band improvement. The metric-aligned v3 uses 240/120-frame attack and 1024/256-frame crest
  objectives, freezes the spectral warm start before joint tuning, and reaches 16.67% passes. Its
  dynamics regression narrows to 15.24% overall and 5.57% on high gain, but transient improvement is
  only 4.95%. CPU RTF is 0.326, leaving little room in the 0.35 single-expert budget. It is rejected,
  generated no Demo and did not open locked-final.
- Primary sources support the current conclusion: real VST-derived data outperforms simplistic
  synthetic distortion, phase-insensitive spectral restoration can disagree with ESR, and EGDB-PG's
  tone encoder uses 3-5 second clips plus same-tone contrastive learning. The next Amp+cab attempt
  must therefore pretrain a Wet-only, content-invariant tone representation and use a true decoder
  or vocoder-like reconstruction stage. More epochs, deeper TCNs and another phase-preserving mask
  are not justified by the current evidence.
- A separate 3-second Wet-only tone encoder was then trained without opening development audio.
  v1 collapsed because its two-dimensional global pooling erased frequency location. The corrected
  frequency-preserving v2 reached 56.25% leave-one-content-out profile retrieval. Wider in-batch
  negatives and more fit performances raised v3 to 63.75% with a 0.050 cosine margin. The final v4
  covers all 48 fit performances through 41 distinct positive-pair offsets and adds a 0.20 hardest
  positive/negative margin objective; it reaches 73.33% retrieval and a 0.113 observed margin.
  Calibration loss improves 60.40%, CPU budget passes, and inputs remain Wet-only, but the frozen
  80% retrieval and 0.20 margin gates both fail. v4 is therefore a useful diagnostic representation,
  not permission to train another inverse. Further tuning on this same content is closed.
- The fit-content contract was expanded once, deterministically, from 48 to 96 disjoint clean
  performances while retaining all calibration, development and locked-final boundaries. Exact
  Range materialization now verifies 3,444/3,444 files and 7,841,019,015 bytes: 1.000559 unique
  clean hours, 22.067111 fit-render hours, 2.778046 calibration hours and 0.517160 development
  hours. All CRC32, format and clean/wet frame-pair checks pass; locked-final remains absent.
- After the one-shot fresh-v1 rejection was consumed as fit evidence, the current audit verifies
  3,540/3,540 files and 8,059,422,253 bytes: 1.190855 unique Clean hours and 25.933206 Wet-render
  hours across 36 profiles. Fresh v2 and locked-final remain absent.
- With architecture and calibration gates frozen, content96 tone encoder v5 selects epoch 24 and
  reaches 83.75% leave-one-content-out retrieval with a 0.204 cosine margin. It passes all six
  representation gates (including Wet-only, CPU and locked-final boundaries). This accepts the
  representation only; it is not a restoration model.
- The accepted v5 encoder was then frozen inside a Wet-only FiLM-GCN inverse whose 3-second tone
  reference is another span of the same Wet recording. The formal content96 decoder selects epoch
  5, improves calibration loss by 15.45% and reaches 22.22% complete unseen-profile passes. Median
  spectrum/high-band/transient reductions are 49.64%/83.99%/8.68%; dynamics still regresses 3.38%.
  CPU RTF is 0.114. This is a meaningful improvement over v3's 16.67% pass and 15.24% dynamics
  regression, but aggregate, per-example, every-category and every-profile gates still fail. It is
  rejected, generated no Demo and did not open locked-final.
- Before the next candidate was trained, a disjoint fresh-validation v1 contract reserved three
  profiles and 24 tracks. The complex magnitude-plus-phase v5 inverse was frozen by hash before the
  corresponding 72 pairs were downloaded. Its one-shot result is rejected at 15.28% complete
  passes: spectrum/high-band/transient improve 59.43%/91.01%/17.47%, but dynamics regresses 11.85%.
  Provenance, checkpoint, runtime, Wet-only/order-free and sealed-data gates pass; quality,
  per-example, every-category and every-profile gates fail. No Demo or locked-final access occurred.
- Fresh v1 is now consumed fit evidence, and a second disjoint three-profile/24-track fresh set is
  reserved but not downloaded. With 33 profiles, complex v7 improves calibration loss 20.80%, runs
  at CPU RTF 0.090 and reaches 23.61% complete passes on the old revealed development set. The v8b
  bounded envelope refiner reaches 29.17% and positive aggregate reductions on all four metrics,
  but only 21/72 examples pass and the frozen calibration-improvement threshold fails. Neither is
  eligible to spend fresh v2.
- Failure-directed follow-ups were bounded and rejected. A gain-category-aware encoder reaches only
  60.42% category accuracy, so hard low/crunch/high routing is unsupported. The v7/v8b oracle union
  is only 31.94%; explicit attack shaping peaks at 30.56%; a long-context TCN worsens calibration;
  and compact frozen or jointly trained U-Nets do not clear the smoke/calibration gate. The next
  justified Amp mechanism is therefore a materially larger end-to-end waveform/complex decoder or
  a two-stage spectral-to-waveform reconstruction model, not another small refiner or selector.
- That larger-capacity check is now complete. Joint complex-plus-Demucs v13 has 8,271,870 parameters,
  a 3.09-second bottleneck receptive field and CPU RTF 0.173. On a sealed long-context MPS smoke it
  improves calibration from 3.6851 to 3.5329 (4.13%) after four epochs, below the frozen 10% gate.
  No development or fresh data was opened. Capacity scaling by itself is therefore closed; the next
  Amp hypothesis must change the factorization to explicit gray-box parameter estimation/inversion
  or a separately justified two-stage spectral-to-waveform reconstruction.
- The explicit gray-box branch then materially outperformed capacity scaling. v14 estimates three
  LTI inverse stages and two monotonic dynamic inverse stages from the frozen Wet tone embedding;
  it improves calibration 25.34%, runs at CPU RTF 0.023 and reaches 22.22% complete old-development
  passes. Replacing its symmetric zero-phase filters with non-symmetric phase-generating LTI stages
  produces v15: 28.88% calibration improvement and 34.72% complete passes, with median
  spectrum/high-band/transient/dynamics reductions of 60.79%/92.32%/20.90%/6.66%. This is the best
  generic software Amp result, but aggregate dynamics remains below 10%, only 25/72 examples pass,
  and no category or unseen profile is accepted. A frozen-v15 explicit dynamics v16 selected epoch
  zero and was stopped at smoke. Fresh v2 and locked-final remain absent; no Demo was generated.
- A no-training reference-duration audit on the revealed development split compares 3.0/5.5/8.0
  seconds from the same Wet recording. Complete pass fractions are 34.72%/34.72%/36.11%; the small
  eight-second gain does not clear dynamics or per-example gates. Longer Wet observation by itself
  is therefore not a promotion path.
- A split-isolated Clean-assisted profile oracle does not rescue v15: fitting one condition on
  support tracks reduces old-development complete passes from 38.89% baseline to 33.33%. A
  deliberately non-promotable same-audio Clean condition reaches only 55.56% after 360 steps, well
  below the 80% individual gate. The official 10,645-byte EGDB-PG preset metadata archive contains
  only UUID-to-low/crunch/high labels, not knobs or internal stages. This proves the limitation is
  shared by parameter estimation and the frozen five-stage processor, not just profile routing.
- Open Riff Box was audited at commit `c980ae874c87e16835d87b265ea58079fc69e7f5`. Its official
  GPL-3.0 source provides a working offline Platinum renderer and taps after V1A, V1B, V2A, V2B,
  V3A, cathode follower, tone stack, power amp and output transformer. GNU's GPL FAQ says ordinary
  program output generally follows input copyright when it does not copy program assets; however,
  the repository's cabinet assets have no per-IR notices and its roadmap calls the older set
  placeholders. The project therefore used only float32, noise-disabled, no-cabinet ephemeral
  renders and kept the source, audio and learned weights research-only.
- The nine-stage research probe learned materially on 9 fit / 3 disjoint calibration Clean tracks.
  v20 reached 66.67% complete passes and positive median reductions on spectrum, high band,
  transient and dynamics; no research weight was retained. This justified architecture selection,
  not product promotion.
- Product v21 was independently reinitialized and trained only on authorized EGDB-PG rows at the
  same budget as v15. It is the strongest generic software Amp diagnostic: 29.25% calibration
  improvement, 38.89% complete old-development passes, 61.12%/92.88%/27.59%/12.84% median
  spectrum/high-band/transient/dynamics reduction and CPU RTF 0.074. All four aggregate metric
  thresholds pass, but only 28/72 examples pass and no category/profile summary is accepted. v22
  ties the pass fraction while worsening high band, transient and dynamics, so v22 is pruned and
  v21 remains diagnostic-only. Fresh v2, locked-final and Demo were not opened.
- RemFX provides relevant architecture evidence—its distortion remover uses Demucs, 5.5-second
  chunks and a strong waveform-plus-multiresolution spectral objective—but not product weights. The
  official repository is Apache-2.0 while Zenodo record 8218621 licenses the separate pretrained
  checkpoints as `cc-nc`; the approximately 1.0-GB distortion checkpoint was not downloaded and is
  research-only under the fail-closed product-weight gate.
- Three obvious pretrained-encoder shortcuts were checked and rejected without downloads. MERT's
  official model card is CC-BY-NC-4.0. AudioMAE's official repository LICENSE is also explicitly
  CC-BY-NC-4.0 and its AudioSet audio is unavailable for copyright reasons. LAION-CLAP code is CC0,
  but the project states most checkpoint training audio cannot be released for copyright reasons
  and does not separately grant product-weight provenance for each checkpoint. Code licensing is
  not substituted for weight/data rights; all three remain metadata-only and gradient-blocked.
- Two further paired-guitar candidates were checked against their dataset records, not inferred from
  paper licenses. GOAT describes 5.9 hours of DI and 29.5 hours of Amp/cab renders, but its official
  Zenodo dataset is restricted, CC-BY-NC-4.0 and explicitly research-only/noncommercial. The paper's
  CC-BY-4.0 notice does not override the dataset license. Five Guitar Dataset is open CC-BY-4.0 and
  records DI/phone/computer channels simultaneously, but it contains named cover performances with
  no composition-clearance statement; its electric DI is an amplifier output rather than raw pickup
  DI, and its acoustic/electric targets are inconsistent. Both remain metadata-only and were not
  downloaded.

### Product-licensed rusty-amp stage supervision

- `rusty-amp` commit `831d6bba4ad3a1a8da958c510cbc96a9e55b73f9` is Apache-2.0 and passed
  181 library tests. The durable builder admits only its no-cabinet Marshall algorithm and records
  eight stage taps; demo audio, cabinets, Audio Units and third-party plugins are excluded.
- v23-v1 product pretraining improves disjoint stage calibration loss by 74.03%. Fine-tuning the
  168,638-parameter Wet-only eight-stage inverse on the physical Marshall contract yields v24 at
  57/135 = 42.22%, the strongest current physical Amp diagnostic, but below every promotion gate.
- Fourfold stage-corpus scaling improves synthetic calibration to 78.24% yet v25 regresses physical
  development to 40.00%. This falsifies synthetic scale-up as the missing ingredient.
- One independent equal-budget EGDB-PG transfer, v26, improves calibration 25.44%
  (`3.3791 -> 2.5194`) but reaches only 22/72 = 30.56% profile-disjoint development passes. Every
  gain category and unseen profile fails; v21 remains stronger at 38.89%. This closes further
  selector/epoch/hyperparameter sweeps on the revealed split.
- No fresh-v2, locked-final or Demo evidence was opened. v24 remains diagnostic only; physical
  source breadth or self-verifying Wet parameter inference is now required.
- The official EG-IPT Zenodo metadata confirms CC-BY-4.0 over the complete archive and simultaneous
  DI plus EVH 5150 III/Mesa V30/SM57 capture. Three ZIP ranges recovered 8,717 paired SM57 files
  (4.7266 h) with CRC verification; three absent upstream Wet members are explicitly excluded.
  Corrected broadband-transient alignment gives 64 frames at 96 kHz with p95 deviation 28.3 frames,
  and all geometry/effect/clipping gates pass. The single fixed hardware profile is admitted fit-only.
- v27 adds this physical source to the unchanged EGDB training contract. It improves the EG-IPT
  fit-only internal diagnostic from 17.54% to 48.54%, proving the path is learned, but reaches only
  36.11% on EGDB and 6.67% on Marshall. The v6/v24/v27 Marshall Clean-oracle ceiling is 49.63%, so
  no selector can repair the family. v27 is rejected and pruned without Demo/fresh-v2/locked-final.
- The corrected per-device interpretation produces v30: one fixed EVH 5150 III/Mesa V30/SM57
  expert with its profile embedded in 59,214 weights and no content-dependent profile inference.
  It reaches 78.65% on the first fit-only internal calibration and 75.80% on 343 windows from 49
  pair-disjoint reserved cells; all aggregate metrics and pickup groups pass. Four short-transient
  techniques fail the first split, and the reservation cannot cover two exhausted techniques.
  A frozen-base transient v31 lowers average loss but regresses pass coverage to 74.05%; its weight
  is pruned. Keep v30 as diagnostic and require independent physical validation before Demo.
- The final bounded Amp architecture check used EG-VAE (arXiv 2608.05513) only as a design
  reference. Its full recipe uses about 758 hours of commercial-plugin renders, 350,000 steps and
  one RTX PRO 6000, while the official article/demo exposes no reusable implementation or product
  checkpoint grant. The bounded local implementation instead kept the product-safe eight-stage
  gray-box inverse and added a training-only forward renderer. Exact same-Clean/different-profile
  EGDB-PG pairs provide cross-tone supervision, but checkpoint selection excludes replay loss.
  v32 smoke improves Clean calibration loss by only 8.22%, reaches 16.67% unseen-profile passes,
  collapses to a 0.000025 same-vs-different tone margin and fails all forward anti-leakage gates.
  Its checkpoint was pruned. No formal run, fresh-v2 opening or Demo is allowed; this closes the
  factorized/cycle-assisted Amp route at the current scale.
- Resource reconciliation deleted 3.6 GiB of unused Guitar-TECHS P1/P2 category ZIPs plus the
  residual P3 archive that the contamination policy already required absent. The two audited
  single-note archives and 382 MiB extracted P1/P2 corpus remain; no training-readable source was
  removed.

## Frozen continuation record

The historical items below document the last admissible continuation boundaries. Amp and Reverb
are now user-frozen, so none of their proposed follow-ups are active until explicitly reopened.

1. Freeze Marshall v24 and EGDB-PG v21. Do not spend fresh v2 on either diagnostic and do not run
   more local post-filter/loss-weight/stage-scale sweeps on the same revealed splits. The licensed
   eight-stage prior helped Marshall only marginally; adding a newly audited physical EG-IPT fit
   domain then learned that domain but transferred negatively to both EGDB and Marshall. A next Amp
   attempt requires domain-invariant or self-verifying Wet parameter estimation, not more source
   mixing or a selector, while retaining Wet-only, order-free inference.
   The first global-Wet-latent conditional forward verifier selected epoch zero on EGDB and failed
   every anti-leakage/identifiability gate; its rejected weights were deleted. Do not attach cycle
   loss to that verifier. Cross-tone factorized gray-box v32 also failed its bounded smoke gate and
   must not receive a formal schedule or sweep. The fixed-profile v30 is now the strongest
   physical-chain diagnostic. The bounded online search found no independently licensed paired
   capture of the same EVH/Mesa/SM57 chain, so promotion is data-blocked; do not tune it further on
   EG-IPT.
2. Freeze and reject Product4 Reverb frequency-profile v3; do not tune it further on revealed
   development, listening or external sets. Its runtime remains reusable engineering evidence only.
3. Keep the now-frozen active-frame foreground metric together with whole-clip nonregression. Tail
   reduction remains restricted to measurable release tails; abstentions are reported and never
   counted as successes.
4. The calibration-only gate rejects v3 and its convex hull. v4 proves the analytic inverse has
   capacity but rejects scalar, MLP, exact-replay and time-frequency audio-prior gates. Do not open
   development or repeat classifier/threshold sweeps; require a different observation model.
5. Listening rejected the existing A/B package. Keep Rochester and locked-final sealed until a new
   candidate passes automatic gates, runtime and listening in that order.
6. Generic Drive and non-Amp/non-Reverb real-chain acceptance remain active, but the current
   finite-grid Drive Graybox route is closed and its selector must not be trained. Amp and Reverb
   are paused regardless of their historical importance; modulation stays lower priority and
   forward Clone/structural OOD remains closed diagnostic infrastructure.
