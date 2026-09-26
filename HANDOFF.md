# Handoff

The full route re-audit and Product4 Reverb continuation boundary are recorded
in `ROUTE_AUDIT_20260904.md`. Its executable conclusions take precedence over
stale historical `cycles/clean*.json` entries still marked `planned`.

## Goal

Build a maintainable, multi-family Wet-to-Clean restoration foundation. It is
not RAT-specific. The client eventually loads independently maintainable family
experts and device adapters, reverses arbitrary effect graphs, supports repeated
families and unusual orders, and preserves explicit Clean profiles.

The current phase is model/data work only. Do not connect UI and do not access
physical audio input/output devices.

## 2026-09-05 active boundary

Reverb and Amp are frozen by explicit user decision. Do not train either family, search for or
download new family-specific corpora, open their locked-final evidence, or schedule another
hypothesis until the user explicitly reopens that family. Historical Reverb and Amp results below
remain rejection/diagnostic audit evidence only. Retain Amp v30 and the rejected v31/v32 records
unchanged. The active path is generic Drive plus Dynamics, Echo, EQ and non-Reverb/non-Amp chain
integration.

## 2026-09-05 foundation expert packaging

Drive, Dynamics, Echo and generic EQ are now four separate development packages under
`models/catalog/foundation/`. The package schema now distinguishes true offline
`loss_preserving_restore` inverse packages from the older analysis-only control-recovery inverse
packages; forward quality is not borrowed. Every runtime rejects missing, extra, non-finite or
out-of-domain controls, preserves the Wet input bytes, and accepts no graph order or neighbouring
effect input.

`runs/foundation/product5-expert-packaging/chain-acceptance.json` applies the three analytic
forward effects in all six permutations, then lets the separate graph executor traverse the known
stages in reverse. Every expert appears in every chain position and all six rows pass. Maximum
absolute Clean error is `2.3526e-6` against a frozen `2e-5` bound; minimum RMS-error reduction is
`99.9985%`. The three manifests materialize, verify all artifact hashes and load independently.
The EQ package retains the accepted epoch-0 checkpoint because its bounded analytic inverse beat
all trained residual epochs on calibration; no further EQ training is justified.

The separately trained Drive v3 checkpoint is now packaged under the same contract. The extended
audit at `runs/foundation/product6-drive-packaging/chain-acceptance.json` covers all 24 permutations
of Drive, Dynamics, Echo and EQ. All 24 rows pass the unchanged quality2 gate, every Drive chain
position passes separately, and aggregate median spectrum/high-band/transient/dynamics error
reductions are 36.68%/95.35%/99.15%/82.36%. The Drive expert receives only its current Wet and
explicit drive/bias/cutoff/level/shape controls; graph order and neighbouring effects are absent.

This closes the synthetic known-control integration task, not the product. It does not infer effect
presence, controls or topology, and does not establish physical-device, listening or locked-final
acceptance. `usable_model` remains null. The next chain gate must use licensed real multi-effect
evidence without changing the four expert contracts.

A bounded follow-up search checked two especially relevant ToneTwist records through the official
Zenodo API. The physical RAT-like Harley Benton Rodent record (`10.5281/zenodo.10796378`) and the
parameterized 808-Scream record (`10.5281/zenodo.10901401`) are both CC-BY-NC-4.0. The latter is a
software emulation; the former is a single physical pedal, and both reuse a heterogeneous,
partly time-varying-gain dry pool. They are registered metadata-only and were not downloaded. They
cannot influence product weights or close the physical multi-effect veto, so the data blocker is
real rather than a reason to keep training on the existing synthetic set.

The later non-Amp/non-Reverb resource audit is frozen at
`runs/foundation/product9-nonamp-nonreverb-resource-audit/audit.json`. It found no public source
that combines product-safe rights, physical multi-effect hardware, aligned Clean/Wet, and enough
stage/control information to evaluate restoration. In particular, the official pOD-set API reports
`CC-BY-NC-4.0` (not an unresolved license): its 27 analog overdrives remain metadata-only and its
45.1-GB collection was not downloaded. DAFx remains software presence/order evidence and EGFxSet
remains normalized single-effect recognition evidence. The physical-chain gate is therefore
data-blocked, not compute-blocked.

The next non-training gate tested whether Drive controls could be narrowed from Wet alone using the
accepted inverse followed by repository-owned forward replay. The frozen MPS audit is
`runs/foundation/product7-drive-control-identifiability/audit.json`: 324 candidates, 12 independent
synthetic examples and a top-12 interval contract fixed before evaluation. It is rejected. Only
1/12 true tuples appeared in top-12, shape top-1 accuracy was 33.33%, cutoff coverage was 33.33%,
and the median replay boundary margin was 1.25%; all five gates failed. Do not tune this result or
emit point knobs. The packaged Drive inverse continues to require explicit controls.

A second frozen audit separated exact knob identification from the actual Wet-to-Clean objective.
`runs/foundation/product11-drive-topk-recoverability/audit.json` ran the same 324 candidates and
12 examples on MPS, ranked candidates only by Wet replay error, and used truth/Clean only after
ranking. It is rejected at the earlier capacity gate: true controls pass quality2 on 8/12 examples,
and even a Clean oracle over all 324 candidates finds a passing restoration on only the same 8/12.
Replay Top-12 contains a passing Clean on 5/12 and Top-1 on 1/12. The four unsalvageable examples
are primarily low-cutoff cases and fail spectrum/high-band recovery, which is consistent with
information removed by the forward low-pass rather than graph order. Therefore selector training
is not justified: it cannot repair the 4/12 cases for which no candidate works. This closes the
current finite-grid inverse-forward Graybox route, not explicit-control Drive or the
order-independent expert contract. A future blind Drive attempt must add genuinely new observation
evidence or a stronger source prior and must be screened by a frozen capacity gate before training.

The accepted Wet-only nonlinear presence path is now separately materializable as
`foundation-drive-presence` under `models/catalog/foundation-presence/`. It packages only the
frozen nonlinear family expert and any-effect Clean gate; the historical shared Ambience head is
not present, and the API has no order or control input/output. Development nonlinear recall is
80.19%, while the 1,716-example external random-position software-chain Drive recall is 98.68%.
However, the revealed synthetic single-effect count-1 stratum reached only 34.78% recall. Therefore
this remains an opt-in development classifier: a positive may propose Drive, but a negative must
not silently bypass restoration.

A single pre-bounded MPS hard-example pass is retained at
`runs/foundation/product8-drive-presence-hard-v1/metrics.json`. It trained only the standalone
nonlinear head for four epochs, selected from epoch 0-4 on calibration, and opened development once.
The selected epoch raised development count-1 recall to 78.26% and low-Drive recall to 85.71%, but
weak-residual recall remained 38.10% and negative false-positive rate rose to 5.84%, above the frozen
5% gate. The attempt is rejected; no checkpoint was written, the packaged expert is unchanged, and
additional epochs or weight tuning on this revealed split are closed. This localizes the remaining
problem to near-Clean observability rather than chain order or a generic lack of training time.

The follow-up impact audit at `runs/foundation/product10-drive-presence-bypass/audit.json` also
rejects treating a negative detection as harmless. It replayed the unchanged packaged nonlinear
expert and any-effect gate over all 640 development examples, then inspected the immediate Drive
stage pair for the 42 missed synthetic positives. Zero of 42 passed the pre-frozen bypass gate.
Median missed-stage gain shift was 8.73 dB, spectral error 0.1913, high-band error 1.7522 and
transient error 0.5147. Therefore the existing `negative_may_bypass: false` rule is empirically
necessary; neither more training on this split nor a no-op shortcut is justified.

## 2026-09-05 fixed-profile physical Amp v30/v31

The EG-IPT physical domain is now modeled as one explicit device expert rather than mixed into a
generic Amp model. `runs/foundation/product3-amp-eg-ipt-fixed/evh-mesa-sm57-v30-formal/model.pt`
is a 59,214-parameter eight-stage inverse for EVH 5150 III 50W 6L6, Mesa 4x12 V30 and close SM57.
The device profile is baked into the weights; content-dependent profile estimation, profile IDs,
Clean, graph order and neighbouring effects are absent at inference. It inherits the licensed
rusty-amp stage prior and was fine-tuned on MPS using only audited CC-BY-4.0 EG-IPT pairs.

On 342 fit-only internal-calibration examples, v30 selects epoch 8, reduces calibration loss from
3.0704 to 1.0141, reaches 78.65% complete per-example passes, and improves median spectrum/high
band/transient/dynamics errors by 63.49%/90.72%/64.15%/52.97%. All three pickup groups pass; four
short-transient techniques fail. CPU RTF is 0.0445. A separate reservation excludes every pair
sampled by v30 training or calibration and retains 49 unseen pickup-technique cells across 17/19
techniques. v30 reaches 75.80% on its 343 deterministic windows with all aggregate metrics passing;
the formal report is `runs/foundation/product3-amp-eg-ipt-fixed/v30-reserved-v2-evaluation.json`.

A frozen-base short-memory transient refiner v31 lowers average reserved loss by only 7.6% and
regresses complete passes to 74.05%; it is rejected and its checkpoint was deleted. The v30 weight
is retained as the strongest fixed physical Amp diagnostic, but EG-IPT is fit-only and the reserved
set lacks two exhausted techniques. No Demo or product promotion is allowed until an independently
licensed physical validation source passes.

The bounded follow-up did not find a second product-safe recording of that same EVH/Mesa/SM57
chain, so this is now an explicit data blocker rather than permission for more EG-IPT tuning. A
single cross-tone factorized gray-box screen was run after reviewing EG-VAE (arXiv 2608.05513).
The product path remained the licensed eight-stage inverse; a small forward renderer existed only
during training, and checkpoint selection used Clean restoration loss only. v32 improved that loss
by 8.22% (`2.4100 -> 2.2118`) but reached only 4/24 = 16.67% unseen-profile passes. Its tone latent
collapsed (same-profile and different-profile cosine similarity were both about 0.9999), and all
four replay anti-leakage gates failed. The checkpoint was deleted; no formal run, fresh-v2 opening
or Demo is allowed. This closes factorized/cycle-assisted Amp work at the current scale.

## 2026-09-05 Product4 frequency-profile Reverb result

The next genuinely different gray-box route is complete through the development gate. It estimates
frequency-bin decay from the current effect's RIR, builds eight bounded response-shortening
candidates, and learns only a convex time-frequency selector with Wet identity available. Clean is
used only by the training oracle/loss. Inference receives Wet, the current Reverb RIR/profile and
its mix/room-gain controls; it has no graph, order or neighbouring-effect input.

The frozen Clean-oracle candidate-bank audit is
`runs/foundation/product4-reverb-frequency-shaping-oracle-v1/metrics.json`: all aggregate, source,
room and decay gates pass. OpenSLR 26 simulated RIRs are admitted under Apache-2.0 with strict
whole-room splits: Room001-120 fit, Room121-150 calibration and Room151-180 development;
Room191-200 are permanently external-only and Room181-190 remain unused. The compact source and
split audit is `runs/foundation/product4-reverb-openslr26-data-audit-v3.json`.

An OpenSLR external-only check of the BUT-trained v2 checkpoint failed 9 of 26 room groups and is
retained as negative evidence. The replacement mixed BUT/OpenSLR MPS run is
`runs/foundation/product4-reverb-frequency-profile-v3-mixed-formal/ambience/metrics.json`. Its
16,120-parameter epoch-4 checkpoint passes the frozen 192-example development gate: 83.13% tail
pass, 59.97% median tail-excess reduction, 59.22% median tail-envelope improvement and zero added
reverb; both Clean sources, all four reported room/category groups and all three decay strata pass.
Checkpoint SHA-256 is
`c5d4e30ed69cf500aa9f9ca9b5e3ad1c8e6a933a352658b93f562b34f9046929`.

This is the strongest Reverb development candidate, not a product model. The follow-up runtime
audit is `runs/foundation/product4-reverb-frequency-runtime-v1/metrics.json`: finite overlap-add
candidate generation matches full convolution within 9.32e-9, and the complete candidate-bank plus
selector path reaches CPU RTF 0.0623 on a 3.62-second window. Profile kernel design is cached once
per current RIR/control state and is reported separately. Runtime is therefore sealed; listening
acceptance, fresh physical-room validation and locked-final remain. The compact two-file listening
package is `demos/reverb-frequency-profile-v3-ab-20260905-v1.zip`; its fixed BUT-measured and
OpenSLR-simulated shaping rows were selected by domain/mode, never by quality score. `usable_model`
stays null.

Fresh physical validation capacity is reserved but unopened. The University of Rochester v3
Figshare record declares 14 measured rooms, 48 kHz 32-bit mono WAV and CC-BY-4.0. It is registered
as `rochester-rir-fresh-v2` with validation allowed and all training/calibration/selection uses
blocked. No files were downloaded; file-level checksums and decay-domain admission must be frozen
before a one-shot evaluation, and that evaluation must wait for the listening decision.

### 2026-09-05 v3 listening rejection

Human listening rejected the frozen package: the BUT measured-room case sounded wrong, while the
OpenSLR simulated-room case produced only a slight effect. Post-decision diagnosis confirms both
fixed rows had no measurable tail and failed their whole-example automatic gates. The BUT row still
received 11.01% of the Wet-to-Clean correction distance and regressed high-band error; the OpenSLR
row corrected only 1.88% and failed meaningful-correction. The package builder exposed the gap by
selecting fixed shaping cases without requiring measurable per-example evidence.

v3 is therefore frozen as `development-gates-passed-listening-rejected-not-promoted`. Do not tune
it on these revealed rows, do not download Rochester, and do not open locked-final. The replacement
calibration-only contract is now executable in `remix/reverb_quality2.py`: when both tail and active
foreground evidence are measurable, processing must pass tail, active-spectral and universal
per-example gates; otherwise deployment must reproduce Wet within 1e-6. Aggregate eligible-tail
medians may no longer hide unassessed examples.

The frozen v3 checkpoint fails that contract on calibration in
`runs/foundation/product4-reverb-frequency-profile-v3-acceptance2-calibration-v1.json`: only 21 of
102 measurable examples are effective, no example hard-bypasses, and only 10.94% of all 192 rows
are safe. This is not merely a weak selector. A whole-candidate Clean oracle and the stronger true
time-frequency convex-hull Clean oracle both attain the same 21/102 (20.59%) effective coverage
after exact Wet fallback; all source, room and decay groups fail. The decisive upper bound is
`runs/foundation/product4-reverb-frequency-profile-acceptance2-tf-oracle-calibration-v1.json`.
Consequently the entire eight-candidate frequency-shortening family is closed. v4 must change the
physical inverse itself and pass this calibration gate before any development, listening, fresh
Rochester or locked-final access.

### 2026-09-05 profile-direct v4 boundary

The materially different v4 direct complex-residual model was trained on MPS with explicit
current-transfer replay consistency. Its 73,891-parameter screen learned on fit (loss 3.17 to 2.40)
but added zero effective non-exact calibration rows across eight epochs, so direct learned
deconvolution from profile features alone is closed as under-conditioned. The negative report is
`runs/foundation/product4-reverb-profile-direct-v4-capacity-smoke-v2/metrics.json`.

Restoring the gray-box analytic prior changes the result: the bounded exact/regularized current-
profile inverse reaches 21/21 effective measurable rows in the 48-row calibration capacity smoke,
with exact Wet fallback for all other rows. On the full 192-row calibration audit, 86 candidate
outputs are effective against 91 measurable rows. The core inverse therefore has enough capacity,
but its deployment gate does not. A frozen observable envelope gate safely releases only 33/91
(36.26%). Two tiny Wet/profile-only safety MLP attempts are also rejected; the corrected shared-
source v2 releases 9/91 (9.89%) while preserving exact hard bypass elsewhere. Reports are
`runs/foundation/product4-reverb-profile-direct-v4-observable-gate-v1.json` and
`runs/foundation/product4-reverb-profile-gate-v4-calibration-v2/metrics.json`.

Do not open development for this candidate and do not train another gate on the same partitions.
The current Reverb blocker is reliable runtime-observable abstention, not inverse capacity. A next
attempt requires materially new evidence or a self-verifying physical signal, not more classifier
capacity or threshold sweeps. Rochester, listening rows and locked-final remain sealed.

Two materially new self-verification attempts now close that possibility on the current evidence.
`runs/foundation/product4-reverb-graybox-replay-audit-v1.json` tests exact re-convolution of each
analytic candidate through its current transfer. Effective and ineffective inverses have strongly
overlapping replay residuals (including near-zero residuals on both sides), demonstrating the
finite single-channel inverse non-uniqueness: forward consistency cannot identify the hidden Clean.

The MPS-trained time-frequency dry-audio prior in
`runs/foundation/product4-reverb-audio-prior-gate-v5-formal/metrics.json` uses Wet, the analytic
candidate, current transfer profile and controls, with no Clean/order/neighbour input at inference.
Its threshold was frozen on a 192-row fit holdout before one-shot evaluation on the existing
calibration split. It safely released only 14/102 measurable holdout rows; calibration released
12/91 and made five unsafe releases (97.40% safe). It is rejected and its checkpoint was deleted.
No development, Rochester, listening or locked-final row was used. Learned gates and physical
replay gates are therefore closed; further Reverb progress requires a different observation model
(for example additional channels or a measured dry reference), not more classifier capacity.

## 2026-09-04 Product4 Reverb addendum

The BUT ReverbDB contract is now whole-room-disjoint, release-event-aware, at least 65,536 target
frames, and balanced across Clean source and every available room-by-decay cell. The earlier v12
and its passing short-window oracle are superseded. The valid long-window v15 single-WPE run and
corrected oracle fail; a four-geometry WPE oracle still fails EGFx and ConferenceRoom2. WPE and
further mask-capacity sweeps are closed.

The replacement Wet-only graybox uses six causal multiband exponential-decay candidates plus Wet
identity. Its corrected 96-example balanced Clean-oracle is
`runs/foundation/product4-reverb-decay-bank-oracle-v4-balanced/metrics.json`: aggregate, both rooms,
medium and long tails pass, but EGFx reaches only 17.39% and short-tail 45.45%, with short-tail
median envelope improvement only 7.29%. It therefore fails before product training. The earlier
MPS v16 selector was trained before decay balancing, is superseded, and is rejected with no Demo.

A balanced WPE-plus-decay union oracle repairs short-tail performance but leaves EGFx at 17.39%,
so the hybrid blind family is also closed. More importantly, the route audit found that
order-independent had been incorrectly treated as profile-independent. The expert may receive its
own explicit RIR/profile and controls; only graph order and neighbouring effects are forbidden.

The corrected primary Reverb route is therefore known/selected-profile inversion. On Product4,
the exact analytic path is perfect but covers 22/96. The regularized fallback can extend coverage,
but the old Wet safety score mistakenly returned safe when no tail was measurable. That bug is
fixed and tested. The valid result is
`runs/foundation/product4-reverb-known-profile-selector-v2-fail-closed/metrics.json`: every quality
axis passes on accepted examples—including both sources, both rooms and all three decay strata—
but calibration coverage is 45.83% and development coverage 40.63%, below the frozen 50% minimum.
It remains rejected, with no Demo and no locked-final access. The invalid pre-fix selector v1 must
never be cited as product evidence.

Three coverage shortcuts were also closed. A shallow calibration-trained risk classifier appeared
to cover 93.75% safely but accepted six unsafe development fallbacks and failed EGFx/E112/medium.
A calibration-selected fixed 5x regularized inverse added reverb on 8.06% of development tails.
A Wet-envelope ceiling reduced added reverb to zero but failed high-band nonregression. Do not
retune these on development. A finite-latency Wiener candidate then selected 32,768 taps/6x on
calibration, but its fresh development added-reverb fraction jumped to 17.74% despite passing global
nonregression. Static finite-kernel Wiener sweeps are closed too. The next Reverb candidate must
change the profile-conditioned inverse itself.

That change is now implemented as finite-lookahead room-response shortening, following the
channel-shortening literature instead of targeting an exact delta response. It preserves the
profile's direct/early response, accelerates only the late decay, caps inverse gain, and applies
only partial low-band correction. The first 2–4 kHz transition version is correctly rejected in
`runs/foundation/product4-reverb-profile-shaping-upper-bound-v1/metrics.json`: aggregate, both
development rooms and the shaping fallback pass with 100% coverage and zero added-reverb cases,
but calibration EGFx/high-band and medium-tail dynamics fail. Its development result was opened
once and must not be used for further tuning.

The narrower 2–3 kHz calibration-only v2 is frozen at the lowest candidate with margin:
`runs/foundation/product4-reverb-profile-shaping-calibration-v2/metrics.json` selects 0.24
low-band strength. All aggregate, exact/fallback, source, room and decay-stratum gates pass on 96
calibration examples; the 76 shaping fallbacks have 81.48% per-example tail pass, 31.75% median
tail-excess reduction and zero added reverb. This is a credible candidate, not product evidence:
canonical development was consumed by v1. A new room-disjoint external measured-RIR validation is
required before runtime work or a Demo. The candidate remains order-independent: it receives only
Wet, the current Reverb profile/RIR and current controls, never graph order or neighbours.

No Reverb model is currently promotable. The next work is to validate the frozen v2 response
shortener on new licensed measured rooms, then seal a partitioned finite-kernel runtime while
preserving fail-closed abstention. A blind fallback is secondary and needs a genuinely different
physical direction plus a passing Clean-oracle upper bound before MPS.
The executable safety contract is currently tail removal plus conservative whole-clip
nonregression; it must not be called an active-frame foreground gate until that metric is actually
implemented and frozen on calibration. The forward Clone/structural-OOD gray-box investigation
remains complete diagnostic work and must not displace the Wet-to-Clean inverse-expert product.

### 2026-09-04 external-room and profile-bank result

The pending external check is now complete and negative. The frozen v2 shortener was evaluated once
on OK5 without selection or tuning: 108 examples from 12 admitted rooms, with 50 exact and 58
shaping cases. Aggregate quality passed, but shaping fallback pass fraction was 53.33% and five
room groups failed. The candidate is rejected and OK5 is consumed as diagnostic evidence.

The requested gray-box direction was then implemented as a four-candidate known-profile shortening
bank plus a 33,176-parameter convex time-frequency selector trained on MPS. It receives only current
Wet, current RIR and current mix/room-gain controls. Exact stable inverses are forced through
unchanged; Wet identity is always selectable; graph order, chain order and neighbouring effects are
absent. Its internal v17 screen passed every old development group, but the frozen checkpoint failed
a one-shot OpenAIR evaluation on 124 examples from four fresh rooms. The shaping subset reached
53.70% tail pass and 26.24% median tail reduction, with dynamics and short-tail failures. There is no
Demo, promotion or locked-final access.

This is not evidence against gray-box modeling in general; it rejects this particular fixed global
time-decay candidate family and selector. Literature on channel shortening and differentiable DSP
supports a next Reverb hypothesis based on frequency-dependent decay shaping with explicit physical
constraints. That hypothesis must first pass a Clean-oracle upper bound and later use another fresh
licensed room set. Near-term training priority returns to Drive/Amp. Formal status remains
`usable_model: null`; the complete end-to-end model count is zero.

## Current verified state (2026-09-03)

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

The archive's measured `preamp` tap is now part of the audited product data
contract at `runs/foundation/product3-amp-marshall/data-audit-stage-v2.json`.
This enabled a genuinely stage-supervised clean-room inverse rather than
another endpoint-only network. The first 6,811-parameter structural cascade
reached 34.07% development pass fraction after calibration-selected
continuation. Applying the intermediate objective jointly from that state
regressed to 3.70%, so it was not promoted.

The corrected stagewise run is
`runs/foundation/product3-amp-marshall/amp-stagewise-graybox-v6`. It first
trains speaker output to the measured preamp boundary, then freezes that
section while training preamp to input, and finally performs a short joint
refinement. The 10,666-parameter MPS checkpoint has SHA-256
`e70c0c85e0491f0208e9b95eab187212e831fb34d2497ef87787c5e5bc64a75a`
and CPU RTF 0.02294. Development median reductions are 84.35% spectrum,
97.41% high band, 11.78% transient and 30.78% dynamics, with 40.00% complete
per-example passes. This is the strongest current Amp single model, but it
still fails aggregate coverage, every-setting and four control-stratum gates.

A Clean-oracle over six frozen Amp candidates reaches 57.78%, while the
deployable ridge selector fitted only on calibration and observing only Wet
features plus B/M/T/Gain reaches 36.30% on development. The oracle is not a
runtime capability and the selector must not be retuned on revealed
development. No Amp demo was generated, locked-final remains unopened, and
the complete accepted Amp-model count remains zero.

The next Amp source is now materialized without opening any final evidence. EGDB-PG
v2 (Zenodo 19789500) is released by the original EGDB author under CC BY 4.0 and
contains 240 clean DI tracks rendered through 256 software Amp+cab presets. The
frozen contract `remix/egdb_pg_subset_v1.json` now admits 30 fit/calibration profiles
(ten per gain class), three disjoint unseen development profiles, three unopened
locked-final profiles and disjoint content IDs. A single frozen content-diversity extension raised
fit content from 48 to 96 tracks. Exact HTTP ranges materialized 3,444 files and
7,841,019,015 valid bytes; all pass CRC32, 44.1-kHz mono and paired-frame
checks, and locked-final remains undownloaded. The initial eight-way download was
rate-limited, then completed as a missing-only four-way retry with backoff.

The EGDB-PG inverse interface is Wet-only: profile/category IDs are used solely for
balanced sampling and reporting, never as model inputs. Waveform v1 was rejected at
epoch zero because the inherited loss was unstable on quiet DI. The stable spectral
v2 trained on MPS and, after expansion to 30 profiles, reached 15.28% complete passes
on three unseen profiles with 60.67% spectrum and 92.89% high-band median reduction.
The warm-started spectral-plus-temporal v3 reaches 16.67%; its dynamics regression
narrows to 15.24% overall and 5.57% for high gain, but transient improvement remains
4.95% and CPU RTF is 0.326. Both remain diagnostic, generated no Demo and did not
open locked-final. The next justified direction is a separately trained Wet-only
content-invariant tone encoder over 3-5 second clips plus a true reconstruction
decoder; deeper TCN/mask sweeps are closed. The tone-encoder prerequisite was then
tested without opening development: frequency-preserving v2 reached 56.25%
cross-content profile retrieval, wide-negative v3 reached 63.75%, and hard-margin
v4 reached 73.33% with a 0.113 cosine margin. v4 still fails the frozen 80%/0.20
representation gates. The one allowed content96 continuation then selected tone encoder v5 at
83.75% retrieval and a 0.204 margin, passing all representation gates. Freezing it inside a Wet-only
FiLM-GCN inverse raises unseen-profile complete passes from 16.67% to 22.22% and cuts dynamics
regression from 15.24% to 3.38%, with 8.68% transient, 49.64% spectrum and 83.99% high-band
improvement at CPU RTF 0.114. The decoder still fails aggregate, per-example, every-category and
every-profile gates, so it is rejected and generated no Demo. Both encoder and decoder are now
frozen; the next attempt requires a materially different reconstruction mechanism and fresh
validation, not more tuning on the same split. This remains software Amp+cab
evidence, not physical-device validation and not a Marshall speaker-output target.

That fresh boundary has now been exercised once. Before any candidate training, three new profiles
and 24 disjoint tracks were reserved as fresh validation v1. Only after the complex-mask v5 model
was frozen by SHA-256 was that audio materialized. The one-shot 72-example result is rejected:
15.28% complete passes, with 59.43% spectrum, 91.01% high-band and 17.47% transient median
reductions but 11.85% dynamics regression. All provenance, Wet-only/order-independent, checkpoint,
runtime and sealed-data gates passed; aggregate dynamics, per-example, every-category and
every-profile gates failed. No Demo was generated and locked-final stayed absent.
After consuming fresh v1 as declared fit evidence, the current corpus audit verifies 3,540/3,540
files and 8,059,422,253 bytes across 36 Wet profiles. Fresh v2 and locked-final remain absent.

Fresh v1 was then explicitly consumed as additional fit data, while a second disjoint three-profile,
24-track fresh validation set was reserved in the contract and left undownloaded. The resulting
complex v7 model trained on MPS, reduced calibration loss by 20.80%, runs at CPU RTF 0.090 and
reaches 23.61% complete passes on the previously revealed development set. Its frozen-base bounded
envelope refiner v8b reaches 29.17% there and makes all four aggregate median reductions positive,
but only 21/72 examples pass and its calibration improvement is below the frozen threshold. Neither
candidate is allowed to open fresh v2. Category-aware routing generalizes to only 60.42%; the v7/v8b
oracle union is only 31.94%; explicit attack shaping, a long-context TCN and compact frozen/joint
U-Nets also failed their smoke/calibration gates. These lines are closed rather than promoted.

The capacity-scaling hypothesis was then tested directly with v13, an 8,271,870-parameter joint
complex-plus-Demucs model with 32,768 target frames, 8,192 frames of context on each side and a
3.09-second bottleneck receptive field. It trained on MPS and remained inside the single-expert CPU
budget at RTF 0.173, but four bounded epochs improved calibration only from 3.6851 to 3.5329
(4.13%), below the frozen 10% prerequisite. Development, fresh v2 and locked-final were not opened.
This closes raw capacity scaling; a future Amp attempt must change the factorization itself, such as
explicit gray-box parameter estimation plus model-based inversion or a separately justified
spectral-to-waveform stage.

That gray-box branch is now implemented rather than merely proposed. v14 uses the frozen Wet tone
encoder only to estimate parameters for three explicit LTI inverse stages and two asymmetric
monotonic dynamic inverse stages. It improves calibration by 25.34%, runs at CPU RTF 0.023 and
reaches 22.22% complete passes on the old development set. Its zero-phase symmetric FIR restriction
left transient recovery at 7.16%, so v15 replaced those filters with non-symmetric, phase-generating
LTI stages. v15 improves calibration by 28.88% and reaches 34.72% complete passes: median spectrum,
high-band, transient and dynamics reductions are 60.79%, 92.32%, 20.90% and 6.66%. It is the best
generic software Amp diagnostic so far, but aggregate dynamics remains below 10% and only 25/72
examples pass. A final frozen-v15 explicit dynamics refiner v16 selected epoch zero on calibration
and was stopped at smoke. No fresh v2, locked-final or Demo was opened. Further generic dynamics
post-filtering is closed; a next gray-box attempt needs better per-example parameter identifiability
or independently supervised internal stages.

A frozen-checkpoint observation-duration audit then compared 3.0, 5.5 and 8.0 seconds of the same
Wet recording on the already revealed development split. Complete pass fractions were 34.72%,
34.72% and 36.11%. Eight seconds slightly improves spectrum but does not clear aggregate dynamics;
longer observation alone is not accepted as the next mechanism and does not justify fresh v2.

Clean-assisted capacity audits then separated parameter estimation from processor capacity without
changing production inference. A support-track Clean profile oracle overfits its support rows and
drops old-development complete passes from the frozen v15 baseline 38.89% to 33.33%. Even the
deliberately contaminated same-audio Clean oracle reaches only 55.56% after 360 steps, still far
below the frozen 80% individual gate. The tiny official EGDB-PG preset archive was also checked: it
maps UUIDs only to `low_gain`, `crunch` and `high_gain`, with no knob, stage or topology parameters.
Both the Wet estimator and the five-stage processor family therefore remain limiting; profile IDs
or the broad gain label cannot repair them.

Open Riff Box commit `c980ae874c87e16835d87b265ea58079fc69e7f5` provided a useful isolated
architecture probe: its GPL-3.0 source includes an offline Platinum renderer with nine internal taps
from V1A through output transformer. The source was built locally, and 12 fit-only EGDB Clean clips
produced 120 distinct float32 no-cabinet, noise-disabled stage files. The registry keeps this source
research-only: the GPL program, temporary renderer, audio and learned parameters cannot enter
product weights. A 166,948-parameter, Wet-only nine-stage inverse trained on MPS with per-stage
waveform, envelope, crest and positive-attack supervision; its three-track research calibration
passes all four aggregate gates at 66.67% complete passes. No research weight was saved.

The same architecture was then completely reinitialized and trained only on product-authorized
EGDB-PG data as v21. At equal v15 budget it improves calibration 29.25% and reaches 38.89% complete
old-development passes, with median spectrum/high-band/transient/dynamics reductions of
61.12%/92.88%/27.59%/12.84% at CPU RTF 0.074. This is now the strongest generic software Amp
checkpoint, and all four aggregate metric thresholds pass, but only 28/72 examples pass; no gain
category or unseen profile passes the 50% summary gate, so product acceptance still fails. A v22
attack-emphasis repeat ties 38.89% but is worse on high band, transient and dynamics, so its weight
was pruned. v21 is retained as diagnostic evidence only. Fresh v2, locked-final and Demo remain
unopened.

External architecture review and the completed v13-v22 evidence close both raw capacity scaling and
generic post-filter sweeps. RemFX is relevant as architecture evidence (Demucs for distortion,
long 5.5-second chunks and a strong waveform loss), but its source code and model weights have
different rights: the repository is Apache-2.0 while Zenodo record 8218621 marks the pretrained
weights `cc-nc`. Those weights are therefore research-only, were not downloaded and cannot enter
product gradients or a redistributable model. Formal status remains `usable_model: null`; accepted
complete Amp and end-to-end model counts remain zero.

### 2026-09-05 product-licensed eight-stage Amp gray-box result

`rusty-amp` was audited at pinned commit
`831d6bba4ad3a1a8da958c510cbc96a9e55b73f9`. Its Apache-2.0 no-cabinet Marshall algorithm passed
181 source tests and exposed eight deterministic stage boundaries. Built-in/external cabinets,
demo audio, Audio Units and third-party plugins were excluded. Because source comments describe
tuning against commercial reference rigs, this is admitted only as a product-compatible
architecture/control prior, never as independent physical validation.

The 168,638-parameter Wet-only reverse model was deeply supervised on ephemeral stage renders and
then evaluated in two independently meaningful target domains. Marshall v24 reaches 57/135 =
42.22% development passes, improving the previous physical v6 result by 2.22 percentage points but
still failing aggregate, per-setting and per-stratum promotion gates. Scaling the synthetic stage
corpus fourfold produces v25 at only 40.00%, so more renderer rows/epochs are closed.

The same accepted v23-v1 pretrain was then fine-tuned once on the frozen EGDB-PG contract as v26.
Calibration improves from 3.3791 to 2.5194, but unseen-profile development falls to 22/72 = 30.56%
versus v21's 28/72 = 38.89%. All three gain categories and all three unseen profiles fail. This is
negative transfer, so selector, epoch and hyperparameter sweeps are forbidden. Fresh v2,
locked-final and Demo remain unopened. Retain v24 as the strongest new physical diagnostic and v21
as the strongest generic software Amp diagnostic; neither is usable. The next Amp attempt requires
an independently licensed physical source/domain or a self-verifying Wet parameter estimator, not
another synthetic-stage scale-up. The model contract remains one Wet-only, order-free effect expert.

An official-source follow-up identified an already licensed but previously DI-only local corpus as
the missing independent physical domain. EG-IPT's official Zenodo record applies CC-BY-4.0 to the
complete archive and documents simultaneous DI plus an EVH 5150 III 50W 6L6, Mesa 4x12 V30 and
2.5-cm SM57 chain. ZIP byte-range extraction downloaded only the three contiguous SM57 blocks
(about 4.0 GB) rather than the 23.8-GB archive. CRC verification recovered 8,717 pairs / 4.7266 h;
three named HB-neck bottleneck-slow SM57 members are absent upstream and explicitly excluded.

The first envelope-lag screen was preserved as rejected because periodic single-note content caused
false correlation peaks. The corrected v1 acquisition-alignment audit uses only broadband scratch
and snap-pizz transients: median lag is 64 frames at 96 kHz, p95 deviation 28.3 frames and maximum
30 frames. All-pair geometry, meaningful-effect and zero-clipping gates pass. The source is admitted
for product Amp fit only; it is never its own product validation because it is one player, guitar,
fixed Amp/cab setting and microphone.

v27 added 864 EG-IPT physical crops per epoch to the frozen v26 EGDB contract. It genuinely learns
the new domain: fit-only internal EG-IPT calibration rises from 17.54% for v23 to 48.54%, with all
four median reductions improving. Generalization nevertheless fails: EGDB development is 26/72 =
36.11%, below v21, and Marshall development is only 9/135 = 6.67%. Even a non-promotable Clean
oracle across v6/v24/v27 reaches only 67/135 = 49.63%, so selector training is closed. v27 is pruned;
the licensed EG-IPT pairs and audit remain for a future domain-invariant or explicitly
self-verifying mechanism. No Demo was generated.

Guitar-TECHS (Zenodo 14963133) is a CC BY 4.0 source containing direct input
and fixed Amp+cab+mic recordings. P1 and P2 passed the corrected pair audit
with configured lags of 35 and 37 frames, but the current project registry
admits Guitar-TECHS only as a Clean/family source—not for `train-amp`. The first
correct-direction diagnostic comparison is now complete at
`runs/foundation/clone-real-guitar-techs-v4/metrics.json`: it maps direct input
to Amp+cab+mic output, excludes P3, uses time-disjoint fit/development rows,
and trains the neural baseline on MPS.  On aggregate development absolute ESR,
the 64-tap FIR is 0.3433, the 4,786-parameter causal LSTM is 0.3937, and the
dynamic gray-box is 1.2298; none is promoted and no checkpoint or demo was
created.

A calibration-only FIR selection was then rerun under a stricter leakage
contract at `runs/foundation/clone-real-guitar-techs-v6/metrics.json`.  The
common 192-tap, ridge-0.01 candidate won calibration at ESR 0.2304 but reached
0.4239 on its untouched development draw; because v4 and v6 use different
random crops, this is not a paired comparison and does not demonstrate an
upgrade over the v4 64-tap baseline.  The
earlier v5 run is retained and explicitly marked invalid because it computed
development probes for every candidate before finalizing selection.  v6 is
diagnostic evidence only.  The follow-up alignment audit at
`runs/foundation/clone-real-guitar-techs-alignment-v1/metrics.json` found
fit-interval residual-lag outliers but stable calibration/development pairs.
A fit-only alignment filter was selected honestly on calibration, yet its
fresh development ESR remained 1.1849; it is retained as rejected diagnostic
evidence at `runs/foundation/clone-real-guitar-techs-alignment-filter-v2`.
The resulting coverage-controlled LSTM run is
`runs/foundation/clone-real-guitar-techs-lstm-coverage-v2/metrics.json`: it
uses 64 fit windows/profile, selects 26 epochs on calibration, and evaluates
64 fresh development windows/profile.  On that same challenge the LSTM's
absolute ESR is 0.2115 versus 0.7997 for the paired 64-tap FIR, with CPU RTF
0.1966.  This is a useful forward-clone signal, not a Wet-to-Clean model or a
product promotion; its checkpoint is diagnostic and `usable_model` remains
null.  The next step is one justified inverse training design, not another
blind capacity sweep.

That inverse step is now recorded at
`runs/foundation/clone-real-guitar-techs-inverse-lstm-v1/metrics.json`.  It
trains the same 4,786-parameter causal LSTM on MPS with the explicit
Wet-to-direct-input direction, 64 fit windows/profile, and calibration-selected
10 epochs.  On 64 fresh development windows/profile it reaches absolute ESR
0.4582 versus 0.6457 for the paired 64-tap FIR; P1 is 0.6146 and P2 is 0.3019.
The frozen Amp gate rejects it on both calibration and development: spectrum
and high-band improve, but transient and dynamics regress and pass fraction is
0.0.  A restoration-aware loss was tested at
`runs/foundation/clone-real-guitar-techs-inverse-restoration-loss-v1`; it still
fails transient/dynamics.  A same-budget residual LSTM reduced development
ESR to 0.1514, but it mostly preserves Wet and fails all four required Amp
improvement gates; the exact checkpoint reload is recorded at
`runs/foundation/clone-real-guitar-techs-inverse-residual-gates-v2`.
These are inverse diagnostic checkpoints, not product models: the named-profile
scope is narrow, the P1 tail remains weak, and listening, physical veto and
locked-final gates have not run.  `usable_model` remains null.  Stop blind
LSTM/loss/capacity sweeps on this Amp line.

The promised gray-box inverse is now recorded at
`runs/foundation/clone-real-guitar-techs-inverse-graybox-v1/metrics.json`.  It
uses the existing 24-branch causal envelope state bank with a calibration-only
choice among five small FIR-memory/ridge configurations, and maps only one
effect Wet plus its fixed named profile to the immediate predecessor.  The
selected 576-parameter `24-tap / ridge 0.1` model reaches development ESR
0.5835 versus 0.6091 for the paired 64-tap FIR, but the frozen Amp gate rejects
both calibration and development: pass fraction is 0%, and spectrum,
high-band, transient and dynamics all fail.  The compressed coefficient
checkpoint was independently reloaded and reproduced development ESR exactly;
there is no gray-box promotion or demo.  This closes the promised gray-box
check rather than hiding its negative result.  A next Amp attempt needs a
newly justified state/forward-replay design or better aligned, product-safe
coverage, not another blind sweep.

The same bounded refresh also confirmed that Guitar-TECHS is not new evidence: its archives and
prior inverse/gray-box failures were already present locally, and the current use-specific gate
still blocks Amp-restoration gradients. EGFxSet remains unsuitable for sample-faithful inversion
because per-file normalization destroyed the original level relation. Neither source was reopened.
The cache was reconciled with the registry: six unused, never-extracted P1/P2 category archives and
the forbidden residual P3 archive were deleted (3.6 GiB). Only the audited P1/P2 single-note ZIPs
and their extracted read-only corpus remain.

The separate repository-owned Structural-OOD forward benchmark is now at
`runs/foundation/clone-structural-ood-v7/metrics.json`.  Its fit-only BLA and
history audit is `runs/foundation/clone-structural-ood-bla-v1/metrics.json`.
The audit finds nearly fixed transfer shape for synthetic DUTs A/C, mild
level/control drift for B, clear history dependence for D (relative output
delta 0.1161), and the strongest level/control drift for E.  This is a
structural diagnosis only: no product audio, external audio, physical device,
analytic inverse or product gradient is involved.

The follow-up complexity-controlled selection is
`runs/foundation/clone-structural-ood-complexity-v1/metrics.json`.  It fits
source IDs 0--5, selects on source IDs 6--7 using the lowest parameter count
within 5% of the best calibration absolute ESR, and only then creates and
reads source-, level-, control- and topology-OOD rows.  The selected known-DUT
experts are A=dynamic gray-box (576 parameters), B=Parallel Hammerstein (448),
C=dynamic gray-box (576), and D/E=Parallel Hammerstein (448).  Source-OOD ESR
is 0.0473 and control-OOD ESR is 0.1359, but level-OOD ESR is 3.5119 with a
188.97 worst case; A-to-hidden-D/E topology transfer is 0.6330.  The result
confirms that calibration fit can look excellent while amplitude extrapolation
is unsafe.  It is diagnostic evidence only and does not create a usable model,
an inverse checkpoint, a Demo or a product promotion.

The targeted bounded-nonlinearity follow-up is
`runs/foundation/clone-structural-ood-bounded-ph-v1/metrics.json`.  It keeps
the same 448-parameter branch budget and replaces the unbounded polynomial PH
terms with bounded symmetric and asymmetric `tanh` branches.  Calibration
selects the bounded candidate for D/E and the original PH for A/B/C.  Level-OOD
ESR improves from 3.5119 to 0.3893, but its p99 remains 11.06 and the worst
case 14.90; B alone has mean 1.57.  Source-OOD is 0.0373, control-OOD is
0.1280, and A-to-hidden-D/E transfer is 0.6535.  The result supports the
diagnosis that unbounded polynomial extrapolation was a real failure mode, but
does not solve hidden-topology or safe out-of-domain behavior.  It remains
diagnostic forward-clone evidence only.

A final tail-aware variant of that screen is retained at
`runs/foundation/clone-structural-ood-bounded-ph-v2/metrics.json`.  Within a
5% calibration-mean tolerance it chooses the lower calibration p99 candidate,
which reduces level-OOD ESR to 0.3489 (p99 9.28, worst 12.44); source-OOD is
0.0375, control-OOD is 0.1305, and topology transfer is 0.6538.  This is a
small stability improvement, not a resolution of unknown topology or a
promotion criterion.  The bounded-PH screen is now closed; further work must
address structural identification/profile replay or new licensed aligned
coverage rather than continue basis or selection sweeps.

The one allowed complexity escalation is retained at
`runs/foundation/clone-structural-ood-residual-v1/metrics.json`.  A fixed
zero-initialized bounded 24-unit causal GRU residual was trained on MPS on the
fit rows.  It was accepted only when calibration absolute ESR mean strictly
improved and p99 did not worsen: A/C/E accepted it, while B/D automatically
fell back to their structured bases.  The final selected result is source-OOD
0.0370, level-OOD 0.3477, control-OOD 0.1262, and A-to-hidden-D/E transfer
0.6557, with CPU RTF 0.1548.  The residual's local gain and small transfer
regression do not justify further neural sweeps.  This closes the
`BLA -> structured identification -> bounded complexity selection -> one tiny
residual` diagnostic route; all outputs remain synthetic forward-clone
evidence, not product weights, inverse checkpoints, Demos or a usable model.

The observable profile-replay audit is
`runs/foundation/clone-profile-replay-v1/metrics.json`.  Each unknown
synthetic DUT is assembled as a separate profile, but topology and split
metadata are stripped before fitting or selecting.  The fixed support guard
accepts source/control OOD at 100% coverage and rejects all level-OOD because
the observed peak is 0.28 with a 5% margin to 0.294.  Its identity fallback
has level-OOD ESR 0.6678; this is an explicit unsupported boundary, not a
restoration-quality guarantee.  Hidden-topology transfer remains 100% admitted
at ESR 0.6557.  Profile replay therefore makes the coverage/abstention
contract explicit but does not establish a universal or product model.

The real-data forward-replay inverse trial is
`runs/foundation/product3-amp-guitar-techs/amp-cab-forward-loss-p1-v1/metrics.json`.
The historical weight-only gate treated P1 as product-weight compatible; the
current use-specific gate rejects the same source for `train-amp`. The trial
froze the fit/calibration-selected causal
forward clone, and added one replay-consistency loss to the existing
1025-tap inverse.  Calibration selected epoch 8, but the fresh development
pass fraction was 70.83%, below the 80% product target.  Candidate replay ESR
was 0.0547 versus 0.1506 for clean passed through the imperfect forward clone;
that inversion of the expected ordering proves replay consistency can reward
forward-model bias and is not an audio-quality gate.  P2 was not sent through
this route because its forward replay was worse than the direct identity
baseline.  The design is closed with no Demo, promotion or `usable_model`.
P3 had already been reserved as locked-final in the local registry, but its
audio was mistakenly decoded during the first alignment audit.  No model,
threshold or selection used P3; it is permanently marked contaminated, its
local audio and archive were deleted, and it must never be used as final
evidence.  P3 remains excluded by the current loader and no current run
decodes it.

The deterministic Reverb follow-up removed the rejected v3 neural residual and
screened bounded frequency-domain profile inverses.  On development, a 4x/6x
gain ceiling could individually pass 22 of the 51 examples rejected by the
exact causal path, but no runtime-observable safety selector generalized.  A
calibration-perfect 25/25 selector fell to 7/17 on disjoint development RIRs
and added reverb on two accepted examples.  It is rejected and must not be
retuned on development.  Evidence is retained at
`runs/foundation/product3-ambience/ambience/deterministic-base-screen.json` and
`deterministic-safety-selector.json`; the existing exact stable-profile
boundary remains the only admissible Reverb path. It is a safe subdomain, not
a promoted whole-product model.

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
- ToneTwist and pOD-set: blocked or research-only because rights are mixed, noncommercial, or
  missing. RemFX source code is Apache-2.0, but its Zenodo pretrained checkpoints are separately
  `cc-nc`; architecture inspection is allowed while the weights remain research-only and absent.
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
  pairs. Only its Clean/family roles are currently product-admitted; P1/P2 Amp
  pairs remain diagnostic and are blocked from `train-amp`. P3 was accidentally
  decoded by an alignment audit before training, is permanently contaminated,
  and has been removed locally; it is forbidden from model selection and final
  evidence.
- EGDB-PG v2: CC BY 4.0 paired DI and software Amp+cab renders. Product use is
  restricted to `train-amp-cab`/`validate-amp-cab` with attribution; source audio
  is not bundled by default. The local subset audit proves 30 fit/calibration and
  three disjoint development profiles while locked-final profiles remain absent.
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
