# Muspector Remixer

This directory is a separate model boundary from `train/`, which remains the
released Inspector family/identity path.

The first milestone is paired reconstruction for Drive, Delay, and Reverb:

- `spec.py` is the versioned chain and physical-control contract;
- `render.py` is deterministic reference DSP for labeled synthetic pairs;
- `data.py` reads the author-provided DAFx25 effect order and generates aligned
  clean/wet pairs whose sampled knob values are retained;
- `model.py` predicts three pairwise precedence relations plus nine normalized
  controls with effect-specific heads; inactive relations and controls are
  explicitly masked. Coarse envelope and lag-correlation features preserve the
  delay/reverb evidence that global spectrogram pooling loses;
- `train.py` is the smoke/full training entrypoint.

`reverb_model.py` is a 74,819-parameter companion head over a deconvolved,
multiscale impulse-energy curve. It restores Reverb Decay, Damping, and Mix;
`train_reverb.py` trains it without changing the order checkpoint.

`drive_model.py` is a 505,091-parameter companion head over a compact
Clean/Wet/Delta time-frequency image plus amplitude quantiles. It restores
Drive Gain, Tone, and Level without changing order or the linear-effect heads;
`train_drive.py` uses calibrate-only checkpoint selection.

`forward_drive.py` is the separate controllable forward-rendering candidate:
Clean audio plus normalized Gain/Tone/Level enters a 4,834-parameter causal
residual LSTM. Gain and Tone condition only signal-multiplicative paths, the
bounded nonlinear stage cannot generate audio from silence, and Level is an
exact physical -18 to +12 dB gain applied last. Recurrent state is explicit and
must be carried between chunks. It never opens an audio device and was trained
only from existing disk WAVs with in-memory reference/alternate/stress renders;
Pedalboard is evaluation-only.

The V7 wide-amplitude pilot passed the fresh guitar-disjoint 160 x 4 one-second
audit. Relative to dry bypass, ESR improved 91.9-98.2% and MAE improved
81.5-87.9%; peak-ratio p95 stayed at or below 1.217 and the worst example at or
below 1.463. A separate 16-cell audit crossed four out-of-training input-level
bands with all four renderer domains: all cells passed, with p95 at or below
1.249 and the worst example at or below 1.376. No runtime normalization was
used. This makes it a synthetic-domain research candidate, not a claim of real
pedal fidelity.

After freezing the V7 checkpoint, the previously excluded `challenge` renderer
was opened once without using the locked Tele/test source split. Its nominal
160-example audit improved ESR by 97.0% and MAE by 81.2%, with peak-ratio p95
1.031 and worst case 1.115. All four challenge input-level bands also passed;
their maximum p95 was 1.096 and worst case 1.209. Challenge results are never
eligible for training or checkpoint selection. `finalize_forward_drive.py`
hash-checks every training, validation, challenge, ONNX, silence, and
audio-device policy gate into one `acceptance.json`.

`export_forward_drive.py` emits a stateful opset-17 ONNX model and refuses the
handoff unless full/chunked ONNX Runtime output matches PyTorch and silence is
exactly zero. The checkpoint, extended audit, ONNX file, and hash-bearing
runtime contract live in `runs/drive-forward-pilot/`.
`tests/drive_forward_tract.rs` separately proves that the generated graph loads
in Muspector's Rust tract runtime, preserves exact silence, and produces the
same nonzero output for full and state-carrying 2,048-frame chunked inference.

`forward_delay.py` is the accepted hybrid Delay forward candidate. Time maps
directly to an integer 48 kHz delay line, Feedback and Mix use the physical
formulas, and only four device parameters are learned: a repeat-path cutoff and
the mixture of identity, first-order low-pass, and second-order low-pass bases.
The pilot learned a 4.60 kHz cutoff. On fresh 2.5-second validation, all four
domains passed: ESR improved 95.6-99.7%, MAE improved 84.7-95.5%, the worst
control-bin ratio was 0.206, and the worst peak ratio was 1.174. The frozen
challenge audit also passed nominal plus four input-level bands; nominal ESR
and MAE improved 96.4% and 85.1%.

The end-to-end inverse-to-forward audit confirms that these gains survive
automatic control recovery. Across reference, alternate, stress, and
Pedalboard domains, oracle-control ESR improved 97.6-99.3%, while recovered
controls still improved ESR by 91.9-95.4% and MAE by 74.1-79.0%. Delay Time
p95 error stayed below 0.100 ms. The stress-domain inverse Feedback and Mix
p95 errors of 0.228 and 0.072 identify control recovery, rather than the frozen
forward renderer, as the next bottleneck. A separately opened challenge
closed-loop audit also passed with 91.6% ESR improvement, 72.9% MAE
improvement, 0.092 ms Time p95 error, and a 1.146 peak ratio. Neither challenge
result was eligible for training or checkpoint selection.

The native streaming contract carries 63 dry-history samples plus the physical
delay line. At 1,024-frame blocks its worst full-versus-streaming error was
2.98e-8; bypass and silence remained exactly zero-error. The learned
coefficients are exported as JSON rather than ONNX because the long memory is a
physical circular buffer, not a neural recurrent graph. `finalize_forward_delay.py`
checks every training, long-horizon, streaming, inverse-to-forward closed-loop,
challenge, hash, and audio-device gate before accepting the candidate.

`forward_reverb.py` is the accepted Phase 1 Reverb forward architecture. It
fits a per-device eight-second impulse profile from a fixed Decay/Damping
calibration grid, compresses the grid to a mean impulse plus eight learned
bases, interpolates the physical Decay and Damping controls, and applies Mix
exactly. The canonical 48 kHz profile is 13 MB. This avoids asking a recurrent
network to memorize 384,000 samples of tail state and gives the native runtime
a stable partitioned-convolution target.

Reverb is evaluated by audible-range Schroeder energy decay and pooled wet-tail
spectrum; sample-aligned ESR remains diagnostic because independently rendered
stochastic tails do not share meaningful sample phase across sample rates. On
the four fitted validation devices, EDC improved 76.4-92.5%, pooled spectrum
55.5-98.6%, and MAE 81.5-92.1% relative to bypass. The architecture-frozen
challenge-device fit improved EDC 94.1%, spectrum 99.0%, and MAE 94.3%. In the
inverse-to-forward closed loop, recovered controls retained 99.1-99.6% EDC and
69.3-87.3% spectral improvement; challenge retained 99.3% and 81.9%.

The 1,024-frame streaming reference matched complete convolution within
3.73e-8, with exact bypass and silence. `finalize_forward_reverb.py` verifies
33 fitting, validation, challenge-adaptation, closed-loop, hash, quality, and
audio-device gates. These synthetic and Pedalboard profiles prove the capture
architecture, not fidelity to an uncaptured physical Reverb device.

```sh
.venv310/bin/python -m remix.evaluate_forward_drive
.venv310/bin/python -m remix.export_forward_drive
.venv310/bin/python -m remix.evaluate_forward_delay
.venv310/bin/python -m remix.export_forward_delay
.venv310/bin/python -m remix.fit_forward_reverb
.venv310/bin/python -m remix.evaluate_forward_reverb
.venv310/bin/python -m remix.export_forward_reverb
```

`delay_model.py` is a 75,464-parameter residual head over the analytic Delay
hints plus the multiscale impulse curve. It keeps deconvolved Time unchanged
and corrects Feedback/Mix across differently colored repeat paths. The current
checkpoint includes Pedalboard replay and improved all four validation domains.
Phase 2 added balanced Delay-only replay without dropping mixed-chain replay.
Its 10,600-example refit reduced mean isolated-control error by 16.2% and mixed
control error by 3.3%. On the larger 40-example-per-domain inverse-to-forward
audit, every domain improved: mean closed-loop ESR fell 5.2% and MAE fell 4.5%
relative to the previous inverse checkpoint. A frozen 40-example challenge
audit then passed with 95.7% ESR and 77.7% MAE improvement. The promoted
checkpoint and rollback copy are hash-checked in `runs/delay-inverse-phase2/`.

All hybrid heads share one paired frequency-domain deconvolution context during
evaluation. `export_bundle.py` validates and packages the four checkpoints into
one artifact plus a hash manifest; this is the model/runtime handoff boundary,
not a UI integration.

`inference.py` is the canonical end-to-end offline reference for that handoff.
Inspector supplies the active effect families; Remixer reads an aligned
Clean/Wet pair without modifying either file and returns the selected topology,
named physical controls, classifier/search evidence, reconstruction error, file
hashes, and the audio-quality policy. It deliberately performs no automatic
time alignment because shifting Wet audio can erase the very Delay evidence the
model must recover.

`differentiable.py` retains an experimental multi-resolution audio
reconstruction loss for Drive and Delay. Its surrogate passes a global
true-versus-wrong audit, but the gradient-training candidates regressed, so the
loss remains disabled by default. Reverb uses the separately validated
deconvolution-feature companion head instead.

The DAFx25 variable-order subset contains all 16 possible topologies over the
three target families. Its varied parameter values are not published, so they
are valid order examples but must never be treated as knob ground truth.

Synthetic controls are rendered across two deterministic DSP domains sharing
one physical `ChainSpec`. Validation reports both domains separately and gives
every control in normalized error plus its physical unit (dB, ms, s, or %).
The first unseen-domain audit promoted `stress` into a third augmentation
renderer after it exposed overfitting. The separate `challenge` renderer is
excluded from every training/calibration path and remains the final synthetic
holdout.

`pedalboard_renderer.py` adds a still more independent, evaluation-only domain
using Spotify Pedalboard 0.9.24 built-ins. It expands semantic Drive into
Distortion/Lowpass/Gain, uses native Delay controls, and maps desired Reverb T60
onto a measured monotonic `room_size` curve. Pedalboard cannot express the
contract's 0.2-0.52 second Reverb range, so those values are excluded rather
than clamped and mislabeled. This optional GPL-3.0 research dependency is not
linked into or shipped with the Muspector application.

The frozen pre-adaptation Pedalboard audit is retained as
`pedalboard-unseen-baseline.json`. Replay adaptation reduced the external
Reverb Decay/Mix and Delay Feedback/Mix errors without touching the locked Tele
split. Reverb Damping remains a documented weakly identifiable control in mixed
chains. Temporal-only and temporal/spectral V2-V4 candidates were audited and
rejected rather than promoted.

Order evaluation additionally supports a -30 dB counterfactual equivalence
mask. It enumerates alternate topologies and masks a relation when flipping it
produces an RMS residual below that threshold. Head-only five-domain refits did
not generalize and were rejected; the production order encoder is unchanged.

`order_search.py` is the accepted precision-order layer. It decodes the nine
recovered controls, enumerates up to six compatible topologies, renders each
through the reference/alternate/stress DSP bank, and ranks gain-aligned Wet
residuals. Search replaces the classifier only when the best residual beats the
runner-up by at least 2%; otherwise it safely falls back. On the expanded
guitar-disjoint validation audit this raised perceptual exact accuracy from
42.9% to 67.9% on reference, 48.0% to 71.6% on alternate, 37.8% to 64.8% on
stress, and 43.4% to 51.6% on Pedalboard while real DAFx moved only 74.7% to
74.4%. `order-search-metrics.json` records the calibration sweep.
The same gate reduced mean gain-aligned waveform reconstruction error from
0.0953 to 0.0863 on reference, 0.1097 to 0.1021 on alternate, 0.1122 to 0.1008
on stress, 0.4624 to 0.4578 on Pedalboard, and 0.72524 to 0.72502 on real DAFx.

Audio quality is a release constraint, not a post-processing preference.
`quality.py` defines it for every model and DSP path: source files are
immutable; mono conversion, 44.1 kHz resampling, and joint dry/wet scaling are
allowed only on disposable analysis copies; rendering must retain frame count,
channel layout, and sample rate; all processing uses float32; and hidden
normalization, limiting, dither, and lossy re-encoding are forbidden. A bypass
chain must be bit-exact. Non-finite audio and geometry changes fail immediately
instead of being silently repaired. Order evaluation also reports the selected
topology's gain-aligned waveform reconstruction error, so an order improvement
cannot be presented without its audio-fidelity result.

These rules do not promise that an intentionally enabled Drive, Delay, or
Reverb leaves samples unchanged—the requested effect is the change. They ensure
that no additional conversion, downmix, gain correction, clipping, or codec
generation loss is introduced around that effect. The Rust edit/save path uses
the same boundary: original files are never overwritten, edits preserve source
rate and channel count in 32-bit float WAV, and project schema v3 stores the
quality policy beside the recovered physical controls.

`NEURAL_DSP_RESEARCH.md` records the primary-source comparison with Neural DSP.
Their published work supports a later conditioned forward renderer trained on
aligned `(Clean, Wet, controls)` real-device triples. The current inverse bundle
and DSP search bank must remain distinct from that future hardware-emulation
model.

`capture_manifest.py` is the admission gate for those future triples. It checks
physical-control bounds, exact post-latency-compensation alignment, source
hashes, relative paths, 48 kHz lossless finite audio, capture-quality evidence,
and device-session/audio/replay-program hash split leakage. Development mode
reads locked-final metadata only: it never resolves, opens, scans, or hashes a
locked-final audio path.
No capture may enter forward-model training without passing this validator.

`data_sources.json` is the source/rights registry. It separates material that
can train a redistributable model from non-commercial evaluation data and
records whose rights are unresolved. `capture_pack.py` then audits the intended
grain (one device/session/setting/replay-program aligned pair), volume, split
integrity, and six-bin physical-control coverage. Unclear rights default to
metadata-only; a repository code license is never treated as an audio license.
Every Capture Pack also carries a top-level dataset identity, source kind,
intended use, and an explicit `owned` or
`authorized-for-device-adapter-training` rights attestation.

`capture_kit.py` is the headless producer for that gate. It keeps the replay
program, loopback recording, and Wet raw recording immutable; estimates session
I/O latency from a low-level chirp recorded with the device bypassed; slices new
sample-aligned Clean/Wet payloads; and refuses files with insufficient source or
Wet level, less than 45 dB capture SNR, less than 6/3 dB Clean/Wet headroom,
clipped samples, digital dropout, or weak latency confidence. It never repairs a
failed capture with gain normalization. The aligned pair is written as float32
WAV only after every gate passes, and its manifest retains hashes for all raw
and derived audio.

Drive collection starts with 192 deterministic Latin-hypercube settings:
144 train and 16 each for calibrate, valid, and locked-final. Every split covers
the full Gain/Tone/Level space independently, uses session-disjoint identifiers,
uses its own replay program, and has one repeated center control per 16-setting
session for hardware-drift checking. Reusing the same performance/program across
splits is leakage even when the device setting differs. Drift anchors are audit
recordings and are not duplicated into model training.

Create the plan after assigning the physical pedal and player identifiers:

```sh
.venv310/bin/python -m remix.capture_kit plan \
  --device-id physical-pedal-serial \
  --player-id player-or-di-source-id \
  --output /path/to/capture-session/drive-plan.json
```

Prepare the assigned split-local 48 kHz replay program, record it once through
the bypass path and once for each setting in that split, then align and admit
each setting:

```sh
.venv310/bin/python -m remix.capture_kit program \
  --source /path/to/48k-diverse-guitar-di.wav \
  --output /path/to/capture-session/raw-program.wav \
  --metadata /path/to/capture-session/raw-program.json

.venv310/bin/python -m remix.capture_kit align \
  --program /path/to/capture-session/raw-program.wav \
  --program-metadata /path/to/capture-session/raw-program.json \
  --latency-capture /path/to/capture-session/raw-loopback.wav \
  --wet-raw /path/to/capture-session/raw-wet.wav \
  --clean-out /path/to/capture-session/aligned-clean.wav \
  --wet-out /path/to/capture-session/aligned-wet.wav \
  --manifest /path/to/capture-session/manifest.json \
  --capture-id physical-pedal-setting-0001 --split train \
  --device-id physical-pedal-serial --session-id train-session-01 \
  --player-id player-or-di-source-id \
  --source-program-id player-or-di-source-id-train-program \
  --rights owned \
  --gain-db 12 --tone 0.5 --level-db -3
```

Run the complete offline data-readiness audit with:

```sh
.venv310/bin/python -m remix.audit_data_readiness
```

The generated acceptance report distinguishes “Capture Pack contract ready”
from “real-device training data ready.” The latter remains false until an
actual manifest passes the development audit with locked-final audio unopened.

After that admission passes, the real-data training boundary is:

```sh
.venv310/bin/python -m remix.train_capture_adapter \
  --manifest /path/to/capture-pack/manifest.json \
  --output /path/to/drive-capture-adapter-run
```

It optimizes only on `train`, selects the checkpoint only on `calibrate`, and
reports `valid` after selection. The generic Drive renderer is frozen. The
manifest and every development audio hash are revalidated after training;
`locked-final` is rejected as a loader split and remains unopened. The complete
192-record synthetic pipeline proof can be rerun with
`.venv310/bin/python -m remix.smoke_capture_adapter`; all temporary audio and
the smoke checkpoint are deleted, leaving only the compact report.

Only after the model, checkpoint-selection rule, and thresholds are frozen may
the one-shot final command be used:

```sh
.venv310/bin/python -m remix.evaluate_capture_adapter \
  --manifest /path/to/capture-pack/manifest.json \
  --checkpoint /path/to/drive-capture-adapter.pt \
  --output /path/to/frozen-run/locked-final.json \
  --open-locked-final
```

The evaluator checks that checkpoint, frozen base, manifest hash, and dataset
identity match. It refuses to open the split without the explicit flag and
refuses to overwrite an existing final report. Once opened, the report marks
model or threshold updates as forbidden; using the result for tuning demotes
that split to development.

The separate ASRNN RAT pilot uses the public 2026 physical-device dataset under
CC BY-NC 4.0. Each 48 kHz one-second stereo WAV stores Dry in channel one and
Wet in channel two; filenames encode RAT Distortion, Filter, and Volume from
0-100. `asrnn_data.py` validates the lossless geometry and maps RAT's clockwise
high-cut Filter to increasing-brightness Tone with `1 - filter / 100`.

```sh
.venv310/bin/python -m remix.asrnn_data \
  data/corpus/asrnn-physical-effects \
  --output remix/runs/asrnn-rat-adapter-pilot/data-audit.json

.venv310/bin/python -m remix.train_asrnn_rat_adapter \
  --corpus data/corpus/asrnn-physical-effects

.venv310/bin/python -m remix.train_asrnn_rat_direct \
  --corpus data/corpus/asrnn-physical-effects

.venv310/bin/python -m remix.import_asrnn_stable \
  --source /path/to/verified/StableLSTM_inf-4-8-GFB.pt \
  --output remix/runs/asrnn-rat-stable-pilot/rat-stable.pt

.venv310/bin/python -m remix.evaluate_asrnn_stable \
  --checkpoint remix/runs/asrnn-rat-stable-pilot/rat-stable.pt \
  --corpus data/corpus/asrnn-physical-effects \
  --output remix/runs/asrnn-rat-stable-pilot/metrics.json

.venv310/bin/python -m remix.benchmark_stable_rat \
  --checkpoint remix/runs/asrnn-rat-stable-pilot/rat-stable.pt \
  --output remix/runs/asrnn-rat-stable-pilot/runtime-benchmark.json

.venv310/bin/python -m remix.train_asrnn_rat_inverse \
  --corpus data/corpus/asrnn-physical-effects \
  --renderer remix/runs/asrnn-rat-stable-pilot/rat-stable.pt

.venv310/bin/python -m remix.refine_asrnn_rat_inverse \
  --corpus data/corpus/asrnn-physical-effects \
  --inverse remix/runs/asrnn-rat-inverse-pilot/rat-inverse.pt \
  --renderer remix/runs/asrnn-rat-stable-pilot/rat-stable.pt \
  --output remix/runs/asrnn-rat-inverse-pilot/refinement.json

.venv310/bin/python -m remix.summarize_asrnn_inverse
```

The official train split is deterministically divided into fit/calibrate;
official eval uses a different guitar and is opened only after checkpoint
selection. This is meaningful real-circuit evidence, but not a product gate:
the license is non-commercial and the dataset has no independent locked-final
split.

The real-data experiment rejected the small residual adapter and retained a
bias-free direct RNN only as a training scaffold. Its full-data official-eval
mean/median ESR is 0.462/0.436; exact silence, zero-volume output, and streaming
parity pass, but waveform fidelity does not. The separately downloaded official
stable 4x8 GFB reference establishes that the task is learnable: the best seed
reaches strict mean/median/p95 ESR 0.063/0.034/0.202, 0.0095 absolute peak-error
p95, exact static and dynamic-control silence, and 5.7e-14 streaming parity.
The selected weight tensors are converted with `weights_only=True` into
Muspector's independent standard-operator runtime; no GPL-3.0 implementation is
embedded. The converted checkpoint reproduces those metrics and is promoted only
as an internal CC BY-NC non-commercial pilot. The frozen decision is recorded in
`runs/asrnn-rat-phase2-summary.json`. No additional public audio is needed now;
the next data gate is a separately licensed Capture Pack for independent
release-compatible training and locked-final evidence.

The canonical pilot has 2,472 parameters (9,888 bytes of Float32 weights) and
64 Float32 state values per mono stream. On the current CPU, full-buffer and
1,024-frame streaming PyTorch realtime factors are approximately 0.447 and
0.443. This passes the offline realtime diagnostic but does not authorize Rust
or UI promotion; export and target-runtime parity remain separate later gates.

Phase 3 evaluated semantic RAT knob recovery rather than inferring success from
waveform reconstruction alone. A learned paired-audio initializer followed by
4,096-frame differentiable analysis-by-synthesis is the selected hybrid path.
On all 128 official-eval clips, Volume reaches 1.76 knob points MAE and 4.23
points p95. On the 106 clips whose Wet peak is at least 1e-3, Distortion reaches
2.66 points MAE and 9.53 points p95. Tone remains outside the frozen gate at
8.85 points MAE and 28.29 points p95, so exact Tone recovery is explicitly
rejected. The other 22 clips are too quiet to identify upstream Distortion or
Tone and are excluded only from those two semantic claims, never from Volume or
closed-loop reporting.

The recovered controls reduce closed-loop MAE versus bypass by 94.5% and reach
mean ESR 0.044, but this is a render-matching result: the optimizer can use
control offsets to compensate for forward-model error. It therefore cannot
override failed semantic knob metrics. An 8,192-frame refinement and a separate
spectral-feature inverse were both rejected. The frozen comparison and decision
are in `runs/asrnn-rat-inverse-summary.json`; no UI integration is authorized.
Resolving Tone needs a separately licensed session-disjoint Capture Pack with
controlled Tone sweeps, multiple source programs, and deliberately audible
output levels, not more blind tuning on the same public split.

Phase 4 generalized the independent stable runtime from RAT-only conditioning
to one-, two-, or three-control physical effects, then evaluated all three
official stable 4x8 seeds for Darkglass Duality Fuzz and Boss CS-3. Their full
data audits cover 576 DFZ and 704 CS-3 lossless Dry/Wet files with zero exact
Dry overlap across official splits. Both best seeds have excellent waveform and
spectral results, exact silence, and streaming parity: DFZ global/mean/p95 ESR
is 0.0185/0.0170/0.0465; CS-3 is 0.00963/0.0130/0.0409.

Neither model is promoted because the frozen absolute-peak-error P95 gate is
0.02: DFZ reaches 0.113 and CS-3 reaches 0.0304. Train-only per-setting gain
cannot close the gap. A final frozen-core, bias-free output-layer refit fails DFZ
on calibration; CS-3 passes calibration but still reaches 0.0264 on its one-shot
official evaluation. All rejected weights are deleted while compact evidence is
retained in `runs/asrnn-multi-device-phase4-summary.json`. Future work requires
peak-aware recurrent training and an independent locked-final Capture Pack, not
post-hoc gain or limiting.

Phase 5 tested whether a small causal residual could repair CS-3 without
touching the stable recurrent core. The adapter is a 12-state bias-free GRU fed
only by Dry, base output, their difference, and audio-multiplied controls, so
zero input remains exactly zero. It trained on 572 peak-centered windows from
286 fit files and was selected on 66 complete calibrate files. Peak-error P95
improved only from 0.02233 to 0.02215, still outside the 0.02 gate. The official
eval split was therefore not opened in this phase, and no checkpoint was kept.
This rejects post-stable residual correction as the next route; future work must
train the recurrent forward model itself with peak-aware objectives. Evidence is
in `runs/asrnn-cs3-phase5-summary.json`.

Phase 6 trained that recurrent model directly. The runtime renders shared-weight
`F(Dry, controls) - F(zero, controls)`, so silence remains exactly zero even
after full recurrent/output fine-tuning; candidate recurrent matrices are
projected to an infinity norm no greater than 0.995. On 286 fit files and 66
complete calibration files, peak-error P95 improved from 0.02229 to 0.01379 and
all calibration/runtime gates passed. Its single frozen official evaluation,
however, reached 0.03087 against the unchanged 0.02 requirement. All other
official waveform, spectral, quiet-input, silence, and streaming gates passed.
The checkpoint and temporary base were deleted. The now-observed official
split is development-only and cannot serve as locked-final evidence. Compact
evidence is retained in `runs/asrnn-cs3-phase6-summary.json`.

Phase 7 completed a bounded robustness experiment with full three-second CS-3
clips, causal TBPTT, two-resolution spectral/peak-envelope losses, and per-example
CVaR. The Dry-feature coverage split contains 263 fit and 89 calibration files;
one near-duplicate performance pair was kept together in calibration. This is
group-disjoint holdout, not exhaustive leave-one-group-out cross-validation.
Two epochs / 306 updates reduced worst-Attack peak P95 from 0.04192 to 0.03748,
but aggregate peak P95 regressed from 0.02792 to 0.03181 (required <=0.02), and
global ESR rose from 0.00748 to 0.00910. Exact static silence and CPU stream
parity (2.24e-8) remain intact. Calibration rejects the candidate; no candidate
development-challenge run is admitted. This does not establish that the small
model has sufficient capacity. The licensed GuitarSet archive adds unpaired
runtime stress coverage, not paired pedal-fidelity evidence. Architecture,
data caveats, and reproducibility are documented in `PHASE7_RESEARCH.md`.
The completed unpaired GuitarSet evaluation covers 180 cases from 60 source
members and passes all ten runtime gates: exact silence and MPS stream parity,
late silence tail <=3.08e-8, and candidate/base delta-RMS P95 0.05286. This is
stability/drift evidence only. The failed candidate and temporary bases were
removed; the authoritative outcome is `runs/asrnn-cs3-phase7-summary.json`.

Phase 8 addresses the capacity/generalization limit through a fixed 50/50
ensemble of the public long-trained stable 4x64 GFB seed-2 and seed-3 models.
Eight single-model screens and one earlier mixed-loss ensemble were retained as
rejected evidence; only the admitted ensemble weights/export are kept. The same
89 calibration files yield peak P95 0.007554 and worst-Attack P95 0.009289.
On all 352 development clips, peak P95 0.016743, worst-Attack P95 0.021991, and
global ESR 0.002676 pass the unchanged gates. The 180-case GuitarSet run passes
all ten checks, with drift P95 0.321869. This is a P95 safety/drift result, not a
claim that every OOD example is close to the reference, nor paired fidelity.

`stable_effect.py` loads either standard checkpoints or fixed convex ensembles
with independent member states. `export_stable_effect.py` preserves float32
weights and exposes all states to ONNX; `onnx_stable_effect.py` enforces CPU-only
execution and checkpoint/graph provenance. Streaming RTF is 0.418 at 1024 frames,
dynamic silence is exactly zero, and tested Torch/ONNX max error is 2.05e-7.
`promote_asrnn_capacity.py` requires matching checkpoint and graph hashes across
the complete calibration/challenge/OOD/runtime evidence. Model card and source
provenance are in `runs/asrnn-cs3-ensemble-pilot/`; scope and reproduction are in
`PHASE8_CAPACITY.md`. No new final audio, UI integration, normalization, limiting,
quantization, lossy output, or physical audio devices are involved.

Live audio-device control remains intentionally outside this first kit: a DAW,
interface utility, or later dedicated recorder may produce the raw files, but
none may bypass the same alignment and quality admission path.

Aligned Clean/Wet Delay Time, Feedback, and Mix are recovered by regularized
frequency-domain deconvolution in `physics.py`. Time comes from the first echo;
Feedback and Mix come from consecutive echo-window energies. They are
deterministic hybrid controls rather than learned logits.

`audit_reverb_decay.py` records the rejected fixed-form Reverb Decay route. A
local-energy T60 fit works on isolated Reverb but is not used because Drive
nonlinearity and Delay feedback make it regress on mixed chains. The learned
companion head retains the full multiscale curve and handles those interactions.

Run the auditable smoke path with:

```sh
.venv310/bin/python -m remix.train --smoke
```

Run the complete offline inference contract with:

```sh
.venv310/bin/python -m remix.inference \
  --clean /path/to/aligned-clean.wav \
  --wet /path/to/aligned-wet.wav \
  --effects drive,delay,reverb
```

Full training is deliberately not automatic. Before a long run, retain the
generated `remix/runs/order-control-paired-public/data-manifest.json`, config,
seed, checkpoint hash, per-topology metrics, and guitar-disjoint test results.
