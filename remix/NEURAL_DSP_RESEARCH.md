# Neural DSP research notes for Muspector Remixer

This note separates Neural DSP's published engineering from product claims and
records what is applicable to Muspector.

## What is public

- Neural Capture V1 measures target-device latency, plays recorded test signals,
  analyzes the response, and trains a compact model on the device. V2 moves
  training to Cortex Cloud and targets higher-resolution dynamic behavior,
  especially fuzz, compression, and tube-amplifier touch response. The exact V1
  and V2 network architecture and loss are not public.
- Neural DSP's controllable-amplifier paper defines aligned training triples
  `(input audio, target output, physical control positions)`. It samples the
  continuous knob space randomly, orders physical knob movements efficiently,
  and trains one conditioned forward model to interpolate unseen settings.
- Its published experiment used roughly 4.5 hours of 48 kHz paired audio in
  one-second segments, a one-layer 32-cell LSTM conditioned on five normalized
  controls, and error-to-signal-ratio loss. Expert listening tests compared it
  with a high-quality offline SPICE model.
- Earlier Neural DSP/Aalto work supports residual sample-level LSTM or causal
  WaveNet models for nonlinear devices. The residual path makes the network
  learn the effect delta rather than regenerate the dry signal. That work also
  found that source diversity matters more than repeating one performance.
- Their tone-stack work uses a control-conditioned network to predict a stable
  state-space filter. This supports a hybrid design rather than forcing every
  linear effect into an opaque recurrent network.

Primary sources:

- https://neuraldsp.com/manual/quad-cortex
- https://neuraldsp.com/news/introducing-neural-capture-version-2
- https://arxiv.org/pdf/2403.08559
- https://acris.aalto.fi/ws/portalfiles/portal/41964332/Real_time_guitar_amplifier_emulation.pdf
- https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_58.pdf
- https://patents.google.com/patent/US20230119557A1
- https://data.epo.org/publication-server/rest/v1.2/patents/EP4375985NWA1/document.pdf

## Consequence for Muspector

The current Remixer bundle is an inverse identifier:

```text
aligned Clean + Wet + active effect families -> order + physical controls
```

It is not yet a Neural DSP-style controllable forward renderer:

```text
Clean + order + physical controls -> reconstructed Wet
```

The reference/alternate/stress DSP bank remains valid for inverse-model search
and audit. It must not be presented as a learned hardware clone. A future
high-fidelity renderer should use the recovered controls as conditioning and be
trained on aligned real-device triples. Start with Drive using a causal residual
model and ESR plus multiresolution spectral validation. Retain analytic or stable
state-space Delay/Reverb paths until real long-memory captures demonstrate a
neural replacement is better.

## Data decision

The 2026 ASRNN physical-device dataset is now the first justified additional
download. Its Zenodo API record explicitly licenses the data CC BY-NC 4.0 and
contains 1.3 GB of 48 kHz stereo Dry/Wet files for ProCo RAT,
Darkglass Duality Fuzz, and Boss CS-3. Local decoded files are one second for
RAT and three seconds for DFZ/CS-3; the paper's general one-second description
must not override the actual file geometry. RAT varies Distortion, Filter, and Volume
over normalized 0-100 controls. The associated paper uses TBPTT with 2,048-sample
steps and batch size 32, and confirms that the official evaluation source uses a
different guitar from training.

This data can support an internal non-commercial real-hardware RAT adapter pilot.
It cannot support commercial/release-compatible weights and it has only official
train/eval splits: calibration must be carved from train, while eval is an
external source-disjoint development test rather than an independent locked
final. RAT Filter is a clockwise high-cut, so Muspector maps Tone as
`1 - filter / 100`.

Rechecked2026-08-31: [the updated ASRNN paper, section3.2](https://arxiv.org/html/2509.15622v2#S3.SS2)
states that DFZ varies **fuzz blend** and Filter, with Level maximum and the
separate dry blend fully wet. Our legacy key `blend` is the fuzz-circuit
balance, not dry/wet mixing. CS-3 varies Attack with Level/Sustain maximum and
Tone minimum. These are metadata clarifications, not changes to normalized
labels, accepted weights or their immutable evidence. The paper includes a
Neural DSP-affiliated coauthor; it is public research, not disclosure of the
proprietary Neural Capture production architecture.

EGFxSet remains useful real-device recognition evidence but normalization and a
single published setting per device exclude it from sample-faithful conditioned
forward training. ToneTwist remains per-record/non-commercial research data.
pOD-set is not downloaded because its Zenodo Rights field is empty and its
per-pedal normalization plus Wet-only structure are weaker for phase-faithful
adapter training.

The final missing evidence is still a new aligned, session-disjoint Capture Pack
with train/calibrate/valid/locked-final. The ASRNN pilot tests whether the current
adapter can learn a real circuit before that collection exists; it does not grant
a product fidelity claim.

## ASRNN RAT pilot result

The downloaded archive passed the published byte count, MD5, ZIP-integrity, and
path-safety checks. The RAT subset contains 1,024 train and 128 official-eval
clips (0.32 hours total), covers all three controls from 0 to 100, and has zero
exact Dry-audio duplicates across official splits. Source PCM was read directly;
no normalization, limiting, resampling, re-encoding, or audio-device access was
used.

The 12-state residual adapter was rejected. A locally trained bias-free direct
1x32 model improved over a physical volume baseline but still reached only
0.462 mean and 0.436 median per-file ESR on official eval, so it is not promoted.
The official results archive was then used strictly as an external reference.
Its stable 4x8 GFB model reached 0.0242 global ESR and 0.063/0.034/0.202
mean/median/p95 per-file ESR, with exact silence under static and time-varying
controls and sample-equivalent streaming. The selected checkpoint was converted
with safe tensor-only loading into an independent runtime made solely from
standard PyTorch operators. The converted model reproduces the reference metrics
and passes Muspector's strict internal pilot gates. The GPL-3.0 author code is not
copied into Muspector's Apache-2.0 source boundary; the model remains CC BY-NC and
cannot become a commercial or release claim. A separately licensed Capture Pack
and independently trained locked-final model remain mandatory for that step.

## RAT inverse-control result

The inverse pilot uses paired Dry/Wet features as an initializer and then holds
the accepted stable 4x8 forward model fixed while optimizing controls against
the observed Wet signal. The selected 4,096-frame refinement recovers Volume on
all official-eval files at 1.76/4.23 knob points MAE/p95. For the 106 audible
files, Distortion reaches 2.66/9.53 points and passes the frozen semantic gate.
Tone reaches 8.85/28.29 points and fails. Twenty-two quiet Wet clips are marked
unidentifiable for upstream Distortion/Tone rather than scored as if their knob
labels could be recovered from an inaudible output.

Closed-loop mean ESR is 0.044 and MAE improves 94.5% over bypass, but recovered
controls can compensate for forward-model error; that reconstruction result is
not evidence of exact semantic settings. Doubling the refinement context to
8,192 frames did not materially improve Tone, and an explicit spectral-transfer
inverse regressed. Exact Tone recovery is therefore frozen as unresolved until
new, separately licensed captures provide controlled audible Tone sweeps across
multiple source programs. The machine-readable decision is
`runs/asrnn-rat-inverse-summary.json`.

## Multi-device stable-forward result

The standard-operator runtime now supports one, two, or three normalized
controls with device-specific direction mapping. The same immutable-source,
silence, streaming, waveform, pre-emphasis, spectral, and peak gates were run
against all three official stable 4x8 seeds for DFZ and CS-3. DFZ reaches
global/mean/p95 ESR 0.0185/0.0170/0.0465; CS-3 reaches
0.00963/0.0130/0.0409. Both retain exact silence and sample-consistent
streaming, but fail the frozen 0.02 absolute peak-error P95 gate at 0.113 and
0.0304 respectively.

Train-derived setting gains still fail, and they worsen ESR. A bias-free output
refit with the stable recurrent core frozen fails DFZ before official evaluation.
CS-3 passes the train/calibrate selection but its one-shot official peak P95 is
still 0.0264. No model is promoted and all rejected checkpoints are deleted.
The result rules out post-hoc gain correction; a credible next attempt needs
peak-aware recurrent training plus an independent locked-final Capture Pack.
Evidence is frozen in `runs/asrnn-multi-device-phase4-summary.json`.

## Peak-aware residual result

A final CS-3 ablation froze the stable 4x8 base and trained a 12-state causal
residual GRU. All controls enter multiplied by Dry or base audio; recurrent and
output layers have no bias, guaranteeing exact zero output from zero audio and
zero state. Training used 286 fit files and 572 peak-centered windows, while 66
complete train-derived calibrate files controlled selection.

The adapter changed peak-error P95 from 0.02233 to only 0.02215 and therefore
failed the unchanged 0.02 gate. Global ESR remained approximately 0.0074 and
streaming/silence stayed exact, but neither compensates for the failed peak
criterion. Official eval was not opened, no checkpoint was retained, and this
post-stable residual route is closed. The next credible experiment must optimize
the recurrent forward model itself with peak-aware training and new locked-final
data. See `runs/asrnn-cs3-phase5-summary.json`.

Additional primary sources:

- https://zenodo.org/records/20406285
- https://arxiv.org/abs/2509.15622
