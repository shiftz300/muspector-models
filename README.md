# muspector-models

Model research, training, evaluation, packaging, and catalog sources for
Muspector. The desktop client remains in the sibling `muspector` repository.

## Boundary

- This repository owns datasets contracts, training code, evaluation gates,
  package manifests, and static model-repository builds.
- `muspector` owns runtime inference, package download/install/activation, the
  embedded base catalog snapshot, and UI.
- Large datasets, checkpoints, caches, generated packages, and previous runs
  are never committed here.
- Training and evaluation read audio files only. They must not enumerate, open,
  or record from physical audio devices.

The local `data` link points to `../muspector/data` to avoid duplicating the
existing corpus. It is intentionally ignored by Git.

## Layout

- `models/catalog/base`: client-facing common pedal catalog and default set.
- `models/catalog/development`: provenance map for optional research packages.
- `models/inspector`: manifests and notices for the base classifier packages;
  large assets stay in the client checkout until runtime embedding is removed.
- `remix`: forward, inverse, order, packaging, and evaluation research code.
- `train`: classifier training code.
- `cycles`: bounded experiment definitions and acceptance decisions.
- `runs`: ignored outputs from the current repository only.

## Environment

The existing environment can be reused without copying it:

```sh
../muspector/.venv310/bin/python -m unittest discover -s remix -t . -p 'test_*.py'
../muspector/.venv310/bin/python -m unittest discover -s tests -t . -p 'test_*.py'
```

For a clean environment, install `remix/requirements-research.txt` and
`train/requirements.txt` with Python 3.10.

## Train

`remix.train_stable_rat` implements the paper-derived stable conditioned LSTM
constraints independently and trains only from paired files. It supports
checkpoint resume, a frozen initialization point, auditory-filter loss, and
optional ESR, peak, and quiet-signal guards. It never opens an audio device.

Round definitions and promotion decisions live in `cycles`; `rat4.json` records
the full-data refinement and its strict non-regression decision.

## Packages

Import a downloaded or locally supplied model from a declared `model.json`:

```sh
../muspector/.venv310/bin/python -m remix.importer path/to/model models
```

The importer verifies every byte and refuses partial or replacement installs.
For NAM, the manifest must declare its manufacturer, model, family, source,
license, and `.nam` hash. A fixed NAM snapshot may have no pedal controls;
knob recovery remains a separate inverse package. Declared identity is retained
in the installed `package.json` and is never inferred from audio.

Build the default base set from the sibling client assets:

```sh
../muspector/.venv310/bin/python -m remix.migrate dist --set base
```

Build optional development packages from explicitly admitted artifacts:

```sh
../muspector/.venv310/bin/python -m remix.migrate dist --set development
```

Build locally owned Remixer packages separately:

```sh
../muspector/.venv310/bin/python -m remix.migrate dist --set remix
```

The optional Remixer set contains the separately versioned `knobs`, `order`,
and `chain` packages. `chain` 1.1 composes the base family model with a per-
family-set abstention gate, Order 2, knob recovery, and an explicit named-device
boundary. The first RAT adapter uses `ratknobs` 1.0 and delivers physical
distortion, filter, and volume only after the canonical chain accepts a single
Drive family. A development replay delivered 21 of 32 examples at 0.0381 macro
MAE and 0.0948 P95, while seven routed but unidentifiable examples safely
abstained. The replay reused existing ASRNN development data, so `chain` remains
development-only. The historical `route` 1.0 source remains reproducible but is
excluded from the installable Remixer collection and the base catalog. Three
independent multi-author A2 seals and a separate fixed-probe seal rejected exact
RAT identity routing across capture domains. Named device identity is therefore
metadata-first and explicit; audio models may propose a family but cannot infer
a brand or authorize delivery. See `ROUTING.md` for the frozen decision.

Both commands verify byte size and SHA-256 before copying. Package manifests
separate classifier, forward, inverse, and order capabilities; an inverse model
never implies that its renderer is admitted as a forward model.

## Audio quality

Forward packages must preserve frame count, channels, and sample rate, process
in float32, avoid automatic normalization/limiting/dither/lossy encoding, and
pass the declared bypass and streaming checks. Source audio is immutable.

## Clean

Wet-to-clean restoration uses the forward order declared by each chain, not a
single order inferred from genre. Stages are removed in reverse. For example,
a shoegaze chain declared as `reverb, drive` is restored as `drive, reverb`.
An inverse stage is trained to recover its immediate predecessor and receives
upstream effects as context, so the Drive inverse may target a reverberant
intermediate rather than dry guitar.

When the order is unknown, candidates combine order evidence, forward replay,
a clean-guitar prior, and intermediate-state validity. A weak score or a small
margin abstains instead of forcing the common pedal order. `cycles/clean.json`
freezes the current development result and its remaining boundaries.

The historical Python CPU reference uses a bounded spectral-scale prepass followed
by 32,768-frame cores with 4,096-frame context on each side. Whole-record model
inference is forbidden because its activation memory grows with duration. On
the current host, the two-thread 60-second probe ran at 0.0611 RTF with a 99 KB,
23,078-parameter checkpoint and 409 MB process peak RSS. All 128 real RAT
development files matched the whole-record reference within 2.99e-8. This is a
background offline runtime, not permission to run PyTorch in an audio callback;
a native portable export must pass the same gates before client integration.

Round 11 kept the original 23,078-parameter width and froze restoration
strength at 0.875. It was trained without EGFx group bucket 2, which was opened
once for the final 66-file seal. The held-out domain improved aligned ESR by
2.83%, aligned MAE by 3.78%, and SI-SDR by 0.454 dB while improving both P95
tails. The ASRNN wet-only gate accepts 83 of 128 files with zero false accepts
for the declared low-energy criterion; its selected restoration improves
aligned ESR by 14.57% and SI-SDR by 1.73 dB. These are development-domain
claims, not universal RAT inversion or proof of the original absolute level.
The later anti-blur audit rejected model11: median pick-transient improvement
was 0.51% and only 5.94% of eligible real RAT examples passed all perceptual
gates.

Round 13 exports only the learned spectral core to ONNX. The client-side
contract keeps the periodic-Hann STFT/iSTFT, whole-clip scale prepass and
bounded overlap chunks in native code; this avoids unsupported spectral graph
operators and keeps activation memory independent of clip duration. The export
is admitted only after end-to-end PyTorch/ONNX parity, exact bypass, two-thread
CPU speed, memory, geometry and finite-output checks.
The portable contract uses 16,384-frame cores plus 4,096-frame halos so the
combined reference stays below the frozen 512 MB process ceiling.

Rounds 14-17 added independent Drive and Reverb inverse stages. Training now
balances both `reverb,drive` and `drive,reverb` forward orders and always
restores the selected order in reverse. Every target is the paired original
performance, not a canonical clean timbre. The split spans DAFx guitars, EGFx
pickups, GuitarSet players and styles, GuitarJam, and TECHS with zero source-
group overlap; weak order margins abstain.

Round 17 is a rejected near-boundary result, not an installable model. On its
previously unopened 120-example Tele seal it improved absolute ESR by 7.35%,
aligned ESR by 0.825%, and SI-SDR by 0.176 dB while improving final P95 and peak
tails. The frozen aligned-ESR gate was 1.0%, so no checkpoint was promoted.
The same candidate improved real DAFx two-order validation by 3.56% absolute
ESR, 1.20% aligned ESR, and 0.145 dB SI-SDR. Its two ONNX cores total 55,784
parameters and 247 KB; the isolated two-thread chain ran at 0.166 RTF and 492
MB peak RSS. Further tuning requires a new independent seal because Tele has
now been opened.

Round 18 used a fresh DAFx test slice but remained below the aligned-ESR and
SI-SDR gates. Round 19 added 320 read-only DAFx train pairs to the composite
fit and sealed on a different, previously unevaluated test slice (records
64:128). It passed all development, order, external, and runtime gates:
absolute ESR improved 4.25% on the seal, aligned ESR 2.29%, and SI-SDR
0.276 dB. The portable Drive/Reverb pair remains small (247 KB total, 55,784
parameters), runs at 0.179 RTF with two CPU threads, and stayed under the
512 MB process ceiling. That automated acceptance is superseded: DAFx is
admitted for order work, not restoration training, and the later listening and
temporal audit rejected the chain. It is not an installable restoration model
or a physical-device claim.

The replacement architecture is family-neutral and documented in
`remix/RESTORE.md`: shared performance content, separate family experts, named
device adapters, explicit clean profiles and a reverse stage graph. RAT is one
Drive-family pilot, not the system boundary.

The next Delay expansion passed the capacity probe (three stages at 0.211 RTF
and 484.6 MB peak RSS), but a direct wet-only Delay inverse was rejected after
SI-SDR degraded to about -18 dB. Long feedback echoes are not safely
identifiable from wet audio alone, so Delay remains a paired Clean/Wet or
explicit-parameter path until it has a separate locked abstention seal. See
`remix/DELAY_BOUNDARY.md`.

Stage round 2 removes the fixed chain-training dependency. `drive` and
`reverb` now have separate optimizers, immediate-predecessor targets, real
single-effect fit data, and no multi-effect chain gradient. Runtime order is a
request property implemented by `remix.restore.Bank`: the selected forward
order is reversed over the stage registry, so the same two checkpoints support
both orders. Its automated seal improved aligned ESR by 3.81% and SI-SDR by
0.509 dB, but the held-out musical-phrase review exposed that those gates were
too permissive. The output stayed 0.989-0.991 correlated with Wet and closed
only 1.27-2.83% of the Wet-to-Clean waveform distance. Round 2 is therefore
`rejected-listening`, not an accepted or demonstrable restoration model.

All later rounds must also close at least 15% of the Wet-to-Clean distance,
apply at least 25% of the required correction, and point that correction toward
Clean with cosine similarity of at least 0.50. These audibility gates supplement
ESR, SI-SDR, P95, peak, runtime, and audio-integrity gates; they do not replace
them. The lossless phrase files remain in `runs/stage/model2/demo` as failure
evidence.

The next blind-loss probe was stopped before a full run: larger correction
weights did not improve correction direction or closure. The underlying inverse
was missing effect controls, so random Drive and Reverb settings presented
contradictory mappings and rewarded an averaged near-identity answer. Stage
round 4 replaces that interface with separate three-control-conditioned inverse
models. Synthetic rows retain their exact controls; real paired rows fit an
effective control pseudo-label only during training. Wet-only runtime searches
control candidates by inverse-forward replay and abstains when replay and clean-
prior margins are weak. A control-conditioned oracle must pass the audibility
gate before any full real-data run or new demo is allowed.
