# Restore

The restoration product is not a RAT remover and not one universal Wet-to-Clean
network. It is a graph of independently testable inverse stages. RAT is one
physical Drive-family calibration domain.

## Target

The default target is the paired original predecessor of the active stage. It
preserves the player's timing, pitch, articulation, guitar, pickup, recording
path, frame count, sample rate and level whenever that information is observable.
It must not silently convert every input into one canonical DI tone.

An optional clean profile is a separate explicit condition:

- `original`: recover the paired or inferred source character; this is default.
- `reference`: match a user-supplied clean reference while preserving performance.
- `preset`: target a declared maintainable profile package.

Profile conversion is never used to hide restoration error. Every model is first
evaluated against `original`.

## Layers

1. `family` owns its inverse physics, content features and decoder. Nonlinear, dynamics,
   modulation, echo and ambience mechanisms do not share an output head.
2. `device` is a small named adapter for control semantics and device response.
   RAT, TS9, BD-2 and Big Muff are adapters or validation domains, not separate
   product architectures. Unknown devices remain on the family expert.
3. `profile` conditions the requested clean character independently of effect
   removal.
4. `graph` owns stage instances and forward order. Restoration always walks the
   chosen graph in reverse; repeated families are valid.

The legacy two-entry Drive/Reverb `Bank` remains only to reproduce historical
experiments. New work uses the stage graph contract.

## Experts

| Mechanism | Families | Required inverse behavior |
| --- | --- | --- |
| nonlinear | drive, distortion, fuzz | recover body deterministically; reconstruct uncertain pick/air bands with a trained prior; report uncertainty |
| dynamics | compressor, gate | invert time-varying gain and envelope only when attack/release state is identifiable |
| spectral | eq, filter, wah | invert magnitude and phase with explicit noise amplification limits |
| modulation | chorus, flanger, phaser, tremolo, vibrato | estimate the modulation trajectory before resynthesis; preserve dry transients |
| echo | delay | estimate taps/time/feedback; use stable analytic cancellation where possible and abstain on ambiguous long feedback |
| ambience | reverb | remove early reflections and late tail temporally; a frequency-only Wiener filter is insufficient |
| pitch | pitch, octave | preserve the original voice where separable; otherwise expose ambiguity rather than inventing an exact source |

## License boundary

Every source has three independent permissions: isolated research use, gradients
for redistributable product weights, and redistribution of source or rendered
audio. Missing or ambiguous permission is denied by default. A dataset admitted
for research is not thereby admitted for product training.

- Product experts train only on CC0/compatible clean performances plus
  repository-owned DSP, or on sources explicitly approved for product weights.
  For DAFx and EGFx this means their Clean subsets only; their archived Wet
  recordings remain excluded from restoration gradients for quality reasons.
- CC BY material requires attribution in every resulting model card and catalog
  entry. It is not silently folded into an unattributed base model.
- CC BY-NC, mixed-rights, no-derivatives and Apple AU captures are isolated
  benchmarks. They cannot train, calibrate, select, distill into, or share a
  learned encoder with product weights.
- A research benchmark may veto a product candidate but cannot improve its
  weights. This keeps useful physical evidence without laundering its license.
- Audio redistribution is a separate decision from weight distribution. The
  default package contains weights, provenance and attribution, not datasets.

This is a conservative engineering policy rather than legal advice; written
permission or legal review may move a source to a less restrictive tier later.

## Data

Existing data is sufficient for isolated evaluation, but the product-training
pairs must be regenerated under the license boundary before fitting begins.

- ASRNN supplies isolated noncommercial RAT, DFZ and CS-3 benchmarks only.
- ToneTwisT supplies internal-research named-device diversity, subject to each
  record's rights and pairing audit.
- EGFxSet spans Drive, delay, echo, modulation and reverb for family transfer and
  evaluation, but its normalization prevents sample-faithful inverse claims.
- Locally rendered Apple AU pairs cover generic Drive, Delay and Reverb as
  isolated research evidence; they do not contribute product gradients.
- Pedalboard remains research-only until render-output and model-weight rights
  are reviewed. Product pairs use repository-owned DSP instead.
- Aachen RIR supports product dereverberation training under CC BY 4.0 with
  attribution, convolved ephemerally with product-eligible clean programs.
- DAFx archived Wet remains order-only. Its CC BY 4.0 Clean subset may train
  product stages with attribution, paired only with repository-owned DSP.

Missing aligned physical coverage is currently modulation, pitch/octave,
filter/wah and broader compressor devices. Software-generated modulation pilots
can validate the architecture first; generate legal pairs from admitted clean
programs before seeking more physical hardware.

## Quality

Every stage must pass universal gates against its immediate predecessor:
multiresolution spectrum, high-band detail, pick transients, dynamics, phase,
pitch/onset preservation, clipping, geometry, runtime and human listening.
Each mechanism adds its own gate: nonlinear residual distortion, compressor gain
envelope, modulation trajectory, echo cancellation or reverb tail reduction.

Passing averages is insufficient. Promotion also requires per-example coverage,
hard-tail non-regression, explicit uncertainty/abstention and a musical-phrase
Wet/Candidate/Original comparison. A chain can pass only when every intermediate
stage and the final output pass.

## Order

Order belongs to each request, never to genre. Shoegaze can declare
`reverb -> drive`; the graph restores `drive -> reverb`. Unknown order uses
top-k candidate graphs, forward replay and intermediate validity. Small margins
abstain. Two effects from the same family use different stage slots and may use
different device adapters.

## Runtime

The client loads only active independent family experts and device adapters.
A shared learned content encoder is a later optional optimization that must
first prove no negative transfer against frozen independent experts.
Training-only teachers may be large; delivered experts must
run on an ordinary CPU in background restoration, preserve arbitrary length by
bounded chunking, and keep the whole active graph at RTF <= 0.5. No model runs
inside the realtime audio callback until a separate realtime contract exists.

## Sequence

1. Freeze the stage/profile/package schemas and source-admission matrix.
2. Prove one nonlinear expert across generic Drive plus multiple named devices,
   one dynamics expert across compressor settings, and one temporal expert for
   reverb or delay.
3. Compare an optional shared content encoder only after all three independent
   experts are frozen; adopt it only if every expert passes the no-negative-transfer gate.
4. Add modulation and spectral experts, then pitch effects.
5. Validate repeated-family and unusual-order graphs.
6. Produce demos only after automatic gates pass and before any package promotion.
