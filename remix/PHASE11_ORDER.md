# Phase 11: admitted order model and full-chain development replay

Work package 3 passes its frozen development gates. Work package 1 (DFZ forward)
is still open; these results do not admit that device's forward renderer or
replace independent work-package-4 evaluation. No UI or physical audio device
is involved.

## Model and runtime

`runs/order-bankfit-bound-phase11/model-card.json` is the admitted package.
The existing inverse network supplies fixed controls and paired embeddings;
interaction/cascade classifiers recover drive relations, and a temporal RBF
classifier recovers the delay/reverb relation. A frozen guarded decision uses
a three-renderer diagnostic bank. Its bounded search changes at most one
diagnostic control (Drive gain +/-3 dB or Delay/Reverb mix +/-7 percentage
points). Those fitted controls and gain NEVER enter the actual ChainSpec.

The package contains only the selected classifiers (about 29 MB compressed).
Head SHA-256:
`280a841a6cff5f93f65be79afee3b903d22b4176260b29bb7924c70aa90d42fc`.
Sources, calibration/development evidence, canonical inverse bundle, numerical
environment and replay file are byte-bound. Unpackaged/unreplayed candidates
are rejected by default. Inference is CPU, Torch threads=2, fixed padded batch8.

## Frozen five-domain development results

| Domain | Eligible examples | Exact order | Pairwise |
| --- | ---: | ---: | ---: |
| Real DAFx | 320 | 79.6875% | 85.8025% |
| Reference | 112 | 83.9286% | 84.8168% |
| Alternate | 148 | 85.1351% | 88.4298% |
| Stress | 196 | 84.6939% | 87.5000% |
| Pedalboard | 122 | 76.2295% | 81.3397% |

The unchanged -30 dB counterfactual relation mask determines eligibility;
unidentifiable records are not relabeled as correct. Real exact increases
5.3125 percentage points and Pedalboard 24.5902 points versus historical
baseline. Every exact/pairwise/no-higher-diagnostic-error condition passes
against both the fixed historical reports and fresh CPU baseline replay.
Calibration selected the menu/head; development did not refit them.

`runtime-replay.json` passes 173/173 raw analysis-pair replays, covering all
118 bank-triggered development examples plus fixed per-domain/family cases.
Order and bank-trigger decisions match exactly; maximum probability difference
is zero and normalized-control difference is 1.1921e-7. Input arrays are intact.

## Actual full-chain audio, not just diagnostic bank scores

`runs/full-chain-phase11-accepted/metrics.json` is passed, with candidate mode
off and the admitted head hash unchanged throughout the run. All 15 nonempty
ordered topologies were checked across reference/alternate/stress/Pedalboard,
then the previously observed challenge renderer.

- Float32 48 kHz / 1024-frame stream maximum error: 5.96e-8 (gate 2e-6).
- Bypass matches input exactly; silence output is exactly zero.
- Recovered reference chain: mean audio improvement versus bypass 64.42%,
  identifiable pairwise accuracy 72.22% (15 cases).
- Recovered challenge chain: mean improvement 63.04%, pairwise 57.14% (6 cases).
- Forward and inverse checkpoints remain byte-identical; sources are read-only.
  No automatic output normalization, limiting, dithering or lossy encoding.

The added same-controls comparison distinguishes bank diagnostics from actual
rendered audio. Reference mean actual error falls from 0.225484 to 0.225028;
challenge rises from 0.404947 to 0.410962 (+1.49%). Thus the frozen phase gates
pass, but a universal audio non-regression claim is NOT supported. The challenge
result is retained without tuning the guard on those examples.

## Input and evidence limits

Inputs are aligned, finite, mono float32 analysis pairs, 44100 Hz and five
seconds. The package uniformly preserves the provided common input level.
Use `TransferRemixerRuntime.load_pair` / `infer_files` for file input, not the
legacy helper's default normalization. Historical DAFx audit pairs were jointly
scaled upstream; generated DSP pairs retained dry peak 0.22. Arbitrary raw-file
level invariance is not established, and per-domain gain guessing is not used.

Synthetic example drive-level preparation and resampling are analysis-fixture
operations, never edits to source recordings or final output gain correction.
The challenge material was already observed before this replay. Tele/locked
final audio stays unopened. These are development results, not arbitrary
hardware-chain cloning, commercial clearance or fresh independent validation.
