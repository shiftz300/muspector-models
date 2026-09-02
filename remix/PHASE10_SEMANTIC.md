# Phase 10: accepted paired semantic inverse development

The three device inverses pass the predeclared semantic and separate closed-loop
development gates. This is not independent final validation, blind separation,
universal pedal cloning, or authorization to integrate UI/audio devices.

| Device / internal control | Development MAE | P95 | Scope |
|---|---:|---:|---|
| RAT Distortion | 0.02665 | 0.09529 | 106 audible of 128 pairs |
| RAT Tone | 0.07214 | 0.21744 | 106 audible of 128 pairs |
| RAT Volume | 0.01774 | 0.05060 | All 128 pairs |
| CS-3 Attack | 0.04474 | 0.15000 | All 352 pairs |
| DFZ Blend | 0.02767 | 0.12500 | All 288 pairs |
| DFZ Filter | 0.02593 | 0.09375 | All 288 pairs |

DFZ's legacy `blend` field denotes the **fuzz-circuit blend**, not dry/wet mix:
[the dataset paper](https://arxiv.org/html/2509.15622v2#S3.SS2) states the separate
dry blend was fixed fully wet and Level at maximum. This clarification does
not change filename labels, normalized estimates or immutable accepted artifacts.

All errors are on the normalized 0–1 control scale. RAT's 22 quiet pairs are
reported, not silently counted as identifiable: upstream Distortion/Filter
requires Wet peak >=0.001. Internal RAT Tone is **1 minus physical Filter**;
the runtime returns both coordinate systems explicitly. CS-3/DFZ reject Wet
below 0.001 because that region is outside their validated inverse domain.

Closed-loop MAE improvements versus unprocessed Dry are 93.84% (RAT), 97.68%
(CS-3), and 91.79% (DFZ). Semantic label accuracy is not inferred from these
audio improvements. In particular, the DFZ search teacher does **not** pass
the separate forward peak-fidelity gate and is not an admitted DFZ clone.

## Architecture and retained packages

- RAT: learned initializer, fixed causal differentiable refinement, and
  independently trained Tone/Volume transfer-feature heads. Distortion retains
  the refined estimate. Package: `runs/rat-identifiable-phase10/model-card.json`;
  entrypoint: `semantic_rat_runtime.RATSemanticRuntime`.
- CS-3: label-free 21-proposal causal waveform search, replaying the complete
  prefix. Package: `runs/cs3-search-gpu-phase10/model-card.json`; entrypoint:
  `semantic_effect_runtime.SemanticEffectRuntime`.
- DFZ: continuous coarse-to-fine two-control waveform search, with steps .25,
  .125, .0625 and .03125 and no label supplied to inference. Package:
  `runs/dfz-search-phase10/model-card.json`; same generic entrypoint.

Inputs are already aligned mono float32 arrays at 48 kHz, one second for RAT
and three seconds for CS-3/DFZ. Inference works on copies, never rewrites source
files, and performs no gain matching, limiting, playback, recording, or device
enumeration. Runtime packages bind accepted evidence and renderer hashes to a
verified compute backend. RAT explicitly uses CPU, never an automatic GPU choice.

CPU audio-only replay matches frozen predictions: RAT maximum difference
2.63e-8 across eight examples; CS-3 and DFZ exact control matches across four
evenly spaced examples each. This is a runtime replay sample, not a claim that
every development pair was re-searched on CPU. Complete semantic and closed-loop
development evidence remains in each run's metrics and acceptance reports.

## Evidence limits

- The observed CS-3 control grid has 11 settings; DFZ has only 3 settings per
  knob (9 combinations). Finer search outputs do not prove interpolation
  fidelity at unseen physical knob settings.
- RAT development has 71/74/74 distinct Distortion/Tone/Volume values over 128
  combinations; its calibration has 63/66/69 over 106 combinations. These are
  finite observed settings, not exhaustive continuous-control validation.
- A 5e-8 comparison tolerance applies only to float32 knob representation:
  CS-3's .15 is represented as .15000000596. Audio thresholds were not changed.
- Public forward teachers may have seen the official training/calibration
  recordings. All currently observed official evaluation data are development
  evidence, not locked-final evidence. No new locked-final audio was opened.
- Model weights/data retain their noncommercial research licensing boundary.

## Reproduction

Run from the repository root with `.venv310/bin/python`. The separate packages
can be replayed using `-m remix.package_rat_semantic --replay-only` and
`-m remix.package_effect_semantic --run <run> --checkpoint <teacher.pt>`.
See each model card for exact retained artifact paths and hashes. Disposable
feature caches and superseded fitted heads have been removed; regeneration
requires the retained audited corpus and corresponding training scripts.
