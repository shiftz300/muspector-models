# Phase 8: capacity upgrade under unchanged quality gates

The user authorized autonomous architecture changes and continued iteration until
the existing quality gates pass. UI, physical audio devices, source-audio changes,
normalization, limiting, and commercial promotion remain outside scope.

## Architecture screen

Use the existing independent standard-PyTorch stable LSTM importer. Screen the
long-trained, seed-1 1x64 and 4x64 public ASRNN GFB/MAE checkpoints. The same
89-file, near-performance-group-isolated train-side partition is used for
selection. Public pretrained weights may already have seen this official train
partition: it is **not independent final evidence**. No local gradient training
or new locked-final audio is involved in this screen.

| Architecture / source loss | Calibration peak P95 | Worst Attack P95 | Admission |
|---|---:|---:|---|
| 1x64 GFB | 0.040101 | 0.060868 | Fail |
| 1x64 MAE | 0.068471 | 0.088041 | Fail |
| 4x64 GFB | 0.008383 | 0.012512 | Pass |
| 4x64 MAE | 0.008156 | 0.011866 | Pass |

The initial calibration choice is MAE. It fails the 352-file development
challenge: peak P95 0.025923 > 0.02, despite global ESR 0.004564. It also fails
the existing unpaired GuitarSet drift check: delta-RMS P95 0.409859 > 0.35.
It must not be promoted. The remaining long-trained seed-2/3 4x64 weights were
then admitted on the same calibration split and evaluated as development trials.
All eight single-model screen results are preserved in
`runs/asrnn-cs3-capacity-pool.json`, which checks the expected architectures,
losses, seeds, source-archive identity, and split-audit identity before declaring
the pool complete. Selection includes development elimination, not a one-shot
final test. No gradient updates use the official evaluation audio.

| 4x64 source | Development peak P95 | Result |
|---|---:|---|
| GFB seed 1 | 0.023328 | Fail |
| MAE seed 1 | 0.025923 | Fail; GuitarSet drift also fails |
| GFB seed 2 | 0.020663 | Fail |
| MAE seed 2 | 0.026260 | Fail |
| GFB seed 3 | 0.021329 | Fail |
| MAE seed 3 | 0.029437 | Fail |

## Fixed-ensemble architecture

Each member retains its own causal LSTM state; output is a convex waveform
average with coefficients fixed before that candidate's challenge evaluation.
The coefficients sum to one, are not estimated from input peaks/loudness, and
are not per-file post-processing gain or limiting. Models, not recurrent weight
matrices, are averaged. Dynamic controls are fed to every member.

Two 50/50 candidates were admitted on calibration: seed-1 MAE+GFB (peak P95
0.007385, worst Attack 0.011003) and GFB seed-2+3 (0.007554, 0.009289).
The mixed-loss pair fails development peak P95 at 0.022042. The GFB pair passes
the complete 352-file challenge and all ten unpaired OOD checks.

The first ensemble reports incorrectly measured the infinity norm across all
four LSTM gates. They have explicit correction notes: unchanged checkpoint
candidate-state rows `[2H:3H]` were remeasured, consistent with the original
ASRNN importer. Audio metrics and weights were not changed.

PyTorch remains the numeric reference. The same float32 weights can be exported
to standard ONNX LSTM operators with explicit hidden/cell state; there is no
quantization, resampling, limiting, or normalization. CPU ONNX Runtime achieves
RTF about 0.42 on the two-head candidates, with observed Torch/ONNX error below
5.3e-7 and exact dynamic silence. This satisfies the original **CPU RTF <1**
requirement through a faster backend, not a relaxed time limit. Full acceptance
requires the graph's SHA-256 to match both challenge and OOD reports in addition
to checkpoint identity. Numeric reference results alone do not admit an export.

## Frozen admission and provenance

Require the complete existing `evaluate_asrnn_effect._quality_gate`, calibration
peak P95 <=0.02 and worst-Attack P95 <=0.03, and all ten original GuitarSet
checks. Additionally require worst-Attack challenge peak P95 <=0.03, explicit
dynamic-control signal/quiet/silence checks, recurrent infinity norms <=0.995001
(float32 tolerance), and an offline CPU real-time factor below one. Match
checkpoint SHA-256 across every report before promotion. No passing value is
fabricated for an unfinished evaluation.

GuitarSet uses the exact same 60 four-second source segments and three controls
as Phase 7. CPU inference and direct stable rendering replace the slower MPS
zero-centered wrapper for these imported models; the verified source response
to zero is exactly zero, so this removes redundant work, not a quality gate.
The reference remains the exact Phase-7 4x8 GFB seed-3 source checkpoint
`b6f2875d6ba594e429c2a3bdc6adfd9af7c2bededc0fdef5c868b680558569ce`.

Preserve all eight single-model calibration rows and failed challenge/OOD reports.
Retain only the finally admitted checkpoint, with original source provenance,
non-commercial license, and explicit development-only scope. The official
challenge is already observed and cannot become a new locked-final set.

## Admitted development result

| Frozen gate | GFB seed-2/3 ensemble | Requirement |
|---|---:|---:|
| Calibration peak P95 / worst Attack | 0.007554 / 0.009289 | <=0.02 / <=0.03 |
| Development peak P95 / worst Attack | 0.016743 / 0.021991 | <=0.02 / <=0.03 |
| Development global / mean / P95 ESR | 0.002676 / 0.003676 / 0.014841 | <=0.05 / <=0.10 / <=0.25 |
| GuitarSet delta-RMS P95 | 0.321869 | <=0.35 |
| GuitarSet peak-ratio P05 / P95 | 0.846441 / 1.454875 | >=0.50 / <=1.50 |
| GuitarSet maximum peak / absolute DC | 0.736542 / 0.000793 | <=2 / <=0.01 |
| GuitarSet HF-energy P95 | 0.000549 | <=0.10 |
| ONNX full / 1024-frame streaming CPU RTF | 0.413210 / 0.418357 | both <1 |
| Tested Torch/ONNX maximum error | 2.05e-7 | <=2e-6 |
| Static and dynamic silence / ONNX streaming error | exact zero | 0 / <=2e-6 |

The same Phase-7 base's 352-file diagnostic had peak P95 0.030448; the ensemble's
0.016743 is about 45% lower. This compares the same development corpus, not the
different calibration partitions from earlier phases. The 236,160-parameter
ensemble has 1,024 state floats per mono stream. All metrics retain the original
three-second clips and amplitude units; no difficult samples were dropped.

OOD admission is percentile-based, not a promise about every excerpt. The most
extreme retained example has reference peak ratio 3.413 and delta-RMS ratio
0.576; its output peak is 0.737, below full scale. These cases remain disclosed
in `guitarset-ood.json`. There is no paired CS-3 truth for GuitarSet, no listening
test in this phase, and no new independent final evaluation. Public pretrained
weights may have seen all official train data. Do not interpret any result as
arbitrary-pedal cloning, independent generalization, or commercial clearance.

## Reproduction and retained artifacts

The final ensemble embeds these source models from the retained ASRNN archive:

- `results/long/cs-3/checkpoints/StableLSTM_inf-4-64-GFB-2026-03-20T12:07:08-2.pt`,
  source SHA-256 `5b8c35fd2ed495391b627d63cbf46af17f8c2a46f460e1b36d55168dc71f26e3`;
- `results/long/cs-3/checkpoints/StableLSTM_inf-4-64-GFB-2026-03-21T08:48:45-3.pt`,
  source SHA-256 `e04752a3d2d76ea540bbd442c0c2c87532018663dfd06e754eea0da1e0537032`.

Use `.venv310/bin/python` (not the older `.venv/bin/python`) and fresh output
directories. `screen_asrnn_capacity` and `merge_asrnn_capacity_screens` reproduce
the eight-candidate screen. `materialize_asrnn_candidate` verifies archive and
converted-weight hashes before restoring the two admitted components.
`build_stable_ensemble --members <seed2> <seed3> --weights 0.5 0.5` builds and
calibrates the ensemble. `export_stable_effect` exports and tests the unchanged
float32 model. Evaluate with `evaluate_asrnn_effect --device cs3 --onnx <graph>`
and `evaluate_phase7_guitarset_ood --stable --backend cpu --onnx <graph>`, using
the same frozen 4x8 reference, GuitarSet archive, and default 60x4-second/3-control
coverage. `promote_asrnn_capacity` refuses incomplete or hash-mismatched evidence.

The final checkpoint/export and model card reside in
`runs/asrnn-cs3-ensemble-pilot/`. The full experimental reports remain under
`runs/asrnn-cs3-capacity-*`; historical temporary/component paths in provenance
reports are not retained files. Eight rejected weight/export files and the
temporary source/reference imports were deleted. They can be reconstructed from
the intact source archive; no source audio or existing accepted RAT artifact was
deleted. No extracted GuitarSet, rendered audio, or extra downloaded data is kept.

Final QA: the independent PyTorch run also passes all 352 development examples
(peak P95 0.016742618 versus ONNX 0.016742751). All 52 Python regression tests,
compile checks, and `git diff --check` pass. After promotion, both model hashes
and all five evidence-file hashes were rechecked, and the relocated ONNX graph
was reloaded with exact silence. No `.pt` or `.onnx` files remain in the capacity
trial directories. The sole admitted PyTorch/ONNX pair totals 1,916,373 bytes.
The older `asrnn-cs3-stable-pilot` directory remains unchanged as Phase-4 rejected
evidence; it is not the new model's location. No commit or push was made.

## Primary references

- [ASRNN data and model record](https://zenodo.org/records/20406285)
- [ASRNN thesis](https://aaltodoc.aalto.fi/items/766ec0e7-064d-43ea-a63a-59012ace0009)
- [GuitarSet source record](https://zenodo.org/records/3371780)

The thesis reports a stability-versus-modeling-performance tradeoff; this
motivates testing capacity, but does not itself prove that any one width or
depth is sufficient for this dataset.
