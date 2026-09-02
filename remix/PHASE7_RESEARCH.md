# Phase 7: cross-guitar transient robustness

## Evidence boundary

The ASRNN record states that its electric-guitar train and evaluation inputs
come from different instruments: an Ibanez RGD-7 ALMS for training and an
Ibanez RG-7321 for evaluation. The Phase-6 CS-3 result therefore measures both
new performances and an instrument-domain shift. Its aggregate waveform and
spectral metrics passed, but absolute peak-error P95 did not.

A fresh per-file diagnostic of the verified stable base found 50/352 challenge
files above 0.02 absolute peak error. The largest errors are underpredicted
high Wet peaks. Peak-error correlation is 0.455 with target peak but only 0.053
with Dry crest factor; the problem must not be described as merely sharp
pick attacks. This diagnostic was collected after the Phase-7 objective and
split rule were frozen and is not used to retune the running experiment.

The Phase-7 audit does not group files by the trailing take number: inspection proved that
the same number under different Attack settings does not reuse the same Dry
audio. Within each of the eleven Attack settings, eight of 32 train files are
initially selected for calibration by maximin coverage over Dry-only peak, RMS,
crest, quantile, and transient features. Exact hashes are insufficient: the
near-performance audit found `80,19.wav` and `80,28.wav` with envelope correlation
0.999817 and waveform correlation 0.989210. Both now belong to calibration,
leaving 263 fit and 89 calibration files. No exact or detected near-performance
pairs cross fit/calibration or train/challenge. This bounded fingerprint check
does not establish complete source/session independence. Wet labels are not
used to construct the split. The earlier, interrupted partition was discarded
without retaining a checkpoint.

The earlier roadmap proposed leave-one-performance-group-out and a mixture of
full clips and reset transient windows. The implemented bounded experiment uses
one fingerprint-group-disjoint coverage holdout, not exhaustive leave-one-group-out
cross-validation. Source-session provenance is unavailable and trailing take IDs
cannot stand in for it. Training uses complete causal clips with local peak-envelope
and peak losses instead of separately resetting peak windows: this preserves the
compressor history at transients. These are deliberate scope refinements, not
claims that the originally sketched cross-validation or window sampler ran.
The earlier assertion that model capacity was already sufficient is not established
by the existing evidence; architecture capacity remains an open hypothesis.

## Architecture decision

Phase 7 retains the zero-centered stable runtime:

```text
Wet prediction = F(Dry, Attack) - F(zeros, Attack)
```

The shared-weight subtraction keeps silence exactly zero after recurrent
fine-tuning. Candidate recurrent matrices remain projected to an infinity norm
of at most 0.995. Training streams complete three-second CS-3 files in causal chunks
and carries detached recurrent state between chunks. At 48 kHz this is 144,000
frames and nine 16,384-frame chunks, with the final chunk shorter. Each chunk
performs an optimizer update; the frozen final run uses two complete epochs,
batch size 16, and learning rate 0.000025 (306 updates). The objective combines:

- waveform and pre-emphasized ESR;
- normalized sample and peak-envelope error;
- absolute and relative peak error;
- 256/1024-bin STFT magnitude error;
- equal weighting of mean batch loss and the worst 25% tail (CVaR);
- an anchor to the verified stable initialization.

The tail objective is per-example CVaR, not a claim to reproduce Group DRO.
Group DRO motivates regularization and early selection when optimizing tails. The
multi-resolution spectral term follows established audio waveform-loss work.

## External pure-guitar audio

GuitarSet v1.1.0 mono pickup audio was selected instead of arbitrary YouTube or
music-platform material. The dataset contains multiple experienced performers,
styles, and excerpts. The official record API independently confirms the dataset
license as `metadata.license.id=cc-by-4.0`; this is not inferred from the paper's
separate copyright notice. The 683,145,360-byte archive is
kept compressed and verified against the official MD5
`aecce79f425a44e2055e46f680e10f6a` and `unzip -t`.

The complete archive audit found 360 unique PCM16 mono WAVs, 3.047 hours of
audio, and 30 comp plus 30 solo files for each of six players. Ninety-four
members have one full-scale sample each. This is recorded as an amplitude
calibration caveat, not asserted to be sustained clipping or silently repaired.

GuitarSet has no matching Boss CS-3 Wet audio. It is therefore used only for
unpaired runtime stress tests: finite output, silence, stream parity, DC,
high-frequency energy, output peaks, and drift from the stable base. It is not
used to train Phase 7 and cannot support a pedal-fidelity claim. Resampling from
44.1 to 48 kHz happens only on in-memory analysis copies; no source or rendered
audio is retained.

## Primary references

- [ASRNN physical-effects data record](https://zenodo.org/records/20406285)
- [Asymptotically stable recurrent neural networks for virtual analog modelling](https://aaltodoc.aalto.fi/items/766ec0e7-064d-43ea-a63a-59012ace0009)
- [GuitarSet v1.1.0 record](https://zenodo.org/records/3371780)
- [GuitarSet dataset license metadata](https://zenodo.org/api/records/3371780)
- [GuitarSet paper](https://archives.ismir.net/ismir2018/paper/000188.pdf)
- [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731)
- [STFT spectral loss for training a neural speech waveform model](https://arxiv.org/abs/1810.11945)
- [Hyper Recurrent Neural Network conditioning for black-box effects](https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_16.pdf)

Hypernetwork conditioning remains a future architecture option. It is not added
in Phase 7 because changing the conditioning topology and the robustness
objective simultaneously would make the source of any gain unidentifiable.

## Calibration result

The frozen two-epoch run completed all 306 updates on MPS. Selection preferred
epoch 2 because it reduced the worst-Attack peak error, but admission rejected it:

| Metric on the same 89-file calibration split | Initialization | Selected epoch 2 | Gate |
|---|---:|---:|---:|
| Aggregate absolute peak-error P95 | 0.027920 | 0.031810 | <=0.020000 |
| Worst-Attack absolute peak-error P95 | 0.041922 | 0.037481 | <=0.030000 |
| Global ESR | 0.007483 | 0.009097 | <=0.050000 |

The worst group improved while aggregate peak error and waveform error regressed.
This run is not an overall quality improvement. Static silence remains exactly
zero and CPU streaming parity is 2.24e-8. The candidate development challenge
was not evaluated because calibration admission failed. The separate base-model
challenge diagnosis remains development evidence. These calibration numbers must
not be directly compared with Phase 6's different 66-file calibration partition.

The final GuitarSet OOD run passed all ten frozen runtime checks across 180
cases (60 four-second members, three Attack settings). Static silence and MPS
stream error were exactly zero; maximum late silence tail was 3.07e-8, maximum
absolute DC 0.001104, and candidate/base delta-RMS ratio P95 0.05286. The latter
is deviation from the base, not CS-3 reconstruction accuracy. Stream/tail values
in the retained example rows are conservative batch maxima. The source archive
remains unchanged. The rejected candidate and temporary imported bases were
deleted; final evidence is `runs/asrnn-cs3-phase7-summary.json`. No candidate
challenge or new locked-final audio was opened. All 47 Python regression tests pass.

## Reproduction and retention

The retained ASRNN results archive supplies
`results/long/cs-3/checkpoints/StableLSTM_inf-4-8-GFB-2026-03-21T07:54:54-3.pt`.
Its SHA-256 is
`b6f2875d6ba594e429c2a3bdc6adfd9af7c2bededc0fdef5c868b680558569ce`.
`import_asrnn_effect.py --device cs3` converts it to the independent stable
runtime. The Phase-7 imported checkpoint SHA-256 is
`e865225ae60b48cb376e9c39236771aa25586b8c2df20b29d328ecfd6b562c44`.

The offline entry points, in execution order, are:

1. `audit_asrnn_phase7_groups.py` and `audit_guitarset_source.py`;
2. `diagnose_asrnn_peak_error.py` for the fixed base on the already-observed development challenge;
3. `train_asrnn_phase7.py` with the frozen configuration above;
4. `evaluate_phase7_guitarset_ood.py` on the selected candidate, diagnostic-only if calibration fails;
5. `evaluate_asrnn_effect.py --centered` only if calibration admits the candidate;
6. `summarize_asrnn_phase7.py` after deleting any rejected candidate weight.

Retain compact JSON evidence and the single compressed GuitarSet source archive.
Remove rejected weights and temporary imported bases; do not retain extracted
GuitarSet copies or rendered audio. OOD evaluates 60 source members at three
Attack settings, using four-second center segments, not all 3.047 hours. No new
locked-final audio, audio hardware, or UI is part of this phase.
