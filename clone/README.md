# Clone Structural-OOD benchmark

This is an isolated black-box forward system-identification benchmark.  The
first round uses repository-owned synthetic DUT observations and compares:

- a 64-tap Linear baseline;
- a 7-branch, 64-tap Parallel Hammerstein estimator;
- a low-order Wiener-Hammerstein estimator;
- a 576-parameter state-bank dynamic gray-box estimator;
- a retained diagnostic variant of the gray-box model plus a 2,041-parameter
  causal residual (not part of the formal v7 comparison);
- a fixed approximately 4.8k-parameter causal LSTM baseline.

The estimators receive only `x`, `y` during fitting, and the scalar control.
They do not receive topology names, renderer internals, hidden parameters or
the benchmark split.  The benchmark reports absolute ESR, pre-emphasis ESR,
multi-resolution spectrum error, MAE, peak error, parameter count, fit time and
ordinary-CPU real-time factor for source-, level-, control- and topology-OOD.
It also runs a quiet-versus-excited-prefix history-dependence probe to identify
when a static model is structurally insufficient.

The next diagnostic stage is a fit-only multi-level/control BLA audit followed
by complexity-controlled selection.  BLA evidence is post-hoc diagnosis only;
it cannot choose a model or see OOD rows:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.audit_bla \
  --workspace /Users/shiftz/dev/muspector-models \
  --output runs/foundation/clone-structural-ood-bla-v1
```

The BLA audit finds nearly fixed transfer shape for A/C, mild level/control
drift for B, clear history dependence for D (relative output delta 0.1161),
and the largest level/control drift for E.  The structured selector then fits
on source IDs 0--5, selects only on source IDs 6--7, and materializes the OOD
splits after selection.  Its rule is the lowest parameter count within 5% of
the best calibration absolute ESR:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.select_structural_model \
  --workspace /Users/shiftz/dev/muspector-models \
  --device mps \
  --output runs/foundation/clone-structural-ood-complexity-v1
```

The selected diagnostic experts are A=dynamic gray-box (576 parameters),
B=Parallel Hammerstein (448), C=dynamic gray-box (576), and D/E=Parallel
Hammerstein (448).  Source-OOD ESR is 0.0473 and control-OOD ESR is 0.1359,
but level-OOD ESR is 3.5119 with a 188.97 worst case; A-to-hidden-D/E
topology transfer is 0.6330.  This exposes unstable amplitude extrapolation,
so it is a negative structural-identification result, not a product model or
an inverse-training checkpoint.

The targeted amplitude follow-up keeps the same 448-parameter budget and
replaces the polynomial PH branches with bounded symmetric and asymmetric
`tanh` branches:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.screen_bounded_parallel \
  --workspace /Users/shiftz/dev/muspector-models \
  --output runs/foundation/clone-structural-ood-bounded-ph-v1
```

Calibration selects the bounded variant for D/E and the original PH for A/B/C.
Level-OOD ESR falls to 0.3893 (p99 11.06, worst 14.90), while source-OOD is
0.0373 and control-OOD is 0.1280.  B remains the weak level-extrapolation
case (mean 1.57, worst 14.90), and hidden topology transfer remains 0.6535;
this is a targeted diagnostic improvement, not a safety or product gate.

A tail-aware rerun is retained at
`runs/foundation/clone-structural-ood-bounded-ph-v2/metrics.json`.  Within a
5% calibration-mean tolerance it chooses the candidate with the lower
calibration p99.  It reduces level-OOD ESR further to 0.3489 (p99 9.28,
worst 12.44), but topology transfer remains 0.6538.  This is the final
predeclared bounded-PH screen; do not interpret it as a universal model.

The only escalation after that structural base is the fixed tiny residual:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.screen_structural_residual \
  --workspace /Users/shiftz/dev/muspector-models \
  --device mps \
  --epochs 12 \
  --output runs/foundation/clone-structural-ood-residual-v1
```

It trains one zero-initialized bounded 24-unit causal GRU residual on fit rows.
The residual is accepted only when both calibration mean and p99 improve;
otherwise the structured base is retained.  It is accepted for A/C/E and
rejected for B/D.  The frozen selected result reaches source-OOD ESR 0.0370,
level-OOD 0.3477, control-OOD 0.1262 and A-to-hidden-D/E transfer 0.6557 at
CPU RTF 0.1548.  The small local gain and slight topology-transfer regression
close this escalation stage; no product checkpoint or inverse model is made.

The profile-replay contract is then audited separately:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.profile_replay \
  --workspace /Users/shiftz/dev/muspector-models \
  --device mps \
  --epochs 12 \
  --output runs/foundation/clone-profile-replay-v1
```

The fitter receives no topology or split metadata.  A fixed support guard
accepts source/control OOD at 100% coverage, but rejects all level-OOD because
the observed peak is 0.28 and the declared support ends at 0.294.  The
identity fallback's level-OOD ESR is 0.6678; it is an explicit unsupported
boundary, not a restoration-quality result.  Hidden-topology transfer is
100% admitted but remains ESR 0.6557.  This closes the profile-replay audit
without claiming a universal or product model.

Run it with the project Python environment:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.run \
  --workspace /Users/shiftz/dev/muspector-models \
  --device mps
```

The output is diagnostic only.  It contains no external audio and makes no
product-model or hardware-fidelity claim.

The first real-data forward-replay inverse trial is recorded at
`runs/foundation/product3-amp-guitar-techs/amp-cab-forward-loss-p1-v1/metrics.json`.
It freezes the licensed P1 forward clone and adds one replay-consistency term
to the existing 1025-tap Amp inverse.  Calibration selected epoch 8, but the
fresh P1 development pass fraction was only 70.83% (below the 80% product
target).  The candidate's replay ESR was 0.0547 while the clean-through-forward
floor was 0.1506, showing that the inverse can exploit forward-model bias;
therefore replay consistency is not an audio-quality gate.  P2 was not trained
on this route because its forward replay was worse than the direct identity
baseline.  Stop this loss design here; no Demo or promotion was made.

The licensed Guitar-TECHS P1/P2 aligned check can be run separately:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.run_real \
  --workspace /Users/shiftz/dev/muspector-models \
  --device mps
```

That run keeps P1 and P2 as independent fixed-profile experts, uses the
repository license gate, and never opens P3.

The real-data FIR selection is deliberately separate.  It chooses one common
tap count and ridge value from the fit/calibration intervals, then reads the
development interval exactly once for the selected candidate:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.select_real_linear \
  --workspace /Users/shiftz/dev/muspector-models \
  --output runs/foundation/clone-real-guitar-techs-v6
```

The v6 result is diagnostic only.  Calibration selected 192 taps with ridge
0.01, but its untouched development draw reached ESR 0.4239.  v4 and v6 use
different random crops, so this is not a paired comparison and does not
demonstrate an upgrade over the v4 64-tap baseline.  The earlier v5 output is
retained as invalid provenance because it computed development probes for all
candidates before finalizing selection.

Before changing the estimator, audit local residual alignment:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.audit_real_alignment \
  --workspace /Users/shiftz/dev/muspector-models \
  --output runs/foundation/clone-real-guitar-techs-alignment-v1
```

The fit interval contains residual-lag outliers, while calibration and
development are stable.  A calibration-selected fit-window filter was tested
at `runs/foundation/clone-real-guitar-techs-alignment-filter-v2/metrics.json`;
it remains diagnostic only because its selected 64-tap FIR reached development
absolute ESR 1.1849.  Filtering helped against the same fresh unfiltered
reference (6.1322), but still did not meet a product gate.

The next controlled run expands the fit coverage and trains only the fixed
4.8k causal LSTM on MPS.  It selects 10/18/26 epochs from calibration and
compares the selected model with a 64-tap FIR on the same fresh development
windows:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.train_real_lstm_coverage \
  --workspace /Users/shiftz/dev/muspector-models \
  --device mps \
  --output runs/foundation/clone-real-guitar-techs-lstm-coverage-v2
```

The v2 LSTM reaches aggregate development ESR 0.2115 versus 0.7997 for the
paired FIR, but it remains a forward diagnostic rather than a Wet-to-Clean
product model.  Its diagnostic checkpoint and hash are recorded in metrics.

The first actual inverse diagnostic uses the same 64/32/64 windows, swaps the
direction explicitly to Wet-to-DI, and trains only the existing LSTM on MPS:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.train_real_inverse_lstm \
  --workspace /Users/shiftz/dev/muspector-models \
  --device mps \
  --output runs/foundation/clone-real-guitar-techs-inverse-lstm-v1
```

It reaches development ESR 0.4582 versus 0.6457 for the paired FIR.  This is
diagnostic evidence only; it is not yet a general Amp inverse, a physical
clone, or a promoted Wet-to-Clean product.

The frozen Amp gate is run from the saved checkpoint with:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.evaluate_real_inverse_gates \
  --workspace /Users/shiftz/dev/muspector-models \
  --checkpoint runs/foundation/clone-real-guitar-techs-inverse-lstm-v1/inverse-diagnostic.pt \
  --output runs/foundation/clone-real-guitar-techs-inverse-gates-v1
```

The MSE inverse fails transient/dynamics.  A restoration-aware loss and a
same-budget `wet + learned correction` residual variant were also tested; the
residual variant reaches ESR 0.1514 but fails all four required Amp improvement
gates because it mostly preserves Wet.  Exact reload evidence is in
`runs/foundation/clone-real-guitar-techs-inverse-residual-gates-v2`.  Stop
blind LSTM/loss/capacity sweeps here.

The promised gray-box inverse is run separately with a small pre-registered
state-bank configuration grid:

```sh
/Users/shiftz/dev/muspector/.venv310/bin/python -u -m clone.benchmark.train_real_inverse_graybox \
  --workspace /Users/shiftz/dev/muspector-models \
  --output runs/foundation/clone-real-guitar-techs-inverse-graybox-v1
```

This uses the existing causal envelope state bank and calibration-only
selection, then reads development.  The selected 576-parameter `24-tap /
ridge 0.1` candidate reaches development ESR 0.5835 versus 0.6091 for the
paired 64-tap FIR, but fails the frozen Amp gate with 0% per-example pass
fraction.  It is diagnostic evidence only; its coefficient checkpoint is
reloaded from the compressed numeric file and no demo is generated.
