# Project archive and model-training status — 2026-09-27

Development and training are paused at the owner's request. This is a snapshot
of the last audited evidence (mostly 2026-09-05), not a newly trained or
re-evaluated model. The historical `HANDOFF.md`, `MODEL_STATUS_20260904.md`,
`ROUTE_AUDIT_20260904.md` and `cycles/` records remain the detailed audit trail.

## Product verdict

- Complete, releasable end-to-end Wet-to-Clean models: **0**.
- Formal promotion state: `cycles/restore.json` has `usable_model: null`.
- Four independent development experts (Drive, Dynamics, Echo and generic EQ)
  have hash-verified package contracts. Their separate graph executor passed
  all 24 permutations in the synthetic, known-control chain. An expert does not
  receive the chain order or neighboring effects. This is not evidence of
  blind chain recognition or physical multi-effect restoration.
- The Drive v3 explicit-control candidate passed its bounded development and
  listening gates. Wet-only Drive presence remains opt-in: a negative result
  cannot safely bypass the stage (0 of 42 missed synthetic positives passed
  the no-op safety gate). Wet-only control replay failed its frozen gate; even
  the Clean oracle over all 324 candidates restored only 8 of 12 examples.
- Amp and Reverb were frozen before archiving. Amp v30 is a fixed-profile
  diagnostic, not a general product model. Reverb v3 passed a development
  screen but was rejected in human listening; v4 has no accepted deployment
  gate. Neither family has a promotable restoration expert.

## Unfinished acceptance gates

There is no rights-cleared, aligned, physical multi-effect chain corpus meeting
the required evaluation contract. Physical-chain veto, general Drive physical
listening, untouched locked-final evidence, final package/runtime seal and
end-to-end blind Wet-to-Clean acceptance are not complete. The planning
estimates in `MODEL_STATUS_20260904.md` (75% engineering foundation, 48%
deliverable product) are historical estimates, not measured product quality.

## Reproducibility and rights boundary

Source code, metadata, cycle decisions, package/catalog manifests, license
registry and written audit conclusions are intended for Git history. The
three Apache-2.0 development checkpoints referenced by the Drive restoration
and Drive-presence package manifests, plus the decisive product5–product11
audit JSONs, are explicitly preserved in this archive. All other ignored
`runs/` outputs are historical local diagnostics, not release artifacts.
The local `data/` corpus, downloaded third-party audio, generated demo
audio and local Python/build environments are **not** part of a normal Git
clone. Source rights are separate for research, model
gradients and audio redistribution; unknown or noncommercial sources are not
silently promoted. If the local dataset and run outputs are removed, historical
models and listening packages will not be reproducible from this repository
alone. No checkpoint hash by itself reconstructs a missing checkpoint.

The application repository contains a development graybox preview and an
attributed GFX classifier, but it does not ship a validated general restoration
model. Keep that distinction when reading old UI or training claims.
