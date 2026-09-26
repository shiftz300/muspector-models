# Human acceptance demos

All audio intended for human Wet/Candidate/Clean acceptance lives under this
directory.  Training checkpoints and machine-only metrics remain under
`runs/`; temporary render diagnostics must not be promoted here merely because
training completed.

Each acceptance package must contain:

- lossless, aligned Wet, Candidate and Original Clean files;
- an explicitly documented playback order;
- `demo.json` with exact checkpoint hashes, controls, split and attribution;
- `SHA256SUMS.txt` covering every delivered artifact;
- honest effect/device labeling and a declaration of locked-final access;
- retained failing or hard-tail examples rather than only cherry-picked wins.

The default human delivery format is the historical compact A/B convention:
one file per effect or explicitly labelled hard case, ordered as Wet, 0.75
seconds of silence, then Restored.  Raw Wet/Candidate/Clean stems and progress
comparisons may remain in the full audit directory, but the delivery ZIP should
exclude them unless a reviewer asks for them.

Package directories are immutable evidence.  A changed model, renderer,
selection or safety rule receives a new directory instead of overwriting an old
one.  Locked-final audio remains sealed until every prerequisite gate allows a
single final opening.

Current packages:

- `drive-product2-demo-20260902-v1`: historical v2 development listening set.
- `drive-product3-safe-demo-20260902-v1`: v2/v3 progress comparisons plus v3
  acceptance triples; automatic development gates passed and human acceptance
  is pending.
- `drive-product3-safe-ab-demo-20260902-v1`: compact human delivery package
  following the historical Wet-to-Restored A/B convention.
- `reverb-frequency-profile-v3-ab-20260905-v1`: two fixed, non-score-selected
  shaping cases (BUT measured and OpenSLR simulated), each in compact Wet/silence/Restored order;
  human acceptance was rejected because the measured-room case sounded wrong and the simulated
  case produced only a slight effect. See `reverb-frequency-profile-v3-human-acceptance-20260905.json`.
