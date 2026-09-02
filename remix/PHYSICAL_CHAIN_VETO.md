# Physical multi-effect veto

This is the next unopened gate for the frozen Blind family-presence stack. It
tests real hardware only. It is not a fit, calibration, threshold-selection, or
model-selection dataset, and it is never reused after a candidate sees its
result.

## Invariant

The task is order-independent multi-label presence. Each inverse family remains
an independent effect remover. The manifest stores only the set of present
families (`nonlinear`, `echo`, `ambience`, `unknown`); it must not contain pedal
order, position, sequence, slot, neighbour, or control labels. Capture routing
may vary as a hidden nuisance condition, but no routing annotation reaches the
model or evaluator.

## Rights and capture boundary

- Use self-recorded or otherwise explicitly authorized audio. Set
  `product_evaluation_authorized` only when the rights holder permits product
  evaluation, and record the basis in `rights_basis`.
- Use two or more genuinely physical hardware groups. Software plugins,
  re-amping through a DAW-only chain, and neural pedal emulations do not close
  this gate.
- Record paired bypass/clean and wet files through the same conversion path.
  Preserve original files; do not peak-normalize, limit, denoise, or edit clips
  independently.
- Reject and recapture clipping, dropouts, clock faults, or mismatched takes.
- Keep the future locked-final capture separate and sealed.

## Minimum evidence

- 73 bypass/clean examples. With zero false positives this is the minimum here
  for the 95% Wilson upper bound to reach the 5% gate.
- At least 12 dry-performance groups and two physical hardware groups.
- At least 25 positive examples for every family overall.
- At least five positives for every family in every hardware group.
- Every wet row contains two or more families. Single-effect captures can be
  useful elsewhere, but cannot satisfy this multi-effect veto.

Use multiple guitars, pickup settings, playing dynamics, chain family subsets,
and hidden routing variants. Assign opaque `hardware_group` and
`capture_variant_id` values so the evaluator cannot infer routing.

## Manifest

Place `manifest.json` beside an `audio/` directory. Paths must stay relative to
the manifest.

```json
{
  "schema": 1,
  "scope": "physical-multi-effect-veto",
  "source_id": "self-capture-physical-v1",
  "physical_hardware": true,
  "product_evaluation_authorized": true,
  "rights_basis": "self-recorded audio authorized for product evaluation",
  "rows": [
    {
      "path": "audio/phrase-001-clean.wav",
      "group": "phrase-001",
      "hardware_group": "bypass-a",
      "capture_variant_id": "v0001",
      "families": []
    },
    {
      "path": "audio/phrase-001-wet-a.wav",
      "group": "phrase-001",
      "hardware_group": "board-a",
      "capture_variant_id": "v0002",
      "families": ["nonlinear", "echo", "ambience"]
    }
  ]
}
```

The reader fails closed if any manifest key contains order, position, sequence,
or slot metadata, if a path escapes the corpus, if rights authorization is not
explicit, or if a wet row is single-effect.

## Frozen run

Run on MPS without changing any checkpoint or threshold:

```sh
python -m remix.evaluate_physical_chain_presence \
  --stack runs/foundation/product2/blind2-independent-family-stack/stack.json \
  --manifest /path/to/physical-veto/manifest.json \
  --device mps \
  --output runs/foundation/product2/blind2-independent-family-stack/physical-veto.json
```

Passing still does not set `usable_model`. Listening must pass next, followed by
one separately sealed locked-final evaluation. A failed veto rejects this
candidate; do not tune against the failed veto and rerun it as if it were fresh.
