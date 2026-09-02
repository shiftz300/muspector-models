# Routing

## Decision

Exact named-device routing is not admitted. The catalog treats manufacturer,
model, controls, license, and capture provenance as declared metadata verified
by hashes. Audio inference may identify a broad effect family and may recover
controls for an explicitly selected compatible model; it must not guess a
brand, choose a named adapter, or authorize automatic delivery.

This boundary applies to both aligned Dry/Wet guitar audio and downloadable NAM
capture models. The historical `route` package remains reproducible research,
but it is excluded from `models/catalog/remix/collection.json` and is no longer
referenced by the base catalog.

## Evidence

The gain-robust paired router passed its ASRNN development domain, then failed
three frozen multi-author A2 seals:

| Model | RAT recall | Other false route | Result |
| --- | ---: | ---: | --- |
| router model3 | 87.50% | 37.50% | rejected |
| router model5 | 100.00% | 18.75% | rejected |
| router model7 | 62.50% | 81.25% | rejected |

The content-reduced signature removed absolute Dry/Wet spectra and used
leave-one-device-group-out validation across 1,452 examples. Its best candidate
reached 78.59% recall but 65.08% false routing, so it was rejected.

The fixed-probe software-model classifier reached 100% recall and 0% false
routing in leave-one-device-group-out development validation. Its untouched
seal then rejected all four new RAT models: 0% recall and 0% false routing.
This disproves promotion and shows that cross-capture/trainer shift remains
larger than the inferred named-device signature.

All runs used in-memory CPU inference only. Source arrays and files were hashed
before and after use; no physical audio device was enumerated or opened, no
rendered audio was retained, and no normalization, limiting, dither, or lossy
encoding was applied.

## Boundary

- `family`: may propose broad Drive, Delay, Reverb, Fuzz, or Compressor labels.
- `identity`: development evidence only; never authorizes named-device delivery.
- `route`: archived research source, not an installable collection member.
- `probe`: rejected research; not packaged.
- `knobs`: valid only when the compatible named model was selected explicitly.
- `forward`: renders only through its declared package and audio-quality contract.

## Reopening

Reopen exact identity work only with a predeclared corpus containing multiple
independent physical devices per positive and negative family, captured through
the same standardized probe and recording protocol. Split by physical device,
session, author, and capture chain before training. A final seal must use unseen
devices and authors and meet at least 75% positive recall, at most 10% false
routing, zero source mutation, and zero automatic delivery.

No user-owned hardware is required: a licensed public or community capture pack
can satisfy this requirement if it publishes the raw standardized probe pairs
and provenance. Ordinary NAM files from mixed trainers do not satisfy it.
