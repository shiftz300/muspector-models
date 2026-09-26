# Foundation presence packages

This opt-in catalog contains analysis-only Wet classifiers. It is separate from
the restoration experts and graph executor: a classifier may report a family,
but it never owns chain order, controls or restored audio.

`foundation-drive-presence` packages only the frozen generic nonlinear expert
and any-effect Clean gate. It deliberately does not expose the historical
shared Ambience output while Reverb is frozen. Its weak single-Drive false
negative limitation remains explicit, so it cannot silently bypass restoration.
