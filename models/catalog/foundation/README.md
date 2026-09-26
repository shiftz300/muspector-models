# Foundation restoration packages

This catalog contains four independent, development-only Wet-to-Clean experts:

- `foundation-drive`: bounded inverse of the repository-owned tanh/atan/cubic Drive;
- `foundation-dynamics`: exact stateful inverse of the repository-owned compressor;
- `foundation-echo`: exact causal inverse of the repository-owned feedback delay;
- `foundation-spectral`: bounded analytic inverse of the repository-owned generic three-band EQ.

Each expert accepts only the current effect Wet plus that effect's explicit controls and private
state. The graph executor, not the expert, owns reverse traversal. The packages do not claim blind
effect discovery, named-device fidelity, physical-chain acceptance or final product usability.

`sources.json` is the canonical, hash-pinned materialization input. `collection.json` deliberately
sets `default_install` to false because all four packages remain at development quality.
