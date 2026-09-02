"""Perceptual equivalence masks for effect-order supervision and audits."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from .spec import ChainSpec, identifiable_order_targets, order_targets


def perceptual_order_targets(
    dry: np.ndarray,
    wet: np.ndarray,
    spec: ChainSpec,
    render: Callable[[np.ndarray, ChainSpec], np.ndarray],
    threshold_db: float = -30.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mask relations flipped by an acoustically near-equivalent permutation.

    A relation remains supervised only when every topology that reverses it is
    more than ``threshold_db`` RMS below the rendered reference difference.
    This extends exact LTI commutation to weak/near-bypass controls without
    changing the underlying nominal labels.
    """

    targets, masks = identifiable_order_targets(spec.topology)
    masks = list(masks)
    if len(spec.effects) < 2:
        return targets, tuple(masks)
    reference_rms = max(float(np.sqrt(np.mean(np.square(wet, dtype=np.float64)))), 1.0e-8)
    for permutation in itertools.permutations(spec.effects):
        if permutation == spec.effects:
            continue
        alternative = ChainSpec(tuple(permutation))
        alternative_targets, _ = order_targets(alternative.topology)
        if all(left == right for left, right in zip(targets, alternative_targets)):
            continue
        counterfactual = render(dry, alternative)
        difference = float(
            np.sqrt(np.mean(np.square(wet - counterfactual, dtype=np.float64)))
        )
        difference_db = 20.0 * np.log10(max(difference / reference_rms, 1.0e-12))
        if difference_db <= threshold_db:
            for index, (target, candidate) in enumerate(
                zip(targets, alternative_targets)
            ):
                if target != candidate:
                    masks[index] = 0.0
    return targets, tuple(masks)
