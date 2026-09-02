"""Model-based topology search using recovered controls and a small DSP bank."""

from __future__ import annotations

import itertools

import numpy as np

from .render import Renderer, render_chain
from .spec import ChainSpec, Delay, Drive, Reverb


SEARCH_RENDERERS: tuple[Renderer, ...] = ("reference", "alternate", "stress")


def decode_controls(topology: tuple[str, ...], controls: np.ndarray) -> tuple:
    """Decode normalized model outputs into effects for the active topology."""

    value = np.clip(np.asarray(controls, dtype=np.float64), 0.0, 1.0)
    effects = {
        "drive": Drive(value[0] * 30.0, value[1], value[2] * 30.0 - 18.0),
        "delay": Delay(40.0 * 25.0 ** value[3], value[4] * 0.9, value[5] * 0.7),
        "reverb": Reverb(0.2 * 40.0 ** value[6], value[7], value[8] * 0.7),
    }
    return tuple(effects[kind] for kind in topology)


def gain_aligned_error(candidate: np.ndarray, target: np.ndarray) -> float:
    """Compare structure after removing renderer-specific output gain."""

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    gain = float(np.dot(left, right) / (np.dot(left, left) + 1.0e-12))
    residual = left * gain - right
    return float(np.mean(residual * residual) / (np.mean(right * right) + 1.0e-12))


def rank_topologies(
    dry: np.ndarray,
    wet: np.ndarray,
    topology: tuple[str, ...],
    controls: np.ndarray,
    sample_rate: int = 44_100,
    renderers: tuple[Renderer, ...] = SEARCH_RENDERERS,
) -> list[tuple[tuple[str, ...], float]]:
    """Rank compatible permutations by the best gain-aligned DSP-bank residual."""

    effects = decode_controls(topology, controls)
    ranked = []
    for permutation in itertools.permutations(effects):
        spec = ChainSpec(tuple(permutation))
        error = min(
            gain_aligned_error(
                render_chain(dry, spec, sample_rate, renderer), wet
            )
            for renderer in renderers
        )
        ranked.append((spec.topology, error))
    return sorted(ranked, key=lambda item: item[1])


def search_margin(ranked: list[tuple[tuple[str, ...], float]]) -> float:
    if len(ranked) < 2:
        return float("inf")
    best, second = ranked[0][1], ranked[1][1]
    return (second - best) / max(best, 1.0e-12)


def convex_renderer_fit(candidates: list[np.ndarray], target: np.ndarray):
    """Fit fixed convex renderer weights on reference audio, not a gain patch.

    Overall gain is removed only for the existing diagnostic residual metric;
    the returned renderer weights always sum to one, and are held fixed during
    subsequent rendering. Individual renderers remain possible candidates.
    """
    from scipy.optimize import nnls
    matrix = np.stack(candidates, axis=1).astype(np.float64)
    expected = np.asarray(target, dtype=np.float64)
    coefficients, _ = nnls(matrix, expected)
    alternatives = list(np.eye(len(candidates)))
    if coefficients.sum() > 1e-12:
        alternatives.append(coefficients / coefficients.sum())
    scored = [(gain_aligned_error(matrix @ weights, expected), weights) for weights in alternatives]
    error, weights = min(scored, key=lambda item: item[0])
    return float(error), weights.tolist()


def rank_topology_blends(dry, wet, topology, controls, sample_rate=44100):
    """Order plus a compact, fixed mixture of three causal DSP renderers."""
    effects = decode_controls(topology, controls)
    ranked = []
    for permutation in itertools.permutations(effects):
        spec = ChainSpec(tuple(permutation))
        candidates = [render_chain(dry, spec, sample_rate, renderer) for renderer in SEARCH_RENDERERS]
        error, weights = convex_renderer_fit(candidates, wet)
        ranked.append({'topology': list(spec.topology), 'error': error, 'renderer_weights': weights})
    return sorted(ranked, key=lambda item: item['error'])
