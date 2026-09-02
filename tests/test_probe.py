import numpy as np

from remix.probe import COUNT, RATE, SECONDS, hashes, probes


def test_probes_are_deterministic_bounded_and_distinct():
    first, second = probes(), probes()
    assert len(first) == COUNT and hashes() == hashes()
    assert len(set(hashes())) == COUNT
    for left, right in zip(first, second):
        assert left.shape == (RATE * SECONDS,)
        assert np.array_equal(left, right)
        assert np.isfinite(left).all() and np.max(np.abs(left)) <= 0.2
