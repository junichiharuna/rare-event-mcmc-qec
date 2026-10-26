import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from diagnostics import split_rhat


def test_iid_chains_give_rhat_near_one():
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(8, 10_000))
    assert abs(split_rhat(chains) - 1.0) < 0.01


def test_shifted_chains_give_large_rhat():
    rng = np.random.default_rng(1)
    chains = rng.normal(size=(4, 5_000))
    chains[:2] += 5.0  # two chains stuck in another mode
    assert split_rhat(chains) > 1.5


def test_odd_length_works():
    rng = np.random.default_rng(2)
    assert abs(split_rhat(rng.normal(size=(4, 10_001))) - 1.0) < 0.02


def test_constant_chains_return_one():
    assert split_rhat(np.ones((4, 100))) == 1.0


def test_single_chain_raises():
    with pytest.raises(ValueError):
        split_rhat(np.zeros((1, 100)))


def test_1d_input_raises():
    with pytest.raises(ValueError):
        split_rhat(np.zeros(100))


def test_known_value():
    x = np.array([[1.0, 2, 3, 4], [2.0, 3, 4, 9]])
    assert split_rhat(x) == pytest.approx(1.354006400773, rel=1e-9)


def test_uint8_input_is_cast_not_overflowed():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 30, size=(8, 4000), dtype=np.uint8)
    assert abs(split_rhat(x) - 1.0) < 0.02
