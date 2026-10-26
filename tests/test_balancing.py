import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from runners import run_rimc_mixed


def test_unknown_balancing_raises():
    with pytest.raises(ValueError, match="Unknown balancing"):
        run_rimc_mixed(
            1e-2,
            100,
            p_zanella=1.0,
            p_single=0.0,
            p_stab=0.0,
            p_logic=0.0,
            n_replicas=1,
            balancing="unknown",
            seed=0,
        )


def test_sqrt_is_default():
    # Should run without error (sqrt is the default)
    _, trace = run_rimc_mixed(
        1e-2,
        200,
        p_zanella=1.0,
        p_single=0.0,
        p_stab=0.0,
        p_logic=0.0,
        n_replicas=1,
        seed=0,
    )
    assert len(trace) == 200


def test_barker_runs_and_returns_correct_types():
    dist, trace = run_rimc_mixed(
        1e-2,
        200,
        p_zanella=1.0,
        p_single=0.0,
        p_stab=0.0,
        p_logic=0.0,
        n_replicas=1,
        balancing="barker",
        seed=0,
    )
    assert isinstance(dist, dict)
    assert isinstance(trace, np.ndarray)
    assert len(trace) == 200
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_barker_converges_to_correct_distribution():
    # Exact E[w] for d=3 toric at p=1e-2: 2.2610 (from compare/exact_ew.py)
    # Tolerance: 3 * std_of_mean. With N=30000 and tau~10, std ~ sqrt(2*10/30000)*std_w ~ 0.05
    _, trace = run_rimc_mixed(
        1e-2,
        30_000,
        p_zanella=1.0,
        p_single=0.0,
        p_stab=0.0,
        p_logic=0.0,
        n_replicas=1,
        balancing="barker",
        seed=7,
    )
    e_w = float(np.mean(trace))
    assert abs(e_w - 2.2610) < 0.15, f"E[w]={e_w:.4f} too far from exact 2.2610"
