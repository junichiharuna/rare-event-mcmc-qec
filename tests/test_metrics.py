import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import iat, running_mean


def test_running_mean_basic():
    # cumulative means of [1,2,3,4] at sample counts 1,2,4
    out = running_mean(np.array([1.0, 2.0, 3.0, 4.0]), [1, 2, 4])
    np.testing.assert_allclose(out, [1.0, 1.5, 2.5])


def test_running_mean_final_equals_full_mean():
    x = np.arange(1, 101, dtype=float)
    out = running_mean(x, [100])
    assert out[0] == x.mean()


def test_running_mean_clamps_oversized_n():
    # n beyond the trace length uses the whole trace (no crash)
    x = np.array([2.0, 4.0])
    out = running_mean(x, [5])
    assert out[0] == 3.0


def test_iat_constant_returns_length():
    """Constant trace → ac[0]=0 → guard returns len(x) instead of dividing by zero."""
    assert iat(np.ones(100)) == float(100)


def test_iat_iid_near_one():
    x = np.random.default_rng(42).standard_normal(10_000)
    tau = iat(x)
    assert 0.8 < tau < 2.5


def test_iat_does_not_mutate_input():
    x = np.array([1.0, 2.0, 3.0] * 30)
    original = x.copy()
    iat(x)
    np.testing.assert_array_equal(x, original)


def test_iat_correlated_exceeds_one():
    """AR(1) with rho=0.9: theoretical tau ≈ 19; check tau > 5."""
    rng = np.random.default_rng(0)
    x = np.zeros(50_000)
    for i in range(1, len(x)):
        x[i] = 0.9 * x[i - 1] + rng.standard_normal()
    assert iat(x) > 5.0


def test_doubling_converged():
    from metrics import doubling_converged
    # change (0.02) below c*se (2*0.05=0.10) -> converged
    assert doubling_converged(2.00, 2.02, 0.05, c=2.0) is True
    # change (0.30) above c*se -> not converged
    assert doubling_converged(2.00, 2.30, 0.05, c=2.0) is False
    # no previous estimate -> not converged
    assert doubling_converged(None, 2.0, 0.05, c=2.0) is False
    # zero se with exact match -> converged
    assert doubling_converged(2.0, 2.0, 0.0, c=2.0) is True
