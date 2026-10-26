"""Tests for src/postprocess.py (burn-in / thinning / ESS / chain_stats).

Synthetic model everywhere: stationary AR(1) x_i = rho x_{i-1} + sqrt(1-rho^2) eps_i,
so Var[x] = 1 and tau_true = (1+rho)/(1-rho). Transients are injected as an additive
shifted block on the first samples. All RNGs are seeded, so every numeric assertion is
deterministic.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from postprocess import burn_in, thin, ess, chain_stats  # noqa: E402
from metrics import iat, running_mean  # noqa: E402


def ar1(seed, k, n, rho):
    """(k, n) stationary AR(1) chains, unit marginal variance."""
    rng = np.random.default_rng(seed)
    s = math.sqrt(1.0 - rho * rho)
    e = rng.standard_normal((k, n))
    x = np.zeros((k, n))
    x[:, 0] = e[:, 0]
    for i in range(1, n):
        x[:, i] = rho * x[:, i - 1] + s * e[:, i]
    return x


def transient_chains(seed=0, k=20, n=8000, rho=0.9, shift=20.0, length=100):
    """AR(1) chains with an additive +shift block on the first `length` samples.

    The block is shorter than c*tau = 10*19 = 190, which is what the c*tau rule
    can actually see; the stationary mean is 0.
    """
    x = ar1(seed, k, n, rho)
    x[:, :length] += shift
    return x


def se_of_chain_means(x):
    m = x.mean(axis=1)
    return float(m.mean()), float(m.std(ddof=1)) / math.sqrt(x.shape[0])


def lag1(v):
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    return float(np.sum(v[:-1] * v[1:]) / np.sum(v * v))


# --------------------------------------------------------------------------
# burn_in: shapes / info
# --------------------------------------------------------------------------

def test_burn_in_frac_drops_half_and_preserves_rank_2d():
    x = ar1(0, 4, 1000, 0.5)
    burned, info = burn_in(x, method="frac", frac=0.5)
    assert burned.shape == (4, 500)
    assert info["n_dropped"] == 500
    np.testing.assert_allclose(burned, x[:, 500:])


def test_burn_in_1d_input_returns_1d():
    x = ar1(0, 1, 1000, 0.5)[0]
    burned, info = burn_in(x, method="frac", frac=0.25)
    assert burned.ndim == 1
    assert burned.shape == (750,)
    assert info["n_dropped"] == 250


def test_burn_in_info_has_all_keys():
    x = ar1(1, 4, 1000, 0.5)
    for method in ("frac", "tau"):
        _, info = burn_in(x, method=method)
        assert set(info) >= {"n_dropped", "tau_hat", "capped", "method"}
        assert info["method"] == method
        assert isinstance(info["capped"], bool)
        assert info["tau_hat"] > 0.0


def test_burn_in_rejects_unknown_method():
    with pytest.raises(ValueError):
        burn_in(ar1(0, 2, 100, 0.0), method="magic")


# --------------------------------------------------------------------------
# burn_in: does it actually remove the transient?
# --------------------------------------------------------------------------

def test_full_trace_mean_is_biased_by_the_transient():
    """(a) Without burn-in the pooled mean sits far from the stationary mean 0."""
    x = transient_chains()
    mean, se = se_of_chain_means(x)
    assert abs(mean - 0.0) > 3.0 * se


def test_burn_in_frac_recovers_the_stationary_mean():
    """(b) frac=0.5 brings the pooled mean back within ~2 sigma of 0."""
    x = transient_chains()
    burned, info = burn_in(x, method="frac", frac=0.5)
    mean, se = se_of_chain_means(burned)
    assert info["n_dropped"] == 4000
    assert abs(mean - 0.0) < 2.0 * se


def test_burn_in_tau_recovers_mean_and_drops_fewer_than_half():
    """(c) method='tau' also recovers the mean, at a much smaller cut than 50%."""
    x = transient_chains()
    burned, info = burn_in(x, method="tau", c=10.0)
    mean, se = se_of_chain_means(burned)
    assert abs(mean - 0.0) < 2.0 * se
    assert info["capped"] is False
    assert 100 < info["n_dropped"] < 0.5 * x.shape[1]   # past the block, well under the cap
    assert info["tau_hat"] == pytest.approx(19.0, rel=0.35)  # tau_true = 19


def test_burn_in_tau_iteration_reaches_a_self_consistent_fixed_point():
    """The returned cut must satisfy its own defining equation, cut = ceil(c*tau)
    with tau measured AFTER the cut -- i.e. the iteration converged rather than
    stopping on the first (transient-inflated) tau estimate."""
    x = transient_chains()
    _, info = burn_in(x, method="tau", c=10.0)
    assert info["capped"] is False
    assert info["n_dropped"] == pytest.approx(math.ceil(10.0 * info["tau_hat"]), rel=0.02)


def test_burn_in_tau_caps_when_the_transient_outlives_c_tau():
    """rho=0.5 (tau=3, c*tau~30) with a 500-sample shifted block: the cut oscillates
    between the cap and ~30 and never reaches a fixed point. The fallback must be the
    cap with capped=True -- NOT the small cut, which would leave the transient in
    place behind a clean flag. Pinned so the result cannot depend on the parity of
    the iteration count."""
    import postprocess

    x = ar1(0, 20, 5000, 0.5)
    x[:, :500] += 5.0
    for iters in (4, 5, 6):                       # any budget -> same answer
        original = postprocess._TAU_ITERS
        postprocess._TAU_ITERS = iters
        try:
            _, info = burn_in(x, method="tau", c=10.0)
        finally:
            postprocess._TAU_ITERS = original
        assert info["capped"] is True, f"_TAU_ITERS={iters} escaped the cap"
        assert info["n_dropped"] == 2500


def test_burn_in_tau_caps_on_a_trapped_chain():
    """A near-frozen chain (rho=0.9999, tau >> N) cannot be fixed by burn-in:
    the tau-implied cut exceeds max_frac*N, so the cap is applied and flagged."""
    x = ar1(1, 4, 2000, 0.9999) * 1e-3 + 3.0
    burned, info = burn_in(x, method="tau", c=10.0, max_frac=0.5)
    assert info["capped"] is True
    assert info["n_dropped"] == 1000
    assert burned.shape == (4, 1000)


def test_burn_in_tau_caps_on_a_constant_chain():
    """iat() returns len(x) for a constant trace -> cut pinned at the cap."""
    _, info = burn_in(np.ones((3, 400)), method="tau", c=10.0)
    assert info["capped"] is True
    assert info["n_dropped"] == 200


def test_burn_in_tau_not_capped_when_well_mixed():
    x = ar1(2, 8, 20000, 0.5)  # tau = 3 -> cut ~ 30 << 10000
    _, info = burn_in(x, method="tau", c=10.0)
    assert info["capped"] is False
    assert info["n_dropped"] < 200


def test_burn_in_max_frac_is_respected():
    _, info = burn_in(np.ones((3, 1000)), method="tau", max_frac=0.2)
    assert info["n_dropped"] == 200
    assert info["capped"] is True


# --------------------------------------------------------------------------
# thin
# --------------------------------------------------------------------------

def test_thin_kills_lag1_autocorrelation():
    x = ar1(2, 1, 50000, 0.9)[0]
    assert lag1(x) > 0.8
    thinned, stride = thin(x)
    assert stride > 5
    assert lag1(thinned) < 0.3
    assert len(thinned) == pytest.approx(len(x) / stride, rel=0.05)


def test_thin_explicit_stride():
    x = np.arange(100.0)
    thinned, stride = thin(x, stride=4)
    assert stride == 4
    np.testing.assert_allclose(thinned, x[::4])


def test_thin_2d_uses_a_common_stride():
    x = ar1(3, 6, 20000, 0.9)
    thinned, stride = thin(x)
    assert thinned.shape == (6, x.shape[1] // stride + (1 if x.shape[1] % stride else 0))
    np.testing.assert_allclose(thinned, x[:, ::stride])


def test_thin_stride_is_at_least_one_for_iid():
    x = np.random.default_rng(5).standard_normal(5000)
    thinned, stride = thin(x)
    assert stride >= 1
    assert len(thinned) <= len(x)


# --------------------------------------------------------------------------
# ess
# --------------------------------------------------------------------------

def test_ess_matches_n_over_tau_true_for_ar1():
    x = ar1(3, 1, 200000, 0.8)[0]   # tau_true = 9
    expect = len(x) / 9.0
    assert 0.7 * expect < ess(x) < 1.3 * expect


def test_ess_iid_is_near_n():
    x = np.random.default_rng(6).standard_normal(20000)
    assert 0.5 * len(x) < ess(x) <= len(x) * 1.2


def test_ess_2d_sums_per_chain_ess():
    x = ar1(4, 5, 20000, 0.8)
    assert ess(x) == pytest.approx(sum(ess(x[k]) for k in range(5)), rel=1e-12)


# --------------------------------------------------------------------------
# chain_stats
# --------------------------------------------------------------------------

REQUIRED_KEYS = {"ew_pool", "se_cross", "sd_cross", "tau_median", "ess_total",
                 "se_ess", "rhat", "burn_info", "n_used"}


def test_chain_stats_has_required_keys():
    st = chain_stats(ar1(4, 20, 4000, 0.5))
    assert REQUIRED_KEYS <= set(st)
    assert st["n_used"] == 2000
    assert st["burn_info"]["method"] == "frac"
    assert st["rhat"] == pytest.approx(1.0, abs=0.05)


def test_chain_stats_se_cross_and_se_ess_agree_for_well_mixed_chains():
    """Both estimate the SE of the same pooled mean, so they must agree to ~1.5x."""
    st = chain_stats(ar1(4, 20, 5000, 0.5))
    ratio = st["se_cross"] / st["se_ess"]
    assert 1 / 1.5 < ratio < 1.5


def test_chain_stats_sd_cross_matches_the_ew_std_convention():
    """sd_cross is the undivided std over chain means (repo-wide `ew_std`)."""
    x = ar1(4, 20, 4000, 0.5)
    st = chain_stats(x)
    means = x[:, 2000:].mean(axis=1)
    assert st["sd_cross"] == pytest.approx(float(means.std(ddof=1)), rel=1e-12)
    assert st["se_cross"] == pytest.approx(st["sd_cross"] / math.sqrt(20), rel=1e-12)
    assert st["ew_pool"] == pytest.approx(float(means.mean()), rel=1e-12)


def test_chain_stats_frac_half_matches_running_mean_identity():
    """For even N, the second-half mean equals 2*rm(N) - rm(N/2)."""
    x = ar1(7, 6, 2000, 0.5)
    st = chain_stats(x, burn="frac", frac=0.5)
    per_chain = []
    for k in range(x.shape[0]):
        rm = running_mean(x[k], [1000, 2000])
        per_chain.append(2.0 * rm[1] - rm[0])
    assert st["ew_pool"] == pytest.approx(float(np.mean(per_chain)), rel=1e-10)


def test_chain_stats_ess_total_is_bounded_by_samples_used():
    st = chain_stats(ar1(8, 10, 4000, 0.9))
    assert 0 < st["ess_total"] <= 10 * st["n_used"]
    assert st["tau_median"] > 5.0


def test_chain_stats_thin_by_tau_shrinks_n_used_but_keeps_the_mean():
    x = ar1(8, 10, 20000, 0.9)
    plain = chain_stats(x)
    thinned = chain_stats(x, thin_by_tau=True)
    assert thinned["n_used"] < plain["n_used"]
    assert thinned["ew_pool"] == pytest.approx(plain["ew_pool"], abs=3 * plain["se_cross"])
    assert thinned["tau_median"] < 3.0        # thinned chain is nearly independent
    assert thinned["stride"] > 5
    assert plain["stride"] is None


def test_chain_stats_tau_burn_removes_the_transient():
    x = transient_chains()
    st = chain_stats(x, burn="tau", c=10.0)
    assert abs(st["ew_pool"]) < 2.0 * st["se_cross"]
    assert st["burn_info"]["capped"] is False


def test_chain_stats_rhat_flags_split_chains():
    x = ar1(9, 6, 5000, 0.5)
    x[:3] += 5.0                      # half the chains stuck in another mode
    st = chain_stats(x)
    assert st["rhat"] > 1.5


def test_chain_stats_requires_2d():
    with pytest.raises(ValueError):
        chain_stats(np.zeros(100))


def test_chain_stats_is_deterministic():
    x = ar1(10, 8, 3000, 0.7)
    a, b = chain_stats(x), chain_stats(x)
    for key in ("ew_pool", "se_cross", "se_ess", "tau_median", "ess_total", "rhat"):
        assert a[key] == b[key]
