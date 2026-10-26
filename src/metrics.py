import numpy as np


def running_mean(x: np.ndarray, ns) -> np.ndarray:
    """Cumulative mean of x using its first n samples, for each n in ns.

    For a chain confined to the faulty manifold, the cold-replica weight trace's
    running mean is the E[w|faulty] estimate vs sample budget (a convergence curve;
    n larger than len(x) clamps to the whole trace).
    """
    x = np.asarray(x, dtype=float)
    csum = np.cumsum(x)
    out = []
    for n in ns:
        m = min(int(n), len(x))
        out.append(csum[m - 1] / m if m > 0 else float("nan"))
    return np.array(out)


def iat(x: np.ndarray) -> float:
    """Geyer-truncated integrated autocorrelation time."""
    x = np.asarray(x, dtype=float).copy()
    x -= x.mean()
    f = np.fft.rfft(x, n=2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[: len(x)].real
    if ac[0] == 0.0:
        return float(len(x))
    ac /= ac[0]
    tau = 1.0
    for k in range(1, len(x) // 2):
        if ac[k] < 0.0:
            break
        tau += 2.0 * ac[k]
    return tau


def doubling_converged(prev_est, cur_est, se, c=2.0) -> bool:
    """Beverland-style stop rule: True when the estimate barely changed across a length
    doubling, i.e. |cur - prev| <= c * se (se = cross-seed standard error). False if there
    is no previous estimate yet."""
    if prev_est is None:
        return False
    return abs(cur_est - prev_est) <= c * se
