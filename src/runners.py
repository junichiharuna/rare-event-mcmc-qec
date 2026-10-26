"""Run functions for RIMC (bit-flip, depolarizing) and MUCA."""

from __future__ import annotations
import numpy as np
from css_code import (
    CSSCode,
    Decoder,
    DepolarizingDecoder,
    make_toric_code,
    make_mwpm_decoder,
    make_depolarizing_mwpm_decoder,
)
from kernels import _mixed_step_with_single, _depolarizing_step, one_bit_flip


def run_rimc_mixed(
    physical_error_rate: float,
    target_samples: int,
    *,
    p_single: float = 0.0,
    p_zanella: float = 1.0,
    p_stab: float = 0.0,
    p_logic: float = 0.0,
    p_cluster: float = 0.0,
    cluster_k: int = 2,
    n_replicas: int = 4,
    max_error_rate: float = 0.3,
    steps_per_swap: int = 10,
    code: CSSCode | None = None,
    decoder: Decoder | None = None,
    seed: int | None = None,
    return_swap_stats: bool = False,
    balancing: str = "sqrt",
    randomize_init: bool = False,
    hot_p_zanella: float | None = None,
    return_all_traces: bool = False,
    cache_cap: int | None = None,
    record_trace: bool = True,
    return_move_stats: bool = False,
) -> tuple[dict[int, float], np.ndarray]:
    """RIMC sampler on the bit-flip (X-error) faulty manifold, with Parallel Tempering.

    Runs ``n_replicas`` chains on a geometric temperature ladder from
    ``physical_error_rate`` (cold, the target) up to ``max_error_rate`` (hot) and
    records the cold replica's error weight ``w``. **The default is Z+PT** — pure
    Zanella (``p_zanella=1.0``) with ``n_replicas=4`` parallel tempering; StabFlip,
    LogicFlip, cluster exchange and hotOBF are opt-in options (off by default).
    Each step picks one kernel from a mixture whose probabilities must sum to 1.0
    (turning an option on means lowering ``p_zanella`` to compensate — no
    auto-normalisation, so a mismatched mixture is caught):

      - ``p_zanella`` — Zanella locally-balanced single-bit flip (∝ √ ratio), optimal
        in the local proposal class. Needs an MWPM decode (syndrome→correction cached).
        **Default 1.0.**
      - ``p_stab``    — StabFlip (option): XOR a random H_X row. Syndrome-preserving → cache hit.
      - ``p_logic``   — LogicFlip (option): XOR a random L_X row. Syndrome-preserving → cache hit.
      - ``p_single``  — uniform single-bit flip Metropolis-Hastings (cached decode); OBF baseline.
      - ``p_cluster`` — cluster exchange (option H): weight-preserving swap of ``cluster_k``
        1↔0 bits; MH ratio 1, accept iff the proposal is still faulty.
      - ``hot_p_zanella=0.0`` — hotOBF (option E): hot replicas (k≥1) use OBF instead of Zanella.

    Every ``steps_per_swap`` steps, adjacent replicas attempt a PT swap.

    Parameters
    ----------
    physical_error_rate : target physical error rate p (cold replica).
    target_samples : number of recorded cold-replica weights (rounded down to a
        multiple of ``steps_per_swap``).
    n_replicas : PT replicas; ladder is ``geomspace(p, max_error_rate, n_replicas)``.
    max_error_rate : hottest replica's p. Follow the p-proportional rule
        ``max_error_rate = R * p`` (R=3 for d<=11, R=2 for d>=13); see CLAUDE.md.
    code, decoder : optional ``CSSCode`` / ``Decoder`` (default: toric d=3 + MWPM).
    balancing : 'sqrt' (default, theoretically optimal) or 'barker' (g(t)=t/(1+t)).
    randomize_init : start each replica in a random non-trivial logical sector
        instead of the fixed L_X[0] coset (diversifies multimodal starts).
    hot_p_zanella : if 0.0, hot replicas (k>=1) skip Zanella and use OBF instead
        (improvement E) — cheaper hot chains with no measured quality loss.
    seed : PRNG seed.
    return_swap_stats : also return per-pair PT swap acceptance rates.
    return_all_traces : also return the ``(n_replicas, n_recorded)`` weight traces of
        *all* replicas plus ``log_p1mp`` (the ladder log-odds), for MBAR estimation.
    return_move_stats : append a cold-replica cluster-move counter dict
        ``{"h_proposed", "h_accepted", "h_sector_changed"}`` as the LAST element.

    Returns
    -------
    ``(faulty_error_distribution, weight_trace)`` — ``faulty_error_distribution`` maps
    weight → empirical probability on the cold replica; ``weight_trace`` is the
    per-step cold-replica weight series (for IAT). Flags append extras:
    ``return_swap_stats`` → ``swap_rates``; ``return_all_traces`` →
    ``all_weight_traces, log_p1mp`` (swap stats come first if both flags are set).
    """
    total = p_single + p_zanella + p_stab + p_logic + p_cluster
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Kernel probabilities must sum to 1.0, got {total:.6f}")

    rng_gen = np.random.default_rng(seed)

    if code is None:
        code = make_toric_code(3)
    if decoder is None:
        decoder = make_mwpm_decoder(code)

    n = code.n
    hx_dense = code.hx
    lx_dense = code.lx
    lz_dense = code.lz
    n_stab = hx_dense.shape[0]
    n_logic = lx_dense.shape[0]
    hz_cols = code.hz.T.astype(np.uint8)

    error_rates = np.geomspace(physical_error_rate, max_error_rate, n_replicas)
    log_p1mp = np.log(error_rates) - np.log(1.0 - error_rates)
    if balancing == "sqrt":
        alphas = np.sqrt((1.0 - error_rates) / error_rates)
        betas = np.sqrt(error_rates / (1.0 - error_rates))
    elif balancing == "barker":
        alphas = 1.0 - error_rates
        betas = error_rates
    else:
        raise ValueError(f"Unknown balancing: {balancing!r}")
    alpha_m_betas = alphas - betas

    lookup = {}
    move_stats = ({"h_proposed": 0, "h_accepted": 0, "h_sector_changed": 0}
                  if return_move_stats else None)

    if randomize_init:
        replica_errors = []
        for _ in range(n_replicas):
            b = int(rng_gen.integers(1, 2**n_logic))  # non-zero: at least one lx row
            init = np.zeros(n, dtype=np.int8)
            for i in range(n_logic):
                if b & (1 << i):
                    init = (init ^ lx_dense[i]).astype(np.int8)
            replica_errors.append(init)
    else:
        init = code.lx[0].astype(np.int8)
        replica_errors = [init.copy() for _ in range(n_replicas)]
    replica_weights = [int(np.sum(e)) for e in replica_errors]
    replica_syndromes = [(code.hz @ e % 2).astype(np.uint8) for e in replica_errors]
    ones_arr_list = [np.where(e == 1)[0].astype(np.int32) for e in replica_errors]

    _init_key = tuple(replica_syndromes[0])
    lookup[_init_key] = decoder.decode_syndrome(replica_syndromes[0])

    n_recorded = (target_samples // steps_per_swap) * steps_per_swap
    faulty_error_frequencies = np.zeros(n + 1, dtype=np.int64)
    weight_trace = np.empty(n_recorded if record_trace else 0, dtype=np.int32)
    swap_attempts = np.zeros(n_replicas - 1, dtype=np.int64)
    swap_accepts = np.zeros(n_replicas - 1, dtype=np.int64)

    if return_all_traces:
        all_weight_traces = np.empty((n_replicas, n_recorded), dtype=np.int32)

    total_swap_rounds = target_samples // steps_per_swap
    recorded = 0

    for swap_round in range(total_swap_rounds):

        # Bound the syndrome->correction cache: decode is deterministic, so clearing
        # only forces re-decodes (results unchanged) while capping memory. Needed for
        # large codes (e.g. concatenated Steane L3) where distinct syndromes explode.
        if cache_cap is not None and len(lookup) > cache_cap:
            lookup.clear()

        for _ in range(steps_per_swap):
            for k in range(n_replicas):
                if k == 0 or hot_p_zanella is None:
                    ps, pl, psi = p_stab, p_logic, p_single
                else:
                    ps, pl, psi = p_stab, p_logic, 1.0 - p_stab - p_logic
                if move_stats is not None and k == 0:
                    prev_acc = move_stats["h_accepted"]
                    prev_err = replica_errors[0]  # reference OK: kernels copy on accept
                (
                    replica_errors[k],
                    replica_weights[k],
                    replica_syndromes[k],
                    ones_arr_list[k],
                ) = _mixed_step_with_single(
                    replica_errors[k],
                    replica_weights[k],
                    replica_syndromes[k],
                    ones_arr_list[k],
                    n,
                    log_p1mp[k],
                    alphas[k],
                    betas[k],
                    alpha_m_betas[k],
                    hx_dense,
                    n_stab,
                    lx_dense,
                    n_logic,
                    hz_cols,
                    ps,
                    pl,
                    psi,
                    decoder,
                    lookup,
                    rng_gen,
                    p_cluster,
                    cluster_k,
                    move_stats if k == 0 else None,
                )
                if (move_stats is not None and k == 0
                        and move_stats["h_accepted"] > prev_acc
                        and ((lz_dense @ (prev_err ^ replica_errors[0])) % 2).any()):
                    move_stats["h_sector_changed"] += 1
            faulty_error_frequencies[replica_weights[0]] += 1
            if record_trace:
                weight_trace[recorded] = replica_weights[0]
            if return_all_traces:
                for k in range(n_replicas):
                    all_weight_traces[k, recorded] = replica_weights[k]
            recorded += 1

        start = swap_round % 2
        for i in range(start, n_replicas - 1, 2):
            j = i + 1
            swap_attempts[i] += 1
            log_acc_swap = (log_p1mp[i] - log_p1mp[j]) * (
                replica_weights[j] - replica_weights[i]
            )
            if rng_gen.random() < min(1.0, np.exp(log_acc_swap)):
                replica_errors[i], replica_errors[j] = (
                    replica_errors[j],
                    replica_errors[i],
                )
                replica_weights[i], replica_weights[j] = (
                    replica_weights[j],
                    replica_weights[i],
                )
                replica_syndromes[i], replica_syndromes[j] = (
                    replica_syndromes[j],
                    replica_syndromes[i],
                )
                ones_arr_list[i], ones_arr_list[j] = ones_arr_list[j], ones_arr_list[i]
                swap_accepts[i] += 1

    total_rec = int(faulty_error_frequencies.sum())
    faulty_error_distribution = {
        int(w): faulty_error_frequencies[w] / total_rec
        for w in np.where(faulty_error_frequencies > 0)[0]
    }
    out = [faulty_error_distribution, weight_trace[:recorded]]
    if return_swap_stats:
        out.append(swap_accepts / np.maximum(swap_attempts, 1))
    if return_all_traces:
        out.append(all_weight_traces[:, :recorded])
        out.append(log_p1mp)
    if return_move_stats:
        out.append(move_stats)
    return tuple(out) if len(out) > 2 else (out[0], out[1])


def run_rimc_depolarizing(
    physical_error_rate: float,
    target_samples: int,
    *,
    p_single: float = 0.0,
    p_zanella: float = 1.0,
    p_stab: float = 0.0,
    p_logic: float = 0.0,
    p_cluster: float = 0.0,
    cluster_k: int = 2,
    n_replicas: int = 4,
    max_error_rate: float = 0.3,
    steps_per_swap: int = 10,
    code: CSSCode | None = None,
    decoder: DepolarizingDecoder | None = None,
    seed: int | None = None,
    return_swap_stats: bool = False,
    return_decode_count: bool = False,
    cache_cap: int | None = None,
    record_trace: bool = True,
) -> tuple[dict[int, float], np.ndarray]:
    """RIMC sampler on the depolarizing faulty manifold, with Parallel Tempering.

    Same PT + mixed-kernel design as :func:`run_rimc_mixed`, but for the depolarizing
    noise model: each qubit carries an X part ``ex`` and a Z part ``ez`` (Y = both),
    decoded independently by X- and Z-MWPM, and the recorded weight is the support
    size ``w = |ex OR ez|``. The per-Pauli rate is p/3, and the Zanella branch
    enumerates all eight single-qubit Pauli transitions with locally-balanced weights.

    The kernel mixture (``p_zanella``/``p_stab``/``p_logic``/``p_single`` must sum to
    1.0) and the PT ladder (``geomspace(p, max_error_rate, n_replicas)``, swap every
    ``steps_per_swap`` steps) match the bit-flip runner. ``code``/``decoder`` default
    to toric d=3 with an independent X/Z MWPM decoder.

    Returns ``(faulty_error_distribution, weight_trace)``; ``return_swap_stats``
    appends per-pair PT swap rates and ``return_decode_count`` appends the total
    number of MWPM decodes (cache size == #decodes, an algorithmic-cost metric),
    in that order. ``cache_cap`` bounds each of the two syndrome->correction caches
    with the same clear-on-full rule as :func:`run_rimc_mixed` (decode is
    deterministic, so clears only force re-decodes and leave the chain unchanged);
    ``record_trace=False`` returns a zero-length trace. ``return_decode_count``
    counts every decode, including the ones repeated after a clear.
    """
    total = p_single + p_zanella + p_stab + p_logic + p_cluster
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Kernel probabilities must sum to 1.0, got {total:.6f}")

    rng_gen = np.random.default_rng(seed)

    if code is None:
        code = make_toric_code(3)
    if decoder is None:
        decoder = make_depolarizing_mwpm_decoder(code)

    n = code.n
    hx_dense = code.hx
    hz_dense = code.hz
    lx_dense = code.lx
    lz_dense = code.lz
    n_stab_x = hx_dense.shape[0]
    n_stab_z = hz_dense.shape[0]
    n_logic = lx_dense.shape[0]
    hz_cols = code.hz.T.astype(np.uint8)
    hx_cols = code.hx.T.astype(np.uint8)

    error_rates = np.geomspace(physical_error_rate, max_error_rate, n_replicas)
    log_p1mp_dep = np.log(error_rates / 3.0) - np.log(1.0 - error_rates)
    alphas_dep = np.sqrt(3.0 * (1.0 - error_rates) / error_rates)
    betas_dep = np.sqrt(error_rates / (3.0 * (1.0 - error_rates)))

    lookup_x: dict = {}
    lookup_z: dict = {}

    init_ex = code.lx[0].astype(np.int8)
    init_ez = np.zeros(n, dtype=np.int8)

    replica_ex = [init_ex.copy() for _ in range(n_replicas)]
    replica_ez = [init_ez.copy() for _ in range(n_replicas)]
    replica_w_any = [int(np.sum(ex | ez)) for ex, ez in zip(replica_ex, replica_ez)]
    replica_sz = [(code.hz @ ex % 2).astype(np.uint8) for ex in replica_ex]
    replica_sx = [(code.hx @ ez % 2).astype(np.uint8) for ez in replica_ez]

    lookup_x[tuple(replica_sz[0])] = decoder.decode_x(replica_sz[0])
    lookup_z[tuple(replica_sx[0])] = decoder.decode_z(replica_sx[0])

    n_recorded = (target_samples // steps_per_swap) * steps_per_swap
    faulty_error_frequencies = np.zeros(n + 1, dtype=np.int64)
    weight_trace = np.empty(n_recorded if record_trace else 0, dtype=np.int32)
    swap_attempts = np.zeros(n_replicas - 1, dtype=np.int64)
    swap_accepts = np.zeros(n_replicas - 1, dtype=np.int64)

    total_swap_rounds = target_samples // steps_per_swap
    recorded = 0
    cleared_decodes = 0

    for swap_round in range(total_swap_rounds):
        if cache_cap is not None and (len(lookup_x) > cache_cap or len(lookup_z) > cache_cap):
            cleared_decodes += len(lookup_x) + len(lookup_z)
            lookup_x.clear()
            lookup_z.clear()
            for k in range(n_replicas):
                kx, kz = tuple(replica_sz[k]), tuple(replica_sx[k])
                if kx not in lookup_x:
                    lookup_x[kx] = decoder.decode_x(replica_sz[k])
                if kz not in lookup_z:
                    lookup_z[kz] = decoder.decode_z(replica_sx[k])

        for _ in range(steps_per_swap):
            for k in range(n_replicas):
                (
                    replica_ex[k],
                    replica_ez[k],
                    replica_w_any[k],
                    replica_sz[k],
                    replica_sx[k],
                ) = _depolarizing_step(
                    replica_ex[k],
                    replica_ez[k],
                    replica_w_any[k],
                    replica_sz[k],
                    replica_sx[k],
                    n,
                    log_p1mp_dep[k],
                    alphas_dep[k],
                    betas_dep[k],
                    hx_dense,
                    n_stab_x,
                    hz_dense,
                    n_stab_z,
                    lx_dense,
                    lz_dense,
                    n_logic,
                    hz_cols,
                    hx_cols,
                    p_stab,
                    p_logic,
                    p_single,
                    decoder,
                    lookup_x,
                    lookup_z,
                    rng_gen,
                    p_cluster,
                    cluster_k,
                )
            faulty_error_frequencies[replica_w_any[0]] += 1
            if record_trace:
                weight_trace[recorded] = replica_w_any[0]
            recorded += 1

        start = swap_round % 2
        for i in range(start, n_replicas - 1, 2):
            j = i + 1
            swap_attempts[i] += 1
            log_acc_swap = (log_p1mp_dep[i] - log_p1mp_dep[j]) * (
                replica_w_any[j] - replica_w_any[i]
            )
            if rng_gen.random() < min(1.0, np.exp(log_acc_swap)):
                replica_ex[i], replica_ex[j] = replica_ex[j], replica_ex[i]
                replica_ez[i], replica_ez[j] = replica_ez[j], replica_ez[i]
                replica_w_any[i], replica_w_any[j] = replica_w_any[j], replica_w_any[i]
                replica_sz[i], replica_sz[j] = replica_sz[j], replica_sz[i]
                replica_sx[i], replica_sx[j] = replica_sx[j], replica_sx[i]
                swap_accepts[i] += 1

    total_rec = int(faulty_error_frequencies.sum())
    faulty_error_distribution = {
        int(w): faulty_error_frequencies[w] / total_rec
        for w in np.where(faulty_error_frequencies > 0)[0]
    }
    # decode is called only on a lookup-cache miss, so cache size == #decodes
    decode_count = cleared_decodes + len(lookup_x) + len(lookup_z)
    out = [faulty_error_distribution, weight_trace[:recorded]]
    if return_swap_stats:
        out.append(swap_accepts / np.maximum(swap_attempts, 1))
    if return_decode_count:
        out.append(decode_count)
    return tuple(out) if len(out) > 2 else (out[0], out[1])


def run_muca(
    p: float,
    n_production: int,
    *,
    code_size: int = 3,
    log_f_initial: float = 1.0,
    log_f_min: float = 1e-6,
    flatness_threshold: float = 0.8,
    wl_check_interval: int = 5_000,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[int, float], dict[int, float]]:
    """Wang-Landau Phase 1 + production Phase 2.

    Returns (log_g, reweighted_dist, muca_raw_dist).
    reweighted_dist is importance-reweighted to the Boltzmann distribution at p.
    muca_raw_dist is the flat-histogram MUCA distribution (uniform over faulty weights).
    """
    rng = np.random.default_rng(seed)
    code = make_toric_code(code_size)
    decoder = make_mwpm_decoder(code)
    n = code.n

    lookup: dict = {}

    def _get_correction(error: np.ndarray) -> np.ndarray:
        key = tuple((code.hz @ error) % 2)
        if key not in lookup:
            lookup[key] = decoder.decode_syndrome(np.array(key, dtype=np.uint8))
        return lookup[key]

    current_error = code.lx[0].astype(np.int8)
    current_weight = int(np.sum(current_error))

    # Phase 1: Wang-Landau
    log_g = np.zeros(n + 1, dtype=np.float64)
    log_f = log_f_initial
    wl_level = 0

    while log_f > log_f_min:
        H = np.zeros(n + 1, dtype=np.int64)
        steps_in_level = 0
        wl_level += 1

        while True:
            proposed, pw = one_bit_flip(current_error, current_weight, rng)

            correction = _get_correction(proposed)
            if decoder.is_faulty(proposed, correction):
                log_acc = log_g[current_weight] - log_g[pw]
                if rng.random() < min(1.0, np.exp(log_acc)):
                    current_error = proposed
                    current_weight = pw

            log_g[current_weight] += log_f
            H[current_weight] += 1
            steps_in_level += 1

            if steps_in_level % wl_check_interval == 0:
                visited = H[H > 0]
                if (
                    len(visited) > 0
                    and visited.min() >= flatness_threshold * visited.mean()
                ):
                    break

        log_f /= 2.0

    active = np.where(log_g > 0)[0]
    if len(active):
        log_g -= log_g[active].min()

    # Phase 2: Production
    faulty_error_frequencies = np.zeros(n + 1, dtype=np.int64)

    for _ in range(n_production):
        proposed, pw = one_bit_flip(current_error, current_weight, rng)

        correction = _get_correction(proposed)
        if decoder.is_faulty(proposed, correction):
            log_acc = log_g[current_weight] - log_g[pw]
            if rng.random() < min(1.0, np.exp(log_acc)):
                current_error = proposed
                current_weight = pw

        faulty_error_frequencies[current_weight] += 1

    # Reweight to Boltzmann at p
    ws = np.where(faulty_error_frequencies > 0)[0]
    log_p = np.log(p)
    log_1mp = np.log(1.0 - p)
    log_rw = np.array(
        [
            np.log(float(faulty_error_frequencies[w]))
            + log_g[w]
            + w * log_p
            + (n - w) * log_1mp
            for w in ws
        ]
    )
    log_Z_rw = np.logaddexp.reduce(log_rw)
    reweighted_dist = {
        int(w): float(np.exp(lrw - log_Z_rw)) for w, lrw in zip(ws, log_rw)
    }

    total_prod = int(faulty_error_frequencies.sum())
    muca_raw_dist = {int(w): int(faulty_error_frequencies[w]) / total_prod for w in ws}

    return log_g, reweighted_dist, muca_raw_dist
