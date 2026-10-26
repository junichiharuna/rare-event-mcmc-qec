"""MCMC step functions for bit-flip and depolarizing RIMC, and MUCA."""

import numpy as np


def one_bit_flip(error: np.ndarray, weight: int, rng) -> tuple[np.ndarray, int]:
    """Propose a single uniformly-chosen bit flip. Returns (proposed, new_weight)."""
    proposed = error.copy()
    idx = rng.integers(0, len(error))
    proposed[idx] ^= 1
    new_weight = weight + 1 if proposed[idx] == 1 else weight - 1
    return proposed, new_weight


def _update_ones_arr(
    ones_arr: np.ndarray, idx: int, flipped_to_one: bool
) -> np.ndarray:
    """O(weight) update of ones_arr after a single bit flip."""
    if flipped_to_one:
        return np.append(ones_arr, idx)
    else:
        pos = int(np.where(ones_arr == idx)[0][0])
        result = ones_arr.copy()
        result[pos] = result[-1]
        return result[:-1]


def _cluster_exchange_step(
    error,
    weight,
    syndrome,
    n,
    hz_cols,
    cluster_k: int,
    decoder,
    lookup,
    rng,
):
    """
    Weight-preserving cluster exchange: swap cluster_k 1-bits with cluster_k 0-bits.
    MH ratio = 1 (weight unchanged). Accept iff proposed is faulty.
    Returns (new_error, new_weight, new_syndrome).
    """
    if weight < cluster_k or (n - weight) < cluster_k:
        return error, weight, syndrome

    ones = np.where(error == 1)[0]
    zeros = np.where(error == 0)[0]

    from_ones = ones[rng.choice(len(ones), cluster_k, replace=False)]
    from_zeros = zeros[rng.choice(len(zeros), cluster_k, replace=False)]

    proposed = error.copy()
    proposed[from_ones] = 0
    proposed[from_zeros] = 1

    new_syndrome = syndrome.copy()
    for idx in from_ones:
        new_syndrome = new_syndrome ^ hz_cols[int(idx)]
    for idx in from_zeros:
        new_syndrome = new_syndrome ^ hz_cols[int(idx)]

    key = tuple(new_syndrome)
    if key not in lookup:
        lookup[key] = decoder.decode_syndrome(new_syndrome)

    if decoder.is_faulty(proposed, lookup[key]):
        return proposed, weight, new_syndrome  # weight unchanged!

    return error, weight, syndrome


_DEPOL_PAULI = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8)  # X, Y, Z


def _depolarizing_cluster_step(
    ex,
    ez,
    w_any,
    sz,
    sx,
    n,
    cluster_k,
    hz_cols,
    hx_cols,
    decoder,
    lookup_x,
    lookup_z,
    rng,
):
    """Support-preserving cluster move for depolarizing RIMC.

    Two symmetric sub-moves, each keeping the support w = |ex OR ez| fixed:
    relocate (clear cluster_k occupied sites, fill cluster_k empty sites with
    uniform Paulis) and relabel (resample the Pauli on cluster_k occupied sites).
    pi depends only on w (preserved) so the MH ratio is 1; the proposal is
    symmetric, so accept iff the candidate is still faulty.
    Returns (new_ex, new_ez, new_w_any, new_sz, new_sx).
    """
    support = ex | ez
    occ = np.where(support == 1)[0]
    if len(occ) < cluster_k:
        return ex, ez, w_any, sz, sx
    emp = np.where(support == 0)[0]

    pex = ex.copy()
    pez = ez.copy()
    changed_x = []  # sites whose ex bit changed -> update sz
    changed_z = []  # sites whose ez bit changed -> update sx

    if rng.random() < 0.5 and len(emp) >= cluster_k:
        clear = occ[rng.choice(len(occ), cluster_k, replace=False)]
        fill = emp[rng.choice(len(emp), cluster_k, replace=False)]
        for idx in clear:
            idx = int(idx)
            if pex[idx]:
                changed_x.append(idx)
            if pez[idx]:
                changed_z.append(idx)
            pex[idx] = 0
            pez[idx] = 0
        for idx in fill:
            idx = int(idx)
            p = _DEPOL_PAULI[rng.integers(0, 3)]
            if p[0] != pex[idx]:
                changed_x.append(idx)
            if p[1] != pez[idx]:
                changed_z.append(idx)
            pex[idx], pez[idx] = p[0], p[1]
    else:
        sites = occ[rng.choice(len(occ), cluster_k, replace=False)]
        for idx in sites:
            idx = int(idx)
            p = _DEPOL_PAULI[rng.integers(0, 3)]
            if p[0] != pex[idx]:
                changed_x.append(idx)
            if p[1] != pez[idx]:
                changed_z.append(idx)
            pex[idx], pez[idx] = p[0], p[1]

    new_sz = sz.copy()
    for idx in changed_x:
        new_sz = new_sz ^ hz_cols[idx]
    new_sx = sx.copy()
    for idx in changed_z:
        new_sx = new_sx ^ hx_cols[idx]

    new_w = int(np.sum(pex | pez))  # == w_any by construction

    kx = tuple(new_sz)
    if kx not in lookup_x:
        lookup_x[kx] = decoder.decode_x(new_sz)
    kz = tuple(new_sx)
    if kz not in lookup_z:
        lookup_z[kz] = decoder.decode_z(new_sx)
    if decoder.is_faulty(pex, pez, lookup_x[kx], lookup_z[kz]):
        return pex, pez, new_w, new_sz, new_sx
    return ex, ez, w_any, sz, sx


def _mixed_step_with_single(
    error,
    weight,
    syndrome,
    ones_arr,
    n,
    log_p1mp,
    alpha,
    beta,
    alpha_m_beta,
    hx_dense,
    n_stab,
    lx_dense,
    n_logic,
    hz_cols,
    p_stab,
    p_logic,
    p_single,
    decoder,
    lookup,
    rng,
    p_cluster: float = 0.0,
    cluster_k: int = 2,
    move_stats: dict | None = None,
) -> tuple:
    """
    One proposal step for a single replica.
    Returns (new_error, new_weight, new_syndrome, new_ones_arr).

    Dispatch order (cumulative thresholds on u ~ Uniform[0,1)):
      u < p_logic                                  -> LogicFlip  (MWPM-free, faultiness may change)
      u < p_logic + p_stab                         -> StabFlip   (MWPM-free, faultiness preserved)
      u < p_logic + p_stab + p_single              -> uniform 1-bit flip MH  (MWPM with cache)
      u < p_logic + p_stab + p_single + p_cluster  -> ClusterExchange  (MWPM with cache)
      else                                         -> Zanella locally-balanced  (MWPM with cache)
    """
    u = rng.random()

    if u < p_logic:
        row_k = rng.integers(0, n_logic)
        proposed = (error ^ lx_dense[row_k]).astype(np.int8)
        pw = int(np.sum(proposed))
        if rng.random() < min(1.0, np.exp((pw - weight) * log_p1mp)):
            correction = lookup[tuple(syndrome)]
            if decoder.is_faulty(proposed, correction):
                new_ones_arr = np.where(proposed == 1)[0].astype(np.int32)
                return proposed, pw, syndrome, new_ones_arr
        return error, weight, syndrome, ones_arr

    elif u < p_logic + p_stab:
        row_i = rng.integers(0, n_stab)
        proposed = (error ^ hx_dense[row_i]).astype(np.int8)
        pw = int(np.sum(proposed))
        if rng.random() < min(1.0, np.exp((pw - weight) * log_p1mp)):
            new_ones_arr = np.where(proposed == 1)[0].astype(np.int32)
            return proposed, pw, syndrome, new_ones_arr
        return error, weight, syndrome, ones_arr

    elif u < p_logic + p_stab + p_single:
        idx = rng.integers(0, n)
        proposed = error.copy()
        proposed[idx] ^= 1
        flipped_to_one = error[idx] == 0
        pw = weight + (1 if flipped_to_one else -1)
        if rng.random() < min(1.0, np.exp((pw - weight) * log_p1mp)):
            proposed_syndrome = syndrome ^ hz_cols[idx]
            key = tuple(proposed_syndrome)
            if key not in lookup:
                lookup[key] = decoder.decode_syndrome(proposed_syndrome)
            if decoder.is_faulty(proposed, lookup[key]):
                new_ones_arr = _update_ones_arr(ones_arr, int(idx), flipped_to_one)
                return proposed, pw, proposed_syndrome, new_ones_arr
        return error, weight, syndrome, ones_arr

    elif u < p_logic + p_stab + p_single + p_cluster:
        new_error, new_weight, new_syndrome = _cluster_exchange_step(
            error, weight, syndrome, n, hz_cols, cluster_k, decoder, lookup, rng
        )
        if move_stats is not None:
            move_stats["h_proposed"] += 1
        if new_error is error:
            return error, weight, syndrome, ones_arr
        if move_stats is not None:
            move_stats["h_accepted"] += 1
        new_ones_arr = np.where(new_error == 1)[0].astype(np.int32)
        return new_error, new_weight, new_syndrome, new_ones_arr

    else:
        Z_cur = weight * alpha + (n - weight) * beta

        p_flip_one = weight * alpha / Z_cur
        if rng.random() < p_flip_one:
            # flip a 1-bit: O(1) using ones_arr
            if weight == 0:
                return error, weight, syndrome, ones_arr
            pos = rng.integers(0, weight)
            idx = int(ones_arr[pos])
            pw = weight - 1
            Z_new = Z_cur - alpha_m_beta
            flipped_to_one = False
        else:
            # flip a 0-bit: rejection sampling O(1) expected at low p
            n_zeros = n - weight
            if n_zeros == 0:
                return error, weight, syndrome, ones_arr
            while True:
                idx = int(rng.integers(0, n))
                if error[idx] == 0:
                    break
            pw = weight + 1
            Z_new = Z_cur + alpha_m_beta
            flipped_to_one = True

        proposed = error.copy()
        proposed[idx] ^= 1
        proposed_syndrome = syndrome ^ hz_cols[idx]

        if rng.random() < min(1.0, np.exp(np.log(Z_cur) - np.log(Z_new))):
            key = tuple(proposed_syndrome)
            if key not in lookup:
                lookup[key] = decoder.decode_syndrome(proposed_syndrome)
            if decoder.is_faulty(proposed, lookup[key]):
                new_ones_arr = _update_ones_arr(ones_arr, idx, flipped_to_one)
                return proposed, pw, proposed_syndrome, new_ones_arr

        return error, weight, syndrome, ones_arr


def _depolarizing_step(
    ex,
    ez,
    w_any,
    sz,
    sx,
    n,
    log_p1mp_dep,
    alpha_dep,
    beta_dep,
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
    rng,
    p_cluster: float = 0.0,
    cluster_k: int = 2,
):
    """
    One proposal step for a single depolarizing replica.
    Returns (new_ex, new_ez, new_w_any, new_sz, new_sx).

    Dispatch order (cumulative thresholds on u ~ Uniform[0,1)):
      u < p_logic                      -> LogicFlip
      u < p_logic + p_stab             -> StabFlip
      u < p_logic + p_stab + p_single  -> Single-bit flip MH
      else                             -> Zanella locally-balanced
    """
    u = rng.random()

    if u < p_logic:
        row_k = rng.integers(0, n_logic)
        if rng.random() < 0.5:
            proposed_ex = (ex ^ lx_dense[row_k]).astype(np.int8)
            proposed_ez = ez
        else:
            proposed_ex = ex
            proposed_ez = (ez ^ lz_dense[row_k]).astype(np.int8)
        pw = int(np.sum(proposed_ex | proposed_ez))
        if rng.random() < min(1.0, np.exp((pw - w_any) * log_p1mp_dep)):
            c_x = lookup_x[tuple(sz)]
            c_z = lookup_z[tuple(sx)]
            if decoder.is_faulty(proposed_ex, proposed_ez, c_x, c_z):
                return proposed_ex, proposed_ez, pw, sz, sx
        return ex, ez, w_any, sz, sx

    elif u < p_logic + p_stab:
        if rng.random() < 0.5:
            row_i = rng.integers(0, n_stab_x)
            proposed_ex = (ex ^ hx_dense[row_i]).astype(np.int8)
            proposed_ez = ez
        else:
            row_i = rng.integers(0, n_stab_z)
            proposed_ex = ex
            proposed_ez = (ez ^ hz_dense[row_i]).astype(np.int8)
        pw = int(np.sum(proposed_ex | proposed_ez))
        if rng.random() < min(1.0, np.exp((pw - w_any) * log_p1mp_dep)):
            return proposed_ex, proposed_ez, pw, sz, sx
        return ex, ez, w_any, sz, sx

    elif u < p_logic + p_stab + p_single:
        if rng.random() < 0.5:
            idx = rng.integers(0, n)
            dw = (1 if ex[idx] == 0 else -1) if ez[idx] == 0 else 0
            pw = w_any + dw
            if rng.random() < min(1.0, np.exp(dw * log_p1mp_dep)):
                proposed_ex = ex.copy()
                proposed_ex[idx] ^= 1
                new_sz = sz ^ hz_cols[idx]
                key_x = tuple(new_sz)
                if key_x not in lookup_x:
                    lookup_x[key_x] = decoder.decode_x(new_sz)
                key_z = tuple(sx)
                if key_z not in lookup_z:
                    lookup_z[key_z] = decoder.decode_z(sx)
                if decoder.is_faulty(proposed_ex, ez, lookup_x[key_x], lookup_z[key_z]):
                    return proposed_ex, ez, pw, new_sz, sx
        else:
            idx = rng.integers(0, n)
            dw = (1 if ez[idx] == 0 else -1) if ex[idx] == 0 else 0
            pw = w_any + dw
            if rng.random() < min(1.0, np.exp(dw * log_p1mp_dep)):
                proposed_ez = ez.copy()
                proposed_ez[idx] ^= 1
                new_sx = sx ^ hx_cols[idx]
                key_z = tuple(new_sx)
                if key_z not in lookup_z:
                    lookup_z[key_z] = decoder.decode_z(new_sx)
                key_x = tuple(sz)
                if key_x not in lookup_x:
                    lookup_x[key_x] = decoder.decode_x(sz)
                if decoder.is_faulty(ex, proposed_ez, lookup_x[key_x], lookup_z[key_z]):
                    return ex, proposed_ez, pw, sz, new_sx
        return ex, ez, w_any, sz, sx

    elif u < p_logic + p_stab + p_single + p_cluster:
        return _depolarizing_cluster_step(
            ex, ez, w_any, sz, sx, n, cluster_k, hz_cols, hx_cols,
            decoder, lookup_x, lookup_z, rng,
        )

    else:
        both = ex & ez
        n_Y = int(both.sum())
        n_X = int(ex.sum()) - n_Y
        n_Z = int(ez.sum()) - n_Y
        n_I = n - n_X - n_Z - n_Y

        Z_cur = 2.0 * beta_dep * n_I + (alpha_dep + 1.0) * (n_X + n_Z) + 2.0 * n_Y

        t1 = beta_dep * n_I
        t2 = t1 + beta_dep * n_I
        t3 = t2 + alpha_dep * n_X
        t4 = t3 + float(n_X)
        t5 = t4 + float(n_Z)
        t6 = t5 + alpha_dep * n_Z
        t7 = t6 + float(n_Y)

        r = rng.random() * Z_cur

        if r < t1:
            cands = np.where((ex == 0) & (ez == 0))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], True
            dw = 1
            Z_new = Z_cur + (alpha_dep + 1.0 - 2.0 * beta_dep)
        elif r < t2:
            cands = np.where((ex == 0) & (ez == 0))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], False
            dw = 1
            Z_new = Z_cur + (alpha_dep + 1.0 - 2.0 * beta_dep)
        elif r < t3:
            cands = np.where((ex == 1) & (ez == 0))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], True
            dw = -1
            Z_new = Z_cur - (alpha_dep + 1.0 - 2.0 * beta_dep)
        elif r < t4:
            cands = np.where((ex == 1) & (ez == 0))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], False
            dw = 0
            Z_new = Z_cur + (1.0 - alpha_dep)
        elif r < t5:
            cands = np.where((ex == 0) & (ez == 1))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], True
            dw = 0
            Z_new = Z_cur + (1.0 - alpha_dep)
        elif r < t6:
            cands = np.where((ex == 0) & (ez == 1))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], False
            dw = -1
            Z_new = Z_cur - (alpha_dep + 1.0 - 2.0 * beta_dep)
        elif r < t7:
            cands = np.where((ex == 1) & (ez == 1))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], True
            dw = 0
            Z_new = Z_cur + (alpha_dep - 1.0)
        else:
            cands = np.where((ex == 1) & (ez == 1))[0]
            idx, flip_ex = cands[rng.integers(0, len(cands))], False
            dw = 0
            Z_new = Z_cur + (alpha_dep - 1.0)

        pw = w_any + dw

        if rng.random() < min(1.0, np.exp(np.log(Z_cur) - np.log(Z_new))):
            if flip_ex:
                proposed_ex = ex.copy()
                proposed_ex[idx] ^= 1
                new_sz = sz ^ hz_cols[idx]
                key_x = tuple(new_sz)
                if key_x not in lookup_x:
                    lookup_x[key_x] = decoder.decode_x(new_sz)
                key_z = tuple(sx)
                if key_z not in lookup_z:
                    lookup_z[key_z] = decoder.decode_z(sx)
                if decoder.is_faulty(proposed_ex, ez, lookup_x[key_x], lookup_z[key_z]):
                    return proposed_ex, ez, pw, new_sz, sx
            else:
                proposed_ez = ez.copy()
                proposed_ez[idx] ^= 1
                new_sx = sx ^ hx_cols[idx]
                key_z = tuple(new_sx)
                if key_z not in lookup_z:
                    lookup_z[key_z] = decoder.decode_z(new_sx)
                key_x = tuple(sz)
                if key_x not in lookup_x:
                    lookup_x[key_x] = decoder.decode_x(sz)
                if decoder.is_faulty(ex, proposed_ez, lookup_x[key_x], lookup_z[key_z]):
                    return ex, proposed_ez, pw, sz, new_sx

        return ex, ez, w_any, sz, sx
