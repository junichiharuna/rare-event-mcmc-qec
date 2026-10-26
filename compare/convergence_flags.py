"""Per-entry convergence flags for the paper's MCMC audit tables.

Today every entry of tab:audit / tab:rect / tab:lowp (and the depolarizing
tab:depol / tab:depollowp) prints a signed bias, and the only stated criterion is
whether the 5*tau burn-in rule had a fixed point (with a half-trace fallback).
This module turns that into an explicit, mechanical per-entry verdict.

VERDICT LOGIC (all thresholds are CLI options)
----------------------------------------------
PRIMARY -- used wherever a sibling `<stem>_chains.csv` supplies the K chain means
at the checkpoints N/16, N/8, N/4, N/2, N (the format cluster_scripts/lowp_audit.py
emits). Four tests, ANDed:

  (S) SCALING. The between-chain SD of the K running means must shrink like
      1/sqrt(N) from the N/4 reference to the final N,
          sd(N) <= scale_factor * sd(N/4) * sqrt((N/4)/N)   [default 1.5]
      i.e. "the statistical error actually halved when the chain quadrupled".
      An ensemble whose spread is dominated by a frozen inter-sector offset does
      not shrink and fails here even when every within-chain diagnostic is healthy.
      The ratio against the earliest checkpoint N/16 is reported as
      `scale_ratio_first` but does NOT decide: for a long-initialized chain the
      N/16 spread is inflated by the transient, so the N/16 reference is biased
      towards passing (UT(3,15) p=1e-2 long OBF: 0.51 against 0.73).

  (D) DRIFT, on late segments only. Cumulative running means cannot separate a
      residual transient from a genuine drift, and -- as the 2026-08-28 report
      noted -- the cumulative test therefore punishes long initialization for
      exactly what the burn-in is there to remove. Note also that "second half vs
      first half" is algebraically the SAME test: with
      m_2nd = 2 m(N) - m(N/2), the difference m_2nd - m(N/2) = 2(m(N) - m(N/2)) and
      its SD is 2 SE(N), so the z is identical. The fix is to drop the first
      quarter entirely and compare the two LATE segments,
          D = mean(N/2..N) - mean(N/4..N/2) = 2 m(N) - 3 m(N/2) + m(N/4),
          Var(D) = SE(N)^2 * N * (1/(N-N/2) + 1/(N/2-N/4)) = 6 SE(N)^2,
      pass iff |D| <= drift_k * sqrt(6) * SE(N)  [default drift_k = 3].
      `--chain-drift-mode cum` restores the cumulative m(N) vs m(N/2) test.

  (M) MULTI-START. Where a condition was run from both the min-weight ("short")
      and the max-weight ("long") initialization, the two REPORTED (5*tau-burned)
      estimates must agree:
          |est_short - est_long| <= ms_k * sqrt(se_short^2 + se_long^2)  [3].
      This test is not optional decoration. It is the ONLY one of the four that
      catches a chain ensemble frozen in the wrong logical sector: UT(3,15) at
      p=1e-4 from the long start sits at w = 8.008 against a reference 2.0097 --
      a bias of +6.0 -- yet its 50 chains scatter by only 0.0026, shrink like
      1/sqrt(N) (S: 1.15), do not drift (D: 1.6) and match their own
      autocorrelation time (Rhat 0.94). All within-ensemble statistics are healthy
      because the chains fluctuate with the RIGHT variance about the WRONG mean.
      Codes with a flat logical-weight basis have a single "uniform" start, so (M)
      is vacuous for them and their certificate is correspondingly weaker.

  (R) DISPERSION. Rhat = s_B / s_W, with a deliberately LOOSE cut
      (`--chain-rhat-max`, default 3.0). Its job in the primary rule is not to be a
      null-calibrated test -- for that see rhat_critical() -- but to catch the one
      failure (S) misses in the other direction: a spread that DOES shrink like
      1/sqrt(N) while being orders of magnitude larger than the chains' own tau
      predicts. [[144,12,12]] at p=1e-3 from the long start passes S (1.35), D (1.4)
      and M (0.9) and is biased by +3.09; Rhat = 34 rejects it.

      s_W is by DEFAULT computed from the CHAINS ALONE (`--rhat-within chain`), so
      that the whole primary rule is an internal consistency check requiring no
      external reference -- which is the point of the mean-weight benchmark. W is
      the textbook Gelman-Rubin within-chain variance, the ARITHMETIC MEAN over the
      K chains of each chain's own variance on its 5*tau-burned segment, which the
      run CSV carries as `w_var_within_mean`; then

          s_W = sqrt(W * tau_factor * tau_burn5 / n_used5)

      in the tau_int convention Var(chain mean) = tau sigma^2 / n -- no factor 2,
      see TAU CONVENTION below.

      `--rhat-within ref` restores the reference form of the fallback rule below,
      s_W = sqrt(Var[w|F] * tau_factor * tau / n_used) with Var[w|F] taken from an
      MCMC-INDEPENDENT reference, and reproduces the 2026-08 verdicts exactly. Rows
      whose CSV has no `w_var_within_mean` column -- every pre-2026-09 file -- fall
      back to the reference form automatically even in `chain` mode. The output
      field `rhat_within_used` records which of the two ('chain' / 'ref' / 'none')
      decided each row, and `rhat_chain` / `rhat_ref` (with `s_within_chain` /
      `s_within_ref`) are always emitted side by side so the two can be compared;
      `rhat`, `s_within`, `vw` and `vw_source` always describe the one USED.

      The chain form is the more conservative of the two. A trapped ensemble
      explores less than the target within each chain, so W UNDERSTATES the true
      Var[w|F] and Rhat_chain typically exceeds Rhat_ref (2026-09 audit on
      lowp_audit_2e6 + lowp_bb144_long: the ratio spans 0.83-4.37, and every row
      that moves across the cut at all moves towards FAILING).

  A fixed-N run with a post-hoc check is preferred over an online/adaptive stopping
  rule because adaptive stopping would break the embarrassingly-parallel 50-chain
  cluster layout: every chain must finish at the same N for the workers to be
  scheduled independently and merged without cross-talk.

FALLBACK -- Gelman-Rubin-style dispersion ratio, used when per-chain checkpoints
  are absent (i.e. for every entry in the paper's 2026-06/08 production CSVs):
      Rhat = s_B / s_W,
      s_B  = observed between-chain SD of the K burned chain means (`ew_burn5_sd`),
      s_W  = sqrt(Var[w|F] * tau_factor * tau / n_used), the SD a single chain mean
             SHOULD have at its own measured autocorrelation time.
  Those CSVs carry no `w_var_within_mean` column, so Var[w|F] here is necessarily
  the MCMC-INDEPENDENT reference (`vw` in subset_sampling.csv, else
  direct_mc_ew.csv) and the fallback rule is unchanged by `--rhat-within`.
  Converged iff Rhat <= rhat_max AND the drift test passes.

AUXILIARY -- reported for every entry, and decisive only in the two places noted:
  (a) fixed-point fraction: capped5 / n_seeds <= cap_frac  (default 10% capped),
  (b) length in autocorrelation times: n_used5 / tau_burn5 >= n_tau_min.

WHERE (a)/(b) ACTUALLY DECIDE
  1. Rhat-CIRCULAR rows. When no reference `vw` exists for a code, Var[w|F] is
     estimated from the best-mixing arm of the same cell; that donor row then has
     Rhat == 1 by construction, which is not evidence. Donor rows are judged by
     (a)+(b) instead and carry rhat_circular=True.
  2. Rows with NO Var[w|F] at all -- neither a reference nor a `w_var_within_mean`
     column. These cannot be certified and are always
     flagged. (a)/(b) alone must never certify them: `p6_depol_bb144_burn.csv`
     p=1e-3/minw/OBF has capped5=0 and n_used/tau=251, so it passes both, yet it
     is +2.99 biased. Refusing to certify without an independent variance is the
     whole point of the flag.

TAU CONVENTION -- IMPORTANT
  src.metrics.iat() returns the Geyer-truncated tau_int = 1 + 2*sum_k rho_k, i.e.
  the convention in which Var(chain mean) = tau * sigma^2 / n. The within-chain SE
  is therefore sqrt(vw * tau / n_used) and NOT sqrt(vw * 2*tau / n_used); the
  latter would divide every Rhat by sqrt(2). Verified empirically: on well-mixed
  arms (UT(3,11) Z+H at 1e-3, BB[[72,12,6]] Z+H) ew_burn5_sd^2 * n_used / tau
  reproduces the reference `vw` to within 1-30%, whereas the 2*tau form is low by
  a factor 2. `--tau-factor 2.0` reproduces the 2*tau reading of the spec.
  lowp_audit.py emits the same quantity per row as `ew_within_se_tau1` (the MEDIAN
  over chains of sqrt(var_c tau_c / n_c)); the two agree to better than a factor 1.3
  on every well-mixed arm, which cross-checks the reference `vw` against the chains
  themselves. `w_var_within_mean` is the pooled (arithmetic-mean-over-chains)
  variance behind the same quantity and is what `--rhat-within chain` consumes.

DONOR GATE
  An MCMC-derived Var[w|F] is accepted only from an arm with capped5 == 0 AND
  n_used5/tau_burn5 >= --min-donor-ntau (default 500), and the MINIMUM vw_hat over
  qualifying arms is taken: trapping inflates the observed spread, never deflates
  it, so the minimum errs on the side of a LARGER Rhat (harder to certify).

Outputs compare/out/convergence_flags.csv and prints a readable table.
`--only lowp_audit,lowp_long` restricts to the 2026-08-28 low-p cross-code job;
`--only lowp_audit_gr,lowp_long_gr` to its 2026-09 rerun, the one that carries
`w_var_within_mean` and therefore exercises the chain form of (R).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "cluster_results")
OUTDIR = os.path.join(ROOT, "compare", "out")

# ---------------------------------------------------------------- pure rules --


def _normal_quantile(q: float) -> float:
    """Acklam rational approximation to the standard normal quantile."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1.0 - 0.02425
    if q < pl:
        t = math.sqrt(-2.0 * math.log(q))
        return ((((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
                / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0))
    if q > ph:
        t = math.sqrt(-2.0 * math.log(1.0 - q))
        return (-(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5])
                / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0))
    t = q - 0.5
    r = t * t
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))


def wilson_hilferty_chi2(k: int, q: float) -> float:
    """Wilson-Hilferty approximation to the q-quantile of chi^2 with k dof.

    Avoids a scipy dependency; accurate to well under 1% for k >= 5 in the upper
    tail, which is all that is needed to turn an alpha into an Rhat cut.
    """
    z = _normal_quantile(q)
    a = 2.0 / (9.0 * k)
    return k * (1.0 - a + z * math.sqrt(a)) ** 3


def rhat_critical(n_chains: int, alpha: float = 1e-3) -> float:
    """Null-calibrated cut for Rhat = s_B / s_W with `n_chains` chains.

    Under perfect mixing (K-1) s_B^2 / s_W^2 ~ chi^2_{K-1}, so the (1-alpha)
    quantile fixes a false-alarm rate rather than an eyeballed constant.
      K=50: 1.24 at alpha=1e-2, 1.32 at alpha=1e-3.
      K=8 : 1.63 at alpha=1e-2, 1.87 at alpha=1e-3.
    """
    if n_chains < 2:
        return float("inf")
    k = n_chains - 1
    return math.sqrt(wilson_hilferty_chi2(k, 1.0 - alpha) / k)


def scaling_verdict(ns: Sequence[float], sds: Sequence[float],
                    scale_factor: float = 1.5,
                    ref_frac: Optional[float] = None) -> Dict[str, object]:
    """Post-hoc 1/sqrt(N) check on the between-chain SD of the chain means.

    `ns`/`sds` are checkpoint lengths and the between-chain SD of the K chain
    means at each. The final checkpoint N is compared with a reference checkpoint
    n_ref:

        pass  iff  sd(N) <= scale_factor * sd(n_ref) * sqrt(n_ref / N)

    i.e. the statistical error actually shrank, to within `scale_factor`. A chain
    ensemble whose spread is dominated by a frozen inter-sector offset does not
    shrink and fails here even when every within-chain diagnostic looks healthy.

    `ref_frac` selects n_ref as the checkpoint closest to ref_frac * N (excluding
    N itself); the default None keeps the earliest checkpoint, which is what the
    two-point call sites in the tests use. The production setting is ref_frac=0.25:
    on a long-initialized chain the EARLIEST checkpoint's spread is inflated by the
    transient, which biases the test towards passing. The ratio against the
    earliest checkpoint is always reported as `ratio_first`.
    """
    pts = sorted((float(n), float(s)) for n, s in zip(ns, sds)
                 if n and s is not None and not math.isnan(float(s)))
    nan = float("nan")
    if len(pts) < 2:
        return {"n_ref": nan, "sd_ref": nan, "n_final": nan, "sd_final": nan,
                "sd_expected": nan, "ratio": nan, "ratio_first": nan, "ok": None}
    n_fin, sd_fin = pts[-1]

    def _ratio(n_ref: float, sd_ref: float) -> float:
        expected = sd_ref * math.sqrt(n_ref / n_fin)
        return float("inf") if expected == 0.0 else sd_fin / expected

    if ref_frac is None:
        n_ref, sd_ref = pts[0]
    else:
        n_ref, sd_ref = min(pts[:-1], key=lambda t: abs(t[0] - ref_frac * n_fin))
    ratio = _ratio(n_ref, sd_ref)
    return {"n_ref": n_ref, "sd_ref": sd_ref, "n_final": n_fin, "sd_final": sd_fin,
            "sd_expected": sd_ref * math.sqrt(n_ref / n_fin), "ratio": ratio,
            "ratio_first": _ratio(*pts[0]), "ok": ratio <= scale_factor}


def late_drift_verdict(ns: Sequence[float], ms: Sequence[float], se_final: float,
                       drift_k: float = 3.0) -> Dict[str, object]:
    """Drift between the two LATE segments of the run, from cumulative means.

    Given cumulative pooled running means m(n1) < m(n2) < m(N) at the last three
    checkpoints, the disjoint-segment means are

        late    = mean over (n2, N]  = (N m(N) - n2 m(n2)) / (N - n2)
        earlier = mean over (n1, n2] = (n2 m(n2) - n1 m(n1)) / (n2 - n1)

    and, writing SE(N) for the cross-chain SE of the pooled mean at N (so that a
    segment of length L has variance SE(N)^2 * N / L, and the two segments are
    disjoint hence treated as independent),

        Var(late - earlier) = SE(N)^2 * N * (1/(N - n2) + 1/(n2 - n1)),

    which is 6 SE(N)^2 on the geometric grid n1 = N/4, n2 = N/2. Dropping the
    first quarter is the point: it is where the initialization transient lives, and
    the burn-in the tables quote has already removed it, so a cumulative test would
    charge a long-initialized entry for a transient it does not report.
    """
    nan = float("nan")
    out = {"drift": nan, "drift_se": nan, "drift_z": nan, "ok": None,
           "seg_late": nan, "seg_early": nan, "n_split": nan}
    pts = sorted((float(n), float(m)) for n, m in zip(ns, ms)
                 if n and m is not None and not math.isnan(float(m)))
    if len(pts) < 3 or not (se_final > 0):
        return out
    (n1, m1), (n2, m2), (nf, mf) = pts[-3], pts[-2], pts[-1]
    if not (n1 < n2 < nf):
        return out
    late = (nf * mf - n2 * m2) / (nf - n2)
    early = (n2 * m2 - n1 * m1) / (n2 - n1)
    se = se_final * math.sqrt(nf * (1.0 / (nf - n2) + 1.0 / (n2 - n1)))
    v = drift_verdict(early, late, se, drift_k)
    out.update(v)
    out.update({"seg_late": late, "seg_early": early, "n_split": n2})
    return out


def multistart_verdict(est_a: float, se_a: float, est_b: float, se_b: float,
                       ms_k: float = 3.0) -> Dict[str, object]:
    """Do two initializations of the same condition report the same number?

    Compares the values the TABLE quotes (the 5*tau-burned chain-mean averages)
    with their cross-chain SEs, not the raw running means: the verdict certifies
    the printed entry, and the raw means still carry the transient that the
    burn-in removes. Vacuous (ok=None) where only one initialization was run.
    """
    if est_b is None or math.isnan(est_b) or math.isnan(est_a):
        return {"ms_diff": float("nan"), "ms_se": float("nan"),
                "ms_z": float("nan"), "ok": None}
    se = combine_se(se_a, se_b)
    d = est_a - est_b
    z = float("inf") if not (se > 0) else abs(d) / se
    return {"ms_diff": d, "ms_se": se, "ms_z": z, "ok": z <= ms_k}


def drift_verdict(m_early: float, m_late: float, se: float,
                  drift_k: float = 3.0) -> Dict[str, object]:
    """|m_late - m_early| <= drift_k * se  ->  no resolvable drift."""
    d = float(m_late) - float(m_early)
    z = float("inf") if not (se > 0) else abs(d) / se
    return {"drift": d, "drift_se": se, "drift_z": z, "ok": z <= drift_k}


def rhat_verdict(s_between: float, s_within: float, rhat_max: float) -> Dict[str, object]:
    """Gelman-Rubin-style dispersion ratio of observed to predicted chain scatter."""
    if s_within is None or math.isnan(s_within) or not (s_within > 0):
        return {"rhat": float("nan"), "ok": None}
    r = s_between / s_within
    return {"rhat": r, "ok": r <= rhat_max}


def select_within(s_chain: float, s_ref: float, mode: str = "chain"
                  ) -> Tuple[float, str]:
    """Pick the within-chain SD that feeds Rhat, and say which one it is.

    `s_chain` is the Gelman-Rubin form built from the run's own pooled within-chain
    variance W (`w_var_within_mean`), `s_ref` the older form built from an
    MCMC-independent Var[w|F]. Returns (s_within, used) with

      mode='chain' (default) -- the chain form when it is finite and positive, else
        the reference form; a CSV predating `w_var_within_mean` therefore keeps its
        old verdict instead of losing the test.
      mode='ref' -- always the reference form, reproducing the 2026-08 verdicts.

    `used` is 'chain', 'ref', or 'none' when neither is usable (Rhat undefined).
    """
    def _ok(x) -> bool:
        return x is not None and not math.isnan(x) and x > 0

    if mode == "chain" and _ok(s_chain):
        return float(s_chain), "chain"
    if _ok(s_ref):
        return float(s_ref), "ref"
    return float("nan"), "none"


def classify(*, scaling_ok: Optional[bool], rhat_ok: Optional[bool],
             rhat_circular: bool, has_variance_ref: bool,
             drift_ok: Optional[bool], aux_a_ok: bool, aux_b_ok: bool
             ) -> Tuple[bool, str, str]:
    """Combine the tests into (converged, verdict_source, reason).

    Precedence: per-chain scaling > Rhat > auxiliary. A row with no independent
    Var[w|F] is never certified (see module docstring). The drift test is an AND
    on every branch; where it could not be computed it is treated as passing.
    """
    d_ok = True if drift_ok is None else bool(drift_ok)
    if scaling_ok is not None:
        ok = bool(scaling_ok) and d_ok
        why = ([] if scaling_ok else ["scaling"]) + ([] if d_ok else ["drift"])
        return ok, "scaling", "ok" if ok else "+".join(why)
    if not has_variance_ref:
        return False, "none", "no_variance_reference"
    if rhat_circular:
        ok = aux_a_ok and aux_b_ok and d_ok
        why = (([] if aux_a_ok else ["fixedpoint"]) + ([] if aux_b_ok else ["short"])
               + ([] if d_ok else ["drift"]))
        return ok, "aux(circular-Rhat)", "ok" if ok else "+".join(why)
    if rhat_ok is None:
        return False, "none", "no_variance_reference"
    ok = bool(rhat_ok) and d_ok
    why = ([] if rhat_ok else ["rhat"]) + ([] if d_ok else ["drift"])
    return ok, "rhat", "ok" if ok else "+".join(why)


def classify_primary(*, scaling_ok: Optional[bool], drift_ok: Optional[bool],
                     ms_ok: Optional[bool], rhat_ok: Optional[bool]
                     ) -> Tuple[bool, str, str]:
    """The four-test primary rule, for rows that have per-chain checkpoints.

    All four are ANDed; a test that could not be computed (None) is treated as
    passing, which is why a single-initialization code gets a weaker certificate.
    """
    names = (("scaling", scaling_ok), ("drift", drift_ok),
             ("multistart", ms_ok), ("rhat", rhat_ok))
    bad = [n for n, v in names if v is False]
    return (not bad), "primary", ("ok" if not bad else "+".join(bad))


def combine_se(*ses: float) -> float:
    return math.sqrt(sum(s * s for s in ses if s is not None and not math.isnan(s)))


# ------------------------------------------------------------------- loading --

# One entry per merged production CSV behind a paper table.
DATASETS = [
    dict(name="rect", csv="rect_audit_zh.csv", noise="bit_flip",
         tables=("tab:rect",)),
    dict(name="zh_rerun", csv="zh_rerun_burn.csv", noise="bit_flip",
         tables=("tab:audit", "tab:lowp", "tab:ablation")),
    dict(name="depol", csv="depol_audit_burn.csv", noise="depolarizing",
         tables=("tab:depol",)),
    dict(name="p6_depol", csv="p6_depol_bb144_burn.csv", noise="depolarizing",
         tables=("tab:depollowp",)),
    # Emitted by cluster_scripts/lowp_audit.sh / lowp_bb144_long.sh; these carry the
    # per-chain checkpoint file, so the scaling criterion decides them. Absent files
    # are skipped silently.
    dict(name="lowp_audit", csv="lowp_audit.csv", noise="bit_flip",
         tables=("tab:audit3", "tab:lowp"), key_d=False),
    dict(name="lowp_audit_2e6", csv="lowp_audit_2e6.csv", noise="bit_flip",
         tables=("tab:audit3", "tab:lowp"), key_d=False),
    dict(name="lowp_long", csv="lowp_bb144_long.csv", noise="bit_flip",
         tables=("tab:lowp",), key_d=False),
    # 2026-09 rerun of the two jobs above. Same schema plus `w_var_within_mean`
    # (and `w_var_within_median`, `n_used5_mean`, `tau_burn5_mean`), i.e. the
    # pooled within-chain variance that makes criterion (R) reference-free.
    dict(name="lowp_audit_gr", csv="lowp_audit_2e6_gr.csv", noise="bit_flip",
         tables=("tab:audit3", "tab:lowp"), key_d=False),
    dict(name="lowp_long_gr", csv="lowp_bb144_long_gr.csv", noise="bit_flip",
         tables=("tab:lowp",), key_d=False),
    dict(name="lowp_audit_k1", csv="lowp_audit_2e6_k1.csv", noise="bit_flip",
         tables=("tab:audit3",), key_d=False),
    dict(name="lowp_bb108_p1e2_4e7", csv="lowp_bb108_p1e2_4e7.csv", noise="bit_flip",
         tables=("tab:lowp",), key_d=False),
    # cluster_scripts/lowp_audit.py --noise depolarizing: the same protocol and
    # schema as lowp_audit_gr; its reference is subset_sampling_depol.csv.
    dict(name="depol_meanw", csv="depol_meanw_2e6.csv", noise="depolarizing",
         tables=("tab:audit3", "tab:lowp"), key_d=False),
]

# Per-chain running-mean column in `<stem>_chains.csv`, best first. lowp_audit.py
# writes both the raw running mean and one with the first 10% of each checkpoint
# discarded; the latter is closer to the burned estimator the tables quote.
# `ew_running` (the plain cumulative mean over the first n steps) is the default
# on purpose: the late-segment drift test recovers disjoint segment means from
# cumulative ones, algebra that only holds for a mean starting at step 0.
# `ew_running_burn10pct` is a mean over [0.1n, n] and breaks it.
CHAIN_VALUE_COLS = ("ew_running", "ew_running_burn10pct", "ew")
# `<stem>_chains.csv` names the checkpoint `checkpoint_n`; `<stem>_conv.csv` and the
# older files name it `n_samples`.
CHAIN_N_COLS = ("checkpoint_n", "n_samples")

# run-CSV code label -> candidate (code_type, d) keys in the reference CSVs.
# The same keys serve both noise models: references are keyed by noise first
# (see load_references), the bit-flip ones in direct_mc_ew.csv / subset_sampling.csv
# and the depolarizing one in subset_sampling_depol.csv.
# hgp_hamming3 and bb_l9m6 have no row in direct_mc_ew.csv; their bit-flip reference
# comes from subset_sampling.csv (and the depolarizing one from subset_sampling_depol.csv).
REF_KEYS = {
    "5x5": [("asymmetric_toric", "5x5"), ("asymmetric_toric", "5")],
    "3x11": [("asymmetric_toric", "3x11")],
    "3x15": [("asymmetric_toric", "3x15")],
    "bb144": [("bb144", "0")],
    "bb72": [("bb72", "0")],
    # added 2026-08-28: subset_sampling.csv now covers all eleven codes, so the
    # older run CSVs (which carry no `code_type` column) can reach the reference too.
    "hgp_hamming3": [("hgp_hamming3", "0")],
    "bb_l9m6": [("bb_l9m6", "0")],
}


def _read(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str, default=float("nan")) -> float:
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_references(results_dir: str = RESULTS
                    ) -> Dict[Tuple[str, str, str, float], dict]:
    """(noise, code_type, d, p) -> {ew, ew_se, vw, p_L, source}; subset sampling wins.

    A row with an empty `noise` field counts as bit_flip.
    """
    out: Dict[Tuple[str, str, str, float], dict] = {}
    for fname, src in (("direct_mc_ew.csv", "directmc"),
                       ("subset_sampling.csv", "subset"),
                       ("subset_sampling_depol.csv", "subset")):
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        for r in _read(path):
            key = (r.get("noise") or "bit_flip", r["code_type"], str(r["d"]),
                   float(r["p"]))
            out[key] = {"ew": _f(r, "ew_mean"), "ew_se": _f(r, "ew_se"),
                        "vw": _f(r, "vw"), "p_L": _f(r, "p_L"), "source": src}
    return out


def lookup_reference(refs, code: str, p: float, noise: str,
                     row: Optional[dict] = None) -> Optional[dict]:
    """Reference row for a run CSV entry.

    The 2026-08-28 CSVs carry `code_type` and `d` verbatim from css_code, so the
    reference can be addressed directly; the older ones need the REF_KEYS table.
    """
    if row is not None and row.get("code_type"):
        hit = refs.get((noise, row["code_type"], str(row.get("d", "")), p))
        if hit is not None:
            return hit
    for ct, d in REF_KEYS.get(code, []):
        hit = refs.get((noise, ct, d, p))
        if hit is not None:
            return hit
    return None


def cond_key(row: dict, use_d: bool = True) -> tuple:
    """Condition identity shared by a run CSV and its `_conv` / `_chains` siblings.

    `d` MUST be part of the key for depol_audit_burn.csv, which carries toric d=3
    and d=5 under the same `code` value; a (code, p, init, method) key would
    silently merge two different codes' convergence curves.

    It must NOT be for the lowp_audit job: its `<stem>_chains.csv` has no `d`
    column at all (the code label there is already unique -- toric5 vs toric7 vs
    ut5x5), so keeping `d` in the key would make every chain lookup miss silently
    and drop the whole dataset back to the Rhat fallback. Hence `key_d` per dataset.
    """
    return (row["code"], str(row.get("d", "")) if use_d else "", float(row["p"]),
            row["init"], row["method"])


def load_conv_curve(results_dir: str, stem: str, use_d: bool = True
                    ) -> Dict[tuple, List[Tuple[float, float]]]:
    """`<stem>_conv.csv` -> cond_key -> [(n_samples, pooled running mean)]."""
    path = os.path.join(results_dir, stem + "_conv.csv")
    if not os.path.exists(path):
        return {}
    out: Dict[tuple, List[Tuple[float, float]]] = {}
    for r in _read(path):
        n = _f(r, "n_samples")
        if math.isnan(n):
            n = _f(r, "checkpoint_n")
        out.setdefault(cond_key(r, use_d), []).append((n, _f(r, "ew_running")))
    for v in out.values():
        v.sort()
    return out


def load_chain_checkpoints(results_dir: str, stem: str,
                           value_col: Optional[str] = None,
                           use_d: bool = True
                           ) -> Dict[tuple, Dict[float, List[float]]]:
    """`<stem>_chains.csv` -> (code, p, init, method) -> {n_samples: [chain means]}.

    Per-chain running means at N/16..N, the format the new lowp_audit job writes.
    Absent for the 2026-06/08 production runs, in which case the scaling test is
    skipped and the Rhat fallback decides.
    """
    path = os.path.join(results_dir, stem + "_chains.csv")
    if not os.path.exists(path):
        return {}
    rows = _read(path)
    if not rows:
        return {}
    ncol = next((c for c in CHAIN_N_COLS if c in rows[0]), None)
    vcol = (value_col if value_col
            else next((c for c in CHAIN_VALUE_COLS if c in rows[0]), None))
    if ncol is None or vcol is None:
        return {}
    out: Dict[tuple, Dict[float, List[float]]] = {}
    for r in rows:
        n, val = _f(r, ncol), _f(r, vcol)
        if math.isnan(n) or math.isnan(val):
            continue
        out.setdefault(cond_key(r, use_d), {}).setdefault(n, []).append(val)
    return out


def _sd(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ------------------------------------------------------------------ analysis --


def _vw_hat(row: dict, tau_factor: float) -> float:
    """Var[w|F] implied by a chain's own scatter: s_B^2 * n_used / (tau_factor*tau)."""
    sd = _f(row, "ew_burn5_sd")
    n = _f(row, "n_used5")
    t = _f(row, "tau_burn5")
    if not (t > 0) or not (n > 0):
        return float("nan")
    return sd * sd * n / (tau_factor * t)


def build_entries(results_dir: str = RESULTS, *, tau_factor: float = 1.0,
                  rhat_alpha: float = 1e-3, rhat_max: Optional[float] = None,
                  cap_frac: float = 0.10, n_tau_min: float = 50.0,
                  drift_k: float = 3.0, drift_mode: str = "cum",
                  scale_factor: float = 1.5, min_donor_ntau: float = 500.0,
                  chain_col: Optional[str] = None,
                  scale_ref_frac: float = 0.25, chain_drift_mode: str = "late",
                  ms_k: float = 3.0, chain_rhat_max: float = 3.0,
                  rhat_within: str = "chain",
                  only: Optional[Sequence[str]] = None,
                  ) -> List[dict]:
    refs = load_references(results_dir)
    entries: List[dict] = []
    for ds in DATASETS:
        if only and ds["name"] not in only:
            continue
        path = os.path.join(results_dir, ds["csv"])
        if not os.path.exists(path):
            # Absent files are skipped; say so when the caller asked for this one
            # by name, so a missing cluster result is not silently an empty table.
            if only:
                print("convergence_flags: dataset %r skipped, no such file: %s"
                      % (ds["name"], path), file=sys.stderr)
            continue
        stem = ds["csv"][:-4]
        use_d = ds.get("key_d", True)
        rows = _read(path)
        conv = load_conv_curve(results_dir, stem, use_d)
        chains = load_chain_checkpoints(results_dir, stem, chain_col, use_d)
        # The multi-start test pairs the two initializations of one condition.
        arms: Dict[tuple, Dict[str, dict]] = {}
        for r in rows:
            arms.setdefault((r["code"], str(r.get("d", "")), float(r["p"]),
                             r["method"]), {})[r["init"]] = r

        # ---- Var[w|F] per cell: reference first, else a gated MCMC donor.
        groups: Dict[tuple, List[dict]] = {}
        for r in rows:
            groups.setdefault((r["code"], float(r["p"])), []).append(r)
        vw_of: Dict[tuple, Tuple[float, str, Optional[dict]]] = {}
        for gkey, grows in groups.items():
            code, p = gkey
            ref = lookup_reference(refs, code, p, ds["noise"], grows[0])
            if ref is not None and ref["vw"] > 0:
                vw_of[gkey] = (ref["vw"], ref["source"], None)
                continue
            best = None
            for r in grows:
                if int(_f(r, "capped5", 0)) != 0:
                    continue
                t, n = _f(r, "tau_burn5"), _f(r, "n_used5")
                if not (t > 0) or n / t < min_donor_ntau:
                    continue
                v = _vw_hat(r, tau_factor)
                if math.isnan(v):
                    continue
                if best is None or v < best[0]:
                    best = (v, "mcmc:%s/%s" % (r["method"], r["init"]), r)
            vw_of[gkey] = best if best is not None else (float("nan"), "none", None)

        for r in rows:
            code, p = r["code"], float(r["p"])
            vw, vw_src, donor = vw_of[(code, p)]
            k = int(_f(r, "n_seeds", 0))
            sd_b = _f(r, "ew_burn5_sd")
            n_used = _f(r, "n_used5")
            tau_med = _f(r, "tau_burn5")
            tau_full = _f(r, "tau")
            capped = int(_f(r, "capped5", 0))
            est = _f(r, "ew_burn5")
            est_se = sd_b / math.sqrt(k) if k > 1 else float("nan")

            # ---- within-chain SD, in both forms. `vw` is the MCMC-independent
            # reference (or a gated donor); `vw_chain` is the run's own pooled
            # Gelman-Rubin within-chain variance W, absent from pre-2026-09 CSVs.
            vw_chain = _f(r, "w_var_within_mean")
            scale_ok = tau_med > 0 and n_used > 0
            s_w_ref = (math.sqrt(vw * tau_factor * tau_med / n_used)
                       if (vw > 0 and scale_ok) else float("nan"))
            s_w_chain = (math.sqrt(vw_chain * tau_factor * tau_med / n_used)
                         if (vw_chain > 0 and scale_ok) else float("nan"))
            s_w, within_used = select_within(s_w_chain, s_w_ref, rhat_within)
            if within_used == "chain":
                vw, vw_src = vw_chain, "chain"
            rhat_chain = sd_b / s_w_chain if s_w_chain > 0 else float("nan")
            rhat_ref = sd_b / s_w_ref if s_w_ref > 0 else float("nan")
            cut = rhat_max if rhat_max is not None else rhat_critical(k, rhat_alpha)
            rv = rhat_verdict(sd_b, s_w, cut)
            # A donor's own Rhat is circular only while it is judged against the
            # variance it donated; the chain form is not borrowed from anyone.
            circular = (donor is not None and donor is r
                        and within_used != "chain")

            # ---- drift from the pooled convergence curve
            ckey = cond_key(r, use_d)
            curve = conv.get(ckey, [])
            n_total = curve[-1][0] if curve else float("nan")
            dv: Dict[str, object] = {"drift": float("nan"), "drift_se": float("nan"),
                                     "drift_z": float("nan"), "ok": None}
            n_half = float("nan")
            if len(curve) >= 2:
                n_fin, m_fin = curve[-1]
                n_half, m_half = min(curve[:-1], key=lambda t: abs(t[0] - n_fin / 2.0))
                se_fin = _f(r, "ew_std") / math.sqrt(k) if k > 1 else float("nan")
                # se_fin is the SE of the pooled mean at N. Writing V(n) = se_fin^2*N/n
                # for the pooled mean over the first n steps per chain, the cumulative
                # difference m(N) - m(n_h) = ((N-n_h)/N)(m_late - m_h) has
                # Var = se_fin^2 * (N-n_h)/n_h, and the disjoint-segment difference
                # m_late - m_h has Var = se_fin^2 * N * (1/n_h + 1/(N-n_h)).
                if drift_mode == "seg" and n_fin > n_half:
                    m_late = (n_fin * m_fin - n_half * m_half) / (n_fin - n_half)
                    se = se_fin * math.sqrt(n_fin * (1.0 / n_half
                                                     + 1.0 / (n_fin - n_half)))
                    dv = drift_verdict(m_half, m_late, se, drift_k)
                else:
                    se = se_fin * math.sqrt((n_fin - n_half) / n_half)
                    dv = drift_verdict(m_half, m_fin, se, drift_k)

            # ---- per-chain checkpoint tests (only if *_chains.csv exists)
            nan = float("nan")
            sv: Dict[str, object] = {"n_ref": nan, "sd_ref": nan, "n_final": nan,
                                     "sd_final": nan, "sd_expected": nan,
                                     "ratio": nan, "ratio_first": nan, "ok": None}
            cdv: Dict[str, object] = {"drift": nan, "drift_se": nan,
                                      "drift_z": nan, "ok": None, "n_split": nan}
            mv: Dict[str, object] = {"ms_diff": nan, "ms_se": nan, "ms_z": nan,
                                     "ok": None}
            ms_partner = ""
            crv: Dict[str, object] = {"rhat": nan, "ok": None}
            cps = chains.get(ckey)
            if cps:
                ns = sorted(cps)
                sds = [_sd(cps[n]) for n in ns]
                ms = [sum(cps[n]) / len(cps[n]) for n in ns]
                sv = scaling_verdict(ns, sds, scale_factor, ref_frac=scale_ref_frac)
                n_total = ns[-1]
                se_fin = sds[-1] / math.sqrt(len(cps[ns[-1]]))
                if chain_drift_mode == "cum" and len(ns) >= 2:
                    n_h, m_h = min(zip(ns[:-1], ms[:-1]),
                                   key=lambda t: abs(t[0] - ns[-1] / 2.0))
                    # Var(m(N) - m(n_h)) = se_fin^2 (N - n_h)/n_h for cumulative means
                    cdv = drift_verdict(m_h, ms[-1],
                                        se_fin * math.sqrt((ns[-1] - n_h) / n_h),
                                        drift_k)
                    cdv["n_split"] = n_h
                else:
                    cdv = late_drift_verdict(ns, ms, se_fin, drift_k)
                # multi-start: compare the values the table quotes
                pair = arms.get((code, str(r.get("d", "")), p, r["method"]), {})
                other = next((v for kk, v in pair.items() if kk != r["init"]), None)
                if other is not None:
                    ms_partner = other["init"]
                    k_o = int(_f(other, "n_seeds", 0))
                    mv = multistart_verdict(
                        est, est_se, _f(other, "ew_burn5"),
                        _f(other, "ew_burn5_sd") / math.sqrt(k_o) if k_o > 1 else nan,
                        ms_k)
                crv = rhat_verdict(sd_b, s_w, chain_rhat_max)

            aux_a = (capped / k <= cap_frac) if k else False
            aux_b = (n_used / tau_med >= n_tau_min) if tau_med > 0 else False
            if cps:
                conv_ok, source, reason = classify_primary(
                    scaling_ok=sv["ok"], drift_ok=cdv["ok"], ms_ok=mv["ok"],
                    rhat_ok=crv["ok"])
            else:
                conv_ok, source, reason = classify(
                    scaling_ok=sv["ok"], rhat_ok=rv["ok"], rhat_circular=circular,
                    has_variance_ref=bool(vw > 0), drift_ok=dv["ok"],
                    aux_a_ok=aux_a, aux_b_ok=aux_b)

            # ---- bias against the best available reference.
            # Only the weight-stratified subset-sampling reference supersedes the
            # run's own truth column: direct_mc_ew.csv is an older, coarser pool
            # (e.g. BB[[72,12,6]] at 1e-2: 3.706(18) there vs 3.690(3) in the run),
            # so it is used for `vw` but never as a better E[w|F].
            ref = lookup_reference(refs, code, p, ds["noise"], r)
            if ref is not None and ref["source"] == "subset" and not math.isnan(ref["ew"]):
                truth, truth_se, truth_src = ref["ew"], ref["ew_se"], "subset"
            else:
                truth, truth_se, truth_src = _f(r, "truth_ew"), _f(r, "truth_se"), "run"
            entries.append({
                "dataset": ds["name"], "noise": ds["noise"], "code": code,
                "d": r.get("d", ""), "p": p,
                "init": r["init"], "method": r["method"], "n_chains": k,
                "n_total": n_total, "est": est, "est_se": est_se,
                "truth": truth, "truth_se": truth_se, "truth_source": truth_src,
                "bias": est - truth, "bias_se": combine_se(est_se, truth_se),
                "truth_old": _f(r, "truth_ew"), "truth_old_se": _f(r, "truth_se"),
                "bias_old": est - _f(r, "truth_ew"),
                "bias_old_se": combine_se(est_se, _f(r, "truth_se")),
                "n_capped": capped, "cap_frac": capped / k if k else float("nan"),
                "tau_med": tau_med, "tau_full": tau_full, "n_used": n_used,
                "n_over_tau_med": n_used / tau_med if tau_med > 0 else float("nan"),
                "n_over_tau_full": n_total / tau_full if tau_full > 0 else float("nan"),
                "vw": vw, "vw_source": vw_src, "s_between": sd_b, "s_within": s_w,
                "s_within_chain": s_w_chain, "s_within_ref": s_w_ref,
                "rhat_chain": rhat_chain, "rhat_ref": rhat_ref,
                "rhat_within_used": within_used,
                "rhat": rv["rhat"], "rhat_cut": cut, "rhat_circular": circular,
                "drift": dv["drift"], "drift_se": dv["drift_se"],
                "drift_z": dv["drift_z"], "drift_ok": dv["ok"],
                "drift_split_n": n_half,
                "scale_ratio": sv["ratio"], "scale_ok": sv["ok"],
                "scale_ratio_first": sv["ratio_first"], "scale_n_ref": sv["n_ref"],
                "scale_sd_ref": sv["sd_ref"], "scale_sd_final": sv["sd_final"],
                "cdrift": cdv["drift"], "cdrift_se": cdv["drift_se"],
                "cdrift_z": cdv["drift_z"], "cdrift_ok": cdv["ok"],
                "cdrift_split_n": cdv["n_split"],
                "ms_partner": ms_partner, "ms_diff": mv["ms_diff"],
                "ms_se": mv["ms_se"], "ms_z": mv["ms_z"], "ms_ok": mv["ok"],
                "chain_rhat_ok": crv["ok"], "chain_rhat_cut": chain_rhat_max,
                "spread": _f(r, "lwt_spread"), "n_qubits": _f(r, "n"),
                "k_logical": _f(r, "k"), "h_accept_rate": _f(r, "h_accept_rate"),
                "wall_per_chain": _f(r, "wall_per_chain"),
                "aux_a_ok": aux_a, "aux_b_ok": aux_b,
                "verdict_source": source, "converged": conv_ok, "reason": reason,
            })
    return entries


ALT_CUTS = (1.05, 1.1, 1.2, 1.32, 1.5, 2.0, 3.0)

# (label, scale_factor, drift_k, ms_k, chain_rhat_max) for --matrix on primary rows.
PRIMARY_SETTINGS = (
    ("S1.5 D3 M3 R3.0 (default)", 1.5, 3.0, 3.0, 3.0),
    ("S1.5 D3 M3 R1.32 (chi2)", 1.5, 3.0, 3.0, 1.32),
    ("S1.5 D3 M3 R1.50", 1.5, 3.0, 3.0, 1.50),
    ("S1.5 D3 M3 R2.00", 1.5, 3.0, 3.0, 2.00),
    ("S1.5 D3 M3 R inf (no R)", 1.5, 3.0, 3.0, float("inf")),
    ("S1.3 D3 M3 R3.0", 1.3, 3.0, 3.0, 3.0),
    ("S2.0 D3 M3 R3.0", 2.0, 3.0, 3.0, 3.0),
    ("S1.5 D2 M3 R3.0", 1.5, 2.0, 3.0, 3.0),
    ("S1.5 D3 M inf R3.0 (no M)", 1.5, 3.0, float("inf"), 3.0),
    ("S inf D3 M3 R3.0 (no S)", float("inf"), 3.0, 3.0, 3.0),
)


def _le(x, cut) -> bool:
    """A test passes when it could not be computed (NaN) -- see classify_primary."""
    return math.isnan(x) or x <= cut


def _n(x, dec: int = 1) -> str:
    return "-" if x is None or math.isnan(x) else "%.*f" % (dec, x)


def verdict_matrix(entries: Sequence[dict], cuts: Sequence[float] = ALT_CUTS
                   ) -> List[dict]:
    """How the Rhat-decided verdicts move as the cut moves. Rows not decided by
    Rhat (scaling-decided, circular, or no reference variance) keep their verdict."""
    out = []
    for e in entries:
        row = {k: e[k] for k in ("dataset", "code", "p", "init", "method", "rhat")}
        for c in cuts:
            if e["verdict_source"] == "rhat":
                ok = (not math.isnan(e["rhat"])) and e["rhat"] <= c
                ok = ok and (e["drift_ok"] is not False)
            else:
                ok = e["converged"]
            row["ok@%g" % c] = ok
        out.append(row)
    return out


# ----------------------------------------------------------------------- CLI --


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--tau-factor", type=float, default=1.0,
                    help="multiplier on tau in the within-chain SE; 1.0 matches "
                         "src.metrics.iat (tau_int), 2.0 reproduces the '2*tau' spec")
    ap.add_argument("--rhat-alpha", type=float, default=1e-3,
                    help="false-alarm rate used to derive the Rhat cut from chi^2")
    ap.add_argument("--rhat-max", type=float, default=None,
                    help="fixed Rhat cut, overriding --rhat-alpha")
    ap.add_argument("--cap-frac", type=float, default=0.10,
                    help="auxiliary (a): max fraction of chains lacking a 5tau fixed point")
    ap.add_argument("--n-tau", type=float, default=50.0,
                    help="auxiliary (b): min n_used / tau_med")
    ap.add_argument("--drift-k", type=float, default=3.0)
    ap.add_argument("--drift-mode", choices=("cum", "seg"), default="cum")
    ap.add_argument("--scale-factor", type=float, default=1.5)
    ap.add_argument("--min-donor-ntau", type=float, default=500.0)
    ap.add_argument("--scale-ref-frac", type=float, default=0.25,
                    help="primary (S): reference checkpoint as a fraction of N")
    ap.add_argument("--chain-drift-mode", choices=("late", "cum"), default="late",
                    help="primary (D): 'late' compares the (N/2,N] and (N/4,N/2] "
                         "segments; 'cum' the cumulative m(N) vs m(N/2)")
    ap.add_argument("--ms-k", type=float, default=3.0,
                    help="primary (M): max |short-long| in combined SEs")
    ap.add_argument("--chain-rhat-max", type=float, default=3.0,
                    help="primary (R): loose dispersion cut, see the module docstring")
    ap.add_argument("--rhat-within", choices=("chain", "ref"), default="chain",
                    help="primary (R): which within-chain SD feeds Rhat. 'chain' "
                         "(default) uses the run's own pooled Gelman-Rubin variance "
                         "`w_var_within_mean`, falling back to the reference where "
                         "that column is absent; 'ref' always uses the "
                         "MCMC-independent Var[w|F] (the 2026-08 behaviour)")
    ap.add_argument("--only", default=None,
                    help="comma-separated dataset names to keep, e.g. lowp_audit,lowp_long")
    ap.add_argument("--chain-col", default=None,
                    help="per-chain running-mean column in <stem>_chains.csv "
                         "(default: ew_running_burn10pct, else ew_running)")


def entries_from_args(a) -> List[dict]:
    return build_entries(
        a.results, tau_factor=a.tau_factor, rhat_alpha=a.rhat_alpha,
        rhat_max=a.rhat_max, cap_frac=a.cap_frac, n_tau_min=a.n_tau,
        drift_k=a.drift_k, drift_mode=a.drift_mode, scale_factor=a.scale_factor,
        min_donor_ntau=a.min_donor_ntau, chain_col=a.chain_col,
        scale_ref_frac=a.scale_ref_frac, chain_drift_mode=a.chain_drift_mode,
        ms_k=a.ms_k, chain_rhat_max=a.chain_rhat_max,
        rhat_within=a.rhat_within,
        only=[x for x in (a.only or "").split(",") if x] or None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "convergence_flags.csv"))
    ap.add_argument("--matrix", action="store_true",
                    help="also print the verdict matrix over alternative Rhat cuts")
    a = ap.parse_args(argv)

    entries = entries_from_args(a)
    if not entries:
        print("no input CSVs found under", a.results)
        return 1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(entries[0].keys()))
        w.writeheader()
        w.writerows(entries)

    print("Rhat cut = %.3f (%s) | tau factor %g | drift '%s' |z|<=%g | "
          "aux (a) cap<=%.0f%%, (b) n/tau>=%g"
          % (entries[0]["rhat_cut"],
             "fixed" if a.rhat_max else "chi2 alpha=%g" % a.rhat_alpha,
             a.tau_factor, a.drift_mode, a.drift_k, 100 * a.cap_frac, a.n_tau))
    prim = [e for e in entries if e["verdict_source"] == "primary"]
    if prim:
        print("\nPRIMARY rule (per-chain checkpoints): S scale<=%g @N/%g | "
              "D '%s' |z|<=%g | M |short-long|<=%g SE | R Rhat<=%g (within '%s')"
              % (a.scale_factor, 1.0 / a.scale_ref_frac, a.chain_drift_mode,
                 a.drift_k, a.ms_k, a.chain_rhat_max, a.rhat_within))
        hdr = ("%-11s %-13s %8s %-7s %-5s %6s %6s %6s %6s %7s %7s %5s %9s %8s %6s  %s"
               % ("dataset", "code", "p", "init", "meth", "S", "S16", "D", "M",
                  "Rhat", "Rref", "src", "bias", "bias_se", "n_sig", "verdict"))
        print(hdr)
        print("-" * len(hdr))
        for e in prim:
            nsig = (e["bias"] / e["bias_se"]) if e["bias_se"] > 0 else float("nan")
            print("%-11s %-13s %8.0e %-7s %-5s %6s %6s %6s %6s %7s %7s %5s %9.4f %8.4f %6.1f  %s"
                  % (e["dataset"], e["code"], e["p"], e["init"], e["method"],
                     _n(e["scale_ratio"]), _n(e["scale_ratio_first"]),
                     _n(e["cdrift_z"]), _n(e["ms_z"]), _n(e["rhat"], 2),
                     _n(e["rhat_ref"], 2), e["rhat_within_used"],
                     e["bias"], e["bias_se"], nsig,
                     ("ok   " if e["converged"] else "DAGGER ") + e["reason"]))
    rest = [e for e in entries if e["verdict_source"] != "primary"]
    if rest:
        print("\nFALLBACK rule (no per-chain checkpoints)")
        hdr = ("%-11s %-13s %8s %-7s %-5s %6s %8s %8s %6s %6s %7s %8s %7s  %-16s %s"
               % ("dataset", "code", "p", "init", "meth", "Rhat", "s_B", "s_W",
                  "cap%", "N/tau", "drift_z", "bias", "bias_se", "vw_src", "verdict"))
        print(hdr)
        print("-" * len(hdr))
        for e in rest:
            print("%-11s %-13s %8.0e %-7s %-5s %6s %8.4f %8.4f %5.0f%% %6.0f %7s %8.3f %7.3f  %-16s %s"
                  % (e["dataset"], e["code"], e["p"], e["init"], e["method"],
                     _n(e["rhat"], 2), e["s_between"], e["s_within"],
                     100 * e["cap_frac"], e["n_over_tau_med"], _n(e["drift_z"]),
                     e["bias"], e["bias_se"], e["vw_source"],
                     ("ok   " if e["converged"] else "DAGGER ") + e["reason"]
                     + ("" if e["verdict_source"] == "rhat"
                        else " [%s]" % e["verdict_source"])))
    n_bad = sum(1 for e in entries if not e["converged"])
    print("\n%d/%d entries flagged as not converged" % (n_bad, len(entries)))
    print("written:", a.out)

    if a.matrix and prim:
        print("\nPRIMARY verdict sensitivity (n daggered / n missed biased entries; "
              "'biased' = |bias| > 3 combined SE; ut5x5 duplicate of toric5 dropped)")
        uniq = [e for e in prim if e["code"] != "ut5x5"]
        print("%-28s %7s %7s %7s" % ("setting", "dagger", "missed", "false+"))
        for label, sf, dk, mk, rc in PRIMARY_SETTINGS:
            n_d = n_m = n_f = 0
            for e in uniq:
                ok = (_le(e["scale_ratio"], sf) and _le(e["cdrift_z"], dk)
                      and _le(e["ms_z"], mk) and _le(e["rhat"], rc))
                biased = abs(e["bias"]) > 3 * e["bias_se"] if e["bias_se"] > 0 else False
                n_d += (not ok)
                n_m += (ok and biased)
                n_f += ((not ok) and not biased)
            print("%-28s %7d %7d %7d" % (label, n_d, n_m, n_f))

    if a.matrix:
        print("\nverdict matrix over Rhat cuts")
        mat = verdict_matrix(entries)
        cols = [c for c in mat[0] if c.startswith("ok@")]
        print("%-9s %-13s %8s %-7s %-5s %7s  %s"
              % ("dataset", "code", "p", "init", "meth", "Rhat",
                 " ".join(c[3:].rjust(6) for c in cols)))
        for m in mat:
            print("%-9s %-13s %8.0e %-7s %-5s %7s  %s"
                  % (m["dataset"], m["code"], m["p"], m["init"], m["method"],
                     "-" if math.isnan(m["rhat"]) else "%.2f" % m["rhat"],
                     " ".join(("ok" if m[c] else "DAG").rjust(6) for c in cols)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
