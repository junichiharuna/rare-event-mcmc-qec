import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "compare"))

import analyze_dmc_speedup as A
from cluster_scripts import dmc_speedup as DS


def test_difficulty_orders_topological_codes_between_hgp_and_steane():
    D = DS.DIFFICULTY
    assert (D.index("hgp_hamming") < D.index("toric:7") < D.index("surface:7")
            < D.index("toric:5") < D.index("surface:5")
            < D.index("concat_steane:2") < D.index("surface:3"))


def test_group_paths_by_run_folds_topo_suffix(tmp_path):
    paths = [tmp_path / "dmc_speedup_run7.csv", tmp_path / "dmc_speedup_run7_bb108.csv",
             tmp_path / "dmc_speedup_run7_topo.csv"]
    for path in paths:
        path.write_text("")
    assert A.group_paths_by_run([str(p) for p in paths]) == [[str(p) for p in paths]]


def test_n_target_basic():
    # relSE = sqrt(vw/N)/ew = 0.02  ->  N = vw / (ew^2 * 0.02^2)
    n = A.n_target(ew=3.0, vw=0.06, releps=0.02)
    assert math.isclose(n, 0.06 / (9.0 * 0.0004), rel_tol=1e-12)


def test_n_target_scales_with_precision():
    # tighter precision (smaller releps) needs quadratically more faulty samples
    assert math.isclose(A.n_target(3.0, 0.06, 0.01) / A.n_target(3.0, 0.06, 0.02), 4.0,
                        rel_tol=1e-12)


def test_dmc_lower_bound_measured():
    # single-core time to N* = N* * (n_cores*wall) / n_faulty; the cutoff run
    # collected n_faulty >= N*, so this is a measurement (converged), not a bound
    t, hard, converged = A.dmc_time_lower_bound(n_faulty=100, wall=300.0, n_cores=96,
                                                ew=3.0, vw=0.06, releps=0.02)
    nstar = A.n_target(3.0, 0.06, 0.02)
    assert nstar <= 100
    assert math.isclose(t, nstar * 96 * 300.0 / 100, rel_tol=1e-12)
    assert hard is False
    assert converged is True


def test_dmc_lower_bound_charged_budget_when_nfaulty_below_nstar():
    # 0 < n_faulty < N* -> the run did not itself certify 2%; charged the budget
    # C = n_cores*wall it spent (a strict floor), flagged not-converged
    t, hard, converged = A.dmc_time_lower_bound(n_faulty=10, wall=300.0, n_cores=96,
                                                ew=3.0, vw=0.6, releps=0.02)
    assert t == 96 * 300.0
    assert hard is False
    assert converged is False


def test_dmc_lower_bound_zero_faulty_uses_one():
    # n_faulty=0 -> substitute 1 (stronger floor), flagged hard, never converged
    t, hard, converged = A.dmc_time_lower_bound(n_faulty=0, wall=300.0, n_cores=96,
                                                ew=3.0, vw=0.06, releps=0.02)
    nstar = A.n_target(3.0, 0.06, 0.02)
    assert math.isclose(t, nstar * 96 * 300.0 / 1, rel_tol=1e-12)
    assert hard is True
    assert converged is False


def _adv_row(code, p, method, length, ew, wall):
    return {"phase": "advantage", "code": code, "p": str(p), "method": method,
            "length": str(length), "ew": str(ew), "wall": str(wall)}


def test_wall_to_precision_picks_first_within_tol():
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 4.0, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.8, 1.0),   # large RMSE -> no
            _adv_row("bb144", 1e-3, "Z+H", 2000, 2.98, 2.5),
            _adv_row("bb144", 1e-3, "Z+H", 2000, 3.02, 2.5),  # sustained RMSE <2%
            _adv_row("bb144", 1e-3, "Z+H", 4000, 2.99, 5.0),
            _adv_row("bb144", 1e-3, "Z+H", 4000, 3.01, 5.0)]
    wall, reached = A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", truth=3.0)
    assert reached is True
    # cost = core-seconds of ALL seeds at that checkpoint (2 x 2.5), not one chain's
    assert math.isclose(wall, 5.0, rel_tol=1e-12)


def test_wall_to_precision_charges_all_chains():
    # Precision comes from the seed-pooled mean, so the cost charged is the sum of
    # the per-chain walls. Unequal walls make sum != any multiple of a single chain.
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 2.99, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.01, 4.0)]
    wall, reached = A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", truth=3.0)
    assert reached is True
    assert math.isclose(wall, 5.0, rel_tol=1e-12)


def test_wall_to_precision_unreached_returns_max():
    rows = [_adv_row("rect:3x11", 1e-3, "OBF", 1000, 4.5, 1.0),
            _adv_row("rect:3x11", 1e-3, "OBF", 1000, 4.4, 1.0),
            _adv_row("rect:3x11", 1e-3, "OBF", 2000, 4.4, 3.0),
            _adv_row("rect:3x11", 1e-3, "OBF", 2000, 4.3, 3.0)]  # never within 2%
    wall, reached = A.wall_to_precision(rows, "rect:3x11", 1e-3, "OBF", truth=2.07)
    assert reached is False
    assert math.isclose(wall, 6.0, rel_tol=1e-12)  # largest-grid cost, 2 x 3.0 (floor)


def test_wall_to_precision_rejects_transient_crossing():
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 2.99, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.01, 1.0),  # transient pass
            _adv_row("bb144", 1e-3, "Z+H", 2000, 3.20, 2.0),
            _adv_row("bb144", 1e-3, "Z+H", 2000, 3.18, 2.0),  # later failure
            _adv_row("bb144", 1e-3, "Z+H", 4000, 2.99, 4.0),
            _adv_row("bb144", 1e-3, "Z+H", 4000, 3.01, 4.0)]
    wall, reached = A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", truth=3.0)
    assert reached is True
    assert math.isclose(wall, 8.0, rel_tol=1e-12)  # 2 x 4.0


def test_wall_to_precision_takes_the_max_not_the_quadrature():
    # At L=1000: ews {3.00, 3.10} -> deviation 0.05 (1.67% of truth) and SE 0.05.
    # The old sqrt(bias^2 + SE^2) = 2.36% rejected it, double counting the variance
    # already inside the realized deviation; max(deviation, SE)/truth < 2% now passes.
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 3.00, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.10, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 2000, 2.99, 2.0),
            _adv_row("bb144", 1e-3, "Z+H", 2000, 3.01, 2.0)]
    wall, reached = A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", truth=3.0)
    assert reached is True
    assert math.isclose(wall, 2.0, rel_tol=1e-12)  # 2 x 1.0, the first checkpoint


def test_speedup_flags_lower_bound_when_obf_unreached():
    su, is_lb = A.speedup(t_ref=100.0, t_zh=2.0, ref_reached=False)
    assert math.isclose(su, 50.0, rel_tol=1e-12)
    assert is_lb is True


def test_render_latex_has_tokens():
    table = [
        {"code": "surface:3", "p": 1e-5, "su_dmc": 1.2e6, "dmc_lb": True,
         "su_obf": 1.0, "obf_lb": False, "dmc_hard": True, "zh_reached": True,
         "truth_source": "stratified"},
        {"code": "bb144", "p": 1e-3, "su_dmc": 400.0, "dmc_lb": False,
         "su_obf": 8.0, "obf_lb": True, "dmc_hard": False, "zh_reached": False,
         "truth_source": "stratified"},
    ]
    tex = A.render_latex(table, n_runs=1, n_seeds=50)
    assert "tab:dmcspeedup" in tex
    assert A.ROW_LABEL["surface:3"] in tex and A.ROW_LABEL["bb144"] in tex
    assert r"\geq" in tex                       # lower-bound marker present
    assert "10^{-5}" in tex                     # p header in exponent notation
    assert r"$\geq 1.2\times10^{6}$" in tex     # sci notation, >= alone, no letter
    assert r"$400{}^{\mathrm{c}}$" in tex       # Z+H-capped mark on the dMC cell
    assert r"$8.0{}^{\mathrm{c}}$" in tex       # OBF-unreached prints no letter; (c) stays
    assert r"\geq 8.0" not in tex               # Z+H capped -> no bound claim either way
    assert "did not reach the $2\\%$ target within its budget" in tex   # dmc caption says what >= means
    assert "did not reach the $2\\%$ target within the largest checkpoint" in tex   # obf caption
    assert "neither an upper nor a lower bound" in tex
    assert "eq:tta" in tex                      # caption defers the definition to the setup
    assert "sec:bench2" in tex                  # run count and reference live in the setup
    assert "(a)" not in tex and "(b)" not in tex and "(d)" not in tex   # >= alone marks a lower bound


def test_lower_bounds_print_no_letter_only_the_capped_case_does():
    # 2026-10-15: >= alone marks a lower bound; only the Z+H-capped case keeps (c).
    few = {"dmc_hard": False, "dmc_lb": True, "obf_lb": True, "zh_reached": True}
    none = {"dmc_hard": True, "dmc_lb": True, "obf_lb": False, "zh_reached": True}
    capped = {"dmc_hard": False, "dmc_lb": False, "obf_lb": True, "zh_reached": False}
    assert A._cell_marks(few) == ("", "")
    assert A._cell_marks(none) == ("", "")
    assert A._cell_marks(capped) == ("c", "c")
    assert set(A._MARK_TEXT) == {"c"}


def test_ratio_of_means_is_not_mean_of_ratios():
    # two runs: reference times 100 and 100, Z+H times 1 and 3
    # mean of ratios = (100 + 33.3)/2 = 66.7; ratio of means = 100/2 = 50
    su, se = A.ratio_of_means([100.0, 100.0], [1.0, 3.0])
    assert math.isclose(su, 50.0, rel_tol=1e-12)
    assert se > 0
    # non-finite pairs are dropped from both means; a single run has se 0
    su1, se1 = A.ratio_of_means([100.0, float("nan")], [1.0, 3.0])
    assert math.isclose(su1, 100.0, rel_tol=1e-12) and se1 == 0.0


def test_combine_runs_uses_ratio_of_means_and_floors_any():
    def row(t_dmc, t_obf, t_zh, dmc_lb=False, obf_lb=False, zh_ok=True):
        return {"code": "bb144", "p": 1e-3, "truth_source": "stratified",
                "ew": 3.0, "vw": 0.1, "truth_se": 0.0, "p_L": 1e-6, "n_faulty": 10,
                "t_dmc": t_dmc, "t_obf": t_obf, "t_zh": t_zh,
                "su_dmc": t_dmc / t_zh, "dmc_lb": dmc_lb, "dmc_hard": False,
                "su_obf": t_obf / t_zh, "obf_lb": obf_lb, "zh_reached": zh_ok}
    runs = [[row(100.0, 10.0, 1.0)], [row(100.0, 10.0, 3.0, obf_lb=True)]]
    (c,) = A.combine_runs(runs)
    assert c["n_runs"] == 2
    assert math.isclose(c["su_dmc"], 50.0, rel_tol=1e-12)   # not 66.7
    assert math.isclose(c["su_obf"], 5.0, rel_tol=1e-12)
    assert c["obf_lb"] is True and c["dmc_lb"] is False      # floor if any run is one
    assert c["su_dmc_se"] > 0 and c["su_obf_se"] > 0


def test_truth_for_prefers_cutoff_then_grid_then_zh():
    # cutoff has >=30 faulty -> use its ew
    rows = [{"phase": "dmc_cutoff", "code": "bb72", "p": "0.1", "method": "directmc",
             "n_trials": "5000", "n_faulty": "500", "s1": "1750.0", "s2": "6200.0",
             "wall": "10", "n_cores": "96"}]
    ew, vw, src, tse = A.truth_for(rows, "bb72", 0.1, dmc_grid={}, strat={})
    assert src == "directmc_cutoff"
    assert math.isclose(ew, 1750.0 / 500, rel_tol=1e-9)


def test_dmc_lower_bound_floors_nstar_at_one_when_vw_missing():
    # vw NaN (no reliable V[w|F], e.g. Z+H-fallback deep-low-p cell) -> N* floors to 1,
    # the honest single-faulty bound, NOT a tiny number from a bogus variance.
    t, hard, converged = A.dmc_time_lower_bound(n_faulty=10, wall=100.0, n_cores=8,
                                                ew=3.0, vw=float("nan"))
    assert math.isclose(t, 1.0 * 8 * 100.0 / 10, rel_tol=1e-12)
    assert hard is False
    assert converged is False       # floored N* is an assumption, never a measurement


def test_truth_for_zh_branch_returns_nan_vw():
    # no cutoff, not in grid -> Z+H mean for the center, vw=NaN (not cross-seed variance)
    rows = [{"phase": "advantage", "code": "surface:3", "p": "1e-05", "method": "Z+H",
             "length": "512000", "ew": "5.0", "wall": "1.0"},
            {"phase": "advantage", "code": "surface:3", "p": "1e-05", "method": "Z+H",
             "length": "512000", "ew": "5.2", "wall": "1.0"}]
    ew, vw, src, tse = A.truth_for(rows, "surface:3", 1e-5, dmc_grid={}, strat={})
    assert src == "zh"
    assert math.isclose(ew, 5.1, rel_tol=1e-9)
    assert math.isnan(vw)


def test_row_order_places_bb108_after_bb144():
    # the paper's LDPC group is ordered by basis-weight spread, as in tab:meanw:
    # bb72 (8) -> bb144 (12) -> BB[[108,8,10]] (18)
    order = A.ROW_ORDER
    assert order.index("bb72") < order.index("bb144") < order.index("bb_l9m6")
    assert A.ROW_LABEL["bb_l9m6"] == r"BB $[[108,8,10]]$"


def test_group_paths_by_run_merges_per_code_topup_files():
    # a per-code top-up file shares run 4's seed stream -> ONE run, not two
    groups = A.group_paths_by_run([
        "cluster_results/dmc_speedup_run4.csv",
        "cluster_results/dmc_speedup_run5.csv",
        "cluster_results/dmc_speedup_run4_bb108.csv",
        "cluster_results/dmc_speedup_run5_bb108.csv"])
    assert groups == [["cluster_results/dmc_speedup_run4.csv",
                       "cluster_results/dmc_speedup_run4_bb108.csv"],
                      ["cluster_results/dmc_speedup_run5.csv",
                       "cluster_results/dmc_speedup_run5_bb108.csv"]]


def test_group_paths_by_run_keeps_unnumbered_files_separate():
    # legacy dmc_speedup.csv carries no run id -> its own group, one per path
    groups = A.group_paths_by_run(["cluster_results/dmc_speedup.csv",
                                   "cluster_results/other.csv",
                                   "cluster_results/dmc_speedup_run4.csv"])
    assert [len(g) for g in groups] == [1, 1, 1]


def test_build_table_omits_bb108_when_absent():
    # ROW_ORDER carries bb_l9m6, but a 7-code run must yield no bb_l9m6 row
    rows = [{"phase": "dmc_cutoff", "code": "bb72", "p": "0.1", "method": "directmc",
             "n_trials": "5000", "n_faulty": "500", "s1": "1750.0", "s2": "6200.0",
             "wall": "10", "n_cores": "96"},
            _adv_row("bb72", 0.1, "Z+H", 1000, 3.5, 1.0),
            _adv_row("bb72", 0.1, "Z+H", 1000, 3.5, 1.0),
            _adv_row("bb72", 0.1, "OBF", 1000, 3.5, 1.0),
            _adv_row("bb72", 0.1, "OBF", 1000, 3.5, 1.0)]
    table = A.build_table(rows, dmc_grid={})
    assert [t["code"] for t in table] == ["bb72"]


def test_wall_to_precision_strict_final_checkpoint_only_is_a_lower_bound():
    # met only at the longest length: nothing follows, so it is not "sustained" by evidence
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 4.0, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.8, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 2000, 2.99, 2.5),
            _adv_row("bb144", 1e-3, "Z+H", 2000, 3.01, 2.5)]
    assert A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", truth=3.0) == (5.0, True)
    wall, reached = A.wall_to_precision_strict(rows, "bb144", 1e-3, "Z+H", truth=3.0)
    assert reached is False and math.isclose(wall, 5.0, rel_tol=1e-12)   # final-checkpoint cost
    # one more sustained checkpoint after the first crossing: reached, same cost
    rows += [_adv_row("bb144", 1e-3, "Z+H", 4000, 2.99, 5.0),
             _adv_row("bb144", 1e-3, "Z+H", 4000, 3.01, 5.0)]
    assert A.wall_to_precision_strict(rows, "bb144", 1e-3, "Z+H", truth=3.0) == (5.0, True)
    # the longest length is the cell's own: an OBF cell extended further does not mask it
    rows_zh = rows[:4]
    rows_obf = [_adv_row("bb144", 1e-3, "OBF", L, ew, 1.0) for L in (1000, 2000, 4000, 8000)
                for ew in (2.99, 3.01)]
    assert A.wall_to_precision_strict(rows_zh + rows_obf, "bb144", 1e-3, "Z+H", truth=3.0)[1] is False
    assert A.wall_to_precision_strict(rows_zh + rows_obf, "bb144", 1e-3, "OBF", truth=3.0)[1] is True
