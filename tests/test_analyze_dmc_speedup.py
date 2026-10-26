"""Reference-tier selection in compare/analyze_dmc_speedup.py.

Covers the ``truth_for`` tier ladder (weight-stratified -> direct-MC cutoff ->
direct-MC grid -> Z+H itself) and the reference-uncertainty term that the
accuracy test folds into its RMSE.  The module lives in ``compare/``, which is
not a package, so it is loaded from its file path.
"""
import importlib.util
import math
import pathlib

import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[1] / "compare"
         / "analyze_dmc_speedup.py")
_spec = importlib.util.spec_from_file_location("analyze_dmc_speedup_mod", _PATH)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)          # must NOT run main(): guarded by __main__


def _cut_row(code, p, n_faulty, s1, s2, n_trials=5000):
    return {"phase": "dmc_cutoff", "code": code, "p": str(p), "method": "directmc",
            "n_trials": str(n_trials), "n_faulty": str(n_faulty), "s1": str(s1),
            "s2": str(s2), "wall": "10", "n_cores": "96"}


def _adv_row(code, p, method, length, ew, wall):
    return {"phase": "advantage", "code": code, "p": str(p), "method": method,
            "length": str(length), "ew": str(ew), "wall": str(wall)}


def test_import_does_not_run_main():
    # exec_module above already proves it; assert the guard is really there
    assert 'if __name__ == "__main__":' in _PATH.read_text(encoding="utf-8")


# ── tier ladder ────────────────────────────────────────────────────────────
def test_truth_for_stratified_wins_over_a_healthy_cutoff():
    # the stratified entry outranks the cutoff even when the cutoff has plenty of
    # faulty draws: the two agree, and the stratified SE is the far smaller one
    rows = [_cut_row("bb72", 0.1, n_faulty=500, s1=1750.0, s2=6200.0)]
    strat = {("bb72", "0", 0.1): dict(ew=3.4999, vw=0.9, ew_se=0.002)}
    ew, vw, src, se = A.truth_for(rows, "bb72", 0.1, dmc_grid={}, strat=strat)
    assert src == "stratified"
    assert math.isclose(ew, 3.4999, rel_tol=1e-12)
    assert math.isclose(vw, 0.9, rel_tol=1e-12)
    assert math.isclose(se, 0.002, rel_tol=1e-12)


def test_truth_for_cutoff_when_no_stratified_entry():
    # no stratified row for this cell -> the cutoff run with n_faulty >=
    # MIN_FAULTY_FOR_TRUTH is the reference
    rows = [_cut_row("bb72", 0.1, n_faulty=500, s1=1750.0, s2=6200.0)]
    ew, vw, src, se = A.truth_for(rows, "bb72", 0.1, dmc_grid={}, strat={})
    assert src == "directmc_cutoff"
    assert math.isclose(ew, 1750.0 / 500, rel_tol=1e-9)
    # truth_se of a cutoff reference is the SE of its own mean, sqrt(vw/n_faulty)
    assert math.isclose(se, math.sqrt(vw / 500), rel_tol=1e-9)


def test_truth_for_stratified_below_min_faulty():
    # cutoff saw < 30 faulty -> the weight-stratified reference carries the cell,
    # with its REAL vw (so N* is a real number, unlike the zh tier) and its ew_se
    rows = [_cut_row("bb144", 1e-5, n_faulty=3, s1=9.0, s2=27.0)]
    strat = {("bb144", "0", 1e-5): dict(ew=3.0005, vw=0.0501, ew_se=1.3e-3)}
    ew, vw, src, se = A.truth_for(rows, "bb144", 1e-5, dmc_grid={}, strat=strat)
    assert src == "stratified"
    assert math.isclose(ew, 3.0005, rel_tol=1e-12)
    assert math.isclose(vw, 0.0501, rel_tol=1e-12)
    assert math.isclose(se, 1.3e-3, rel_tol=1e-12)


def test_truth_for_stratified_matches_p_with_tolerance():
    # p comes from a CSV round-trip on both sides; match must not be exact-float
    rows = [_cut_row("rect:3x11", 1e-5, n_faulty=0, s1=0.0, s2=0.0)]
    strat = {("asymmetric_toric", "3x11", 1.0000000000000002e-05):
             dict(ew=2.0001, vw=0.0002, ew_se=1e-9)}
    ew, vw, src, se = A.truth_for(rows, "rect:3x11", 1e-5, dmc_grid={}, strat=strat)
    assert src == "stratified"
    assert math.isclose(ew, 2.0001, rel_tol=1e-12)


def test_truth_for_falls_to_grid_when_no_stratified_entry():
    rows = [_cut_row("bb144", 1e-3, n_faulty=5, s1=15.0, s2=46.0)]
    grid = {("bb144", 1e-3): (3.07, 0.089, 0.004)}
    ew, vw, src, se = A.truth_for(rows, "bb144", 1e-3, dmc_grid=grid, strat={})
    assert src == "directmc_grid"
    assert math.isclose(ew, 3.07, rel_tol=1e-12)
    assert math.isclose(vw, 0.089, rel_tol=1e-12)
    assert math.isclose(se, 0.004, rel_tol=1e-12)


def test_truth_for_falls_to_zh_when_nothing_else():
    rows = [_cut_row("surface:3", 1e-5, n_faulty=2, s1=4.0, s2=8.0),
            _adv_row("surface:3", 1e-5, "Z+H", 512000, 5.0, 1.0),
            _adv_row("surface:3", 1e-5, "Z+H", 512000, 5.2, 1.0)]
    ew, vw, src, se = A.truth_for(rows, "surface:3", 1e-5, dmc_grid={}, strat={})
    assert src == "zh"
    assert math.isclose(ew, 5.1, rel_tol=1e-9)
    assert math.isnan(vw)
    assert se == 0.0


def test_truth_for_ignores_stratified_of_another_code():
    # the STRAT_CODE mapping must be respected: hgp_hamming -> hgp_hamming3 only
    rows = [_cut_row("hgp_hamming", 1e-5, n_faulty=1, s1=2.0, s2=4.0),
            _adv_row("hgp_hamming", 1e-5, "Z+H", 1000, 2.0, 1.0),
            _adv_row("hgp_hamming", 1e-5, "Z+H", 1000, 2.0, 1.0)]
    strat = {("bb144", "0", 1e-5): dict(ew=3.0, vw=0.05, ew_se=1e-3)}
    _, _, src, _ = A.truth_for(rows, "hgp_hamming", 1e-5, dmc_grid={}, strat=strat)
    assert src == "zh"


def test_strat_code_maps_every_table_row():
    assert set(A.STRAT_CODE) == set(A.ROW_ORDER)


def test_row_order_is_the_twelve_paper_codes():
    assert A.ROW_ORDER == ["surface:3", "surface:5", "surface:7", "toric:5", "toric:7",
                           "concat_steane:2", "hgp_hamming", "rect:3x11", "rect:3x15",
                           "bb72", "bb144", "bb_l9m6"]
    assert set(A.STRAT_CODE) == set(A.ROW_ORDER) == set(A.ROW_LABEL)
    for family, d in (("surface", "5"), ("surface", "7"), ("toric", "5"), ("toric", "7")):
        key = f"{family}:{d}"
        assert A.STRAT_CODE[key] == (family, d)
        assert A.ROW_LABEL[key] == f"{family} $d={d}$"


def test_stratified_has_1e5_rows_for_topological_codes():
    strat = A.load_stratified(A.STRAT)
    for ct, d in (("toric", "5"), ("toric", "7"), ("surface", "5"), ("surface", "7")):
        assert (ct, d, 1e-5) in strat


# ── stratified loader ──────────────────────────────────────────────────────
def test_load_stratified_reads_real_csv():
    strat = A.load_stratified(A.STRAT)
    assert ("bb144", "0", 1e-3) in strat
    e = strat[("bb144", "0", 1e-3)]
    assert e["ew"] > 0 and e["vw"] >= 0 and e["ew_se"] >= 0


def test_load_stratified_missing_file_is_empty():
    assert A.load_stratified("no/such/file.csv") == {}


# ── reference uncertainty in the accuracy test ─────────────────────────────
def test_wall_to_precision_ignores_the_reference_se():
    # deviation 0 with a chain SE of 1.5%: passes whatever the reference SE
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 3.0 - 0.045, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.0 + 0.045, 1.0)]
    for tse in (0.0, 0.06):
        wall, reached = A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", truth=3.0,
                                            truth_se=tse)
        assert reached is True
        assert math.isclose(wall, 2.0, rel_tol=1e-12)


def test_wall_to_precision_requires_the_chain_se_below_target():
    # deviation 0 but a chain SE of 2.5%: fails, as direct MC would at that SE
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 3.0 - 0.075, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.0 + 0.075, 1.0)]
    wall, reached = A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", truth=3.0)
    assert reached is False


def test_wall_to_precision_truth_se_zero_is_the_old_behaviour():
    rows = [_adv_row("bb144", 1e-3, "Z+H", 1000, 2.99, 1.0),
            _adv_row("bb144", 1e-3, "Z+H", 1000, 3.01, 1.0)]
    assert (A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", 3.0)
            == A.wall_to_precision(rows, "bb144", 1e-3, "Z+H", 3.0, truth_se=0.0))


# ── direct-MC budget floor ─────────────────────────────────────────────────
# ew=1, vw=0.04 -> N* = vw/(ew^2 releps^2) = 0.04/0.0004 = 100; C = 96*10 = 960
_EW, _VW, _WALL, _NC = 1.0, 0.04, 10.0, 96


def test_n_target_fixture_is_100():
    assert math.isclose(A.n_target(_EW, _VW), 100.0, rel_tol=1e-12)


def test_dmc_time_converged_is_the_measurement():
    t, hard, conv = A.dmc_time_lower_bound(200, _WALL, _NC, _EW, _VW)
    assert math.isclose(t, 100.0 * 960.0 / 200, rel_tol=1e-12)
    assert conv is True and hard is False


def test_dmc_time_partial_is_the_spent_budget():
    t, hard, conv = A.dmc_time_lower_bound(50, _WALL, _NC, _EW, _VW)
    assert t == _NC * _WALL
    assert hard is False and conv is False


def test_dmc_time_hard_is_nstar_budgets():
    t, hard, conv = A.dmc_time_lower_bound(0, _WALL, _NC, _EW, _VW)
    assert math.isclose(t, 100.0 * 960.0, rel_tol=1e-12)
    assert hard is True and conv is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_val_unc_pdg_rule():
    assert A._val_unc(0.4912, 0.052) == "0.49(5)"
    assert A._val_unc(3.62, 0.16) == "3.62(16)"
    assert A._val_unc(0.68, 0.098) == "0.68(10)"
    assert A._val_unc(631.8, 82.6) == "630(80)"
    assert A._val_unc(207.3, 1.2) == "207.3(12)"
    assert A._val_unc(70.4, 10.2) == "70(10)"


def test_fmt_lower_bound_rounds_down():
    assert A._fmt(595.26, True) == r"$\geq 590$"
    assert A._fmt(11672.6, True) == r"$\geq 1.1\times10^{4}$"
    assert A._fmt(8.87, True) == r"$\geq 8.8$"
    assert A._fmt(24.9, True) == r"$\geq 24$"
    assert A._fmt(595.26, False) == r"$600$"                 # a measured value rounds to nearest


def test_mean_se_bootstrap_over_runs():
    assert A.mean_se([2.0]) == 0.0
    assert math.isnan(A.mean_se([float("nan")]))
    v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    se = A.mean_se(v + [float("nan")])
    assert abs(se - 2.8723 / math.sqrt(10)) < 0.03      # ~ population SD / sqrt(n)


def test_combine_runs_carries_time_se():
    base = dict(code="surface:3", p=0.1, truth_source="s", ew=2.0, vw=0.1, truth_se=0.0,
                p_L=1e-3, n_faulty=1, t_obf=2.0, dmc_lb=False, dmc_hard=False,
                obf_lb=False, zh_reached=True)
    tabs = [[dict(base, t_dmc=10.0 + i, t_zh=1.0 + 0.1 * i)] for i in range(5)]
    (row,) = A.combine_runs(tabs)
    assert row["t_zh_se"] > 0 and row["t_dmc_se"] > 0
    assert abs(row["t_dmc_se"] - 10 * row["t_zh_se"]) < 1e-9 * row["t_dmc_se"] + 0.05
