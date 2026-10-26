# tests/test_analyze_kernel_param_tta.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compare"))
import analyze_kernel_param_tta as A  # noqa: E402


def _rows(ews_by_len, pc=0.3, k=2, code="bb144", p=1e-3):
    out = []
    for L, ews in ews_by_len.items():
        for s, ew in enumerate(ews):
            out.append(dict(code=code, group="trap", p=repr(p), p_cluster=repr(pc), cluster_k=str(k),
                            init="long", seed=str(7000 + s), length=str(L), ew=repr(ew),
                            wall=repr(0.001 * L), cpu=repr(0.001 * L)))
    return out


def test_arm_time_is_sustained_and_averaged_over_runs():
    truth = 3.0
    run1 = _rows({1000: [2.0, 4.0, 2.5], 2000: [3.0, 3.01, 2.99], 4000: [3.0, 3.0, 3.0]})
    run2 = _rows({1000: [3.0, 3.0, 3.0], 2000: [2.0, 4.0, 2.5], 4000: [3.0, 3.0, 3.0],
                  8000: [3.0, 3.0, 3.0]})
    t, se, reached, n_runs = A.arm_times([run1, run2], "bb144", 1e-3, 0.3, 2, truth, 0.0)
    # run1 sustains from L=2000 (cost 3 chains x 2 s), run2 only from L=4000 (3 x 4 s)
    assert abs(t - (6.0 + 12.0) / 2) < 1e-9 and se > 0 and reached is True and n_runs == 2


def test_target_first_met_at_the_final_checkpoint_is_a_lower_bound():
    # the last length has no successor: a crossing there is not "sustained" by any evidence
    rows = _rows({1000: [2.0, 4.0, 2.5], 2000: [2.0, 4.0, 2.5], 4000: [3.0, 3.0, 3.0]})
    arm = A._arm_rows(rows, 0.3, 2)
    t, ok = A.wall_to_precision_strict(arm, "bb144", 1e-3, "arm", 3.0, truth_se=0.0)
    assert ok is False and abs(t - 12.0) < 1e-9          # cost of the final checkpoint
    t, se, reached, n_runs = A.arm_times([rows], "bb144", 1e-3, 0.3, 2, 3.0, 0.0)
    assert reached is False and abs(t - 12.0) < 1e-9
    # one earlier sustained checkpoint is enough, and T is unchanged by the stricter rule
    rows = _rows({1000: [2.0, 4.0, 2.5], 2000: [3.0, 3.0, 3.0], 4000: [3.0, 3.0, 3.0]})
    t, ok = A.wall_to_precision_strict(A._arm_rows(rows, 0.3, 2), "bb144", 1e-3, "arm", 3.0, truth_se=0.0)
    assert ok is True and abs(t - 6.0) < 1e-9
    # a single-length grid can never certify anything
    rows = _rows({1000: [3.0, 3.0, 3.0]})
    assert A.wall_to_precision_strict(A._arm_rows(rows, 0.3, 2), "bb144", 1e-3, "arm", 3.0, truth_se=0.0)[1] is False
    # the longest length of OTHER arms in the same run must not mask this arm's final checkpoint
    rows = _rows({1000: [2.0, 4.0, 2.5], 2000: [3.0, 3.0, 3.0]})            # (0.3, 2): met only at its last L
    rows += _rows({1000: [3.0, 3.0, 3.0], 2000: [3.0, 3.0, 3.0], 4000: [3.0, 3.0, 3.0]}, k=3)
    t, se, reached, n_runs = A.arm_times([rows], "bb144", 1e-3, 0.3, 2, 3.0, 0.0)
    assert reached is False and abs(t - 6.0) < 1e-9


def test_n_runs_counts_runs_with_a_finite_time():
    strat = {("bb144", "0", 1e-3): dict(ew=3.0, vw=0.1, ew_se=0.0)}
    run1 = _rows({1000: [3.0, 3.0, 3.0]})
    run2 = _rows({1000: [3.0, 3.0, 3.0]}, k=3)           # no (0.3, 2) rows in this run
    table = A.build_table_from_rows([run1, run2], strat)
    by = {t["cluster_k"]: t for t in table}
    assert by[2]["n_runs"] == 1 and by[3]["n_runs"] == 1


def test_unreached_arm_is_a_lower_bound():
    run = _rows({1000: [2.0, 4.0, 2.5], 2000: [2.0, 4.0, 2.5]})
    t, se, reached, n_runs = A.arm_times([run], "bb144", 1e-3, 0.3, 2, 3.0, 0.0)
    assert reached is False and abs(t - 6.0) < 1e-9 and n_runs == 1


def test_table_rows_and_tex_ratios():
    strat = {("bb144", "0", 1e-3): dict(ew=3.0, vw=0.1, ew_se=0.0)}
    runs = []
    for _ in range(2):
        rows = []
        for (pc, k), ews in {(0.3, 2): [3.0, 3.0, 3.0], (0.3, 3): [2.0, 4.0, 2.5]}.items():
            rows += _rows({1000: ews, 2000: ews}, pc=pc, k=k)
        runs.append(rows)
    table = A.build_table_from_rows(runs, strat)
    by = {(t["code"], t["p"], t["p_cluster"], t["cluster_k"]): t for t in table}
    assert by[("bb144", 1e-3, 0.3, 2)]["reached_all"] is True
    assert by[("bb144", 1e-3, 0.3, 3)]["reached_all"] is False
    tex = A.render_clusterk_tex(table, p=1e-3)
    # m=2 reached: the row is emitted with its absolute time and the m=3 arm as a ratio lower bound
    assert r"\geq" in tex and r"\mathbf" not in tex and "BB $[[144,12,12]]$" in tex
    # the other four codes emit no row: no stacked rules
    lines = tex.splitlines()
    assert lines.count(r"\midrule") == 1 and lines[-2] == r"\bottomrule"
    assert A.render_clusterk_tex(table).count(r"\\") == 1            # default p=1e-4: header line only
    assert all(not (a.endswith("rule") and b.endswith("rule")) for a, b in zip(lines, lines[1:]))


def test_table_omits_rows_whose_m2_arm_is_a_lower_bound():
    strat = {("bb144", "0", 1e-3): dict(ew=3.0, vw=0.1, ew_se=0.0)}
    rows = []
    for (pc, k), ews in {(0.3, 2): [2.0, 4.0, 2.5], (0.3, 3): [3.0, 3.0, 3.0]}.items():
        rows += _rows({1000: ews, 2000: ews}, pc=pc, k=k)
    table = A.build_table_from_rows([rows], strat)
    tex = A.render_clusterk_tex(table, p=1e-3)
    assert "BB $[[144,12,12]]$" not in tex and r"\geq" not in tex


def test_fmt_ratio():
    assert A._fmt_ratio(0.4912, 0.052, False) == "$0.49(5)$"
    assert A._fmt_ratio(48.3, 6.6, False) == "$48(7)$"
    assert A._fmt_ratio(122.4, 1.0, True) == r"$\geq 120$"
    assert A._fmt_ratio(7.63, 0.1, True) == r"$\geq 7.6$"
    assert A._fmt_ratio(84.9, 1.0, True) == r"$\geq 84$"     # a lower bound rounds down
    assert A._fmt_ratio(0.5, 0.1, False) == "$0.50(10)$"     # PDG: SE 100-354 keeps two digits
    assert A._fmt_ratio(float("nan"), None, False) == "--"


def _dmc_rows(ews_by_len, code="bb144", p=1e-3, method="Z+H"):
    return [dict(phase="advantage", code=code, p=repr(p), method=method, seed=str(7000 + s),
                 length=str(L), ew=repr(ew)) for L, ews in ews_by_len.items() for s, ew in enumerate(ews)]


def test_cross_check_dmc_matches_and_flags_mismatch():
    run = _rows({1000: [3.0, 3.1], 2000: [3.0, 3.2], 1024000: [9.0, 9.0]}) \
        + _rows({1000: [1.0, 1.0]}, pc=0.1)
    dmc = _dmc_rows({1000: [3.0, 3.1], 2000: [3.0, 3.2]}) + _dmc_rows({1000: [7.0, 7.0]}, method="direct")
    n, bad, unm = A.cross_check_dmc(run, dmc)
    assert n == 4 and bad == [] and unm == 0     # L > 512000 without a dmc row is not "unmatched"; other arms ignored
    dmc_long = dmc + _dmc_rows({1024000: [9.0, 9.5]})   # a bb144 top-up row at long L is now cross-checked
    n, bad, unm = A.cross_check_dmc(run, dmc_long)
    assert n == 6 and len(bad) == 1 and unm == 0
    dmc2 = _dmc_rows({1000: [3.0, 3.5], 2000: [3.0, 3.2]})
    n, bad, unm = A.cross_check_dmc(run, dmc2)
    assert n == 4 and len(bad) == 1 and unm == 0
    dmc3 = _dmc_rows({1000: [3.0, 3.1]})          # the L=2000 rows have no counterpart
    n, bad, unm = A.cross_check_dmc(run, dmc3)
    assert n == 2 and bad == [] and unm == 2


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_group_paths_by_run():
    ps = ["/x/kernel_param_tta_run4.csv", "/x/kernel_param_tta_run4_ext.csv",
          "/x/kernel_param_tta_run5.csv", "/x/other.csv"]
    assert A.group_paths_by_run(ps) == {4: ps[:2], 5: [ps[2]], "other.csv": [ps[3]]}


def test_load_rows_by_run_concatenates_base_and_ext(tmp_path):
    base, ext = tmp_path / "kernel_param_tta_run4.csv", tmp_path / "kernel_param_tta_run4_ext.csv"
    _write_csv(base, _rows({1000: [3.0, 3.0, 3.0]}))
    _write_csv(ext, _rows({2000: [3.0, 3.0, 3.0]}))
    by_run = A.load_rows_by_run([str(base), str(ext)])
    assert list(by_run) == [4] and len(by_run[4]) == 6
    assert {int(r["length"]) for r in by_run[4]} == {1000, 2000}
    strat = {("bb144", "0", 1e-3): dict(ew=3.0, vw=0.1, ew_se=0.0)}
    table = A.build_table_from_rows(list(by_run.values()), strat)
    assert len(table) == 1 and table[0]["n_runs"] == 1       # one run, not two
    # sustained from L=1000 (3 chains x 1 s) only if both lengths are seen
    assert abs(table[0]["t_mean"] - 3.0) < 1e-9


def test_check_dmc_merges_bb108_file(tmp_path):
    res = tmp_path / "cluster_results"
    res.mkdir()
    _write_csv(res / "dmc_speedup_run4.csv", _dmc_rows({1000: [3.0, 3.1]}))
    _write_csv(res / "dmc_speedup_run4_bb108.csv", _dmc_rows({1000: [5.0, 5.1]}, code="bb_l9m6"))
    dmc = A.load_dmc_rows(4, str(res))
    assert {r["code"] for r in dmc} == {"bb144", "bb_l9m6"} and len(dmc) == 4
    run = _rows({1000: [3.0, 3.1]}) + _rows({1000: [5.0, 5.1]}, code="bb_l9m6")
    n, bad, unm = A.cross_check_dmc(run, dmc)
    assert n == 4 and bad == [] and unm == 0
    assert A.load_dmc_rows(5, str(res)) is None and A.load_dmc_rows(None, str(res)) is None


def test_check_dmc_exits_nonzero_on_mismatch_or_no_match(tmp_path, monkeypatch):
    import pytest
    res = tmp_path / "cluster_results"
    res.mkdir()
    _write_csv(res / "dmc_speedup_run4.csv", _dmc_rows({1000: [3.0, 3.5]}))
    run_csv = res / "kernel_param_tta_run4.csv"
    _write_csv(run_csv, _rows({1000: [3.0, 3.1]}))
    monkeypatch.setattr(A, "ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as e:                  # one mismatch
        A.main(["--check-dmc", str(run_csv)])
    assert e.value.code == 1
    _write_csv(res / "dmc_speedup_run4.csv", _dmc_rows({1000: [3.0, 3.1]}, code="bb72"))
    with pytest.raises(SystemExit) as e:                  # nothing matched
        A.main(["--check-dmc", str(run_csv)])
    assert e.value.code == 1
    assert not (tmp_path / "paper").exists() and not (tmp_path / "docs").exists()


def test_plot_pcluster_looks_up_p_with_tolerance(tmp_path):
    p_rt = 1e-3 * (1 + 1e-12)                     # not == 1e-3, but within _near's tolerance
    assert p_rt != 1e-3
    table = [dict(code="bb144", p=p_rt, p_cluster=pc, cluster_k=2, n_runs=2,
                  t_mean=10.0 + pc, t_se=1.0, reached_all=pc < 0.5, truth=3.0, truth_se=0.0)
             for pc in A.PCS]
    assert len(A._select_arm_points(table, "bb144", 1e-3, 2)) == len(A.PCS)
    rel = A._rel_times(table, "bb144", 1e-3)
    assert [x for x, _, _, _ in rel] == A.PCS
    assert min(r for _, r, _, ok in rel if ok) == 1.0                  # best reached value sits at 1
    assert [ok for _, _, _, ok in rel] == [pc < 0.5 for pc in A.PCS]
    assert rel[0][2] == 0.0                                             # the reference cell has no SE
    assert abs(rel[1][2] - 1.01 * (0.1 ** 2 / 1.01 ** 2 + 0.1 ** 2) ** 0.5) < 1e-12
    out = tmp_path / "f.png"
    fig, ax = A.plot_pcluster(table, str(out), rows=[("bb144", 1e-3)])
    assert out.stat().st_size > 0
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == len(A.PCS)                                     # one labelled cell per arm of the one row
    assert sum(t.startswith("≥") for t in texts) == sum(pc >= 0.5 for pc in A.PCS)
    assert A._fmt_rel(1.234, True) == "1.2" and A._fmt_rel(12.3, False) == "≥12"
    assert A._fmt_rel(1.234, True, 0.14) == "1.23(14)" and A._fmt_rel(12.3, False, 2.0) == "≥12"
    assert texts[0] == "1" and texts[1] == "1.01(14)"                # reference cell, then value(SE)


def test_codes_order_and_labels():
    assert A.CODES == ["surface:5", "hgp_hamming", "rect:3x11", "bb144", "bb_l9m6"]
    assert set(A.LABEL) == set(A.CODES)


def test_group_paths_by_run_merges_add_files():
    ps = ["/x/kernel_param_tta_run4.csv", "/x/kernel_param_tta_run4_ext.csv",
          "/x/kernel_param_tta_run4_add.csv", "/x/kernel_param_tta_run5_add.csv"]
    assert A.group_paths_by_run(ps) == {4: ps[:3], 5: [ps[3]]}


def test_check_dmc_merges_topo_file(tmp_path):
    res = tmp_path / "cluster_results"
    res.mkdir()
    _write_csv(res / "dmc_speedup_run4.csv", _dmc_rows({1000: [3.0, 3.1]}))
    _write_csv(res / "dmc_speedup_run4_topo.csv", _dmc_rows({1000: [2.0, 2.1]}, code="surface:5"))
    dmc = A.load_dmc_rows(4, str(res))
    assert {r["code"] for r in dmc} == {"bb144", "surface:5"} and len(dmc) == 4
    run = _rows({1000: [2.0, 2.1]}, code="surface:5")
    n, bad, unm = A.cross_check_dmc(run, dmc)
    assert n == 2 and bad == [] and unm == 0


def test_plot_pcluster_default_rows(tmp_path):
    table = [dict(code=c, p=p, p_cluster=pc, cluster_k=2, n_runs=2, t_mean=10.0 + pc,
                  t_se=1.0, reached_all=True, truth=3.0, truth_se=0.0)
             for c in A.CODES for p in A.PS for pc in A.PCS]
    fig, ax = A.plot_pcluster(table, str(tmp_path / "f.png"))
    assert A.FIG_ROWS == [(c, 1e-4) for c in A.CODES]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == len(A.FIG_ROWS)
    assert labels == [A.LABEL[c] for c, _ in A.FIG_ROWS]          # one p, so no p in the labels
    assert len(ax.texts) == len(A.FIG_ROWS) * len(A.PCS)
    assert [t.get_text() for t in ax.get_xticklabels()] == [f"{x:g}" for x in A.PCS]


def test_tex_first_block_is_surface5():
    strat = {("surface", "5", 1e-3): dict(ew=3.0, vw=0.1, ew_se=0.0),
             ("bb144", "0", 1e-3): dict(ew=3.0, vw=0.1, ew_se=0.0)}
    rows = _rows({1000: [3.0] * 3, 2000: [3.0] * 3}, code="surface:5") \
        + _rows({1000: [3.0] * 3, 2000: [3.0] * 3})
    table = A.build_table_from_rows([rows], strat)
    assert [t["code"] for t in table] == ["surface:5", "bb144"]
    tex = A.render_clusterk_tex(table, p=1e-3)
    body = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and not ln.startswith("code")]
    assert body[0].startswith(A.LABEL["surface:5"]) and len(body) == 2
    assert tex.splitlines().count(r"\midrule") == 1 and tex.splitlines()[-2] == r"\bottomrule"
