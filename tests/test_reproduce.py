"""reproduce.py: the step list, the dmc_speedup input set, and the byte comparison."""
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce as R  # noqa: E402


def test_steps_cover_every_table_and_figure():
    labels = " ".join(label for label, _ in R.steps(str(ROOT)))
    for key in ("tab:meanw", "tab:lowp", "tab:depol", "tab:dmcspeedup", "tab:plcost",
                "tab:clusterk", "fig:pcluster", "fig:ttawalltime", "fig:intro"):
        assert key in labels
    assert len(R.TABLES) == 6 and len(R.FIGURES) == 3
    assert R.TABLES[0] == "meanw_table.tex" and "dmc_speedup_table.tex" in R.TABLES


def test_steps_run_from_the_given_root():
    argv = dict(R.steps(str(ROOT)))["tab:plcost"]
    assert argv[0] == sys.executable and argv[1] == "compare/make_plcost_table.py"


def test_dmc_run_files_are_the_thirty_in_natural_order():
    files = [os.path.basename(f) for f in R.dmc_run_files(str(ROOT))]
    assert len(files) == 30
    assert files[:3] == ["dmc_speedup_run4.csv", "dmc_speedup_run4_bb108.csv",
                         "dmc_speedup_run4_topo.csv"]
    assert files[-3:] == ["dmc_speedup_run13.csv", "dmc_speedup_run13_bb108.csv",
                          "dmc_speedup_run13_topo.csv"]
    assert "dmc_speedup.csv" not in files and "dmc_speedup_table.csv" not in files


def test_compare_tables_reports_only_differences(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x.tex").write_bytes(b"1\n2\n")
    (b / "x.tex").write_bytes(b"1\n2\n")
    (a / "y.tex").write_bytes(b"1\n2\n")
    (b / "y.tex").write_bytes(b"1\n3\n")
    bad = R.compare_tables(str(a), str(b), ["x.tex", "y.tex"])
    assert [n for n, _ in bad] == ["y.tex"]
    assert "-2" in bad[0][1] and "+3" in bad[0][1]


def test_compare_tables_is_byte_exact_on_line_endings(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x.tex").write_bytes(b"1\r\n")
    (b / "x.tex").write_bytes(b"1\n")
    assert [n for n, _ in R.compare_tables(str(a), str(b), ["x.tex"])] == ["x.tex"]


def test_compare_tables_names_a_line_ending_only_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x.tex").write_bytes(b"1\r\n2\r\n")
    (b / "x.tex").write_bytes(b"1\n2\n")
    assert R.compare_tables(str(a), str(b), ["x.tex"]) == [
        ("x.tex", "line endings differ (CRLF vs LF); content identical")]


def test_remove_outputs_deletes_only_the_nine_outputs(tmp_path):
    tables, figs = tmp_path / "paper" / "tables", tmp_path / "docs" / "figures"
    tables.mkdir(parents=True), figs.mkdir(parents=True)
    for n in R.TABLES:
        (tables / n).write_bytes(b"")
    for n in R.FIGURES:
        (figs / n).write_bytes(b"")
    (tables / "keep.tex").write_bytes(b"")
    removed = R.remove_outputs(str(tmp_path))
    assert len(removed) == 9
    assert [p.name for p in tables.iterdir()] == ["keep.tex"]
    assert list(figs.iterdir()) == []


@pytest.mark.slow
def test_check_passes_on_the_committed_tree():
    assert R.check(str(ROOT)) == 0
