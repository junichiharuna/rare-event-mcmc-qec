import pathlib
import sys

CS = pathlib.Path(__file__).resolve().parents[1] / "cluster_scripts"
sys.path.insert(0, str(CS))

import dmc_speedup as W


def test_dmc_cutoff_worker_row_shape():
    # surface:3 at high p: cutoff gets faulty draws quickly
    row = W.dmc_cutoff_worker(("surface:3", 0.1, 0, 0.3, 50_000))
    assert row["phase"] == "dmc_cutoff"
    assert row["method"] == "directmc"
    assert int(row["n_trials"]) > 0
    assert int(row["n_cores"]) == 1
    for k in ("n_faulty", "s1", "s2", "wall"):
        assert k in row


def test_advantage_worker_row_shape():
    row = W.advantage_row(("surface:3", 0.1, "Z+H", 0, 2000, 50_000))
    assert row["phase"] == "advantage"
    assert row["method"] == "Z+H"
    assert float(row["ew"]) >= 0
    assert float(row["wall"]) >= 0
