import csv
import subprocess
import sys

import pytest

from cluster_scripts import dmc_speedup as DS


@pytest.mark.parametrize("missing_checkpoint", [False, True])
def test_expect_ten_checkpoints_marks_control_file_complete(tmp_path, missing_checkpoint):
    path = tmp_path / "control.csv"
    rows = []
    for p in (1e-1, 1e-3, 1e-5):
        rows.append(dict(phase="dmc_cutoff", code="toric:5", p=p, method="directmc",
                         n_faulty=1, n_trials=100, s1=3, s2=9, n_cores=1, wall=1))
        for method in ("OBF", "Z+H"):
            for i in range(10):
                rows.append(dict(phase="advantage", code="toric:5", group="ctrl_uniform",
                                 p=p, method=method, init="long", seed=7000,
                                 length=1000 * 2 ** i, ew=3.0, wall=1, truth_kind="directmc"))
    if missing_checkpoint:
        rows.pop()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, DS.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    out = subprocess.run([sys.executable, "cluster_scripts/check_dmc_speedup_complete.py",
                          str(path), "--n-seeds", "1", "--expect", "OBF:10,Z+H:10"],
                         check=True, capture_output=True, text=True)
    assert out.stdout.strip() == ("INCOMPLETE" if missing_checkpoint else "COMPLETE")
