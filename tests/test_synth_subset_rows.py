import csv
import io
import pathlib
import subprocess
import sys

import pytest

import compare.synth_subset_rows as S


def test_reproduces_existing_1e4_row():
    rows = S.synthesize(["toric5"], [1e-4])
    assert abs(float(rows[0]["ew_mean"]) - 3.00824535) < 1e-6


def test_1e5_values():
    want = {"toric5": 3.000825, "toric7": 4.0017422,
            "surface5": 3.0003954, "surface7": 4.0009621}
    rows = {r["code_type"] + r["d"]: r for r in S.synthesize(list(want), [1e-5])}
    for key, value in want.items():
        assert abs(float(rows[key]["ew_mean"]) - value) < 1e-5


def test_rejects_large_p():
    with pytest.raises(ValueError, match="truncation"):
        S.synthesize(["toric5"], [1e-1])


def test_rejects_prior_dependent_decoder():
    with pytest.raises(ValueError, match="BP-OSD"):
        S.synthesize(["bb144"], [1e-5])


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_append_is_idempotent_and_preserves_existing_bytes(tmp_path, newline):
    # Remove the target in a private fixture so this tests appending before AND after
    # the repository's four production rows have been added.
    with open(S.SS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        existing = [r for r in reader if not (r["code_type"] == "toric" and r["d"] == "5"
                                             and float(r["p"]) == 1e-5)]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, lineterminator=newline)
    writer.writeheader()
    writer.writerows(existing)
    before = buffer.getvalue().encode("utf-8")
    path = tmp_path / "ss.csv"
    path.write_bytes(before)
    rows = S.synthesize(["toric5"], [1e-5])
    assert S.append_rows(path, rows + rows) == 1
    after = path.read_bytes()
    assert after.startswith(before)
    suffix = after[len(before):]
    assert suffix.endswith(newline.encode()) and suffix.count(b"\n") == 1
    assert suffix.count(b"\r\n") == (1 if newline == "\r\n" else 0)
    assert S.append_rows(path, rows) == 0
    assert path.read_bytes() == after


def test_cli_dry_leaves_csv_unchanged(tmp_path):
    path = tmp_path / "ss.csv"
    before = pathlib.Path(S.SS_CSV).read_bytes()
    path.write_bytes(before)
    result = subprocess.run([sys.executable, "compare/synth_subset_rows.py", "--codes", "toric5",
                             "--p-list", "1e-5", "--csv", str(path), "--fw-csv", S.FW_CSV,
                             "--dry"], check=True, capture_output=True, text=True)
    assert "ew_mean" in result.stdout and "appended" not in result.stdout
    assert path.read_bytes() == before
