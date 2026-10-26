import csv as _csv
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cluster_scripts"))

from code_advantage_map import CODE_REGISTRY, build_code_decoder  # noqa: E402


def test_registry_has_all_specs():
    # The twelve paper codes, including the topological time-to-accuracy controls.
    expect = {"rect:3x11", "rect:3x15", "bb72", "bb144", "bb_l9m6",
              "hgp_hamming", "surface:3", "concat_steane:2",
              "toric:5", "toric:7", "surface:5", "surface:7"}
    assert set(CODE_REGISTRY) == expect


def test_build_light_codes_and_decoders():
    # build the light codes (avoid bb144 in the unit test for speed)
    for spec in ["surface:3", "concat_steane:2", "hgp_hamming", "rect:3x11"]:
        code, dec = build_code_decoder(spec, p=0.01)
        assert code.n > 0
        assert hasattr(dec, "decode_syndrome")
        assert hasattr(dec, "is_faulty")


@pytest.mark.parametrize("key,group", [("toric:5", "ctrl_uniform"),
                                      ("toric:7", "ctrl_uniform"),
                                      ("surface:5", "ctrl_k1"),
                                      ("surface:7", "ctrl_k1")])
def test_topological_codes_registered(key, group):
    entry = CODE_REGISTRY[key]
    assert entry.group == group and entry.truth_kind == "directmc"
    code = entry.make_code()
    _, reference_decoder = build_code_decoder("surface:3", 1e-3)
    assert type(entry.make_decoder(code, 1e-3)).__name__ == type(reference_decoder).__name__


def test_groups_are_tagged():
    assert CODE_REGISTRY["bb144"].group == "trap"
    # bb_l9m6 must be "trap" so dmc_speedup.py gives its OBF chains the long --L-max
    assert CODE_REGISTRY["bb_l9m6"].group == "trap"
    assert CODE_REGISTRY["bb_l9m6"].truth_kind == "consensus"
    assert CODE_REGISTRY["hgp_hamming"].group == "ctrl_uniform"
    assert CODE_REGISTRY["surface:3"].group == "ctrl_k1"
    assert CODE_REGISTRY["rect:3x11"].truth_kind == "directmc"
    assert CODE_REGISTRY["bb144"].truth_kind == "consensus"


def test_task_key_identities():
    from code_advantage_map import _task_key
    assert _task_key(("truth", ("rect:3x11", 0.01, 1000, 200, 12345))) == \
        ("truth", "rect:3x11", "0.01", "directmc", "12345", "0")
    assert _task_key(("consensus", ("bb144", 0.01, "OBF", 1000, 2000, 200000))) == \
        ("consensus", "bb144", "0.01", "OBF", "1000", "2000")
    assert _task_key(("advantage", ("surface:3", 0.001, "Z+H", 7003, 4000, 200000))) == \
        ("advantage", "surface:3", "0.001", "Z+H", "7003", "4000")


def test_load_done_skips_partial_last_line(tmp_path):
    from code_advantage_map import _load_done, FIELDS
    p = tmp_path / "x.csv"
    with open(p, "w", newline="") as f:
        f.write(",".join(FIELDS) + "\n")
        f.write("advantage,surface:3,ctrl_k1,0.01,OBF,long,7000,1000,2.0,0.1,directmc\n")
        f.write("advantage,surface:3,ctrl_k1,0.01,OBF,long,7000,2000,2.0\n")  # truncated kill
    done, rows = _load_done(str(p))
    assert len(rows) == 1                          # partial last line dropped
    assert ("advantage", "surface:3", "0.01", "OBF", "7000", "1000") in done


def test_driver_resume_is_incremental(tmp_path):
    out = tmp_path / "r.csv"
    repo = os.path.join(os.path.dirname(__file__), "..")
    base = [
        sys.executable, os.path.join(repo, "cluster_scripts", "code_advantage_map.py"),
        "--specs", "surface:3", "--p-list", "1e-1", "--m-consensus", "2",
        "--consensus-len-obf", "2000", "--consensus-len-zh", "2000",
        "--n-seeds", "2", "--L0", "1000", "--L-max", "2000", "--kernel-L-max", "2000",
        "--truth-mmax", "20000", "--truth-batch", "20000",
        "--n-cores", "1", "--out-csv", str(out),
    ]
    subprocess.run(base, check=True, cwd=repo)
    full = list(_csv.DictReader(open(out)))
    keep = full[: len(full) // 2]                  # simulate a kill mid-run
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(full[0].keys()))
        w.writeheader()
        for r in keep:
            w.writerow(r)
    res = subprocess.run(base, check=True, cwd=repo, capture_output=True, text=True)
    assert "pending" in res.stdout                 # reported resume
    after = list(_csv.DictReader(open(out)))

    def key(r):
        return (r["phase"], r["code"], r["p"], r["method"], r["seed"], r["length"])
    keys = [key(r) for r in after]
    assert len(keys) == len(set(keys))             # no duplicate work
    assert set(keys) == {key(r) for r in full}     # full set restored


def test_driver_end_to_end_light(tmp_path):
    out = tmp_path / "tiny.csv"
    repo = os.path.join(os.path.dirname(__file__), "..")
    cmd = [
        sys.executable, os.path.join(repo, "cluster_scripts", "code_advantage_map.py"),
        "--specs", "surface:3,concat_steane:2",
        "--p-list", "1e-1",
        "--m-consensus", "2",
        "--consensus-len-obf", "2000", "--consensus-len-zh", "2000",
        "--n-seeds", "2", "--L0", "1000", "--L-max", "2000", "--kernel-L-max", "2000",
        "--truth-mmax", "20000", "--truth-batch", "20000",
        "--n-cores", "1", "--out-csv", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=repo)
    rows = list(_csv.DictReader(open(out)))
    phases = {r["phase"] for r in rows}
    assert "consensus" in phases and "advantage" in phases
    assert "truth" in phases                      # surface:3 is directmc
    # every advantage row for surface (k=1, no trap) has finite positive ew
    surf_adv = [r for r in rows if r["code"] == "surface:3" and r["phase"] == "advantage"]
    assert surf_adv and all(float(r["ew"]) > 0 for r in surf_adv)


def test_bb_l9m6_is_the_108_8_10_code():
    # same builder as cluster_scripts/lowp_audit.py::_bb_l9m6, whose rows in
    # cluster_results/lowp_audit_2e6_gr.csv report n=108, k=8, logical weights 12..30
    code, dec = build_code_decoder("bb_l9m6", p=1e-3)
    assert code.n == 108 and code.lx.shape[0] == 8
    assert sorted(int(r.sum()) for r in code.lx) == [12, 12, 12, 12, 20, 22, 28, 30]
    assert hasattr(dec, "decode_syndrome") and hasattr(dec, "is_faulty")
