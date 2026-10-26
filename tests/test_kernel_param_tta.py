# tests/test_kernel_param_tta.py
import csv
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "cluster_scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import kernel_param_tta as KP  # noqa: E402


def test_arms():
    assert KP.ARMS == [(0.0, 2), (0.1, 2), (0.2, 2), (0.3, 2), (0.5, 2), (0.7, 2), (0.9, 2),
                       (0.3, 3), (0.3, 4), (0.3, 6)]


@pytest.mark.parametrize("pc,k", KP.ARMS)
def test_arm_kernel_kwargs_sum_to_one(pc, k):
    # Review focus 4: run_rimc_mixed raises unless the mixture sums to 1.0 exactly.
    kw = KP.kernel_kwargs(pc, k)
    assert abs(kw["p_single"] + kw["p_zanella"] + kw["p_cluster"] - 1.0) < 1e-12
    assert kw["cluster_k"] == k and kw["n_replicas"] == 1


def test_build_tasks_seeds_and_lengths():
    tasks = KP.build_tasks(["rect:3x11"], [1e-3], [(0.3, 2)], n_seeds=2, L0=1000,
                           L_max=4000, cache_cap=10, run_id=3)
    assert len(tasks) == 2 * 3                      # 2 seeds x {1000, 2000, 4000}
    assert {t[4] for t in tasks} == {7000 + 2_000_000, 7001 + 2_000_000}
    assert KP._task_key(tasks[0]) == ("rect:3x11", "0.001", "0.3", "2", str(7000 + 2_000_000), "1000")


def test_worker_row_on_tiny_chain():
    row = KP._worker(("rect:3x11", 1e-2, 0.3, 2, 7000, 500, 10))
    assert list(row) == KP.FIELDS
    assert row["code"] == "rect:3x11" and row["length"] == 500 and row["init"] == "long"
    assert 2.0 < row["ew"] < 20.0 and row["wall"] > 0 and row["cpu"] >= 0


def test_load_done_drops_truncated_rows(tmp_path):
    # Review focus 5: a kill mid-flush leaves a truncated line; it must be recomputed.
    p = tmp_path / "x.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KP.FIELDS)
        w.writeheader()
        w.writerow(dict(code="bb144", group="trap", p="0.001", p_cluster="0.3", cluster_k="2",
                        init="long", seed="7000", length="1000", ew="3.1", wall="1.0", cpu="1.0"))
        f.write("bb144,trap,0.001,0.3,2,long,7001,1000\n")
    done, rows = KP._load_done(str(p))
    assert len(rows) == 1 and done == {("bb144", "0.001", "0.3", "2", "7000", "1000")}


def test_worker_reproduces_benchmark_zh_chain():
    import code_advantage_map as CAM
    ew = KP._worker(("rect:3x11", 1e-3, 0.3, 2, 7000, 2000, 200000))["ew"]
    ref = CAM._advantage_worker(("rect:3x11", 1e-3, "Z+H", 7000, 2000, 200000))[8]
    assert ew == ref


def test_cost_penalizes_bposd_decoders():
    # longest-first ordering: BP-OSD codes (bb*, hgp_hamming) cost ~20x per step
    n_of = {"bb144": 144, "hgp_hamming": 58, "surface:5": 41, "rect:3x11": 66}

    def c(spec):
        return KP._cost((spec, 1e-3, 0.3, 2, 7000, 1000, 10), n_of)

    assert c("bb144") == 1000 * 20 * 144
    assert c("hgp_hamming") == 1000 * 20 * 58
    assert c("surface:5") == 1000 * 41
    assert c("rect:3x11") == 1000 * 66
