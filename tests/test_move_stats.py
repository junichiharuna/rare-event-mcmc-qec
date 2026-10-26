# tests/test_move_stats.py
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from runners import run_rimc_mixed


def test_move_stats_counts_cluster_moves():
    _, _, stats = run_rimc_mixed(
        0.1, 2_000, p_zanella=0.5, p_cluster=0.5, cluster_k=2, n_replicas=1,
        max_error_rate=0.1, seed=7, return_move_stats=True)
    assert stats["h_proposed"] > 0
    assert 0 <= stats["h_accepted"] <= stats["h_proposed"]
    assert 0 <= stats["h_sector_changed"] <= stats["h_accepted"]


def test_move_stats_zero_without_cluster():
    _, _, stats = run_rimc_mixed(
        0.1, 1_000, p_zanella=1.0, n_replicas=1, max_error_rate=0.1,
        seed=7, return_move_stats=True)
    assert stats == {"h_proposed": 0, "h_accepted": 0, "h_sector_changed": 0}


def test_default_return_shape_unchanged():
    out = run_rimc_mixed(0.1, 1_000, p_zanella=0.5, p_cluster=0.5, n_replicas=1,
                         max_error_rate=0.1, seed=7)
    assert len(out) == 2
