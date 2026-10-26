import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compare"))
import plot_tta_walltime as P  # noqa: E402


@pytest.mark.parametrize("n_codes", [len(P.ROW_ORDER), 3])
def test_panel_grid_covers_every_code(tmp_path, monkeypatch, n_codes):
    codes = P.ROW_ORDER[:n_codes]
    monkeypatch.setattr(P, "ROW_ORDER", codes)
    rows = [dict(code=code, p=p, t_dmc=10.0, t_zh=1.0, t_obf=5.0, t_dmc_se=0.5, t_zh_se=0.1,
                 dmc_lb=False, dmc_hard=False, obf_lb=False, zh_reached=True)
            for code in codes for p in P.P_ORDER]
    source = tmp_path / "synthetic.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(P, "IN", str(source))
    saved = []
    monkeypatch.setattr(Figure, "savefig",
                        lambda fig, path, **kwargs: saved.append((fig, path, kwargs)))
    previous_figures = set(plt.get_fignums())
    try:
        with plt.rc_context():
            P.main()
        assert len(saved) == 1
        fig, path, kwargs = saved[0]
        assert len(fig.axes) == len(P.P_ORDER)
        assert [ax.get_title() for ax in fig.axes] == [
            r"$p=10^{-1}$", r"$p=10^{-3}$", r"$p=10^{-5}$"]
        ax = fig.axes[0]
        # Sort by displayed height so the assertion also checks the row direction.
        labels_by_height = sorted(
            (ax.transData.transform((1, y))[1], label.get_text())
            for y, label in zip(ax.get_yticks(), ax.get_yticklabels()))
        assert [label for _, label in reversed(labels_by_height)] == [
            P.TITLE[code] for code in codes]
        assert fig.get_size_inches() == pytest.approx([7.0, 3.2])
        assert all(ax.get_xlabel() == r"$T_{2\%}$ (core-seconds)"
                   for ax in fig.axes)
        assert path == P.OUT and kwargs == {"dpi": 300}
        # one horizontal SE bar per measured point: Z+H and direct MC on every row
        assert len(fig.axes[0].containers) == 2 * len(codes)
    finally:
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)


def test_unreached_zh_cell_is_rejected(tmp_path, monkeypatch):
    code = P.ROW_ORDER[0]
    monkeypatch.setattr(P, "ROW_ORDER", [code])
    rows = [dict(code=code, p=p, t_dmc=10.0, t_zh=1.0, t_obf=5.0, t_dmc_se=0.5, t_zh_se=0.1,
                 dmc_lb=False, dmc_hard=False, obf_lb=False,
                 zh_reached=(p != P.P_ORDER[0]))
            for p in P.P_ORDER]
    source = tmp_path / "synthetic.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(P, "IN", str(source))
    with pytest.raises(ValueError, match="did not reach the target"):
        P.main()
