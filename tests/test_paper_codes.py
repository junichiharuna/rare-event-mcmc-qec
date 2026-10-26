# tests/test_paper_codes.py
"""The single code registry every NEW driver uses. It must build the SAME code objects
as the drivers that produced the published CSVs (lowp_audit.CODES,
code_advantage_map.CODE_REGISTRY) and as src/subset_sampling.build_code_and_decoder."""
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "cluster_scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import paper_codes as PC  # noqa: E402
import lowp_audit as LA  # noqa: E402
import code_advantage_map as CAM  # noqa: E402
import subset_sampling as ss  # noqa: E402

_CAM_NAMES = {"rect:3x11": "ut3x11", "rect:3x15": "ut3x15", "hgp_hamming": "hgp_hamming3",
              "surface:3": "surface3", "concat_steane:2": "concat_steane2",
              "toric:5": "toric5", "toric:7": "toric7",
              "surface:5": "surface5", "surface:7": "surface7",
              "bb72": "bb72", "bb144": "bb144", "bb_l9m6": "bb_l9m6"}


def _same(a, b):
    return np.array_equal(a.hz, b.hz) and np.array_equal(a.lz, b.lz) and a.n == b.n


def test_registry_lists():
    assert len(PC.PAPER_CODES) == 12          # tab:codes: surface 3/5/7, toric 5/7, AT 3x11/3x15, Steane, HGP, BB x3
    assert len(PC.LADDER_CODES) == 8          # toric 9/13/17/21, AT 3x5/3x9/3x13/3x17
    assert set(PC.MEANW_CODES) == {"toric5", "toric7", "surface5", "surface7", "ut3x11",
                                   "ut3x15", "hgp_hamming3", "bb72", "bb144", "bb_l9m6"}
    assert set(PC.MEANW_CODES) < set(PC.PAPER_CODES)


@pytest.mark.parametrize("name", list(LA.CODES))
def test_matches_lowp_audit(name):
    code, dec = PC.build(name, "bit_flip", 1e-3)
    assert _same(code, LA.build_code(name))
    assert type(dec).__name__ == type(LA.build_decoder(name, code, 1e-3)).__name__
    sp = PC.spec(name)
    assert (sp.code_type, sp.d) == (LA.CODES[name][2], LA.CODES[name][3][0])


@pytest.mark.parametrize("cam_name,name", list(_CAM_NAMES.items()))
def test_matches_code_advantage_map(cam_name, name):
    code, _ = PC.build(name, "bit_flip", 1e-3)
    ref, _ = CAM.build_code_decoder(cam_name, 1e-3)
    assert _same(code, ref)


@pytest.mark.parametrize("name", ["toric5", "ut3x11", "bb144", "hgp_hamming3", "concat_steane2"])
def test_matches_subset_sampling_builder(name):
    sp = PC.spec(name)
    code, _ = PC.build(name, "bit_flip", 1e-2)
    ref, _ = ss.build_code_and_decoder("bit_flip", sp.code_type, sp.d, 1e-2)
    assert _same(code, ref)


def test_ladder_names_parse():
    assert PC.spec("toric21").d == "21" and PC.build("toric21", "bit_flip", 1e-2)[0].n == 882
    assert PC.spec("ut3x17").d == "3x17" and PC.build("ut3x17", "bit_flip", 1e-2)[0].n == 102
    with pytest.raises(ValueError):
        PC.spec("toricX")


def test_depolarizing_decoders():
    _, dec = PC.build("toric5", "depolarizing", 1e-2)
    assert hasattr(dec, "decode_x") and hasattr(dec, "decode_z")
    _, dec = PC.build("bb72", "depolarizing", 1e-2)
    assert type(dec).__name__ == "_DepolarizingBPOSDDecoder"
    with pytest.raises(ValueError):
        PC.build("concat_steane2", "depolarizing", 1e-2)


def test_meta_and_inits():
    m = PC.meta("ut3x11")
    assert sorted(m["lwts"]) == [3, 11] and m["spread"] == 8
    assert [t for t, _ in PC.inits("ut3x11", ["short", "long"])] == ["short", "long"]
    assert [t for t, _ in PC.inits("ut3x11", ["maxw"])] == ["long"]
    assert PC.inits("toric5", ["short", "long"]) == [("uniform", 0)]
    code = PC.build("ut3x11", "bit_flip", 1e-2)[0]
    hi = m["hi"]
    assert int(PC.reorder_logicals(code, hi).lx[0].sum()) == 11
