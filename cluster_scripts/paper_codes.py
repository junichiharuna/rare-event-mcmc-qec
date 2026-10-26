"""The one code registry for the appendix-redo drivers (2026-10).

Names follow cluster_scripts/lowp_audit.py ("toric5", "ut3x11", "bb_l9m6", ...) and
accept size ladders by regex. Every builder is the same css_code factory the published
drivers used (tests/test_paper_codes.py pins hz/lz equality), so a chain started here
and a reference row in cluster_results/subset_sampling*.csv share one failure definition.

noise="bit_flip"      -> MWPM / BP-OSD(prior p) / concat-Steane hard decoder
noise="depolarizing"  -> independent X/Z MWPM or BP-OSD(prior p), as direct_mc does;
                         concat_steane raises (no depolarizing decoder exists).
"""
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from css_code import (
    CSSCode, make_asymmetric_toric_code, make_bb_code, make_bb_code_72, make_bb_code_144,
    make_bposd_decoder, make_concat_steane_hard_decoder, make_concatenated_steane_code,
    make_depolarizing_bposd_decoder, make_depolarizing_mwpm_decoder, make_hgp_hamming,
    make_mwpm_decoder, make_surface_code, make_toric_code,
)


@dataclass(frozen=True)
class CodeSpec:
    name: str
    code_type: str          # code_type column of cluster_results/subset_sampling*.csv
    d: str                  # d column there ("5", "3x11", "0", "2")
    family: str             # "mwpm" | "bposd" | "lookup"
    make_code: Callable[[], CSSCode]


def _bb_l9m6():
    return make_bb_code(9, 6, a_terms=[(3, 0), (0, 1), (0, 2)],
                        b_terms=[(0, 3), (1, 0), (2, 0)])


_FIXED = {
    "hgp_hamming3": CodeSpec("hgp_hamming3", "hgp_hamming3", "0", "bposd", make_hgp_hamming),
    "bb72": CodeSpec("bb72", "bb72", "0", "bposd", make_bb_code_72),
    "bb144": CodeSpec("bb144", "bb144", "0", "bposd", make_bb_code_144),
    "bb_l9m6": CodeSpec("bb_l9m6", "bb_l9m6", "0", "bposd", _bb_l9m6),
}
_RX = [
    (re.compile(r"^toric(\d+)$"),
     lambda m: CodeSpec(m.string, "toric", m[1], "mwpm", lambda: make_toric_code(int(m[1])))),
    (re.compile(r"^surface(\d+)$"),
     lambda m: CodeSpec(m.string, "surface", m[1], "mwpm",
                        lambda: make_surface_code(int(m[1])))),
    (re.compile(r"^ut(\d+)x(\d+)$"),
     lambda m: CodeSpec(m.string, "asymmetric_toric", f"{m[1]}x{m[2]}", "mwpm",
                        lambda: make_asymmetric_toric_code(int(m[1]), int(m[2])))),
    (re.compile(r"^concat_steane(\d+)$"),
     lambda m: CodeSpec(m.string, "concat_steane", m[1], "lookup",
                        lambda: make_concatenated_steane_code(int(m[1])))),
]

# tab:codes (12) and the acceptance-measurement size ladders (8).
PAPER_CODES = ["surface3", "surface5", "surface7", "toric5", "toric7", "ut3x11", "ut3x15",
               "concat_steane2", "hgp_hamming3", "bb72", "bb144", "bb_l9m6"]
LADDER_CODES = ["toric9", "toric13", "toric17", "toric21", "ut3x5", "ut3x9", "ut3x13", "ut3x17"]
MEANW_CODES = ["toric5", "toric7", "surface5", "surface7", "ut3x11", "ut3x15",
               "hgp_hamming3", "bb72", "bb144", "bb_l9m6"]


def spec(name: str) -> CodeSpec:
    if name in _FIXED:
        return _FIXED[name]
    for rx, mk in _RX:
        m = rx.match(name)
        if m:
            return mk(m)
    raise ValueError(f"unknown code name {name!r}")


def build(name: str, noise: str, p: float):
    """-> (code, decoder) for `noise` in {"bit_flip", "depolarizing"}."""
    sp = spec(name)
    code = sp.make_code()
    if noise == "bit_flip":
        if sp.family == "bposd":
            return code, make_bposd_decoder(code, p)
        if sp.family == "lookup":
            return code, make_concat_steane_hard_decoder(int(sp.d))
        return code, make_mwpm_decoder(code)
    if noise == "depolarizing":
        if sp.family == "bposd":
            return code, make_depolarizing_bposd_decoder(code, p)
        if sp.family == "lookup":
            raise ValueError(f"{name}: no depolarizing decoder for the concatenated Steane code")
        return code, make_depolarizing_mwpm_decoder(code)
    raise ValueError(f"unknown noise {noise!r}")


def meta(name: str) -> dict:
    c = spec(name).make_code()
    lwts = [int(r.sum()) for r in c.lx]
    return dict(n=c.n, k=c.lx.shape[0], lwts=lwts, lo=int(np.argmin(lwts)),
                hi=int(np.argmax(lwts)), spread=max(lwts) - min(lwts))


def inits(name: str, requested) -> list:
    """[(tag, logical index)] with tags canonicalised to short/long; a flat basis gives
    the single ("uniform", argmin) sector (same rule as lowp_audit._inits_for)."""
    m = meta(name)
    if m["spread"] == 0:
        return [("uniform", m["lo"])]
    avail = {"short": m["lo"], "long": m["hi"], "minw": m["lo"], "maxw": m["hi"]}
    out, seen = [], set()
    for tag in requested:
        if tag not in avail:
            raise ValueError(f"unknown init {tag!r} (use short/long)")
        canon = "short" if avail[tag] == m["lo"] else "long"
        if canon not in seen:
            seen.add(canon)
            out.append((canon, avail[tag]))
    return out


def reorder_logicals(code: CSSCode, first_idx: int) -> CSSCode:
    """Put logical `first_idx` first: both runners start every chain at lx[0]."""
    order = [first_idx] + [i for i in range(code.lx.shape[0]) if i != first_idx]
    return CSSCode(hx=code.hx, hz=code.hz, lx=code.lx[order].copy(),
                   lz=code.lz[order].copy(), n=code.n)
