"""Synthesise weight-stratified reference rows from the existing f_w table.

MWPM and concat-Steane hard-decoder f_w values are p-independent, so this requires
no new decoding. Refuse p > 1e-2: the w_max=16 truncation is invalid at p=1e-1.
"""
import argparse
import csv
import io
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import subset_sampling as ss  # noqa: E402
from cluster_scripts.paper_codes import spec as paper_spec  # noqa: E402

P_MAX = 1e-2
SS_CSV = "cluster_results/subset_sampling.csv"
FW_CSV = "cluster_results/subset_sampling_fw.csv"


def synthesize(codes, ps, ss_csv=SS_CSV, fw_csv=FW_CSV):
    """Return reference rows without writing files or constructing decoders."""
    ps = [float(p) for p in ps]
    if any(p > P_MAX for p in ps):
        raise ValueError(f"p must be <= {P_MAX}: the w_max truncation is invalid above it")
    if any(not math.isfinite(p) or p <= 0 for p in ps):
        raise ValueError("p must be finite and positive")
    with open(ss_csv, newline="", encoding="utf-8") as f:
        ss_rows = list(csv.DictReader(f))
    with open(fw_csv, newline="", encoding="utf-8") as f:
        fw_rows = list(csv.DictReader(f))
    out = []
    for name in codes:
        sp = paper_spec(name)
        if sp.family == "bposd":
            raise ValueError(f"{name}: f_w depends on p for BP-OSD codes")
        fw = [dict(r) for r in fw_rows if r["code_type"] == sp.code_type
              and r["d"] == sp.d and not r["p_decoder"]
              and (r.get("noise") or "bit_flip") == "bit_flip"]
        tmpl = next((r for r in ss_rows if r["code_type"] == sp.code_type
                     and r["d"] == sp.d
                     and (r.get("noise") or "bit_flip") == "bit_flip"), None)
        if not fw or tmpl is None:
            raise ValueError(f"{name}: missing p-independent f_w rows or bit-flip reference row")
        for row in fw:
            for key in ("n", "w", "n_samples", "n_fail"):
                row[key] = int(row[key])
            for key in ("f_w", "f_w_se"):
                row[key] = float(row[key])
        meta = {k: tmpl[k] for k in ("noise", "code_type", "d", "decoder", "n_cores",
                                    "n_target", "wall_clock")}
        meta["w_max"] = max(r["w"] for r in fw)
        rows = ss.rows_for_p_grid(fw, fw[0]["n"], ps, meta)
        for row in rows:
            if row["feasible"] and tmpl["feasible"].strip().lower() in ("true", "1"):
                row["feasible"] = tmpl["feasible"]
        out.extend(rows)
    return out


def _row_key(row):
    return row["code_type"], str(row["d"]), f"{float(row['p']):.12g}"


def append_rows(ss_csv, rows):
    """Atomically append unseen (code_type, d, p) rows, preserving existing bytes."""
    path = Path(ss_csv)
    original = path.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode("utf-8"), newline=""))
    fields = reader.fieldnames
    have = {_row_key(r) for r in reader}
    new = []
    for row in rows:
        key = _row_key(row)
        if key not in have:
            new.append(row)
            have.add(key)
    if not new:
        return 0
    newline = "\r\n" if b"\r\n" in original else "\n"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore",
                            lineterminator=newline)
    writer.writerows(new)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(original)
            if original and not original.endswith(b"\n"):
                f.write(newline.encode("utf-8"))
            f.write(buffer.getvalue().encode("utf-8"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return len(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--p-list", required=True)
    ap.add_argument("--csv", default=SS_CSV)
    ap.add_argument("--fw-csv", default=FW_CSV)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    rows = synthesize(args.codes.split(","), [float(x) for x in args.p_list.split(",")],
                      args.csv, args.fw_csv)
    for row in rows:
        print({k: row[k] for k in ("code_type", "d", "p", "p_L", "ew_mean", "ew_se", "vw")})
    if not args.dry:
        print("appended", append_rows(args.csv, rows))


if __name__ == "__main__":
    main()
