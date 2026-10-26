import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sweep_direct_mc_stats_decoder_kind(tmp_path):
    out = tmp_path / "rec.jsonl"
    cmd = [
        sys.executable, os.path.join(ROOT, "cluster_scripts", "sweep_direct_mc_stats.py"),
        "--noise", "bit_flip", "--code-type", "toric", "--code-size", "3",
        "--p", "0.08", "--n-target", "10", "--m-max", "100000", "--n-cores", "1",
        "--sampler", "binomial", "--decoder-kind", "bp",
        "--out-jsonl", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["decoder_kind"] == "bp"
    assert rec["code_type"] == "toric"
