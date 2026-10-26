# rare-event-mcmc-qec

Code, cluster job scripts, simulation outputs, and reference estimates for

> Junichi Haruna and Kotaro Murakami, *Fast and accurate rare-event simulation by Markov chain Monte Carlo for general quantum codes* (2026).

The splitting method estimates logical error rates of quantum error-correcting codes deep in the rare-event regime, where direct Monte Carlo sampling is prohibitively expensive.
It relies on a Markov chain Monte Carlo (MCMC) sampler of faulty configurations, usually driven by one-bit-flip proposals, which have been reported to be trapped on some codes.
The paper proposes a kernel that combines locally balanced one-bit-flip proposals with a weight-preserving cluster-exchange move, and benchmarks both kernels and direct Monte Carlo on twelve codes under code-capacity bit-flip noise.
The new kernel agrees with an MCMC-independent reference on every code, including those where one-bit flips are trapped.
It reaches a fixed accuracy faster than one-bit flips wherever they are trapped, by a factor of 21 where one-bit flips reach the target and by at least 530 to 1400 elsewhere, and is 2 to 5 times slower where one-bit flips are not trapped.
At the lowest error rate it is faster than direct Monte Carlo on every code, by more than four orders of magnitude on some.

This repository holds everything behind the paper's numbers:

- `src/`: the sampler (`runners.py`, `kernels.py`), the codes and decoders (`css_code.py`), the MCMC-independent references (`direct_mc.py`, `subset_sampling.py`, `concat_steane_recursion.py`), and the Bravyi--Vargo splitting estimator (`splitting_pl.py`).
- `cluster_scripts/`: the PBS job scripts and drivers that produced every simulation output.
- `cluster_results/`: those outputs as CSV, plus the reference estimates.
- `compare/`: the scripts that turn the CSV files into the paper's tables and figures.
- `paper/tables/`, `docs/figures/`: the generated tables and figures as the paper includes them.
- `tests/`: the test suite.
- `SOURCE_COMMIT`: the private commit this snapshot was exported from.

## Install

Python 3.11.

```
pip install -r requirements.txt        # or requirements-lock.txt for the exact versions we used
python -m pytest tests -q -n auto      # about two minutes
```

## Reproduce the tables and figures

```
python reproduce.py          # rewrites paper/tables/*.tex and docs/figures/*.png from cluster_results/
python reproduce.py --check  # regenerates in a temporary copy and requires the six tables to match byte for byte
```

Both read only the CSV files and take about two minutes; no Monte Carlo runs.
`REPRODUCING.md` maps every table and figure of the paper to its generator, its input files, and the job script that produced them.

## Re-running the simulations

The job scripts in `cluster_scripts/*.sh` were written for a PBS cluster (mostly one node of 96 cores per job, several hours each (3 to 16 h in the `#PBS` headers)) and write the CSV files of `cluster_results/`.
They are provided as the record of how the data were produced.
Most have a `*_smoke.sh` variant that runs a reduced configuration.

## Citation

See `CITATION.cff`.
Every release of this repository is archived on Zenodo; cite the DOI of the version you used.

## License

MIT, for the code and the data alike. See `LICENSE`.
