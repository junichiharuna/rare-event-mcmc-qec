# Reproducing the tables and figures

`python reproduce.py` runs the generators below in this order, from the repository root, reading only `cluster_results/`.
`python reproduce.py --check` does the same in a temporary copy and requires the six `.tex` files to match the committed ones byte for byte; the three figures are regenerated but not compared.

| Paper | Output | Generator | Input files in `cluster_results/` |
|---|---|---|---|
| Table: mean-weight comparison at p=1e-4 | `paper/tables/meanw_table.tex` | `compare/build_audit_tables.py --paper-tables` | `lowp_audit_2e6_gr.csv`, `lowp_audit_2e6_k1.csv` (+ `_chains`, `_conv`, `_seeds`), reference `subset_sampling.csv` |
| Table: BB codes in the rare-event regime | `paper/tables/lowp_table.tex` | same run | `lowp_audit_2e6_gr.csv` (+ companions), reference `subset_sampling.csv` |
| Table: depolarizing test on BB[[144,12,12]] | `paper/tables/depol_table.tex` | same run | `depol_audit_burn.csv`, `p6_depol_bb144_burn.csv`, reference `subset_sampling_depol.csv` |
| Table: time to accuracy vs direct Monte Carlo | `paper/tables/dmc_speedup_table.tex` | `compare/analyze_dmc_speedup.py dmc_speedup_run{4..13}{,_bb108,_topo}.csv` | the 30 run files, references `subset_sampling.csv`, `direct_mc_ew.csv` |
| Table: time to accuracy vs the one-bit-flip baseline | second table in the same `dmc_speedup_table.tex` | same run | same |
| Table: cost of a full splitting estimate | `paper/tables/plcost_estimate_table.tex` | `compare/make_plcost_table.py` | `dmc_speedup_table.csv` (written by the previous step) |
| Table: exchange size m | `paper/tables/clusterk_tta_table.tex` | `compare/analyze_kernel_param_tta.py` | `kernel_param_tta_run{4..8}{,_add,_ext}.csv`, reference `subset_sampling.csv` |
| Figure: cluster fraction sweep | `docs/figures/fig_pcluster_tta.png` | same run | same |
| Figure: wall time to 2% | `docs/figures/fig_tta_walltime.png` | `compare/plot_tta_walltime.py` | `dmc_speedup_table.csv` |
| Figure: speedup at p=1e-5 (introduction) | `docs/figures/fig_intro_tta.png` | `compare/plot_intro_tta.py` | `dmc_speedup_table.csv` |

## Where the input files came from

| Files | Job script (PBS) | Driver |
|---|---|---|
| `lowp_audit_2e6_gr*.csv` | `cluster_scripts/lowp_audit_2e6_gr.sh` | `cluster_scripts/lowp_audit.py` |
| `lowp_audit_2e6_k1*.csv` | `cluster_scripts/lowp_audit_k1.sh` | `cluster_scripts/lowp_audit.py` |
| `depol_audit_burn.csv` | `cluster_scripts/depol_audit_burn.sh` | `cluster_scripts/depol_audit.py` |
| `p6_depol_bb144_burn.csv` | `cluster_scripts/p6_depol_bb144_burn.sh` | `cluster_scripts/depol_audit.py` |
| `dmc_speedup_run*.csv` | `cluster_scripts/dmc_speedup10.sh`, `dmc_speedup_bb108.sh`, `dmc_speedup_topo.sh` | `cluster_scripts/dmc_speedup.py` |
| `kernel_param_tta_run*.csv` | `cluster_scripts/kernel_param_tta.sh`, `kernel_param_tta_add.sh`, `kernel_param_tta_ext.sh` | `cluster_scripts/kernel_param_tta.py` |
| `subset_sampling.csv`, `subset_sampling_depol.csv` (weight-stratified references) | `cluster_scripts/subset_sampling*.sh`; the p=1e-5 rows of toric and surface d=5, 7 were computed locally from the stored f_w table, without a job | `cluster_scripts/subset_sampling.py`; `compare/synth_subset_rows.py` for the computed rows |
| `direct_mc_ew.csv` (direct Monte Carlo reference grid) | `cluster_scripts/direct_mc_ew.sh` (the 14 rows with `n_cores=20`, one per line of `direct_mc_conditions.txt`), `cluster_scripts/direct_mc_truth_ext.sh` (d=7, p=1e-3 rows), `cluster_scripts/direct_mc_bb144_lowp.sh` (BB[[144,12,12]] at p=1e-3, 1e-4) | `cluster_scripts/sweep_direct_mc_ew.py`, `merge_direct_mc.py` for the first; `sweep_direct_mc_stats.py`, `merge_direct_mc_stats.py` for the other two |

Most job scripts have a `*_smoke.sh` variant with a reduced configuration.
The scripts assume one node of 96 cores, except the three direct Monte Carlo jobs, which are arrays of 20-core subjobs pooled by `merge_direct_mc.py` or `merge_direct_mc_stats.py`; the scripts write their CSV next to the others in `cluster_results/`, and the exact walltimes and memory requests are in the `#PBS` headers.
