# results/

This directory contains the aggregated results and analysis outputs behind the thesis. The raw per-episode traces (81,000 episodes, roughly 10 GB) are not committed; this package holds everything from the episode-level metrics table onwards, so every number and figure in the thesis can be traced or recomputed from here.

## Committed contents

- `peds_sfm/`, `peds_pysf/`, `peds_jupedsim/` — one directory per reactive pedestrian layer:
  - `summary_all.csv` — the episode-level metrics table (27,000 episodes per layer). This is the input to all statistical models.
  - `batch_summary.json` — run bookkeeping for the batch (episode counts, exclusions).
  - `stats_combo/` — per-layer statistical outputs: success rates, binomial GLMM odds ratios and variance components (`success_glmm_*.csv`), linear mixed models per continuous metric (`lmm_*.csv`), per-combination means (`means_*.csv`), failure taxonomy, bootstrap ranking stability, and the corresponding forest plots and figures.
- `peds_sfm/plots_map5_t04/` — the sfm-layer trajectory plots for the representative real-map task (map5_ucl, t04): per-combination campus views and lateral envelopes, occupancy, and the divergence comparison. The full trajectory-plot set (4.2 GB across the three layers) is not committed.
- `stats_pooled/stats_combo/` — the pooled three-layer models, with the pedestrian layer as a fixed effect.
- `figs_peds_sfm/`, `figs_peds_pysf/`, `figs_peds_jupedsim/` — ranking-transfer outputs per layer: synthetic-versus-real rankings over the 54 combinations (`ranking_sim_vs_real*.csv`), success heatmaps, and slope graphs.
- `cross_layer_tau.csv` — Kendall's tau-b between the layer rankings, produced by `analysis/cross_layer_tau.py`.

## Generated layout (not committed)

When the simulation itself runs, it writes per-episode outputs here:

```
results/<map>[__<route>][__g-<global>]/<mode>/<algorithm>/seed_<k>/
    robot_trace.csv, robot_metrics.json, scenario.json
```
