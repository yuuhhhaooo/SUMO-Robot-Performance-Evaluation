"""Paired self-implemented vs upstream comparison (supervisor item 1).

Six families appear twice in the grid, once self-implemented and once
as an upstream/library variant. Both run on the same maps, tasks,
seeds, and global planners, so the comparison is fully paired. For
each family and layer this script reports the success rate of both
variants, the mean paired difference (upstream minus self), and a 95%
cluster-bootstrap interval that resamples the (map, task, seed)
scenarios (B = 2000, fixed RNG seed).

Usage:
    python analysis/upstream_paired.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {  # family -> (self-implemented, upstream/library)
    "SARL": ("sarl", "sarl_upstream"),
    "CADRL": ("cadrl", "cadrl_upstream"),
    "LSTM-RL": ("lstm_rl", "lstm_rl_upstream"),
    "TEB": ("teb", "teb_upstream"),
    "MPC": ("mpc", "mpc_dompc"),
    "ORCA": ("orca_heuristic", "orca"),
}
LAYERS = {
    "sfm": "results_sfm/peds_sfm/summary_all.csv",
    "pysf": "results_pysf/peds_pysf/summary_all.csv",
    "jupedsim": "results_jupedsim/peds_jupedsim/summary_all.csv",
}
B = 2000
RNG_SEED = 0


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["termination_reason"].astype(str).str.split(":").str[0]
            != "sumo_crash"].copy()
    df["success"] = (df["success"].astype(str).str.lower()
                     .isin(("true", "1", "1.0"))).astype(int)
    return df


def main():
    rows = []
    for layer, path in LAYERS.items():
        df = load(path)
        for family, (self_name, up_name) in PAIRS.items():
            sub = df[df["algorithm"].isin([self_name, up_name])]
            cell = ["map", "task", "seed", "global_planner"]
            wide = (sub.pivot_table(index=cell, columns="algorithm",
                                    values="success", aggfunc="first")
                    .dropna(subset=[self_name, up_name]))
            diff = (wide[up_name] - wide[self_name]).astype(float)
            idxf = wide.index.to_frame(index=False)
            scen = (idxf["map"].astype(str) + "|"
                    + idxf["task"].astype(str) + "|"
                    + idxf["seed"].astype(str)).values
            groups = pd.Series(diff.values).groupby(scen).mean()
            rng = np.random.default_rng(RNG_SEED)
            g = groups.values
            idx = rng.integers(0, len(g), size=(B, len(g)))
            boot = g[idx].mean(axis=1)
            rows.append({
                "layer": layer, "family": family,
                "n_pairs": len(wide),
                "rate_self": round(float(wide[self_name].mean()), 3),
                "rate_upstream": round(float(wide[up_name].mean()), 3),
                "diff": round(float(diff.mean()), 3),
                "ci_lo": round(float(np.percentile(boot, 2.5)), 3),
                "ci_hi": round(float(np.percentile(boot, 97.5)), 3),
            })
            print(f"{layer:9s} {family:8s} self={rows[-1]['rate_self']:.3f} "
                  f"up={rows[-1]['rate_upstream']:.3f} "
                  f"diff={rows[-1]['diff']:+.3f} "
                  f"[{rows[-1]['ci_lo']:+.3f}, {rows[-1]['ci_hi']:+.3f}] "
                  f"(n={rows[-1]['n_pairs']})")
    out = pd.DataFrame(rows)
    out.to_csv("upstream_paired.csv", index=False)
    print("wrote upstream_paired.csv")


if __name__ == "__main__":
    main()
