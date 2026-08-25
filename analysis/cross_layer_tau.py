#!/usr/bin/env python3
"""Cross-layer ranking agreement (RQ3, REAL data).

Reads each layer's unified 54-combination table
(figs_peds_<layer>/ranking_sim_vs_real_unified54.csv, written by
map_sensitivity_figs.py) and reports the Kendall tau-b between every
pair of layers, computed on the SUCCESS RATES (tie-aware), the same
convention as the sim-vs-real taus inside a layer.

Writes <out>/cross_layer_tau.csv with one row per layer pair and per
column (synthetic-pooled rates and real-map rates).

Usage:
    python analysis/cross_layer_tau.py
    python analysis/cross_layer_tau.py --figs figs_peds_sfm,figs_peds_pysf,figs_peds_jupedsim
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs",
                    default="figs_peds_sfm,figs_peds_pysf,figs_peds_jupedsim",
                    help="comma-separated figs dirs, one per layer")
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args()

    layers = {}
    for d in args.figs.split(","):
        d = Path(d.strip())
        name = d.name.replace("figs_peds_", "")
        layers[name] = (pd.read_csv(d / "ranking_sim_vs_real_unified54.csv")
                        .set_index("combination"))
        print(f"  {name}: {len(layers[name])} combinations from {d}")

    rows = []
    names = list(layers)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = layers[a].join(layers[b], lsuffix="_a", rsuffix="_b",
                               how="inner")
            for col, lab in [("sim_success", "synthetic_pooled"),
                             ("real_success", "real_map")]:
                t = kendalltau(j[f"{col}_a"], j[f"{col}_b"]).statistic
                rows.append({"layer_a": a, "layer_b": b, "ranking": lab,
                             "n": len(j), "kendall_tau_b": round(float(t), 3)})
                print(f"  {a} vs {b} ({lab}): tau-b = {t:.3f}  (n={len(j)})")

    out = Path(args.out) / "cross_layer_tau.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
