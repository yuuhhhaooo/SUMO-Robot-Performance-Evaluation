"""Logit-scale variance decomposition (supervisor item 3).

Replaces the extreme-spread-over-SD argument: the spread between the
best and worst of 54 units is an extreme order statistic whose size
grows with the number of units, so it cannot be compared against one
noise SD. Instead, this script reports how much of the latent
logit-scale variance sits with the algorithm, the map, the task, the
seed, and the episode-level residual (pi^2/3 for the logistic model).
The algorithm variance is the population variance of the 54 fixed
effects (reference included at zero).

Usage:
    python analysis/variance_decomposition.py
"""
import numpy as np
import pandas as pd

LAYERS = ["sfm", "pysf", "jupedsim"]


def main():
    rows = []
    for layer in LAYERS:
        base = f"results_{layer}/peds_{layer}/stats_combo"
        d = pd.read_csv(f"{base}/success_glmm_odds_ratios.csv")
        alg = d[d["term"].str.startswith("C(algorithm)")]["log_odds"]
        effects = list(alg) + [0.0]          # reference at zero
        vc = (pd.read_csv(f"{base}/success_glmm_variance_components.csv")
              .set_index("component")["sd_posterior_mean"])
        var = {
            "algorithm": float(np.var(effects)),
            "task": float(vc["task"] ** 2),
            "seed": float(vc["seed"] ** 2),
            "map": float(vc["map"] ** 2),
            "residual": float(np.pi ** 2 / 3),
        }
        total = sum(var.values())
        for comp, v in var.items():
            rows.append({"layer": layer, "component": comp,
                         "variance": round(v, 3),
                         "sd": round(float(np.sqrt(v)), 3),
                         "share": round(v / total, 3)})
        print(layer, {c: f"{v / total * 100:.0f}%" for c, v in var.items()})
    pd.DataFrame(rows).to_csv("variance_decomposition.csv", index=False)
    print("wrote variance_decomposition.csv")


if __name__ == "__main__":
    main()
