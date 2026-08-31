"""Seed-to-seed variability of the success rate per global planner.

Checked whether the RRT global planner, which replans its route under
every crowd seed, shows larger seed-to-seed variability than the
deterministic A* and Dijkstra globals. It does not: the median SD is
0.072 under RRT against 0.078 (A*) and 0.080 (Dijkstra), so the
Discussion states the corridor-informed mechanism as speculation
instead of claiming extra outcome variability.

For each layer and each (global, local) combination, the success rate
is computed per seed over all maps and tasks; the SD over the 10 seeds
is then summarised per global planner.

Usage:
    python analysis/global_seed_sd.py
"""
import pandas as pd

LAYERS = {
    "sfm": "results_sfm/peds_sfm/summary_all.csv",
    "pysf": "results_pysf/peds_pysf/summary_all.csv",
    "jupedsim": "results_jupedsim/peds_jupedsim/summary_all.csv",
}


def main():
    rows = []
    for layer, path in LAYERS.items():
        df = pd.read_csv(path, usecols=["algorithm", "global_planner",
                                        "seed", "success",
                                        "termination_reason"],
                         low_memory=False)
        df = df[df["termination_reason"].astype(str).str.split(":").str[0]
                != "sumo_crash"]
        df["succ"] = (df["success"].astype(str).str.lower()
                      .isin(("true", "1", "1.0"))).astype(int)
        per_seed = (df.groupby(["global_planner", "algorithm", "seed"])
                    ["succ"].mean().reset_index())
        sd = (per_seed.groupby(["global_planner", "algorithm"])["succ"]
              .std(ddof=1).reset_index())
        sd["layer"] = layer
        rows.append(sd)
    out = pd.concat(rows).rename(columns={"succ": "seed_sd"})
    print("median seed-to-seed SD per global planner:")
    print(out.groupby("global_planner")["seed_sd"].median().round(3)
          .to_string())
    out.to_csv("global_seed_sd.csv", index=False)
    print("wrote global_seed_sd.csv")


if __name__ == "__main__":
    main()
