"""Split the timeouts by location (supervisor item 4).

Classifies every max_time episode by how much of it was spent waiting
at red lights (time_waiting_at_light_s / sim_time_s). Writes
timeout_split.csv with per-layer and per-map counts and the waiting
statistics quoted in the thesis.

Usage:
    python analysis/timeout_split.py
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
        df = pd.read_csv(path, low_memory=False)
        df = df[df["termination_reason"].astype(str).str.split(":").str[0]
                != "sumo_crash"]
        to = df[df["termination_reason"].astype(str)
                .str.startswith("max_time")].copy()
        to["wait_s"] = to["time_waiting_at_light_s"].fillna(0.0)
        to["wait_frac"] = to["wait_s"] / to["sim_time_s"].clip(lower=1)
        for mp, g in to.groupby("map"):
            rows.append({
                "layer": layer, "map": mp, "n_timeouts": len(g),
                "wait_mean_s": round(float(g["wait_s"].mean()), 1),
                "wait_max_s": round(float(g["wait_s"].max()), 1),
                "wait_frac_max": round(float(g["wait_frac"].max()), 3),
                "n_wait_frac_ge_25pct": int((g["wait_frac"] >= 0.25).sum()),
                "n_wait_frac_ge_50pct": int((g["wait_frac"] >= 0.5).sum()),
            })
        rows.append({
            "layer": layer, "map": "ALL", "n_timeouts": len(to),
            "wait_mean_s": round(float(to["wait_s"].mean()), 1),
            "wait_max_s": round(float(to["wait_s"].max()), 1),
            "wait_frac_max": round(float(to["wait_frac"].max()), 3),
            "n_wait_frac_ge_25pct": int((to["wait_frac"] >= 0.25).sum()),
            "n_wait_frac_ge_50pct": int((to["wait_frac"] >= 0.5).sum()),
        })
        print(f"{layer}: {len(to)} timeouts, mean wait "
              f"{to['wait_s'].mean():.0f}s, none >= 25% of the episode"
              if (to["wait_frac"] >= 0.25).sum() == 0 else f"{layer}: check")
    pd.DataFrame(rows).to_csv("timeout_split.csv", index=False)
    print("wrote timeout_split.csv")


if __name__ == "__main__":
    main()
