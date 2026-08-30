"""Five-map mean success per pedestrian layer (tab:layers, Mean column).

The mean is the share of successful episodes among all non-crash
episodes of a layer, over all five maps and all 54 combinations, plus
the pooled value over the three layers together.

Usage:
    python analysis/layer_means.py
"""
import pandas as pd

LAYERS = {
    "sfm": "results_sfm/peds_sfm/summary_all.csv",
    "pysf": "results_pysf/peds_pysf/summary_all.csv",
    "jupedsim": "results_jupedsim/peds_jupedsim/summary_all.csv",
}


def main():
    rows = []
    tot_n = tot_s = 0
    for layer, path in LAYERS.items():
        df = pd.read_csv(path, usecols=["success", "termination_reason"],
                         low_memory=False)
        df = df[df["termination_reason"].astype(str).str.split(":").str[0]
                != "sumo_crash"]
        succ = df["success"].astype(str).str.lower().isin(
            ("true", "1", "1.0"))
        rows.append({"layer": layer, "n_episodes": len(df),
                     "mean_success": round(float(succ.mean()), 3)})
        tot_n += len(df)
        tot_s += int(succ.sum())
        print(rows[-1])
    rows.append({"layer": "pooled", "n_episodes": tot_n,
                 "mean_success": round(tot_s / tot_n, 3)})
    print(rows[-1])
    pd.DataFrame(rows).to_csv("layer_means.csv", index=False)
    print("wrote layer_means.csv")


if __name__ == "__main__":
    main()
