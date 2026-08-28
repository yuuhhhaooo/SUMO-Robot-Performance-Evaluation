"""Bootstrap intervals for the ranking-transfer statistics
(supervisor item 2).

The thesis reports Kendall tau-b between the synthetic-pooled and the
real-map success-rate rankings as single numbers computed from the
seed-averaged rates. This script resamples the 10 crowd seeds with
replacement (B = 2000, fixed RNG seed), recomputes the rates, and
reports the median with a 95% percentile interval for
  - the sim-to-real tau of every layer,
  - the cross-layer taus (synthetic-pooled and real-map columns),
plus two top-weighted measures for sim-to-real transfer: the top-10
overlap and rank-biased overlap (RBO, p = 0.9), because tau weights
mid-table ties the same as reversals at the top.

Usage:
    python analysis/tau_bootstrap.py
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

LAYERS = {
    "sfm": "results_sfm/peds_sfm/summary_all.csv",
    "pysf": "results_pysf/peds_pysf/summary_all.csv",
    "jupedsim": "results_jupedsim/peds_jupedsim/summary_all.csv",
}
SYNTH = ("map1_straight", "map2_crossing", "map3_grid", "map4_london")
REAL = "map5_ucl"
B = 2000
RNG_SEED = 0


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["termination_reason"].astype(str).str.split(":").str[0]
            != "sumo_crash"].copy()
    df["success"] = (df["success"].astype(str).str.lower()
                     .isin(("true", "1", "1.0"))).astype(int)
    df["combo"] = (df["global_planner"].astype(str) + "+"
                   + df["algorithm"].astype(str))
    return df


def seed_tables(df):
    """Per (combo, seed) success sums and counts, for sim and real."""
    tabs = {}
    for label, mask in (("sim", df["map"].isin(SYNTH)),
                        ("real", df["map"] == REAL)):
        part = df[mask]
        g = part.groupby(["combo", "seed"])["success"]
        tabs[label] = (g.sum().unstack("seed").fillna(0.0),
                       g.count().unstack("seed").fillna(0.0))
    return tabs


def rates_from_tables(tabs, seeds=None):
    out = {}
    for label, (s, c) in tabs.items():
        if seeds is None:
            out[label] = s.sum(axis=1) / c.sum(axis=1)
        else:
            mult = pd.Series(seeds).value_counts()
            w = pd.Series(0.0, index=s.columns)
            w.loc[mult.index] = mult.values
            out[label] = (s @ w) / (c @ w)
    return out["sim"], out["real"]


def top_k_overlap(a, b, k=10):
    ta = set(a.sort_values(ascending=False).index[:k])
    tb = set(b.sort_values(ascending=False).index[:k])
    return len(ta & tb) / k


def rbo(a, b, p=0.9):
    ra = list(a.sort_values(ascending=False).index)
    rb = list(b.sort_values(ascending=False).index)
    n = len(ra)
    sa, sb = set(), set()
    total = 0.0
    for d in range(1, n + 1):
        sa.add(ra[d - 1])
        sb.add(rb[d - 1])
        total += (p ** (d - 1)) * len(sa & sb) / d
    return (1 - p) / (1 - p ** n) * total


def summarise(vals):
    v = np.asarray(vals, dtype=float)
    return (round(float(np.median(v)), 3),
            round(float(np.percentile(v, 2.5)), 3),
            round(float(np.percentile(v, 97.5)), 3))


def main():
    rng = np.random.default_rng(RNG_SEED)
    layers = {name: load(path) for name, path in LAYERS.items()}
    seed_pool = sorted(layers["sfm"]["seed"].unique())
    draws = [rng.choice(seed_pool, size=len(seed_pool), replace=True)
             for _ in range(B)]

    rows = []
    boot_cache = {}
    for name, df in layers.items():
        tabs = seed_tables(df)
        sim0, real0 = rates_from_tables(tabs)
        sim0, real0 = sim0.align(real0, join="inner")
        stats0 = {"tau": kendalltau(sim0, real0).statistic,
                  "top10": top_k_overlap(sim0, real0),
                  "rbo": rbo(sim0, real0)}
        boots = {"tau": [], "top10": [], "rbo": []}
        per_draw = []
        for seeds in draws:
            sim, real = rates_from_tables(tabs, seeds)
            sim, real = sim.align(real, join="inner")
            boots["tau"].append(kendalltau(sim, real).statistic)
            boots["top10"].append(top_k_overlap(sim, real))
            boots["rbo"].append(rbo(sim, real))
            per_draw.append((sim, real))
        boot_cache[name] = per_draw
        for stat in ("tau", "top10", "rbo"):
            med, lo, hi = summarise(boots[stat])
            rows.append({"comparison": f"{name} sim vs real",
                         "statistic": stat,
                         "point": round(float(stats0[stat]), 3),
                         "median": med, "ci_lo": lo, "ci_hi": hi})
            print(f"{name:9s} sim-vs-real {stat:6s} "
                  f"point={stats0[stat]:.3f} median={med:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]")

    names = list(layers)
    for i, a in enumerate(names):
        for b_name in names[i + 1:]:
            sim_a0, real_a0 = rates_from_tables(seed_tables(layers[a]))
            sim_b0, real_b0 = rates_from_tables(seed_tables(layers[b_name]))
            for col, la in (("sim", "synthetic"), ("real", "real")):
                x0 = {"sim": sim_a0, "real": real_a0}[col]
                y0 = {"sim": sim_b0, "real": real_b0}[col]
                x0, y0 = x0.align(y0, join="inner")
                point = kendalltau(x0, y0).statistic
                vals = []
                for k in range(B):
                    xa = boot_cache[a][k][0 if col == "sim" else 1]
                    yb = boot_cache[b_name][k][0 if col == "sim" else 1]
                    xa, yb = xa.align(yb, join="inner")
                    vals.append(kendalltau(xa, yb).statistic)
                med, lo, hi = summarise(vals)
                rows.append({"comparison": f"{a} vs {b_name} ({la})",
                             "statistic": "tau",
                             "point": round(float(point), 3),
                             "median": med, "ci_lo": lo, "ci_hi": hi})
                print(f"{a} vs {b_name} ({la}) tau point={point:.3f} "
                      f"median={med:.3f} [{lo:.3f}, {hi:.3f}]")

    pd.DataFrame(rows).to_csv("tau_bootstrap.csv", index=False)
    print("wrote tau_bootstrap.csv")


if __name__ == "__main__":
    main()
