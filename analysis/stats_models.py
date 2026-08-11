#!/usr/bin/env python3
"""Statistical analysis of benchmark results (supervisor protocol).

Implements the analysis mandated in the 2026-08 feedback:

  * success            -> binomial GLMM (variational), random intercepts for
                          seed and map, fixed effects algorithm + mode
                          (+ global planner / task when present)
  * sim_time_s (goal
    runs), path_length -> linear mixed models, same random structure
  * effect sizes       -> odds ratios / coefficients with 95% CIs against a
                          reference algorithm (default: dwa)
  * failure taxonomy   -> per-algorithm termination_reason breakdown
  * ranking stability  -> bootstrap over seeds: rank intervals and P(top-1)

Input: a results directory containing summary_all.csv (written by
benchmark_batch.py) or per-run robot_metrics.json files.

Output: <results>/stats/*.csv + model_summaries.txt + two figures.

Usage:
    python analysis/stats_models.py --results results
    python analysis/stats_models.py --results results --reference dwa
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_rows(results: Path) -> pd.DataFrame:
    """Adds task geometry features (from configs/tasks_<map>.json) as
    columns task_* -- entered as standardized fixed effects so the models
    report WHICH topology properties drive performance changes."""
    csv = results / "summary_all.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        rows = [json.loads(f.read_text())
                for f in results.glob("*/*/*/seed_*/robot_metrics.json")]
        if not rows:
            raise SystemExit(f"no results under {results}")
        df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(bool).astype(int)
    for col, default in (("route", "default"), ("global_planner", "fixed"),
                         ("reactive_peds", "off"), ("task", "t0")):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    # merge per-task geometry features (Option B protocol)
    feats = {}
    for mp in df["map"].unique():
        tf = results.parent / "configs" / f"tasks_{mp}.json"
        tf2 = Path("configs") / f"tasks_{mp}.json"
        use = tf if tf.exists() else tf2
        if use.exists():
            for t in json.loads(use.read_text())["tasks"]:
                feats[(mp, t["id"])] = t
    FEATS = ["path_length_m", "n_turns", "min_sidewalk_width_m",
             "n_signalised_junctions"]
    for c in FEATS:
        col = f"task_{c}"
        df[col] = [feats.get((m, t), {}).get(c)
                   for m, t in zip(df["map"], df["task"])]
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mean())
        else:
            df.drop(columns=[col], inplace=True)
    df["cell_map"] = df.apply(
        lambda r: r["map"] if r["route"] in ("default",)
        else f"{r['map']}[{r['route']}]", axis=1)
    return df


def _fixed_formula(df: pd.DataFrame, reference: str) -> str:
    df["algorithm"] = pd.Categorical(
        df["algorithm"],
        categories=[reference] + sorted(set(df["algorithm"]) - {reference}))
    terms = ["C(algorithm)"]
    for extra in ("mode", "global_planner", "reactive_peds"):
        if df[extra].nunique() > 1:
            terms.append(f"C({extra})")
    for cont in ("task_path_length_m", "task_n_turns",
                 "task_min_sidewalk_width_m",
                 "task_n_signalised_junctions", "osm_mode_flow_ph"):
        if cont in df.columns and pd.api.types.is_numeric_dtype(df[cont]) \
                and df[cont].nunique() > 1:
            df[cont] = df[cont].fillna(df[cont].mean())
            terms.append(f"standardize({cont})")
    return " + ".join(terms)


def fit_success_glmm(df, reference, out):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    fixed = _fixed_formula(df, reference)
    vc = {}
    if df["seed"].nunique() > 1:
        vc["seed"] = "0 + C(seed)"
    if df["cell_map"].nunique() > 1:
        vc["map"] = "0 + C(cell_map)"
    if df["task"].nunique() > 1:
        vc["task"] = "0 + C(task)"
    model = BinomialBayesMixedGLM.from_formula(
        f"success ~ {fixed}", vc, df)
    fit = model.fit_vb()
    names = model.exog_names
    means = fit.fe_mean
    sds = fit.fe_sd
    rows = []
    for name, m, s in zip(names, means, sds):
        lo, hi = m - 1.96 * s, m + 1.96 * s
        rows.append({
            "term": name, "log_odds": round(m, 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "odds_ratio": round(float(np.exp(m)), 3),
            "or_ci_lo": round(float(np.exp(lo)), 3),
            "or_ci_hi": round(float(np.exp(hi)), 3),
        })
    res = pd.DataFrame(rows)
    res.to_csv(out / "success_glmm_odds_ratios.csv", index=False)
    vcp = pd.DataFrame({
        "component": list(vc.keys()),
        # vcp_mean is the posterior mean of LOG-sd -> report sd = exp(.)
        "sd_posterior_mean": [round(float(np.exp(x)), 3)
                              for x in fit.vcp_mean],
    })
    vcp.to_csv(out / "success_glmm_variance_components.csv", index=False)
    # forest plot: odds ratios with 95% CI (algorithm terms), log scale
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ar = res[res["term"].str.startswith("C(algorithm)")].copy()
    if len(ar):
        ar["label"] = (ar["term"].str.extract(r"\[T\.(.+)\]")[0]
                       .fillna(ar["term"]))
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(ar) + 1.8))
        y = range(len(ar))
        ax.errorbar(ar["odds_ratio"], y,
                    xerr=[ar["odds_ratio"] - ar["or_ci_lo"],
                          ar["or_ci_hi"] - ar["odds_ratio"]],
                    fmt="o", capsize=4, color="#1565c0")
        ax.axvline(1.0, color="0.4", ls="--", lw=1)
        ax.set_yticks(list(y), ar["label"])
        ax.set_xscale("log")
        ax.set_xlabel("odds ratio vs reference (95% CI, log scale)")
        ax.set_title("Success GLMM: algorithm effect sizes")
        fig.tight_layout()
        fig.savefig(out / "success_glmm_forest.png", dpi=150)
        plt.close(fig)
    return fit, res, vcp


def fit_lmm(df, endog, reference, out, subset_success=False):
    import statsmodels.formula.api as smf
    d = df.copy()
    if subset_success:
        d = d[d["success"] == 1]
    d = d[np.isfinite(d[endog])]
    if len(d) < 20 or d["algorithm"].nunique() < 2:
        return None
    fixed = _fixed_formula(d, reference)
    vc = {}
    if d["cell_map"].nunique() > 1:
        vc["map"] = "0 + C(cell_map)"
    if d["task"].nunique() > 1:
        vc["task"] = "0 + C(task)"
    groups = d["seed"].astype(str) if d["seed"].nunique() > 1 \
        else pd.Series(["g"] * len(d), index=d.index)
    model = smf.mixedlm(f"{endog} ~ {fixed}", d, groups=groups,
                        vc_formula=vc if vc else None)
    fit = model.fit(reml=True, method="lbfgs")
    ci = fit.conf_int()
    res = pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.round(3).values,
        "ci_lo": ci[0].round(3).values,
        "ci_hi": ci[1].round(3).values,
        "p": fit.pvalues.round(4).values,
    })
    res.to_csv(out / f"lmm_{endog}.csv", index=False)
    # coefficient forest for algorithm terms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ar = res[res["term"].str.startswith("C(algorithm)")].copy()
    if len(ar):
        ar["label"] = (ar["term"].str.extract(r"\[T\.(.+)\]")[0]
                       .fillna(ar["term"]))
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(ar) + 1.8))
        y = range(len(ar))
        ax.errorbar(ar["coef"], y,
                    xerr=[ar["coef"] - ar["ci_lo"],
                          ar["ci_hi"] - ar["coef"]],
                    fmt="s", capsize=4, color="#2e7d32")
        ax.axvline(0.0, color="0.4", ls="--", lw=1)
        ax.set_yticks(list(y), ar["label"])
        ax.set_xlabel(f"{endog}: coefficient vs reference (95% CI)")
        ax.set_title(f"LMM {endog}: algorithm effects")
        fig.tight_layout()
        fig.savefig(out / f"lmm_{endog}_forest.png", dpi=150)
        plt.close(fig)
    return fit


def failure_taxonomy(df, out):
    REASONS = ["goal", "collision", "max_time", "stalled",
               "global_plan_failed"]
    tab = (df.groupby(["algorithm", "termination_reason"]).size()
             .unstack(fill_value=0))
    for r in REASONS:                       # full taxonomy, zero-filled
        if r not in tab.columns:
            tab[r] = 0
    tab = tab[[c for c in REASONS if c in tab.columns]
              + [c for c in tab.columns if c not in REASONS]]
    tab["n"] = tab.sum(axis=1)
    for c in [c for c in tab.columns if c != "n"]:
        tab[f"{c}_pct"] = (100.0 * tab[c] / tab["n"]).round(1)
    tab.to_csv(out / "failure_taxonomy.csv")
    # stacked bar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reasons = ["goal", "collision", "max_time", "stalled",
               "global_plan_failed"]
    algos = tab.index.tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(algos))
    colors = {"goal": "#2e7d32", "collision": "#c62828",
              "max_time": "#f9a825", "stalled": "#6a1b9a",
              "global_plan_failed": "#455a64"}
    for r in reasons:
        vals = (tab[r] / tab["n"] * 100.0).values
        ax.bar(algos, vals, bottom=bottom, label=r,
               color=colors.get(r, "0.5"))
        bottom += vals
    ax.set_ylabel("% of runs")
    ax.set_title("Failure taxonomy by algorithm")
    ax.legend(ncols=len(reasons), fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / "failure_taxonomy.png", dpi=150)
    plt.close(fig)
    return tab


def ranking_stability(df, out, B=2000, rng_seed=0):
    """Bootstrap over SEEDS: how stable is the success-rate ranking?"""
    rng = np.random.default_rng(rng_seed)
    seeds = sorted(df["seed"].unique())
    algos = sorted(df["algorithm"].unique())
    per = df.pivot_table(index="seed", columns="algorithm",
                         values="success", aggfunc="mean")
    per = per.reindex(seeds)
    ranks = np.zeros((B, len(algos)), dtype=int)
    top1 = np.zeros(len(algos))
    for b in range(B):
        take = rng.choice(len(seeds), size=len(seeds), replace=True)
        m = per.iloc[take].mean(axis=0).reindex(algos).values
        order = (-m).argsort(kind="stable")
        rk = np.empty(len(algos), dtype=int)
        rk[order] = np.arange(1, len(algos) + 1)
        ranks[b] = rk
        top1[order[0]] += 1
    rows = []
    for i, a in enumerate(algos):
        rows.append({
            "algorithm": a,
            "mean_success": round(float(per[a].mean()), 3),
            "rank_median": int(np.median(ranks[:, i])),
            "rank_ci_lo": int(np.percentile(ranks[:, i], 2.5)),
            "rank_ci_hi": int(np.percentile(ranks[:, i], 97.5)),
            "P_top1": round(float(top1[i] / B), 3),
        })
    res = pd.DataFrame(rows).sort_values("rank_median")
    res.to_csv(out / "ranking_stability.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    res2 = res.reset_index(drop=True)
    ax.errorbar(res2.index, res2["rank_median"],
                yerr=[res2["rank_median"] - res2["rank_ci_lo"],
                      res2["rank_ci_hi"] - res2["rank_median"]],
                fmt="o", capsize=4)
    for i, r in res2.iterrows():
        ax.annotate(f"P(top1)={r['P_top1']:.2f}",
                    (i, r["rank_median"]), textcoords="offset points",
                    xytext=(8, -4), fontsize=7, color="0.3")
    ax.set_xticks(res2.index, res2["algorithm"], rotation=30, ha="right")
    ax.set_ylabel("success-rate rank (bootstrap 95% CI)")
    ax.invert_yaxis()
    ax.set_title("Ranking stability across seed resamples")
    fig.tight_layout()
    fig.savefig(out / "ranking_stability.png", dpi=150)
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--reference", default="dwa",
                    help="reference algorithm for effect sizes")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    results = Path(args.results)
    out = results / "stats"
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(results)
    if args.reference not in set(df["algorithm"]):
        args.reference = sorted(df["algorithm"])[0]
    lines = [f"n runs = {len(df)}; algorithms = {sorted(set(df['algorithm']))}",
             f"maps = {sorted(set(df['cell_map']))}; "
             f"modes = {sorted(set(df['mode']))}; "
             f"seeds = {df['seed'].nunique()}",
             f"reference algorithm = {args.reference}", ""]

    # 1) success GLMM
    try:
        fit, res, vcp = fit_success_glmm(df, args.reference, out)
        lines.append("== Binomial GLMM (success) ==")
        lines.append(res.to_string(index=False))
        lines.append("variance components (posterior sd):")
        lines.append(vcp.to_string(index=False))
    except Exception as exc:
        lines.append(f"success GLMM skipped: {exc}")
    lines.append("")

    # 2) linear mixed models
    for endog, subset in (("sim_time_s", True), ("path_length_m", True),
                          ("min_pedestrian_distance_m", False)):
        if endog not in df.columns:
            continue
        try:
            fit = fit_lmm(df, endog, args.reference, out,
                          subset_success=subset)
            if fit is not None:
                lines.append(f"== LMM {endog}"
                             f"{' (successful runs)' if subset else ''} ==")
                lines.append(str(fit.summary().tables[1]))
                lines.append("")
        except Exception as exc:
            lines.append(f"LMM {endog} skipped: {exc}\n")

    # 3) failure taxonomy + 4) ranking stability
    tab = failure_taxonomy(df, out)
    lines.append("== Failure taxonomy (% of runs) ==")
    lines.append(tab.to_string())
    lines.append("")
    res = ranking_stability(df, out, B=args.bootstrap)
    lines.append("== Ranking stability (bootstrap over seeds) ==")
    lines.append(res.to_string(index=False))

    (out / "model_summaries.txt").write_text("\n".join(lines))
    print(f"stats -> {out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
