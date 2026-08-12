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
import sys
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


def _effective_reference(d: pd.DataFrame, reference: str) -> str:
    """Return a reference level that is actually PRESENT in this subset.

    fit_lmm restricts to successful runs and to finite endog values, which
    can leave the requested reference algorithm with zero observations.
    Forcing it as the baseline anyway makes patsy emit a baseline level with
    no data: the design matrix becomes rank-deficient and every reported
    contrast is taken against an empty cell.
    """
    present = set(d["algorithm"].astype(str))
    if reference in present:
        return reference
    fallback = d["algorithm"].astype(str).value_counts().idxmax()
    print(f"note: reference '{reference}' has no rows in this subset "
          f"({sorted(present)}); using '{fallback}' as reference instead")
    return fallback


def _fixed_formula(df: pd.DataFrame, reference: str) -> str:
    reference = _effective_reference(df, reference)
    df["algorithm"] = pd.Categorical(
        df["algorithm"],
        categories=[reference] + sorted(set(df["algorithm"]) - {reference}))
    terms = ["C(algorithm)"]
    for extra in ("mode", "global_planner", "reactive_peds"):
        if df[extra].nunique() > 1:
            terms.append(f"C({extra})")
    kept = []
    for cont in ("task_path_length_m", "task_n_turns",
                 "task_min_sidewalk_width_m",
                 "task_n_signalised_junctions", "osm_mode_flow_ph"):
        if cont in df.columns and pd.api.types.is_numeric_dtype(df[cont]) \
                and df[cont].nunique() > 1:
            df[cont] = df[cont].fillna(df[cont].mean())
            # collinearity guard: with few distinct tasks the geometry
            # features are (near-)perfectly correlated with each other;
            # keep only one representative per correlated cluster so the
            # mixed models stay identifiable
            if any(abs(df[cont].corr(df[k])) > 0.95 for k in kept):
                continue
            kept.append(cont)
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


def _abs_means_forest(d, endog, out):
    """Per-algorithm absolute mean +/- 95% CI (no reference algorithm)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # absolute-level forest: per-algorithm mean +/- 95% CI (no reference)
    g = d.groupby("algorithm")[endog]
    means = g.mean()
    sem = g.std(ddof=1) / np.sqrt(g.count())
    ci95 = 1.96 * sem.fillna(0.0)
    order = means.sort_values().index.tolist()
    abs_tab = pd.DataFrame({
        "algorithm": order,
        "n": g.count().reindex(order).values,
        "mean": means.reindex(order).round(3).values,
        "ci_lo": (means - ci95).reindex(order).round(3).values,
        "ci_hi": (means + ci95).reindex(order).round(3).values,
    })
    abs_tab.to_csv(out / f"means_{endog}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1.8))
    y = range(len(order))
    ax.errorbar(means.reindex(order).values, y,
                xerr=ci95.reindex(order).values,
                fmt="s", capsize=4, color="#1565c0")
    ax.set_yticks(list(y), order)
    ax.set_xlabel(f"{endog}: mean per algorithm (95% CI)")
    ax.set_title(f"Absolute means: {endog}"
                 + (f" (n={len(d)}, small sample)" if len(d) < 20 else ""))
    fig.tight_layout()
    fig.savefig(out / f"means_{endog}_forest.png", dpi=150)
    plt.close(fig)


def fit_lmm(df, endog, reference, out, subset_success=False):
    """Fit a linear mixed model, degrading gracefully on small samples.

    Returns (fit, note).  fit is None only when there is truly nothing to
    fit; note explains what happened (subset size, fallback used, ...).
    """
    import statsmodels.formula.api as smf
    d = df.copy()
    if subset_success:
        d = d[d["success"] == 1]
    d = d[np.isfinite(d[endog])]
    n = len(d)
    if n >= 2 and d["algorithm"].nunique() >= 1:
        try:
            _abs_means_forest(d, endog, out)
        except Exception:
            pass
    if n < 8 or d["algorithm"].nunique() < 2:
        return None, (f"skipped: only {n} usable rows "
                      f"({d['algorithm'].nunique()} algorithms)"
                      + (" after restricting to successful runs"
                         if subset_success else "")
                      + " - need >=8 rows and >=2 algorithms; "
                        "run more seeds/maps")
    # the reference must exist AFTER the success/finite subsetting above,
    # otherwise the Treatment() fallbacks below reference an absent level
    reference = _effective_reference(d, reference)
    fixed_full = _fixed_formula(d, reference)
    vc = {}
    if d["cell_map"].nunique() > 1:
        vc["map"] = "0 + C(cell_map)"
    if d["task"].nunique() > 1:
        vc["task"] = "0 + C(task)"
    groups = d["seed"].astype(str) if d["seed"].nunique() > 1 \
        else pd.Series(["g"] * len(d), index=d.index)

    attempts = [(fixed_full, vc if vc else None, "LMM"),
                (fixed_full, None, "LMM (no variance components)"),
                (f"C(algorithm, Treatment('{reference}'))", None,
                 "LMM (algorithm-only fixed effects)")]
    last_exc = None
    for fixed, vcf, label in attempts:
        try:
            model = smf.mixedlm(f"{endog} ~ {fixed}", d, groups=groups,
                                vc_formula=vcf)
            fit = model.fit(reml=True, method="lbfgs")
            if not np.all(np.isfinite(fit.params.values)):
                raise ValueError("non-finite coefficients")
            note = label + (f" (small sample: n={n})" if n < 20 else "")
            break
        except Exception as exc:
            last_exc = exc
            fit = None
    if fit is None:
        # final fallback: plain OLS, clearly labelled
        try:
            model = smf.ols(
                f"{endog} ~ C(algorithm, Treatment('{reference}'))", d)
            fit = model.fit()
            note = (f"OLS fallback (mixed model failed: {last_exc}; "
                    f"n={n}, seeds treated as independent)")
        except Exception as exc:
            return None, f"skipped: {exc}"
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
    ar = res[res["term"].str.startswith("C(algorithm")].copy()
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
        ax.set_title(f"{note}: {endog}")
        fig.tight_layout()
        fig.savefig(out / f"lmm_{endog}_forest.png", dpi=150)
        plt.close(fig)

    return fit, note


def failure_taxonomy(df, out):
    REASONS = ["goal", "collision", "max_time", "stalled",
               "global_plan_failed", "planner_error"]
    df = df.copy()
    # the runner emits "planner_error:<ExcType>"; collapse the exception type
    # so these runs land in a real taxonomy category instead of a long tail
    # that the stacked bar silently omits (percentages then miss those runs)
    df["termination_reason"] = (df["termination_reason"].astype(str)
                                .str.split(":").str[0])
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
    # plot EVERY reason present, not a hard-coded five: an unlisted category
    # made the bars silently sum to less than 100% of runs
    reasons = REASONS + [c for c in tab.columns
                         if c != "n" and not c.endswith("_pct")
                         and c not in REASONS]
    reasons = [r for r in reasons if r in tab.columns]
    algos = tab.index.tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(algos))
    colors = {"goal": "#2e7d32", "collision": "#c62828",
              "max_time": "#f9a825", "stalled": "#6a1b9a",
              "global_plan_failed": "#455a64",
              "planner_error": "#00838f"}
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
    from scipy.stats import rankdata
    ranks = np.zeros((B, len(algos)), dtype=float)
    top1 = np.zeros(len(algos))
    for b in range(B):
        take = rng.choice(len(seeds), size=len(seeds), replace=True)
        m = per.iloc[take].mean(axis=0).reindex(algos).values.astype(float)
        # an algorithm absent from this resample ranks last, not first
        m = np.where(np.isnan(m), -np.inf, m)
        # AVERAGE ranks, and top-1 credit SHARED among ties. Exact ties are
        # common here (whole resamples where several algorithms score 0.0 or
        # 1.0). A stable argsort broke those ties by array position -- i.e.
        # alphabetically -- which fabricated a definitive winner with
        # P(top-1)=1.0 and zero-width rank CIs, the exact statistic this
        # table exists to report honestly.
        ranks[b] = rankdata(-m, method="average")
        best = np.flatnonzero(m == m.max())
        top1[best] += 1.0 / len(best)
    rows = []
    for i, a in enumerate(algos):
        rows.append({
            "algorithm": a,
            "mean_success": round(float(per[a].mean()), 3),
            # float: average ranks are fractional when algorithms tie
            "rank_median": round(float(np.median(ranks[:, i])), 2),
            "rank_ci_lo": round(float(np.percentile(ranks[:, i], 2.5)), 2),
            "rank_ci_hi": round(float(np.percentile(ranks[:, i], 97.5)), 2),
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
    ap.add_argument("--unit", choices=["algorithm", "combo", "both"],
                    default="both",
                    help="comparison unit: local algorithm, "
                         "global+local combination, or both (default)")
    args = ap.parse_args()
    if args.unit == "both":
        import subprocess
        for u in ("algorithm", "combo"):
            subprocess.run([sys.executable, __file__,
                            "--results", args.results,
                            "--reference", args.reference,
                            "--bootstrap", str(args.bootstrap),
                            "--unit", u], check=True)
        return
    results = Path(args.results)
    df = load_rows(results)
    if args.unit == "combo":
        df["algorithm"] = (df["global_planner"].astype(str) + "+"
                           + df["algorithm"].astype(str))
        df["global_planner"] = "combined"   # absorbed into the unit
        out = results / "stats_combo"
    else:
        out = results / "stats"
    out.mkdir(parents=True, exist_ok=True)
    if args.reference not in set(df["algorithm"]):
        # reference may be a local algo, a global planner, or a full combo
        cands = sorted(a for a in set(df["algorithm"])
                       if a.endswith("+" + args.reference)
                       or a.startswith(args.reference + "+"))
        args.reference = cands[0] if cands else sorted(set(df["algorithm"]))[0]
    lines = [f"unit = {args.unit}",
             f"n runs = {len(df)}; units = {sorted(set(df['algorithm']))}",
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
                          ("min_pedestrian_distance_m", False),
                          ("ped_delay_s_mean", False),
                          ("ped_deflection_m_mean", False),
                          ("ped_personal_space_s_total", False),
                          ("social_work", False),
                          ("social_force_on_agents", False),
                          ("social_force_on_robot", False)):
        if endog not in df.columns:
            continue
        try:
            fit, note = fit_lmm(df, endog, args.reference, out,
                                subset_success=subset)
            lines.append(f"== LMM {endog}"
                         f"{' (successful runs)' if subset else ''} ==")
            lines.append(f"[{note}]")
            if fit is not None:
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
