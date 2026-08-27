"""Cross-check of the success GLMM (supervisor request, item 5).

The thesis fits the success model with statsmodels' BinomialBayesMixedGLM
via mean-field variational Bayes (fit_vb), which is known to underestimate
posterior variance. This script rebuilds the EXACT model frame used by
stats_models.py for the sfm layer (combo unit) and

  step export : writes model_frame_sfm_combo.csv for glmmTMB
  step compare: merges glmmTMB estimates (from glmmtmb_check.R) with the
                published VB estimates and reports agreement

Run:
  python analysis/glmmtmb_check.py export --results results_sfm/peds_sfm
  Rscript analysis/glmmtmb_check.R <outdir>
  python analysis/glmmtmb_check.py compare --results results_sfm/peds_sfm
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import stats_models as sm_mod


def build_frame(results: Path, reference: str = "dwa"):
    df = sm_mod.load_rows(results)
    if "termination_reason" in df.columns:
        infra_mask = (df["termination_reason"].astype(str)
                      .str.split(":").str[0].isin(sm_mod.INFRA_REASONS))
    else:
        infra_mask = pd.Series(False, index=df.index)
    df = df[~infra_mask].copy()
    # combo unit, exactly as stats_models.main() does it
    df["algorithm"] = (df["global_planner"].astype(str) + "+"
                       + df["algorithm"].astype(str))
    df["global_planner"] = "combined"
    if reference not in set(df["algorithm"]):
        cands = sorted(a for a in set(df["algorithm"])
                       if a.endswith("+" + reference)
                       or a.startswith(reference + "+"))
        reference = cands[0] if cands else sorted(set(df["algorithm"]))[0]
    # covariate selection incl. the collinearity guard, via the same code
    fixed = sm_mod._fixed_formula(df, reference)
    covs = re.findall(r"standardize\(([^)]+)\)", fixed)
    extras = [e for e in ("mode", "reactive_peds")
              if df[e].nunique() > 1]
    keep = (["success", "algorithm", "seed", "cell_map", "task"]
            + extras + covs)
    frame = df[keep].copy()
    frame["success"] = frame["success"].astype(int)
    frame["map_task"] = (frame["cell_map"].astype(str) + ":"
                         + frame["task"].astype(str))
    return frame, reference, covs, extras


def do_export(results: Path):
    frame, reference, covs, extras = build_frame(results)
    out = results / "stats_combo"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "model_frame_sfm_combo.csv", index=False)
    meta = {"reference": reference, "covariates": ",".join(covs),
            "extras": ",".join(extras), "n": len(frame)}
    pd.Series(meta).to_csv(out / "model_frame_meta.csv")
    print(f"exported {len(frame)} rows, reference={reference}, "
          f"covariates={covs}, extras={extras}")


def do_compare(results: Path):
    out = results / "stats_combo"
    vb = pd.read_csv(out / "success_glmm_odds_ratios.csv")
    tmb = pd.read_csv(out / "glmmtmb_fixed_effects.csv")
    # term name harmonisation: statsmodels "C(algorithm)[T.astar+orca]"
    # vs R "algorithmastar+orca"; "standardize(x)" vs "scale_x"
    def key(t):
        t = str(t)
        m = re.match(r"C\(algorithm\)\[T\.(.+)\]$", t)
        if m:
            return "algo:" + m.group(1)
        m = re.match(r"algorithm(.+)$", t)
        if m:
            return "algo:" + m.group(1)
        m = re.match(r"standardize\((.+)\)$", t)
        if m:
            return "cov:" + m.group(1)
        m = re.match(r"scale_(.+)$", t)
        if m:
            return "cov:" + m.group(1)
        if t in ("Intercept", "(Intercept)"):
            return "intercept"
        return "other:" + t

    vb = vb.assign(k=vb["term"].map(key))
    vb["vb_sd"] = (vb["ci_hi"] - vb["ci_lo"]) / (2 * 1.96)
    tmb = tmb.assign(k=tmb["term"].map(key))
    merged = vb.merge(tmb, on="k", suffixes=("_vb", "_tmb"))
    algo = merged[merged["k"].str.startswith("algo:")]
    r = np.corrcoef(algo["log_odds"], algo["estimate"])[0, 1]
    mad = float(np.mean(np.abs(algo["log_odds"] - algo["estimate"])))
    se_ratio = float(np.median(algo["vb_sd"] / algo["se"]))
    cols = ["k", "log_odds", "vb_sd", "estimate", "se"]
    merged[cols].to_csv(out / "glmmtmb_vs_vb_sfm.csv", index=False)
    vcp = pd.read_csv(out / "success_glmm_variance_components.csv")
    tvc = pd.read_csv(out / "glmmtmb_variance_components.csv")
    lines = [
        f"VB (paper) vs glmmTMB refit, sfm success model, "
        f"{len(algo)} combination fixed effects",
        f"corr(log-odds) = {r:.4f}",
        f"mean |log-odds diff| = {mad:.3f}",
        f"median SD ratio VB/glmmTMB = {se_ratio:.3f} "
        f"(< 1 means VB understates uncertainty)",
        "",
        "variance-component SDs, VB (paper):",
        vcp.to_string(index=False),
        "variance-component SDs, glmmTMB:",
        tvc.to_string(index=False),
    ]
    text = "\n".join(lines)
    (out / "glmmtmb_agreement_summary.txt").write_text(text,
                                                      encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "compare"])
    ap.add_argument("--results", default="results_sfm/peds_sfm")
    args = ap.parse_args()
    if args.step == "export":
        do_export(Path(args.results))
    else:
        do_compare(Path(args.results))


if __name__ == "__main__":
    main()
