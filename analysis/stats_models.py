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

import matplotlib
_BOLD = {"font.weight": "bold", "axes.titleweight": "bold",
         "axes.labelweight": "bold", "figure.titleweight": "bold"}
matplotlib.rcParams.update(_BOLD)   # every title/label/tick/legend in bold

warnings.filterwarnings("ignore")
# statsmodels force-resets ConvergenceWarning to "always" at its own import,
# overriding the blanket ignore above -- re-silence it AFTER that import.
# The verdict on any non-converged fit still surfaces through the explicit
# [NON-CONVERGED] tag in figure titles and model_summaries.txt.
try:
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning as _SMConvWarn,
        ValueWarning as _SMValWarn)
    warnings.simplefilter("ignore", _SMConvWarn)
    # fires while summary() builds an omnibus F we discard (the 57-dim
    # joint test is undefined under cluster-robust cov; per-coefficient
    # stats are 1-dim constraints and unaffected)
    warnings.simplefilter("ignore", _SMValWarn)
except Exception:
    pass

# Infrastructure outcomes are measurements of the RIG, not of an algorithm:
# they are excluded from every model/table below, with printed accounting
# (same convention as quicklook.py / the audit protocol).
INFRA_REASONS = {"sumo_crash"}   # global_plan_failed now counts AGAINST the combination (user ruling): the global half failing to route is a failure of the deployed stack, reported as its own category

# Endogs modelled on log1p a priori: non-negative and right-skewed
# (durations, forces, exposure totals). Keeps the reported scale identical
# across the 18-unit and 54-unit views instead of convergence-dependent.
_LOG_SCALE_ENDOGS = {
    "sim_time_s", "path_length_m", "ped_delay_s_mean",
    "ped_deflection_m_mean", "ped_personal_space_s_total",
    "social_work", "social_force_on_agents", "social_force_on_robot",
}

_NAME_MODE = "short"          # --names full: written-out algorithm names
_LAYOUT = "page"              # --layout twocol: design at IEEE \textwidth
_SAVE_PDF = False             # --pdf: vector twin next to every png


def _savefig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if _SAVE_PDF:
        fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")

try:                          # single naming source when deployed alongside
    from algo_names import display as _algo_short            # noqa: E402
except ImportError:           # graceful fallback: unknown ids pass through
    _algo_short = None
_FALLBACK_DISP = {
        "dwa": "DWA", "astar": "A*", "dijkstra": "Dijkstra", "rrt": "RRT",
        "orca_heuristic": "ORCA (heuristic)", "mpc": "MPC", "teb": "TEB",
        "sarl": "SARL", "cadrl": "CADRL", "lstm_rl": "LSTM-RL",
        "orca": "ORCA (RVO2)", "mpc_dompc": "MPC (do-mpc)",
        "teb_upstream": "TEB (upstream)", "sarl_upstream": "SARL (upstream)",
        "cadrl_upstream": "CADRL (upstream)",
        "lstm_rl_upstream": "LSTM-RL (upstream)",
        "crowdnav_dsrnn": "CrowdNav DS-RNN",
        "crowdnav_attngraph": "CrowdNav AttnGraph",
    }

_FALLBACK_FULL = {
    "dwa": "Dynamic Window Approach (DWA)",
    "astar": "A* (search)", "dijkstra": "Dijkstra (search)",
    "rrt": "Rapidly-exploring Random Tree (RRT)",
    "orca_heuristic": "Optimal Reciprocal Collision Avoidance (heuristic)",
    "orca": "Optimal Reciprocal Collision Avoidance (RVO2)",
    "mpc": "Model Predictive Control (MPC)",
    "mpc_dompc": "Model Predictive Control (do-mpc)",
    "teb": "Timed Elastic Band (TEB)",
    "teb_upstream": "Timed Elastic Band (upstream)",
    "sarl": "Socially Attentive RL (SARL)",
    "sarl_upstream": "Socially Attentive RL (upstream)",
    "cadrl": "Collision Avoidance with Deep RL (CADRL)",
    "cadrl_upstream": "Collision Avoidance with Deep RL (upstream)",
    "lstm_rl": "LSTM-based RL (LSTM-RL)",
    "lstm_rl_upstream": "LSTM-based RL (upstream)",
    "crowdnav_dsrnn": "CrowdNav Decentralized Structural RNN (DS-RNN)",
    "crowdnav_attngraph": "CrowdNav Attention Graph (AttnGraph)",
}

if _algo_short is None:
    def _algo_short(u):                                      # noqa: E301
        return _FALLBACK_DISP.get(u, u)
try:
    from algo_names import full as _algo_full                # noqa: E402
except Exception:
    def _algo_full(u):
        return _FALLBACK_FULL.get(u, _algo_short(u))


def _algo_display(u):
    return _algo_full(u) if _NAME_MODE == "full" else _algo_short(u)


try:
    from algo_names import global_display as _global_disp   # noqa: E402
except ImportError:
    def _global_disp(u):
        return _algo_display(u)


def disp(u) -> str:
    """Display name for a unit: plain local algo or 'global+local' combo.
    The global half stays bare (textbook searches); the local half may
    carry the (ours) mark."""
    u = str(u)
    if "+" in u:
        g, a = u.split("+", 1)
        return f"{_global_disp(g)} + {_algo_display(a)}"
    return _algo_display(u)


ENDOG_LABEL = {
    "sim_time_s": "Travel Time [s]",
    "path_length_m": "Path Length [m]",
    "min_pedestrian_distance_m": "Min. Pedestrian Distance [m]",
    "ped_delay_s_mean": "Mean Pedestrian Delay [s]",
    "ped_deflection_m_mean": "Mean Pedestrian Deflection [m]",
    "ped_personal_space_s_total": "Personal-Space Intrusion [s]",
    "social_work": "Social Work",
    "social_force_on_agents": "Social Force on Pedestrians",
    "social_force_on_robot": "Social Force on Robot",
}


def endog_disp(e) -> str:
    """Paper-facing metric label with units; unwraps np.log1p(...)."""
    e = str(e)
    if e.startswith("np.log1p(") and e.endswith(")"):
        inner = e[len("np.log1p("):-1]
        return f"log(1+{ENDOG_LABEL.get(inner, inner)})"
    return ENDOG_LABEL.get(e, e)


def wilson(k: float, n: int, z: float = 1.96):
    """(p, lo, hi) Wilson 95% interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return p, max(0.0, c - h), min(1.0, c + h)



def _forest_figure(labels, est, lo, hi, values, *, null, logx, pos_col,
                   neg_col, xlabel, title, caption, value_header,
                   out_path, dpi, ref_label=None, ref_val="",
                   noise_bands=None, ref_lines=None):
    """Publication forest: rows sorted by effect size, significance
    colouring (CI clear of the null), alternating bands, axis clipped to a
    readable window (</> marks off-scale estimates) and a numeric
    'est [95% CI]' column on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.DataFrame({"label": list(labels),
                      "est": np.asarray(est, float),
                      "lo": np.asarray(lo, float),
                      "hi": np.asarray(hi, float),
                      "val": list(values)}).dropna(subset=["est"])
    d["_ref"] = False
    if ref_label is not None:      # the baseline itself, pinned at the null
        d = pd.concat([d, pd.DataFrame([{
            "label": ref_label, "est": float(null), "lo": float(null),
            "hi": float(null), "val": ref_val, "_ref": True}])],
            ignore_index=True)
    d = d.sort_values("est", ascending=False).reset_index(drop=True)
    n = len(d)
    # adaptive density: a 53-row forest at 0.34 in/row would be ~20 in tall;
    # tighten rows and fonts past ~30 units so the figure stays one page.
    # twocol layout designs at IEEE \textwidth (7.05 in) with FINAL point
    # sizes, to be included 1:1 -- no downscaling, no shrinking fonts.
    if _LAYOUT == "twocol":
        fig_w = 7.05
        row_h = max(0.128, min(0.30, 7.3 / max(n, 1)))
        fs_lab = 8.5 if n <= 26 else 7.6 if n <= 40 else 7.0
        fs_val = max(6.8, fs_lab - 0.2)
    else:
        fig_w = 11.2
        row_h = 0.34 if n <= 30 else max(0.185, 10.5 / n)
        fs_lab = 8.5 if n <= 30 else 7.0
        fs_val = 8.4 if n <= 30 else 7.2
    fig, ax = plt.subplots(figsize=(fig_w, row_h * n + 2.0))
    if logx:
        xmin = max(1e-3, min(float(d["est"].min()) * 0.4, null / 1.5))
        xmax = min(1e3, max(float(d["est"].max()) * 2.5, null * 1.5))
    else:
        span = float(d["hi"].max() - d["lo"].min()) or 1.0
        xmin = float(d["lo"].min()) - 0.04 * span
        xmax = float(d["hi"].max()) + 0.04 * span
    tf = ax.get_yaxis_transform()
    for i, r in d.iterrows():
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
        if r["_ref"]:
            ax.plot([null], [i], marker="D", ms=5 if n <= 30 else 4,
                    mfc="white", mec="0.3", color="0.3", zorder=3)
            ax.text(1.02, i, r["val"], transform=tf, fontsize=fs_val,
                    va="center", family="monospace", color="0.35",
                    style="italic")
            continue
        col = (pos_col if r["lo"] > null else
               neg_col if r["hi"] < null else "0.55")
        px = float(np.clip(r["est"], xmin, xmax))
        mk = "<" if r["est"] < xmin else ">" if r["est"] > xmax else "o"
        ax.errorbar(px, i,
                    xerr=[[max(px - max(r["lo"], xmin), 0.0)],
                          [max(min(r["hi"], xmax) - px, 0.0)]],
                    fmt=mk, color=col,
                    ms=(5 if n <= 30 else 3.8) if _LAYOUT != "twocol"
                    else (4 if n <= 30 else 3.0),
                    elinewidth=1.2 if n <= 30 else 1.0,
                    capsize=2 if n <= 30 else 1.2, zorder=3)
        ax.text(1.02, i, r["val"], transform=tf, fontsize=fs_val,
                va="center", family="monospace", color="0.15")
    ax.text(1.02, -1.1, value_header, transform=tf, fontsize=fs_val,
            va="center", family="monospace", fontweight="bold")
    ax.axvline(null, color="0.35", ls="--", lw=1, zorder=2)
    # RQ2 overlays: shaded noise band(s) (variance components on the same
    # scale as the contrasts) and dotted reference line(s) for the
    # literature-typical effect size -- estimates inside a band are not
    # distinguishable from that noise source.
    _legend_handles = []
    from matplotlib.patches import Patch
    for b in sorted(noise_bands or [], key=lambda b: b["lo"]):
        col = b.get("color", "#f2c94c")
        ax.axvspan(b["lo"], b["hi"], color=col,
                   alpha=b.get("alpha", 0.3), zorder=1, lw=0)
        # solid edge lines in the band's own colour: with nested
        # translucent bands the fills mix, but the edges stay crisp
        for xv in (b["lo"], b["hi"]):
            ax.axvline(xv, color=col, lw=1.2, alpha=0.9, zorder=2)
        _legend_handles.append(Patch(facecolor=col,
                                     alpha=max(b.get("alpha", 0.3), 0.45),
                                     edgecolor=col, linewidth=1.2,
                                     label=b["label"]))
    for xv, lab in (ref_lines or []):
        ln = ax.axvline(xv, color="0.2", ls=":", lw=1.1, zorder=2)
        if lab:
            ln.set_label(lab)
            _legend_handles.append(ln)
    if _legend_handles:
        # below the axes, horizontal: never overlaps rows or the value
        # column, and reads as a caption line
        fig.legend(handles=_legend_handles, loc="lower center",
                   ncol=2 if _LAYOUT == "twocol" else 3,
                   fontsize=7.0 if _LAYOUT == "twocol" else 8.0,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    if logx:
        ax.set_xscale("log")
        cand = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                1, 2, 5, 10, 20, 50, 100]
        ticks = [t for t in cand if xmin <= t <= xmax] or [null]
        # narrow canvases: thin the ticks so labels never collide
        max_t = 7 if _LAYOUT == "twocol" else 11
        if len(ticks) > max_t:
            keep = {0.001, 0.01, 0.1, 1, 10, 100} | {null}
            ticks = [t for t in ticks if t in keep] or ticks[::2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks],
                           fontsize=9.5 if _LAYOUT == "twocol" else 8.5)
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.tick_params(axis="x",
                       labelsize=9.5 if _LAYOUT == "twocol" else 8.5)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(range(n), d["label"], fontsize=fs_lab)
    for t, is_ref in zip(ax.get_yticklabels(), d["_ref"]):
        if is_ref:
            t.set_style("italic")
            t.set_color("0.35")
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel(xlabel,
                  fontsize=10 if _LAYOUT == "twocol" else 9)
    # left-aligned: keeps the title clear of the value-column header
    ax.set_title(title, fontsize=11 if _LAYOUT == "twocol" else 10,
                 pad=16)               # lifted clear of the value header
    ax.grid(axis="x", lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # the caption footnote is no longer rendered on the figure (colour
    # coding and model notes belong in the thesis caption; the model
    # note itself stays in model_summaries.txt). `caption` is kept in
    # the signature so callers stay unchanged.
    del caption
    # twocol: keep the value column INSIDE the 7.05 in canvas, so the
    # tight bounding box never grows past IEEE \textwidth
    right = 0.84 if _LAYOUT == "twocol" else 1.0
    bottom = 0.05 if _legend_handles else 0.02   # room for the legend row
    fig.tight_layout(rect=(0, bottom, right, 1))
    _savefig(fig, out_path, dpi)
    plt.close(fig)


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
    # A blank success field (DictWriter writes "" for a key a run omitted)
    # makes the column object dtype, and bool("") is False but bool(nan) is
    # True -- a missing outcome would silently count as a SUCCESS. Map
    # explicitly and default the unknown case to failure.
    df["success"] = (df["success"]
                     .map({True: 1, False: 0, 1: 1, 0: 0, "True": 1,
                           "False": 0, "true": 1, "false": 0, "1": 1, "0": 0})
                     .fillna(0).astype(int))
    for col, default in (("route", "default"), ("global_planner", "fixed"),
                         ("reactive_peds", "off"), ("task", "t0")):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    # merge per-task geometry features (Option B protocol)
    feats = {}
    for mp in df["map"].unique():
        # resolve configs/ independent of the caller's CWD: a silent miss
        # here drops the four geometry covariates from every model with no
        # error (observed when the script was launched from another
        # directory), so anchor on the repo root via __file__ and only
        # then fall back to CWD-relative
        cands = [results.parent / "configs" / f"tasks_{mp}.json",
                 results.parent.parent / "configs" / f"tasks_{mp}.json",
                 Path(__file__).resolve().parent.parent / "configs"
                 / f"tasks_{mp}.json",
                 Path("configs") / f"tasks_{mp}.json"]
        use = next((p for p in cands if p.exists()), None)
        if use is not None:
            for t in json.loads(use.read_text())["tasks"]:
                feats[(mp, t["id"])] = t
        else:
            print(f"WARNING: tasks_{mp}.json not found (searched "
                  f"{[str(p) for p in cands]}); task geometry covariates "
                  f"will be missing for map {mp}")
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
    # realised demand (pedestrian density): total personFlow arrival rate
    # parsed from the demand.rou.xml SUMO actually consumed -- ONE source
    # with the same semantics on synthetic and OSM maps (the scenario-json
    # records differ between the two map families). Exogenous: drawn before
    # the run and identical for every local algorithm in the cell, unlike
    # encounter counts, which depend on the robot's own behaviour. Entered
    # as a standardized covariate, so the seed variance component measures
    # seed noise net of demand level (supervisor request: density as a
    # fixed effect). OSM demand depends on the robot's route, so the cache
    # key includes the combo directory (map__task__g-global), not just map.
    import xml.etree.ElementTree as ET

    def _rou_total_ph(p):
        # synthetic maps: <personFlow period="exp(rate)"> -> rate*3600
        # OSM maps: INDIVIDUAL <person depart=...> walkers drawn uniformly
        # over the demand window -> count / span; standing pedestrians
        # (stop-only, no personTrip/walk) are not arrival demand
        tot, n_walk, max_dep = 0.0, 0, 0.0
        try:
            for _, el in ET.iterparse(p):
                if el.tag == "personFlow":
                    per = el.get("period", "")
                    if per.startswith("exp(") and per.endswith(")"):
                        tot += float(per[4:-1]) * 3600.0
                    else:
                        try:
                            v = float(per)
                            if v > 0:
                                tot += 3600.0 / v
                        except ValueError:
                            pass
                elif el.tag == "person":
                    if any(ch.tag in ("personTrip", "walk") for ch in el):
                        n_walk += 1
                        try:
                            max_dep = max(max_dep,
                                          float(el.get("depart", "0")))
                        except ValueError:
                            pass
                el.clear()
        except (ET.ParseError, OSError):
            return None
        if n_walk >= 2 and max_dep > 0:
            tot += n_walk * 3600.0 / max_dep
        return tot

    dem = {}
    for f in results.glob("*/*/*/seed_*/demand.rou.xml"):
        sd_dir = f.parent
        key = (sd_dir.parent.parent.parent.name,   # map__task__g-global
               sd_dir.parent.parent.name,          # mode
               sd_dir.name)                        # seed_N
        if key in dem:
            continue
        v = _rou_total_ph(f)
        if v is not None:
            dem[key] = v
    if dem:
        if "mode" not in df.columns:
            df["mode"] = "all"

        def _dkey(m, t, g, mo, s):
            try:
                return dem.get((f"{m}__{t}__g-{g}",
                                mo if pd.notna(mo) else "all",
                                f"seed_{int(s)}"))
            except (TypeError, ValueError):
                return None
        df["demand_total_ph"] = [
            _dkey(*z) for z in zip(df["map"], df["task"],
                                   df["global_planner"], df["mode"],
                                   df["seed"])]
        n_miss = int(df["demand_total_ph"].isna().sum())
        if n_miss:
            print(f"note: demand_total_ph missing for {n_miss}/{len(df)} "
                  f"rows (mean-filled)")
        if df["demand_total_ph"].notna().any():
            df["demand_total_ph"] = df["demand_total_ph"].fillna(
                df["demand_total_ph"].mean())
        else:
            df.drop(columns=["demand_total_ph"], inplace=True)
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
                 "task_n_signalised_junctions", "demand_total_ph",
                 "osm_mode_flow_ph"):
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


def fit_success_glmm(df, reference, out, dpi=200):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    fixed = _fixed_formula(df, reference)
    # task is NESTED in map (supervisor item 6): t04 on map1 and t04 on
    # map5 are different routes that merely share a label, so the task
    # random intercepts are per (map, task) pair -- 50 levels, not 10.
    # The component keeps the name "task" for the bands and tables.
    df["map_task"] = (df["cell_map"].astype(str) + ":"
                      + df["task"].astype(str))
    vc = {}
    if df["seed"].nunique() > 1:
        vc["seed"] = "0 + C(seed)"
    if df["cell_map"].nunique() > 1:
        vc["map"] = "0 + C(cell_map)"
    if df["map_task"].nunique() > 1:
        vc["task"] = "0 + C(map_task)"
    # pooled runs only (supervisor ruling 2026-08-28): the layer may
    # change what a seed, a map, or a route does, so the pooled model
    # carries the full symmetric set of layer interactions. The
    # coarse layer:map term is kept for hierarchical completeness
    # (empirically it does NOT collapse: SD ~0.16, alongside ~0.15 for
    # layer:map_task and ~0.06 for layer:seed).
    if df["reactive_peds"].nunique() > 1:
        lay = df["reactive_peds"].astype(str)
        df["layer_seed"] = lay + ":" + df["seed"].astype(str)
        df["layer_map"] = lay + ":" + df["cell_map"].astype(str)
        df["layer_map_task"] = lay + ":" + df["map_task"].astype(str)
        vc["layer_seed"] = "0 + C(layer_seed)"
        vc["layer_map"] = "0 + C(layer_map)"
        vc["layer_task"] = "0 + C(layer_map_task)"
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
    # label from the model's own vcp_names (insertion order for the VB
    # class, source-verified) rather than assuming dict order
    _vcp_names = list(getattr(model, "vcp_names", vc.keys()))
    vcp = pd.DataFrame({
        "component": _vcp_names,
        # vcp_mean is the posterior mean of LOG-sd -> report sd = exp(.)
        "sd_posterior_mean": [round(float(np.exp(x)), 3)
                              for x in fit.vcp_mean],
        # supervisor item 6: report the uncertainty of the variance
        # components too (95% interval from the posterior of log-sd)
        "sd_ci_lo": [round(float(np.exp(m - 1.96 * sd)), 3)
                     for m, sd in zip(fit.vcp_mean, fit.vcp_sd)],
        "sd_ci_hi": [round(float(np.exp(m + 1.96 * sd)), 3)
                     for m, sd in zip(fit.vcp_mean, fit.vcp_sd)],
    })
    vcp.to_csv(out / "success_glmm_variance_components.csv", index=False)
    # RQ2 overlays for the forest: +/-1 SD of the seed and task variance
    # components mapped to the odds-ratio scale, plus the literature-typical
    # improvement (~3 percentage points at an 80% success baseline =
    # logit(0.83) - logit(0.80) ~= 0.20 log-odds).
    _sd = {r["component"]: float(r["sd_posterior_mean"])
           for _, r in vcp.iterrows()}
    noise_bands = []
    if "map" in _sd:
        # drawn as a band like the others: RQ2 lists map variance among
        # the noise sources. Caveat (disclosed in text): estimated from
        # only 5 map levels, so this SD is the least precise of the three.
        noise_bands.append({"lo": float(np.exp(-_sd["map"])),
                            "hi": float(np.exp(_sd["map"])),
                            "color": "#2ca02c", "alpha": 0.14,
                            "label": f"±1 map SD "
                                     f"(OR {np.exp(-_sd['map']):.2f}--"
                                     f"{np.exp(_sd['map']):.2f})"})
    if "task" in _sd:
        noise_bands.append({"lo": float(np.exp(-_sd["task"])),
                            "hi": float(np.exp(_sd["task"])),
                            "color": "#1f77b4", "alpha": 0.14,
                            "label": f"±1 task SD "
                                     f"(OR {np.exp(-_sd['task']):.2f}--"
                                     f"{np.exp(_sd['task']):.2f})"})
    if "seed" in _sd:
        noise_bands.append({"lo": float(np.exp(-_sd["seed"])),
                            "hi": float(np.exp(_sd["seed"])),
                            "color": "#e6550d", "alpha": 0.16,
                            "label": f"±1 crowd-seed SD "
                                     f"(OR {np.exp(-_sd['seed']):.2f}--"
                                     f"{np.exp(_sd['seed']):.2f})"})
    # the literature-typical-gain reference line was dropped from the
    # figure (author decision 2026-08-25): the comparison against typical
    # published gains is made in the text, with percentage-point
    # conversions of the noise SDs, after the source tables are verified
    ref_lines = []
    # forest plot: odds ratios with 95% CI (algorithm terms), log scale
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ar = res[res["term"].str.startswith("C(algorithm)")].copy()
    if len(ar):
        ar["label"] = (ar["term"].str.extract(r"\[T\.(.+)\]")[0]
                       .fillna(ar["term"]).map(disp))
        # pooled runs carry the pedestrian-layer fixed effect: show those
        # rows in the same forest, clearly labelled against their baseline
        lay = res[res["term"].str.startswith("C(reactive_peds)")].copy()
        if len(lay) and "reactive_peds" in df.columns:
            _lref = str(sorted(df["reactive_peds"].astype(str).unique())[0])
            lay["label"] = ("Pedestrian Layer: "
                            + lay["term"].str.extract(r"\[T\.(.+)\]")[0]
                            + f" (vs {_lref})")
            ar = pd.concat([ar, lay], ignore_index=True)
        # _fixed_formula may have silently fallen back to another baseline;
        # read the one actually used from the Categorical, never the arg
        ref_shown = (str(df["algorithm"].cat.categories[0])
                     if isinstance(df["algorithm"].dtype,
                                   pd.CategoricalDtype) else reference)

        def _fmt(o, l, h):
            if 0.01 <= o <= 99 and l >= 0.01 and h <= 999:
                return f"{o:6.2f}  [{l:6.2f}, {h:6.2f}]"
            return f"{o:9.2e}  [{l:8.1e}, {h:8.1e}]"
        vals = [_fmt(o, l, h) for o, l, h in
                zip(ar["odds_ratio"], ar["or_ci_lo"], ar["or_ci_hi"])]
        _forest_figure(ar["label"], ar["odds_ratio"], ar["or_ci_lo"],
                       ar["or_ci_hi"], vals, null=1.0, logx=True,
                       pos_col="#2e7d32", neg_col="#c62828",
                       xlabel=f"Odds Ratio vs {disp(ref_shown)} "
                              f"(95% CrI, Log Scale)",
                       title="Success: Algorithm Effect Sizes",
                       caption="green = credibly higher success than the "
                               "reference, red = credibly lower, grey = CrI "
                               "crosses 1; shaded bands = +/-1 SD of the "
                               "seed / task variance components (estimates "
                               "inside a band are within that noise "
                               "source), dotted line = literature-typical "
                               "improvement; axis clipped to a readable "
                               "window, </> marks off-scale estimates -- "
                               "exact numbers in the right-hand column",
                       value_header="    OR    [95% CrI]",
                       out_path=out / "success_glmm_forest.png", dpi=dpi,
                       ref_label=f"{disp(ref_shown)}   (reference)",
                       ref_val="  1.00   (reference)",
                       noise_bands=noise_bands, ref_lines=ref_lines)
    return fit, res, vcp


def _abs_means_forest(d, endog, out, dpi=200):
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
        "algorithm_name": [disp(a) for a in order],
        "n": g.count().reindex(order).values,
        "mean": means.reindex(order).round(3).values,
        "ci_lo": (means - ci95).reindex(order).round(3).values,
        "ci_hi": (means + ci95).reindex(order).round(3).values,
    })
    abs_tab.to_csv(out / f"means_{endog}.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(order) + 1.6))
    y = range(len(order))
    for i in y:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
    ax.errorbar(means.reindex(order).values, y, zorder=3,
                xerr=ci95.reindex(order).values,
                fmt="s", capsize=3, ms=4, color="#1565c0")
    ax.set_yticks(list(y), [disp(a) for a in order], fontsize=8.5)
    ax.set_xlabel(f"{endog_disp(endog)}: Mean per Unit (95% CI)",
                  fontsize=9)
    ax.set_title(f"{endog_disp(endog)}: Absolute Means"
                 + (f" (n={len(d)}, small sample)" if len(d) < 20 else ""),
                 fontsize=10)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, out / f"means_{endog}_forest.png", dpi)
    plt.close(fig)


def fit_lmm(df, endog, reference, out, subset_success=False, dpi=200):
    """Fit a linear mixed model, degrading gracefully on small samples.

    Returns (fit, note).  fit is None only when there is truly nothing to
    fit; note explains what happened (subset size, fallback used, ...).
    """
    import statsmodels.formula.api as smf
    d = df.copy()
    # behavioural endogs are undefined for episodes that never started:
    # a failed global plan has zero simulation steps
    d = d[d["termination_reason"].astype(str)
          .str.split(":").str[0].ne("global_plan_failed")]
    if subset_success:
        d = d[d["success"] == 1]
    # an all-null column arrives as object dtype on the JSON path, where
    # np.isfinite raises "ufunc not supported for the input types"
    d[endog] = pd.to_numeric(d[endog], errors="coerce")
    d = d[np.isfinite(d[endog])]
    n = len(d)
    if n >= 2 and d["algorithm"].nunique() >= 1:
        try:
            _abs_means_forest(d, endog, out, dpi=dpi)
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
    # CROSSED random effects for seed / map / task, as the README claims.
    #
    # The previous form passed groups=seed together with map/task variance
    # components. That is wrong twice over. First, statsmodels' MixedLM drops
    # the group random intercept entirely when re_formula is None and a
    # vc_formula is supplied (exog_re becomes None, k_re 0), so there was no
    # seed effect at all -- visible in the output as a "map Var" row with no
    # "Group Var" row. Second, variance components are realised WITHIN each
    # group, so "map" was a per-(seed, map) effect, i.e. a seed x map
    # interaction nested inside seed, not a crossed map main effect.
    #
    # The standard statsmodels recipe for crossed effects is a single constant
    # group with every factor as a variance component. Validated by recovering
    # injected effects from synthetic data: the crossed form returns
    # map 1.363 / seed 1.309 / task 0.030 against a truth of a +2 map shift
    # plus a seed effect, where the nested form reported no seed at all and an
    # inflated map variance of 2.189.
    # task nested in map, as in the GLMM (supervisor item 6)
    d["map_task"] = (d["cell_map"].astype(str) + ":" + d["task"].astype(str))
    vc = {}
    if d["seed"].nunique() > 1:
        vc["seed"] = "0 + C(seed)"
    if d["cell_map"].nunique() > 1:
        vc["map"] = "0 + C(cell_map)"
    if d["map_task"].nunique() > 1:
        vc["task"] = "0 + C(map_task)"
    if d["reactive_peds"].nunique() > 1:   # pooled runs: layer interactions
        lay = d["reactive_peds"].astype(str)
        d["layer_seed"] = lay + ":" + d["seed"].astype(str)
        d["layer_map"] = lay + ":" + d["cell_map"].astype(str)
        d["layer_map_task"] = lay + ":" + d["map_task"].astype(str)
        vc["layer_seed"] = "0 + C(layer_seed)"
        vc["layer_map"] = "0 + C(layer_map)"
        vc["layer_task"] = "0 + C(layer_map_task)"
    groups = pd.Series(["all"] * len(d), index=d.index)

    can_log = bool((d[endog] >= 0).all())   # log1p only for nonneg endogs
    base_vc = dict(vc) if vc else None

    # cluster key = the shared-randomness scenario: episodes with the
    # same (map, task, seed) face the SAME pedestrian realisation, so
    # that is the level residuals are correlated at (~500 clusters,
    # comfortably above the parameter count -- clustering on the 5 maps
    # alone gives a rank-4 covariance and unusable robust SEs)
    clus = (d["cell_map"].astype(str) + ":" + d["task"].astype(str)
            + ":s" + d["seed"].astype(str))

    def _fit_once(lhs, fixed, vcf):
        if vcf:
            # marginal fits differ only in the optimizer's ability to
            # walk the flat boundary region, so retry the SAME model
            # (REML throughout -- mixing estimators across endogs would
            # be its own inconsistency) with progressively more robust
            # optimizers; converged lbfgs fits are untouched
            model = smf.mixedlm(f"{lhs} ~ {fixed}", d, groups=groups,
                                vc_formula=vcf)
            last = last_exc = None
            for method in ("lbfgs", "powell", "cg"):
                try:
                    cand = model.fit(reml=True, method=method)
                except Exception as exc:
                    last_exc = exc
                    continue
                last = cand
                if bool(getattr(cand, "converged", False)):
                    return cand
            if last is not None:
                return last
            raise last_exc
        # no random structure left: OLS with cluster-robust
        # (Liang-Zeger) SEs on the map x task x seed scenario
        if clus.nunique() > 1:
            return smf.ols(f"{lhs} ~ {fixed}", d).fit(
                cov_type="cluster", cov_kwds={"groups": clus})
        return smf.ols(f"{lhs} ~ {fixed}", d).fit()
        # no random structure left: OLS with cluster-robust
        # (Liang-Zeger) SEs on the map x task x seed scenario
        if clus.nunique() > 1:
            return smf.ols(f"{lhs} ~ {fixed}", d).fit(
                cov_type="cluster", cov_kwds={"groups": clus})
        return smf.ols(f"{lhs} ~ {fixed}", d).fit()

    def _pruned(vcf, cand):
        """Variance components this failed fit estimated at the zero
        boundary; returns the surviving subset, or None.

        statsmodels stores components in ITS OWN (name-sorted) order,
        NOT the vc_formula dict order -- verified empirically. Read the
        names back from the fitted model; never assume dict order, or
        the wrong component gets pruned silently."""
        try:
            vals = np.asarray(getattr(cand, "vcomp", []))
            try:
                names = list(cand.model.exog_vc.names)
            except Exception:
                names = sorted(vcf)          # statsmodels' actual order
            if len(vals) != len(names):
                return None
            keep = {k: vcf[k] for k, v in zip(names, vals)
                    if k in vcf and v > 1e-8}
            return keep if 0 < len(keep) < len(vcf) else None
        except Exception:
            return None

    # pre-registered remedy ladder, dynamically extended: a failed rung
    # inserts its own targeted remedy (drop exactly the boundary
    # component(s) it exposed, else fall through to OLS). Converged
    # endogs never leave rung 1: byte-identical to the plain pipeline.
    #
    # Scale is PRE-SPECIFIED per endog, not convergence-dependent: letting
    # the remedy pick the scale left e.g. sim_time_s on log1p in the
    # 18-unit view but raw in the 54-unit view (6 of 9 endogs disagreed
    # between views), which is unreportable. Right-skewed non-negative
    # endogs are modelled on log1p a priori; min_pedestrian_distance_m
    # stays raw (bounded, roughly symmetric, converges everywhere).
    prespec_log = endog in _LOG_SCALE_ENDOGS and can_log
    if prespec_log:
        ladder = [(f"np.log1p({endog})", fixed_full, base_vc,
                   "LMM on log1p (pre-specified scale for right-skewed "
                   "outcome; read coefficients multiplicatively)")]
        ladder.append((endog, fixed_full, base_vc,
                       "LMM on raw scale (fallback: log1p fit "
                       "non-converged)"))
    else:
        ladder = [(endog, fixed_full, base_vc, "LMM")]
        if can_log:
            ladder.append((f"np.log1p({endog})", fixed_full, base_vc,
                           "LMM on log1p (auto remedy: raw-scale fit "
                           "non-converged; read coefficients "
                           "multiplicatively)"))
    ladder.append((endog, f"C(algorithm, Treatment('{reference}'))",
                   None, "OLS, algorithm-only, cluster-robust SEs "
                         "on map x task x seed (last resort)"))
    last_exc = None
    fit = None
    shown_endog = endog
    first_fit = first_note = first_shown = None
    k = 0
    while k < len(ladder):
        lhs, fixed, vcf, label = ladder[k]
        k += 1
        try:
            cand = _fit_once(lhs, fixed, vcf)
            if not np.all(np.isfinite(cand.params.values)):
                raise ValueError("non-finite coefficients")
            note = label + (f" (small sample: n={n})" if n < 20 else "")
            # the reference can be swapped per-endog when the requested
            # one has no rows in that subset; record the baseline used
            note += f" [reference={reference}]"
            if bool(getattr(cand, "converged", True)):
                fit = cand
                shown_endog = lhs
                try:
                    if len(getattr(fit, "vcomp", [])) and \
                            np.any(np.asarray(fit.vcomp) <= 1e-10):
                        note += " [singular RE covariance]"
                except Exception:
                    pass
                break
            if first_fit is None:       # fullest spec, kept as fallback
                first_fit = cand
                first_note = note + " [NON-CONVERGED]"
                first_shown = lhs
            if vcf:                     # targeted remedy for THIS rung
                pv = _pruned(vcf, cand)
                if pv is not None:
                    gone = sorted(set(vcf) - set(pv))
                    ladder.insert(k, (lhs, fixed, pv,
                                      label + " (dropped boundary "
                                      f"components: {gone})"))
                else:
                    # OLS is a bigger leap than the remaining
                    # pre-registered remedies (log1p keeps the mixed
                    # structure intact): queue it as a LATE resort,
                    # just ahead of the algorithm-only final rung
                    ladder.insert(len(ladder) - 1,
                                  (lhs, fixed, None,
                                   label + " -> OLS, cluster-robust SEs "
                                   "on map x task x seed (all random "
                                   "components at the boundary)"))
        except Exception as exc:
            last_exc = exc
    if fit is None and first_fit is not None:
        fit = first_fit
        shown_endog = first_shown
        note = (first_note + " -- no remedy converged; cite the "
                "descriptive means figure instead")
    if fit is None:
        # final fallback: plain OLS, clearly labelled
        try:
            model = smf.ols(
                f"{endog} ~ C(algorithm, Treatment('{reference}'))", d)
            _ck = (d["cell_map"].astype(str) + ":" + d["task"].astype(str)
                   + ":s" + d["seed"].astype(str))
            if _ck.nunique() > 1:
                fit = model.fit(cov_type="cluster",
                                cov_kwds={"groups": _ck})
            else:
                fit = model.fit()
            note = (f"OLS fallback, cluster-robust SEs on map x task x seed "
                    f"(mixed model failed: {last_exc}; n={n})")
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
                       .fillna(ar["term"]).map(disp))
        lay = res[res["term"].str.startswith("C(reactive_peds)")].copy()
        if len(lay) and "reactive_peds" in df.columns:
            _lref = str(sorted(df["reactive_peds"].astype(str).unique())[0])
            lay["label"] = ("Pedestrian Layer: "
                            + lay["term"].str.extract(r"\[T\.(.+)\]")[0]
                            + f" (vs {_lref})")
            ar = pd.concat([ar, lay], ignore_index=True)
        vals = [f"{c:+9.2f}  [{l:8.2f}, {h:8.2f}]" for c, l, h in
                zip(ar["coef"], ar["ci_lo"], ar["ci_hi"])]
        _forest_figure(ar["label"], ar["coef"], ar["ci_lo"], ar["ci_hi"],
                       vals, null=0.0, logx=False,
                       pos_col="#0072B2", neg_col="#D55E00",
                       xlabel=f"{endog_disp(shown_endog)}: Coefficient "
                              f"vs {disp(reference)} (95% CI)",
                       title=f"{endog_disp(endog)}: Effect Sizes",
                       caption=f"model: {note}. blue = credibly above the "
                               "reference, vermillion = credibly below, "
                               "grey = CI crosses 0; exact numbers in "
                               "the right-hand column",
                       value_header="   coef    [95% CI]",
                       out_path=out / f"lmm_{endog}_forest.png", dpi=dpi,
                       ref_label=f"{disp(reference)}   (reference)",
                       ref_val="  +0.00   (reference)")

    return fit, note


def failure_taxonomy(df, out, dpi=200):
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
    if "goal" in tab.columns:
        order = (tab["goal"] / tab["n"]).sort_values(ascending=True).index
        tab = tab.loc[order]                   # worst overall on top
    algos = tab.index.tolist()
    m = len(algos)
    if _LAYOUT == "twocol":
        fig_w = 7.05
        row_h = 0.20 if m <= 30 else max(0.14, 7.6 / m)
    else:
        fig_w = 8.2
        row_h = 0.24 if m <= 30 else max(0.165, 8.9 / m)
    fig, ax = plt.subplots(figsize=(fig_w, row_h * m + 1.7))
    left = np.zeros(m)
    colors = {"goal": "#2e7d32", "collision": "#c62828",
              "max_time": "#f9a825", "stalled": "#6a1b9a",
              "global_plan_failed": "#455a64",
              "planner_error": "#00838f"}
    LEG = {"max_time": "timeout (max_time)", "stalled": "stuck (stalled)",
           "planner_error": "planner error",
           "global_plan_failed": "no global plan"}
    for r in reasons:
        vals = (tab[r] / tab["n"] * 100.0).values
        ax.barh(range(m), vals, left=left, label=LEG.get(r, r),
                color=colors.get(r, "0.5"), height=0.78)
        left += vals
    ax.set_yticks(range(m), [disp(a) for a in algos],
                  fontsize=8.5 if m <= 30 else 7.0)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of Runs", fontsize=9)
    ax.set_title("Outcome Composition per Unit (Worst on Top)",
                 fontsize=11)
    ax.legend(ncols=3, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.045), frameon=False,
              columnspacing=1.6, handletextpad=0.6)
    fig.tight_layout()
    _savefig(fig, out / "failure_taxonomy.png", dpi)
    plt.close(fig)
    return tab



# Mark's taxonomy vocabulary, mapped from the runner's internal ids
CATEGORY_MAP = {"collision": "collision", "max_time": "timeout",
                "stalled": "stuck",
                "global_plan_failed": "no global plan"}
CATEGORY_ORDER = ["collision", "timeout", "stuck", "no global plan",
                  "other (goal not reached)"]
CAT_COL = {"collision": "#c62828", "timeout": "#f9a825",
           "stuck": "#6a1b9a", "no global plan": "#1565c0",
           "other (goal not reached)": "#455a64"}


def failure_rates_by_category(df, out, dpi=200):
    """The 'report those separately' deliverable: one panel per failure
    mode (collision / timeout / stuck / other goal-not-reached), each
    unit's rate with a Wilson 95% CI, shared row order (worst overall on
    top). Infrastructure outcomes are excluded upstream."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.copy()
    base = d["termination_reason"].astype(str).str.split(":").str[0]
    d["_cat"] = np.where(base.eq("goal"), "goal",
                         base.map(CATEGORY_MAP)
                         .fillna("other (goal not reached)"))
    nn = d.groupby("algorithm").size()
    rows = []
    for u in sorted(nn.index):
        du = d[d["algorithm"] == u]
        rec = {"unit": u, "unit_name": disp(u), "n": int(nn[u])}
        for c in ["goal"] + CATEGORY_ORDER:
            key = c.split(" ")[0]
            k = int((du["_cat"] == c).sum())
            p, lo, hi = wilson(k, int(nn[u]))
            rec[f"{key}_n"] = k
            rec[f"{key}_rate"] = round(p, 4)
            rec[f"{key}_lo"] = round(lo, 4)
            rec[f"{key}_hi"] = round(hi, 4)
        rows.append(rec)
    tab = pd.DataFrame(rows).sort_values("goal_rate")   # worst on top
    tab.to_csv(out / "failure_rates_by_category.csv", index=False)

    m = len(tab)
    row_h = 0.32 if m <= 30 else max(0.185, 10.0 / m)
    fs = 8.5 if m <= 30 else 7.0
    fig, axes = plt.subplots(1, len(CATEGORY_ORDER), sharey=True,
                             figsize=(12.8, row_h * m + 2.2))
    axes = np.atleast_1d(axes)
    xmax = max(float(tab[f"{c.split(' ')[0]}_hi"].max())
               for c in CATEGORY_ORDER)
    xmax = min(1.0, xmax * 1.08 + 0.02)
    for ax, c in zip(axes, CATEGORY_ORDER):
        key = c.split(" ")[0]
        for i in range(m):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
        ax.errorbar(tab[f"{key}_rate"], range(m),
                    xerr=[tab[f"{key}_rate"] - tab[f"{key}_lo"],
                          tab[f"{key}_hi"] - tab[f"{key}_rate"]],
                    fmt="o", ms=3.8, elinewidth=1.0, capsize=1.5,
                    color=CAT_COL[c], ls="", zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_title(c.title(), fontsize=9.5, color=CAT_COL[c])
        ax.grid(axis="x", lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(m), tab["unit_name"], fontsize=fs)
    axes[0].set_ylim(m - 0.5, -0.5)
    fig.suptitle("Failure Modes, Reported Separately", fontsize=11,
                 y=0.995)
    fig.supxlabel("Share of Runs (Wilson 95% CI)", fontsize=9.5,
                  y=0.040)
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    _savefig(fig, out / "failure_rates_by_category.png", dpi)
    plt.close(fig)
    return tab


def ranking_stability(df, out, B=2000, rng_seed=0, dpi=200):
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
    # vertical forest layout: one row per unit, best rank on top; the sea of
    # P(top1)=0.00 annotations is folded into the label ONLY when non-zero
    res2 = res.reset_index(drop=True)
    labels = [disp(a) + (f"   [P(top1)={p:.2f}]" if p >= 0.005 else "")
              for a, p in zip(res2["algorithm"], res2["P_top1"])]
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(res2) + 1.6))
    y = range(len(res2))
    ax.errorbar(res2["rank_median"], y,
                xerr=[res2["rank_median"] - res2["rank_ci_lo"],
                      res2["rank_ci_hi"] - res2["rank_median"]],
                fmt="o", capsize=3, ms=4, color="#1565c0")
    ax.set_yticks(list(y), labels, fontsize=8.5)
    ax.invert_yaxis()                       # best (lowest rank) on top
    ax.set_xlim(0.5, len(res2) + 0.5)
    ax.set_xlabel("Rank by Success Rate Across Seed Resamples "
                  "(Median, Bootstrap 95% CI; 1 = Best)", fontsize=9)
    ax.set_title("Ranking Stability"
                 + (" -- Combinations (Global + Local)"
                    if any("+" in str(a) for a in res2["algorithm"])
                    else ""), fontsize=10)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _savefig(fig, out / "ranking_stability.png", dpi)
    plt.close(fig)
    return res


def success_rates_figure(df, out, dpi=200):
    """Absolute success rates with Wilson 95% CIs -- the readable
    replacement for the old 54-bar wall.

    combo unit ("g+a"): 18 rows (local algorithm) x one coloured marker per
    global planner -- all 54 combinations in one page-width figure.
    algorithm unit: one dot row per algorithm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = df.groupby("algorithm")["success"]
    stats = {u: wilson(float(s.sum()), int(s.count()))
             for u, s in g}                        # unit -> (p, lo, hi)
    ns = g.count().to_dict()
    units = list(stats)
    combo = any("+" in u for u in units)
    rows_tab = pd.DataFrame({
        "unit": units,
        "unit_name": [disp(u) for u in units],
        "n": [ns[u] for u in units],
        "success_rate": [round(stats[u][0], 4) for u in units],
        "wilson_lo": [round(stats[u][1], 4) for u in units],
        "wilson_hi": [round(stats[u][2], 4) for u in units],
    }).sort_values("success_rate", ascending=False)
    rows_tab.to_csv(out / "success_rates.csv", index=False)

    if combo:
        locs = sorted({u.split("+", 1)[1] for u in units})
        gpls = sorted({u.split("+", 1)[0] for u in units},
                      key=lambda x: (["astar", "dijkstra", "rrt"].index(x)
                                     if x in ("astar", "dijkstra", "rrt")
                                     else 99, x))
        pooled = {a: np.mean([stats[f"{gp}+{a}"][0] for gp in gpls
                              if f"{gp}+{a}" in stats]) for a in locs}
        order = sorted(locs, key=lambda a: -pooled[a])
        GCOL = {"astar": "#0072B2", "dijkstra": "#E69F00", "rrt": "#009E73"}
        GMRK = {"astar": "o", "dijkstra": "s", "rrt": "^"}
        fig, ax = plt.subplots(figsize=(7.6, 0.40 * len(order) + 1.8))
        for i, a in enumerate(order):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0)
            for j, gp in enumerate(gpls):
                u = f"{gp}+{a}"
                if u not in stats:
                    continue
                p, lo, hi = stats[u]
                yy = i + (j - (len(gpls) - 1) / 2) * 0.24
                ax.errorbar(p, yy, xerr=[[p - lo], [hi - p]],
                            fmt=GMRK.get(gp, "D"),
                            color=GCOL.get(gp, "0.3"), ms=4.5,
                            elinewidth=1.1, capsize=2, zorder=3)
        ax.set_yticks(range(len(order)),
                      [_algo_display(a) for a in order], fontsize=8.5)
        ax.set_ylim(len(order) - 0.5, -0.5)
        handles = [plt.Line2D([], [], marker=GMRK.get(gp, "D"),
                              color=GCOL.get(gp, "0.3"), ls="", ms=6,
                              label=f"g:{_algo_display(gp)}") for gp in gpls]
        ax.legend(handles=handles, loc="lower right", frameon=False,
                  fontsize=8.5, title="global planner", title_fontsize=8.5)
    else:
        order = rows_tab["unit"].tolist()
        fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(order) + 1.6))
        for i, u in enumerate(order):
            p, lo, hi = stats[u]
            ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o",
                        color="#0072B2", ms=4.5, elinewidth=1.1,
                        capsize=2, zorder=3)
        ax.set_yticks(range(len(order)), [disp(u) for u in order],
                      fontsize=8.5)
        ax.set_ylim(len(order) - 0.5, -0.5)
    n_med = int(np.median(list(ns.values())))
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Success Rate (Wilson 95% CI, n ~ {n_med}/Unit)",
                  fontsize=9)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Success Rate"
                 + (" by Combination (Global + Local)" if combo
                    else " by Algorithm"), fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out / "success_rates.png", dpi)
    plt.close(fig)
    return rows_tab


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
    ap.add_argument("--dpi", type=int, default=200,
                    help="figure resolution (use 300 for print)")
    ap.add_argument("--names", choices=["short", "full"], default="short",
                    help="axis labels: standardized abbreviations (short) "
                         "or written-out algorithm names (full)")
    ap.add_argument("--layout", choices=["page", "twocol"], default="page",
                    help="page: thesis/A4 sizing (default); twocol: design "
                         "the forest figures at IEEE textwidth (7.05 in) "
                         "with FINAL point sizes -- include 1:1, never "
                         "rescale")
    ap.add_argument("--pdf", action="store_true",
                    help="also write a vector .pdf next to every .png")
    ap.add_argument("--out", default=None,
                    help="output root for the stats[_combo] folders; "
                         "required feel: defaults to the results dir for a "
                         "single input, and to ./stats_pooled when several "
                         "comma-separated results dirs are pooled")
    args = ap.parse_args()
    global _NAME_MODE, _LAYOUT, _SAVE_PDF
    _NAME_MODE = args.names
    _LAYOUT = args.layout
    if args.layout == "twocol" and args.names == "full":
        print("note: --names full cannot fit the two-column canvas; "
              "using short names (full names belong in the naming table)")
        _NAME_MODE = "short"
    _SAVE_PDF = args.pdf
    if args.unit == "both":
        import subprocess
        for u in ("algorithm", "combo"):
            subprocess.run([sys.executable, __file__,
                            "--results", args.results,
                            "--reference", args.reference,
                            "--bootstrap", str(args.bootstrap),
                            "--dpi", str(args.dpi),
                            "--names", args.names,
                            "--layout", args.layout,
                            "--unit", u]
                           + (["--pdf"] if args.pdf else [])
                           + (["--out", args.out] if args.out else []),
                           check=True)
        return
    # --results accepts a comma-separated list of results trees. Pooling
    # the three reactive-pedestrian layers gives C(reactive_peds) more
    # than one level, so the fixed-effects formula picks it up
    # automatically and the models estimate the pedestrian-model effect
    # (supervisor: "pedestrian type as a fixed effect"). Demand parsing
    # and config lookup run per tree.
    paths = [Path(p) for p in args.results.split(",") if p.strip()]
    if len(paths) == 1:
        df = load_rows(paths[0])
        results = Path(args.out) if args.out else paths[0]
    else:
        parts = []
        for p in paths:
            d = load_rows(p)
            lv = sorted(d["reactive_peds"].astype(str).unique())
            print(f"pooled: {len(d)} rows from {p} (reactive_peds = {lv})")
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        results = Path(args.out) if args.out else Path("stats_pooled")
        print(f"pooled analysis: {len(df)} rows from {len(paths)} trees "
              f"-> {results}")
    # ---- infrastructure outcomes: excluded from EVERY model below,
    # ---- with visible accounting (never silent)
    if "termination_reason" in df.columns:
        infra_mask = (df["termination_reason"].astype(str)
                      .str.split(":").str[0].isin(INFRA_REASONS))
    else:
        infra_mask = pd.Series(False, index=df.index)
    infra = df[infra_mask].copy()
    df = df[~infra_mask].copy()
    print(f"{len(infra)} infrastructure episodes (termination_reason in "
          f"{sorted(INFRA_REASONS)}) EXCLUDED from all statistics; "
          f"{len(df)} analysed")
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
    if len(infra):
        cols = [c for c in ("map", "task", "global_planner", "algorithm",
                            "mode", "seed", "termination_reason")
                if c in infra.columns]
        infra[cols].to_csv(out / "excluded_infrastructure.csv", index=False)
    lines = [f"unit = {args.unit}",
             f"n runs = {len(df)}; units = {sorted(set(df['algorithm']))}",
             f"maps = {sorted(set(df['cell_map']))}; "
             f"modes = {sorted(set(df['mode']))}; "
             f"seeds = {df['seed'].nunique()}",
             f"infrastructure episodes excluded = {len(infra)} "
             f"(see excluded_infrastructure.csv)" if len(infra) else
             "infrastructure episodes excluded = 0",
             f"reference algorithm = {args.reference}", ""]

    # 0) absolute success rates (Wilson CIs) -- the readable headline figure
    try:
        sr = success_rates_figure(df, out, dpi=args.dpi)
        lines.append("== Success rates (Wilson 95% CI) ==")
        lines.append(sr.to_string(index=False))
        lines.append("")
    except Exception as exc:
        lines.append(f"success-rate figure skipped: {exc}\n")

    # 1) success GLMM
    try:
        fit, res, vcp = fit_success_glmm(df, args.reference, out,
                                         dpi=args.dpi)
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
                                subset_success=subset, dpi=args.dpi)
            lines.append(f"== LMM {endog}"
                         f"{' (successful runs)' if subset else ''} ==")
            lines.append(f"[{note}]")
            if fit is not None:
                _t1 = fit.summary().tables[1]
                # a DataFrame str() silently truncates to head+tail rows
                # (the 54-unit tables have 60+ rows); render in full
                lines.append(_t1.to_string()
                             if hasattr(_t1, 'to_string') else str(_t1))
            lines.append("")
        except Exception as exc:
            lines.append(f"LMM {endog} skipped: {exc}\n")

    # 3) failure taxonomy + 4) ranking stability
    tab = failure_taxonomy(df, out, dpi=args.dpi)
    lines.append("== Failure taxonomy (% of runs) ==")
    lines.append(tab.to_string())
    lines.append("")
    cat = failure_rates_by_category(df, out, dpi=args.dpi)
    lines.append("== Failure rates by category (Wilson 95% CI) ==")
    lines.append(cat.to_string(index=False))
    lines.append("")
    res = ranking_stability(df, out, B=args.bootstrap, dpi=args.dpi)
    lines.append("== Ranking stability (bootstrap over seeds) ==")
    lines.append(res.to_string(index=False))

    (out / "model_summaries.txt").write_text("\n".join(lines))
    print(f"stats -> {out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
