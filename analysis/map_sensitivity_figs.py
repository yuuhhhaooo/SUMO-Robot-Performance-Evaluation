#!/usr/bin/env python3
"""Map-sensitivity figures over a completed sweep (REAL data, not mock-ups).

Reads every robot_metrics.json under --results, EXCLUDES infrastructure
outcomes (see INFRA_REASONS) with printed accounting, and writes:

    <out>/success_heatmap_54x5_by_global.png   18 rows x maps, 3 panels
                                               (one per global planner),
                                               family-paired row order,
                                               real map separated + bold
    <out>/ranking_slopegraph_sim_vs_real_unified54.png
                                               PRIMARY: all 54 (global, local)
                                               combinations ranked together,
                                               simulated (pooled) vs real
    <out>/ranking_sim_vs_real_unified54.csv    rates, ranks, rank shift for
                                               the unified 54-combination table
    <out>/ranking_slopegraph_sim_vs_real_by_global.png
                                               secondary: ranks within each
                                               global planner (isolates one
                                               stack's own transfer)
    <out>/ranking_sim_vs_real_by_global.csv
    <out>/ranking_slopegraph_sim_vs_real.png   appendix: 18 locals pooled
                                               over globals
    <out>/ranking_sim_vs_real.csv
    <out>/success_by_combo_map.csv             n + success rate per
                                               (map, global, algorithm)

Row order (paired families) lives in FAMILY_ORDER below -- single source.
Algorithms present in the data but absent from FAMILY_ORDER are appended
(sorted) and reported; absent ones are skipped. Display names come from
analysis/algo_names.py when importable, else from the built-in fallback.

Usage:
    python analysis/map_sensitivity_figs.py --results results_pysf
    python analysis/map_sensitivity_figs.py --results results_sfm --out figs_sfm
    # real map defaults to map5_ucl; override with --real-map <name>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_BOLD = {"font.weight": "bold", "axes.titleweight": "bold",
         "axes.labelweight": "bold", "figure.titleweight": "bold"}
matplotlib.rcParams.update(_BOLD)   # every title/label/tick/legend in bold

INFRA_REASONS = {"sumo_crash"}   # global_plan_failed now counts AGAINST the combination (user ruling): the global half failing to route is a failure of the deployed stack, reported as its own category

FAMILY_ORDER = [("sarl", "sarl_upstream"),
                ("cadrl", "cadrl_upstream"),
                ("lstm_rl", "lstm_rl_upstream"),
                ("teb", "teb_upstream"),
                ("mpc", "mpc_dompc"),
                ("orca", "orca_heuristic"),
                ("crowdnav_dsrnn", "crowdnav_attngraph"),
                ("dwa",), ("astar",), ("dijkstra",), ("rrt",)]

FALLBACK_DISP = {
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
_NAME_MODE = "short"

try:                                    # single naming source when deployed
    from algo_names import display as _algo_short        # noqa: E402
except ImportError:                     # graceful fallback, ids never lost
    def _algo_short(a):                                  # noqa: D103
        return FALLBACK_DISP.get(a, a)
try:
    from algo_names import full as _algo_full            # noqa: E402
except Exception:
    def _algo_full(a):
        return FALLBACK_FULL.get(a, _algo_short(a))


def algo_disp(a):
    return _algo_full(a) if _NAME_MODE == "full" else _algo_short(a)


FALLBACK_FULL = {
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


GDISP = {"astar": "Global: A*", "dijkstra": "Global: Dijkstra",
         "rrt": "Global: RRT"}
GLOBAL_PREF = ["astar", "dijkstra", "rrt"]
SHIFT_UP, SHIFT_DOWN, NEUTRAL = "#0072B2", "#D55E00", "0.72"


# --------------------------------------------------------------------------
def load(results_root: Path) -> pd.DataFrame:
    rows, bad = [], 0
    files = sorted(results_root.rglob("robot_metrics.json"))
    for f in files:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            bad += 1
            continue
        rows.append({"map": m.get("map"), "task": m.get("task"),
                     "global_planner": m.get("global_planner"),
                     "algorithm": m.get("algorithm"), "seed": m.get("seed"),
                     "success": bool(m.get("success", False)),
                     "termination_reason": m.get("termination_reason")})
    if not rows:
        sys.exit(f"no robot_metrics.json found under {results_root}")
    df = pd.DataFrame(rows)
    infra = df[df["termination_reason"].isin(INFRA_REASONS)]
    clean = df[~df["termination_reason"].isin(INFRA_REASONS)].copy()
    print(f"  scanned {len(files)} episodes, parse failures {bad}")
    print(f"  {len(infra)} infrastructure episodes "
          f"(termination_reason in {sorted(INFRA_REASONS)}) EXCLUDED")
    for combo, k in Counter(zip(infra["map"], infra["global_planner"],
                                infra["algorithm"])).most_common():
        print(f"      {combo} x{k}")
    print(f"  analysed {len(clean)} episodes")
    return clean


def row_order(present):
    rows = [a for fam in FAMILY_ORDER for a in fam if a in present]
    extra = sorted(present - set(rows))
    if extra:
        print(f"  [note] algorithms outside FAMILY_ORDER appended: {extra}")
        rows += extra
    seps, y = [], 0
    for fam in FAMILY_ORDER:
        k = sum(1 for a in fam if a in present)
        if k:
            y += k
            seps.append(y - 0.5)
    return rows, seps[:-1] if seps else []


MAP_DISP = {"map4_london": "map4_london_block"}   # display-only rename


def maplab(m, real):
    lab = MAP_DISP.get(m, m) + (" (real)" if m == real else "")
    return lab


# --------------------------------------------------------------------------
def fig_heatmap(df, rows, seps, globals_, maps, real_map, out, label, dpi):
    ncol = len(maps)
    fig, axes = plt.subplots(1, len(globals_),
                             figsize=(1.9 + 2.4 * len(globals_) * ncol / 5,
                                      2.2 + 0.28 * len(rows)),
                             sharey=True)
    axes = np.atleast_1d(axes)
    pv_all = df.pivot_table(values="success", index="algorithm",
                            columns=["global_planner", "map"], aggfunc="mean")
    im = None
    for ax, g in zip(axes, globals_):
        M = np.full((len(rows), ncol), np.nan)
        for i, a in enumerate(rows):
            for j, m in enumerate(maps):
                try:
                    M[i, j] = pv_all.loc[a, (g, m)]
                except KeyError:
                    pass
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(rows)):
            for j in range(ncol):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}",
                            ha="center", va="center", fontsize=6.8)
        if seps:
            ax.hlines(seps, -0.5, ncol - 0.5, color="white", lw=2.4)
        if real_map in maps and maps.index(real_map) > 0:
            ax.axvline(maps.index(real_map) - 0.5, color="black", lw=1.6)
        ax.set_xticks(range(ncol), [maplab(m, real_map) for m in maps],
                      fontsize=7.5, rotation=30, ha="right")
        if real_map in maps:
            ax.get_xticklabels()[maps.index(real_map)].set_fontweight("bold")
        ax.set_title(GDISP.get(g, f"Global: {g}"), fontsize=10)
    axes[0].set_yticks(range(len(rows)), [algo_disp(a) for a in rows],
                       fontsize=8.5)
    fig.colorbar(im, ax=list(axes), shrink=0.72, label="Success Rate",
                 pad=0.015)
    _x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.suptitle(f"Success Rate per Combination (Global + Local) and Map"
                 f"  ({label})", fontsize=10, y=0.995, x=_x_mid)
    p = out / "success_heatmap_54x5_by_global.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


try:
    from algo_names import global_display as _global_disp    # noqa: E402
except ImportError:
    def _global_disp(u):
        return algo_disp(u)


def disp_combo(u):
    """Display name for 'global+local' combo strings (and plain ids)."""
    u = str(u)
    if "+" in u:
        g, a = u.split("+", 1)
        return f"{_global_disp(g)} + {algo_disp(a)}"
    return algo_disp(u)


def _slope_panel(ax, sub, rows, fs=8.5):
    """One slopegraph panel on ax from an episode subset; returns
    (rank_sim, rank_real, sim_rate, real_rate, kendall_tau, n_sim, n_real)."""
    sim_df = sub[sub["_side"] == "sim"]
    real_df = sub[sub["_side"] == "real"]
    sim = sim_df.groupby("algorithm")["success"].mean()
    real = real_df.groupby("algorithm")["success"].mean()
    rows = [a for a in rows if a in sim.index and a in real.index]
    n_sim = sim_df.groupby("algorithm").size().reindex(rows)
    n_real = real_df.groupby("algorithm").size().reindex(rows)

    def ranks(s):
        o = sorted(rows, key=lambda a: (-s[a], a))
        return {a: i + 1 for i, a in enumerate(o)}
    r_sim, r_real = ranks(sim), ranks(real)
    # Kendall tau-b on the success RATES (not the integer ranks): tie-aware,
    # and the statistic the thesis reports (kendall1938).
    tau = (float(kendalltau([sim[a] for a in rows],
                            [real[a] for a in rows])[0])
           if len(rows) > 2 else 0.0)
    for a in rows:
        y0, y1 = r_sim[a], r_real[a]
        dr = y0 - y1
        col = SHIFT_UP if dr >= 3 else SHIFT_DOWN if dr <= -3 else NEUTRAL
        lw, z = (2.0, 3) if col != NEUTRAL else (1.1, 2)
        tc = "0.15" if col == NEUTRAL else col
        ax.plot([0, 1], [y0, y1], color=col, lw=lw, zorder=z)
        ax.plot([0], [y0], "o", color=col, ms=3.5, zorder=z)
        ax.plot([1], [y1], "o", color=col, ms=3.5, zorder=z)
        ax.text(-0.05, y0, f"{_algo_short(a)}  {sim[a]:.2f}  ({y0})",
                ha="right", va="center", fontsize=fs, color=tc)
        ax.text(1.05, y1, f"({y1})  {real[a]:.2f}  {_algo_short(a)}",
                ha="left", va="center", fontsize=fs, color=tc)
    ax.set_xlim(-0.85, 1.85)
    ax.set_ylim(len(rows) + 0.6, -1.8)
    ax.axis("off")
    return r_sim, r_real, sim, real, tau, n_sim, n_real


def _tag_sides(df, real_map, sim_maps):
    d = df[df["map"].isin(set(sim_maps) | {real_map})].copy()
    d["_side"] = np.where(d["map"] == real_map, "real", "sim")
    return d


def fig_slopegraph(df, rows, real_map, sim_maps, out, label, dpi):
    """Pooled-over-globals variant (appendix; the primary comparison is the
    unified 54-combination table, see *_unified54)."""
    d = _tag_sides(df, real_map, sim_maps)
    fig, ax = plt.subplots(figsize=(7.4, 1.6 + 0.36 * df["algorithm"].nunique()))
    r_sim, r_real, sim, real, tau, n_sim, n_real = _slope_panel(ax, d, rows)
    ax.text(0, 0.0, f"Simulated Maps\n({', '.join(sim_maps)})",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.text(1, 0.0, f"Real-World Map\n({real_map})",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_title(f"Local-Algorithm Ranking, Simulated vs Real-World"
                 f"  ({label})", fontsize=9.5)
    fig.text(0.5, 0.012,
             f"APPENDIX VARIANT, pooled over globals & seeds (primary "
             f"comparison is the unified 54-combination table, "
             f"see *_unified54) -- "
             f"n = {int(n_sim.median())}/algo (sim), "
             f"{int(n_real.median())}/algo (real); "
             f"colour = rank shift >= 3; Kendall tau-b = {tau:.2f}",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "ranking_slopegraph_sim_vs_real.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {p}   (pooled appendix variant, tau = {tau:.2f})")

    rows_p = [a for a in rows if a in sim.index and a in real.index]
    tab = pd.DataFrame({
        "algorithm": rows_p,
        "algorithm_name": [algo_disp(a) for a in rows_p],
        "sim_n": n_sim.values, "sim_success": [round(sim[a], 4) for a in rows_p],
        "sim_rank": [r_sim[a] for a in rows_p],
        "real_n": n_real.values,
        "real_success": [round(real[a], 4) for a in rows_p],
        "real_rank": [r_real[a] for a in rows_p],
        "rank_shift": [r_sim[a] - r_real[a] for a in rows_p],
    }).sort_values("sim_rank")
    p = out / "ranking_sim_vs_real.csv"
    tab.to_csv(p, index=False)
    print(f"  wrote {p}")


def fig_slopegraph_by_global(df, rows, real_map, sim_maps, out, label, dpi):
    """SECONDARY view (primary is the unified 54-combination table): one
    panel per global planner, ranks computed within each panel -- no pooling
    across globals, so every line is one specific (global, local)
    combination and the panel isolates that stack's own transfer."""
    gpls = sorted(df["global_planner"].dropna().unique(),
                  key=lambda x: (["astar", "dijkstra", "rrt"].index(x)
                                 if x in ("astar", "dijkstra", "rrt")
                                 else 99, x))
    if len(gpls) < 2:
        return
    n_alg = df["algorithm"].nunique()
    fig, axes = plt.subplots(1, len(gpls),
                             figsize=(5.9 * len(gpls),
                                      1.9 + 0.34 * n_alg))
    axes = np.atleast_1d(axes)
    recs = []
    for ax, gp in zip(axes, gpls):
        d = _tag_sides(df[df["global_planner"] == gp], real_map, sim_maps)
        r_sim, r_real, sim, real, tau, n_sim, n_real = \
            _slope_panel(ax, d, rows, fs=7.6)
        ax.set_title(f"Global: {algo_disp(gp)}   "
                     f"(Kendall tau-b = {tau:.2f})", fontsize=9.5)
        ax.text(0, -0.1, "Sim (Pooled)", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
        ax.text(1, -0.1, real_map, ha="center", va="bottom",
                fontsize=8, fontweight="bold")
        for a in sim.index:
            if a in real.index:
                recs.append({"global_planner": gp, "algorithm": a,
                             "algorithm_name": algo_disp(a),
                             "sim_n": int(n_sim.get(a, 0)),
                             "sim_success": round(float(sim[a]), 4),
                             "sim_rank": r_sim[a],
                             "real_n": int(n_real.get(a, 0)),
                             "real_success": round(float(real[a]), 4),
                             "real_rank": r_real[a],
                             "rank_shift": r_sim[a] - r_real[a]})
    fig.suptitle(f"Combination (Global + Local) Ranking, Simulated vs "
                 f"Real-World  ({label})", fontsize=10, y=0.995)
    fig.text(0.5, 0.010,
             f"every line is one (global, local) combination; sim = "
             f"{', '.join(sim_maps)} pooled, real = {real_map}; "
             f"colour = rank shift >= 3 within the panel",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    p = out / "ranking_slopegraph_sim_vs_real_by_global.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}   (secondary, per-global panels)")
    tab = pd.DataFrame(recs).sort_values(["global_planner", "sim_rank"])
    p = out / "ranking_sim_vs_real_by_global.csv"
    tab.to_csv(p, index=False)
    print(f"  wrote {p}")


# --------------------------------------------------------------------------
def fig_slopegraph_unified54(df, real_map, sim_maps, out, label, dpi):
    """PRIMARY ranking view: all (global, local) combinations ranked
    TOGETHER (1..N) on both sides -- the ranked unit is the deployed stack,
    54 combinations in the full design. Caveat baked into the caption: in a
    unified league table a combination's rank shift also reflects OTHER
    combinations moving around it; the per-global panels isolate a stack's
    own transfer."""
    d = df.copy()
    d["unit"] = d["global_planner"].astype(str) + "+" + d["algorithm"]
    d = _tag_sides(d.rename(columns={"algorithm": "_local",
                                     "unit": "algorithm"}),
                   real_map, sim_maps)
    units = sorted(d["algorithm"].unique())
    n = len(units)
    thr = max(3, int(round(n * 0.15)))          # colour only big movers
    sim = d[d["_side"] == "sim"].groupby("algorithm")["success"].mean()
    real = d[d["_side"] == "real"].groupby("algorithm")["success"].mean()
    units = [u for u in units if u in sim.index and u in real.index]

    def ranks(s):
        o = sorted(units, key=lambda a: (-s[a], a))
        return {a: i + 1 for i, a in enumerate(o)}
    r_sim, r_real = ranks(sim), ranks(real)
    # tau-b on the rates over the units present on BOTH sides (the old
    # hand-rolled Spearman used the pre-filter n and ignored ties)
    tau = (float(kendalltau([sim[u] for u in units],
                            [real[u] for u in units])[0])
           if len(units) > 2 else 0.0)
    fig, ax = plt.subplots(figsize=(9.0, 0.24 * n + 2.4))
    for u in units:
        y0, y1 = r_sim[u], r_real[u]
        dr = y0 - y1
        col = SHIFT_UP if dr >= thr else SHIFT_DOWN if dr <= -thr \
            else NEUTRAL
        lw, z = (1.8, 3) if col != NEUTRAL else (0.9, 2)
        tc = "0.15" if col == NEUTRAL else col
        ax.plot([0, 1], [y0, y1], color=col, lw=lw, zorder=z)
        ax.text(-0.05, y0, f"{disp_combo(u)}  {sim[u]:.2f}  ({y0})",
                ha="right", va="center", fontsize=10.0,
                fontweight="bold", color=tc)
        ax.text(1.05, y1, f"({y1})  {real[u]:.2f}  {disp_combo(u)}",
                ha="left", va="center", fontsize=10.0,
                fontweight="bold", color=tc)
    ax.text(0, -0.6, f"Sim ({len(sim_maps)} Maps Pooled)", ha="center",
            va="bottom", fontsize=10, fontweight="bold")
    ax.text(1, -0.6, f"Real ({real_map})", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.set_xlim(-0.95, 1.95)
    ax.set_ylim(n + 1.0, -2.4)
    ax.axis("off")
    ax.set_title(f"Unified Ranking of All {n} Combinations (Global + Local), "
                 f"Simulated vs Real-World  ({label}; Kendall tau-b = {tau:.2f})",
                 fontsize=11)
    fig.tight_layout()
    p = out / "ranking_slopegraph_sim_vs_real_unified54.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}   (PRIMARY, unified league table, tau = {tau:.2f})")

    n_sim = d[d["_side"] == "sim"].groupby("algorithm").size()
    n_real = d[d["_side"] == "real"].groupby("algorithm").size()
    tab = pd.DataFrame({
        "combination": units,
        "global_planner": [u.split("+", 1)[0] for u in units],
        "local_planner": [u.split("+", 1)[1] for u in units],
        "combination_name": [disp_combo(u) for u in units],
        "sim_n": [int(n_sim.get(u, 0)) for u in units],
        "sim_success": [round(float(sim[u]), 4) for u in units],
        "sim_rank": [r_sim[u] for u in units],
        "real_n": [int(n_real.get(u, 0)) for u in units],
        "real_success": [round(float(real[u]), 4) for u in units],
        "real_rank": [r_real[u] for u in units],
        "rank_shift": [r_sim[u] - r_real[u] for u in units],
    }).sort_values("sim_rank")
    p = out / "ranking_sim_vs_real_unified54.csv"
    tab.to_csv(p, index=False)
    print(f"  wrote {p}")


def fig_scatter54(df, real_map, sim_maps, out, label, dpi,
                  hue="global"):
    """Unified view WITHOUT rank relativity: absolute success rates,
    x = simulated (pooled), y = real map, one point per combination."""
    GCOL = {"astar": "#0072B2", "dijkstra": "#E69F00", "rrt": "#009E73"}
    GMRK = {"astar": "o", "dijkstra": "s", "rrt": "^"}
    d = _tag_sides(df, real_map, sim_maps)
    g = (d.groupby(["global_planner", "algorithm", "_side"])["success"]
         .mean().unstack("_side").dropna())
    wide = hue == "local"
    fig, ax = plt.subplots(figsize=(9.8 if wide else 7.0, 7.0))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=1)
    locs = sorted(g.index.get_level_values(1).unique())
    cmap = plt.get_cmap("tab20")
    LCOL = {a: cmap(i % 20) for i, a in enumerate(locs)}
    diffs = (g["real"] - g["sim"]).abs().sort_values(ascending=False)
    flag = set(diffs.index[:8])                  # annotate biggest movers
    for (gp, a), row in g.iterrows():
        ax.scatter(row["sim"], row["real"], s=44,
                   color=LCOL[a] if wide else GCOL.get(gp, "0.3"),
                   marker=GMRK.get(gp, "D"), zorder=3,
                   edgecolors="white", linewidths=0.4)
        if (gp, a) in flag and not wide:
            k = list(flag).index((gp, a))
            dxy = [(6, 5), (6, -10), (-6, 5), (-6, -10)][k % 4]
            ax.annotate(disp_combo(f"{gp}+{a}"),
                        (row["sim"], row["real"]),
                        textcoords="offset points", xytext=dxy,
                        ha="left" if dxy[0] > 0 else "right",
                        fontsize=8.2, color="0.2")
    if wide:
        ch = [plt.Line2D([], [], marker="o", color=LCOL[a], ls="", ms=7,
                         label=algo_disp(a)) for a in locs]
        leg1 = ax.legend(handles=ch, loc="upper left",
                         bbox_to_anchor=(1.02, 1.0), frameon=False,
                         fontsize=8.5, title="local planner",
                         title_fontsize=9)
        ax.add_artist(leg1)
        sh = [plt.Line2D([], [], marker=GMRK[k], color="0.35", ls="",
                         ms=7, label=f"Global: {algo_disp(k)}")
              for k in ("astar", "dijkstra", "rrt") if k in
              set(g.index.get_level_values(0))]
        ax.legend(handles=sh, loc="lower left", bbox_to_anchor=(1.02, 0.0),
                  frameon=False, fontsize=8.5, title="global planner",
                  title_fontsize=9)
    else:
        handles = [plt.Line2D([], [], marker=GMRK[k], color=GCOL[k], ls="",
                              ms=6, label=f"Global: {algo_disp(k)}")
                   for k in ("astar", "dijkstra", "rrt") if k in
                   set(g.index.get_level_values(0))]
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Success Rate, Simulated Maps Pooled", fontsize=10.5)
    ax.set_ylabel("Success Rate, Real-World Map", fontsize=10.5)
    ax.set_title(f"All Combinations, Simulated vs Real  ({label})",
                 fontsize=11)
    ax.tick_params(labelsize=10.5)
    ax.grid(lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = out / ("success_scatter_sim_vs_real_by_algo.png" if wide
               else "success_scatter_sim_vs_real.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--real-map", default="map5_ucl")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--names", choices=["short", "full"], default="short",
                    help="heat-map / scatter labels: abbreviations or "
                         "written-out names (slopegraph labels stay short)")
    ap.add_argument("--scatter-hue", choices=["global", "local"],
                    default="global",
                    help="scatter colouring: by global planner (default, "
                         "with outlier labels) or one colour per local "
                         "algorithm with a side legend")
    args = ap.parse_args()
    global _NAME_MODE
    _NAME_MODE = args.names

    root = Path(args.results)
    if not root.exists():
        sys.exit(f"results root not found: {root}")
    label = root.name
    out = Path(args.out) if args.out else Path(f"figs_{label}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"== map-sensitivity figures ({label}) ==")
    df = load(root)

    maps_all = sorted(df["map"].dropna().unique())
    if args.real_map not in maps_all:
        sys.exit(f"real map '{args.real_map}' not in data; maps present: "
                 f"{maps_all}  (use --real-map)")
    sim_maps = [m for m in maps_all if m != args.real_map]
    maps = sim_maps + [args.real_map]                     # real map last
    globals_ = ([g for g in GLOBAL_PREF
                 if g in set(df["global_planner"].dropna())] or
                sorted(df["global_planner"].dropna().unique()))
    rows, seps = row_order(set(df["algorithm"].dropna()))

    tab = (df.groupby(["map", "global_planner", "algorithm"])
             .agg(n=("success", "size"), success_rate=("success", "mean"))
             .round(4).reset_index())
    p = out / "success_by_combo_map.csv"
    tab.to_csv(p, index=False)
    print(f"  wrote {p}")

    fig_heatmap(df, rows, seps, globals_, maps, args.real_map,
                out, label, args.dpi)
    fig_slopegraph_by_global(df, rows, args.real_map, sim_maps,
                             out, label, args.dpi)
    fig_slopegraph_unified54(df, args.real_map, sim_maps,
                             out, label, args.dpi)
    fig_scatter54(df, args.real_map, sim_maps, out, label,
                  args.dpi, hue=args.scatter_hue)
    fig_slopegraph(df, rows, args.real_map, sim_maps, out, label, args.dpi)
    print(f"  done -> {out.resolve()}")


if __name__ == "__main__":
    main()
