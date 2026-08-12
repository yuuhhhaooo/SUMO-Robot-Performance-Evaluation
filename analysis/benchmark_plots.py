#!/usr/bin/env python3
"""Plots for the v7 benchmark results tree.

For every (map, mode) found under --results:
  overlay_seed<k>_<map>_<mode>.png   all algorithms, one seed, on the map
  paths_<algo>_<map>_<mode>.png      one algorithm, all seeds, on the map
  metrics_<map>_<mode>.png           6-metric bar chart across algorithms
Plus results-wide:
  success_overall.png                success rate per algorithm per map
"""
from __future__ import annotations

import argparse
import sys
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent      # analysis/
REPO = ROOT.parent
import sys
sys.path.insert(0, str(REPO / "sim"))
from benchmark_runner import make_legs, Frame, in_rect  # noqa: E402

ALGO_ORDER = ["dwa", "astar", "dijkstra", "rrt", "orca", "mpc", "teb",
              "sarl", "cadrl", "lstm_rl"]
CMAP = plt.get_cmap("tab10")
ALGO_COLOR = {a: CMAP(i % 10) for i, a in enumerate(ALGO_ORDER)}

# units are either plain local algorithms ("dwa") or global+local
# combinations ("astar+dwa").  colour encodes the local algorithm,
# linestyle (lines) / hatch (bars) encodes the global planner.
_LS_CYCLE = ["-", "--", ":", "-."]
_HATCH_CYCLE = ["", "//", "xx", ".."]
_GPL_SEEN: dict = {}


def unit_local(u):
    return u.split("+")[-1]


def unit_gpl(u):
    return u.split("+")[0] if "+" in u else ""


def _gpl_idx(g):
    if g not in _GPL_SEEN:
        _GPL_SEEN[g] = len(_GPL_SEEN)
    return _GPL_SEEN[g]


def unit_color(u):
    a = unit_local(u)
    if a not in ALGO_COLOR:
        ALGO_COLOR[a] = CMAP(len(ALGO_COLOR) % 10)
    return ALGO_COLOR[a]


def unit_ls(u):
    return _LS_CYCLE[_gpl_idx(unit_gpl(u)) % len(_LS_CYCLE)]


def unit_hatch(u):
    return _HATCH_CYCLE[_gpl_idx(unit_gpl(u)) % len(_HATCH_CYCLE)]


def unit_key(u):
    a = unit_local(u)
    ai = ALGO_ORDER.index(a) if a in ALGO_ORDER else 99
    return (ai, unit_gpl(u))

METRIC_DEFS = [
    ("success", "success rate", True),
    ("sim_time_s", "time to finish [s]", False),
    ("path_length_m", "path length [m]", False),
    ("min_pedestrian_distance_m", "min pedestrian distance [m]", False),
    ("time_waiting_at_light_s", "waiting at lights [s]", False),
    ("collision", "collision rate", True),
]


def draw_map(ax, spec):
    x0, y0, x1, y1 = spec["extent"]
    ax.add_patch(Rectangle((x0 - 6, y0 - 6), x1 - x0 + 12, y1 - y0 + 12,
                           color="#3f7d3f", zorder=0))
    for pl in spec.get("plazas", []):
        a, b, c, d = pl["rect"]
        ax.add_patch(Rectangle((a, b), c - a, d - b, color="#b9d8a9", zorder=1))
    for r in spec["roads"]:
        if r["axis"] == "h":
            ax.add_patch(Rectangle((r["lo"], r["c"] - 3.2), r["hi"] - r["lo"],
                                   6.4, color="#1c1c1c", zorder=2))
        else:
            ax.add_patch(Rectangle((r["c"] - 3.2, r["lo"]), 6.4,
                                   r["hi"] - r["lo"], color="#1c1c1c", zorder=2))
    for s in spec["sidewalks"]:
        a, b, c, d = s["rect"]
        ax.add_patch(Rectangle((a, b), c - a, d - b, color="#c8c8c8", zorder=3))
    for cr in spec.get("crossings", []):
        a, b, c, d = cr["rect"]
        ax.add_patch(Rectangle((a, b), c - a, d - b, facecolor="white",
                               edgecolor="#666666", lw=0.5, alpha=0.9,
                               zorder=4))
    wps = spec["robot"]["waypoints"]
    ax.plot([w[0] for w in wps], [w[1] for w in wps], ls="--", lw=0.9,
            color="black", alpha=0.55, zorder=5)
    ax.set_xlim(x0 - 6, x1 + 6)
    ax.set_ylim(y0 - 6, y1 + 6)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")


def read_trace(p: Path):
    xs, ys = [], []
    if p.exists():
        with p.open() as f:
            for row in csv.DictReader(f):
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
    return xs, ys


def read_trace_rows(p: Path):
    if not p.exists():
        return []
    with p.open() as f:
        return [(float(r["x"]), float(r["y"]), int(r.get("leg", 0)))
                for r in csv.DictReader(f)]


def legs_for(metrics, spec):
    wps = [tuple(w) for w in metrics["waypoints"]]
    return make_legs(wps, spec)


def leg_crossing_spans(leg, spec):
    """Signalised-crossing spans [a0, a1] (local along) intersecting a leg."""
    fr = Frame(leg)
    spans = []
    for cr in spec.get("crossings", []):
        r = cr["rect"]
        corners = [fr.to_local(r[0], r[1]), fr.to_local(r[0], r[3]),
                   fr.to_local(r[2], r[1]), fr.to_local(r[2], r[3])]
        lys = [c[1] for c in corners]
        if max(lys) < -0.3 or min(lys) > leg["W"] + 0.3:
            continue
        lo = min(c[0] for c in corners)
        hi = max(c[0] for c in corners)
        if hi < -0.5 or lo > leg["len"] + 0.5:
            continue
        spans.append((max(lo, 0.0), min(hi, leg["len"])))
    return spans


def strip_axes(fig_title, legs, spec):
    """One horizontal strip subplot per leg: x along, y across the band."""
    n = len(legs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.9 * n + 1.6),
                             squeeze=False)
    axes = [a[0] for a in axes]
    for i, (ax, leg) in enumerate(zip(axes, legs)):
        bw = leg["W"]
        ax.axhspan(0.0, bw, color="#e2e2e2", zorder=0)
        ax.hlines([0.0, bw], 0.0, leg["len"], colors="black", lw=1.3, zorder=2)
        ax.hlines([bw / 2.0], 0.0, leg["len"], colors="#9a9a9a", lw=0.7,
                  linestyles="dashed", zorder=2)
        for a0, a1 in leg_crossing_spans(leg, spec):
            ax.axvspan(a0, a1, color="white", zorder=1)
            ax.axvspan(a0, a1, facecolor="none", edgecolor="#4d8f4d",
                       hatch="//", lw=0.8, zorder=1)
        w0, w1 = leg["w0"], leg["w1"]
        ax.set_ylabel(f"leg {i + 1}" + "\nacross band / m", fontsize=8)
        ax.set_xlim(-3, leg["len"] + 3)
        ax.set_ylim(-0.2, bw + 0.2)
        ax.set_title(f"leg {i + 1}: ({w0[0]:.0f},{w0[1]:.0f}) -> "
                     f"({w1[0]:.0f},{w1[1]:.0f})   [hatched = signalised "
                     f"crossing]", fontsize=8, loc="left")
        ax.grid(True, axis="x", lw=0.3, alpha=0.35)
    axes[-1].set_xlabel("distance along leg / m")
    fig.suptitle(fig_title)
    return fig, axes


def plot_strip_trace(axes, legs, rows, color, lw=1.4, alpha=0.9, label=None):
    first = True
    for i, leg in enumerate(legs):
        fr = Frame(leg)
        xs = [fr.to_local(x, y)[0] for x, y, lg in rows if lg == i]
        ys = [fr.to_local(x, y)[1] for x, y, lg in rows if lg == i]
        if xs:
            axes[i].plot(xs, ys, lw=lw, alpha=alpha, color=color, zorder=6,
                         label=(label if first else None))
            first = False
    return


def strip_outcome(axes, legs, rows, m, color):
    if not rows:
        return
    x, y, lg = rows[-1]
    lg = min(lg, len(legs) - 1)
    lx, ly = Frame(legs[lg]).to_local(x, y)
    ax = axes[lg]
    if m.get("collision"):
        ax.scatter([lx], [ly], marker="x", s=64, color=color, linewidths=2.0,
                   zorder=8)
    elif m.get("success"):
        ax.scatter([lx], [ly], marker="o", s=30, color=color,
                   edgecolors="black", linewidths=0.4, zorder=8)
    else:
        ax.scatter([lx], [ly], marker="s", s=30, color=color,
                   edgecolors="black", linewidths=0.4, zorder=8)


def strip_start_goal(axes, legs):
    s0 = Frame(legs[0]).to_local(*legs[0]["w0"])
    g0 = Frame(legs[-1]).to_local(*legs[-1]["w1"])
    axes[0].scatter([s0[0]], [s0[1]], marker="o", s=95, facecolor="white",
                    edgecolor="black", zorder=9)
    axes[-1].scatter([g0[0]], [g0[1]], marker="*", s=230, facecolor="gold",
                     edgecolor="black", zorder=9)


def outcome_marker(ax, m, xs, ys, color):
    if not xs:
        return
    if m.get("collision"):
        ax.scatter([xs[-1]], [ys[-1]], marker="x", s=60, color=color,
                   linewidths=2.0, zorder=8)
    elif m.get("success"):
        ax.scatter([xs[-1]], [ys[-1]], marker="o", s=26, color=color,
                   edgecolors="black", linewidths=0.4, zorder=8)
    else:
        ax.scatter([xs[-1]], [ys[-1]], marker="s", s=26, color=color,
                   edgecolors="black", linewidths=0.4, zorder=8)


def start_goal(ax, m):
    (sx, sy), (gx, gy) = m["waypoints"][0], m["waypoints"][-1]
    ax.scatter([sx], [sy], marker="o", s=90, facecolor="white",
               edgecolor="black", zorder=9, label="start")
    ax.scatter([gx], [gy], marker="*", s=210, facecolor="gold",
               edgecolor="black", zorder=9, label="goal")




# ---- median-path + quantile-envelope helpers (supervisor item 7) --------
# Metric data is NEVER simplified: projection/median/quantiles use the raw
# trace; Douglas-Peucker (shapely simplify) is applied only to what is DRAWN.
def _route_arrays(wps):
    import numpy as np
    W = np.asarray(wps, float)
    seg = np.diff(W, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    L[L < 1e-9] = 1e-9
    S = np.concatenate([[0.0], np.cumsum(L)])
    T = seg / L[:, None]
    return W, S, T, L


def _project_traj(xs, ys, W, S, T, L):
    import numpy as np
    P = np.stack([xs, ys], axis=1)
    s_out = np.empty(len(P)); d_out = np.empty(len(P))
    for i, q in enumerate(P):
        rel = q[None, :] - W[:-1]
        t = np.clip((rel * T).sum(1) / L, 0.0, 1.0)
        proj = W[:-1] + (t * L)[:, None] * T
        dd = np.hypot(*(q - proj).T)
        j = int(dd.argmin())
        s_out[i] = S[j] + t[j] * L[j]
        n = np.array([-T[j, 1], T[j, 0]])
        d_out[i] = float((q - proj[j]) @ n)
    s_out = np.maximum.accumulate(s_out)   # monotone progress
    return s_out, d_out


def _route_point(sv, W, S, T):
    import numpy as np
    j = np.clip(np.searchsorted(S, sv, side="right") - 1, 0, len(T) - 1)
    base = W[j] + (sv - S[j])[:, None] * T[j]
    n = np.stack([-T[j, 1], T[j, 0]], axis=1)
    return base, n


def envelope_figure(spec, items, algo, mp, mode, color, dpi, out_png,
                    traces=None):
    """Median lateral path + 10-90% envelope across seeds, drawn in world
    coordinates around the planned route. Answers WHERE trajectories
    diverge instead of overplotting every run.

    ``traces`` is an optional list of already-parsed ``(xs, ys)`` tuples,
    parallel to ``items``; pass it when the caller has read the same
    robot_trace.csv files already, so they are not parsed twice."""
    import numpy as np
    import matplotlib.pyplot as plt
    wps = items[0][1].get("waypoints")
    if not wps or len(items) < 3:
        return False
    W, S, T, L = _route_arrays(wps)
    grid = np.linspace(0.0, S[-1], 140)
    D = np.full((len(items), len(grid)), np.nan)
    for k, (_seed, _metrics, d) in enumerate(items):
        if traces is not None:
            xs, ys = traces[k]
        else:
            xs, ys = read_trace(d / "robot_trace.csv")
        if len(xs) < 3:
            continue
        sv, dv = _project_traj(np.asarray(xs), np.asarray(ys), W, S, T, L)
        mask = grid <= sv[-1] + 1e-6
        D[k, mask] = np.interp(grid[mask], sv, dv)
    n_ok = np.sum(~np.isnan(D), axis=0)
    keep = n_ok >= 3
    if keep.sum() < 5:
        return False
    med = np.full(len(grid), np.nan)
    q10 = np.full(len(grid), np.nan)
    q90 = np.full(len(grid), np.nan)
    med[keep] = np.nanmedian(D[:, keep], axis=0)
    q10[keep] = np.nanpercentile(D[:, keep], 10, axis=0)
    q90[keep] = np.nanpercentile(D[:, keep], 90, axis=0)
    base, nvec = _route_point(grid, W, S, T)
    from shapely.geometry import LineString
    lo_pts = base + nvec * q10[:, None]
    hi_pts = base + nvec * q90[:, None]
    allp = np.vstack([lo_pts[keep], hi_pts[keep], np.asarray(wps)])
    vx0, vx1 = allp[:, 0].min() - 5, allp[:, 0].max() + 5
    vy0, vy1 = allp[:, 1].min() - 5, allp[:, 1].max() + 5
    fig_h = max(2.8, min(9.0, 13.0 * (vy1 - vy0) / (vx1 - vx0) + 1.2))
    fig, ax = plt.subplots(figsize=(13.0, max(fig_h, 4.0)))
    # everything past this point must not leak the figure: main() catches
    # exceptions from this function and falls back to an overlay, so an
    # un-closed figure here would accumulate for the whole run.
    try:
        draw_map(ax, spec)
        ax.set_xlim(vx0, vx1)
        ax.set_ylim(vy0, vy1)
        ax.set_aspect("auto")   # vertical exaggeration for the thin band
        ax.plot(*np.asarray(
            LineString(wps).simplify(0.3).coords).T,
            color="0.35", lw=1.0, ls="--", zorder=6, label="planned route")
        lo = base + nvec * q10[:, None]
        hi = base + nvec * q90[:, None]
        band = np.vstack([lo[keep], hi[keep][::-1]])
        ax.fill(band[:, 0], band[:, 1], color=color, alpha=0.25, zorder=6,
                label="10-90% envelope")
        mid = base + nvec * med[:, None]
        mline = np.asarray(LineString(mid[keep]).simplify(0.15).coords)
        ax.plot(mline[:, 0], mline[:, 1], color=color, lw=2.2, zorder=8,
                label="median path")
        succ = sum(1 for _, m, _ in items if m.get("success"))
        ax.set_title(f"{algo.upper()} on {mp} | mode={mode} | median + "
                     f"10-90% envelope over {len(items)} seeds "
                     f"({succ} success, y stretched)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--maps-dir", default=str(REPO / "maps"))
    ap.add_argument("--dpi", type=int, default=190)
    ap.add_argument("--full-map", action="store_true",
                    help="old whole-map view instead of route-strip view")
    ap.add_argument("--unit", choices=["algorithm", "combo", "both"],
                    default="both",
                    help="compare local algorithms (per global planner), "
                         "global+local combinations, or both (default)")
    args = ap.parse_args()
    if args.unit == "both":
        import subprocess
        for u in ("algorithm", "combo"):
            cmd = [sys.executable, __file__, "--results", args.results,
                   "--maps-dir", args.maps_dir, "--dpi", str(args.dpi),
                   "--unit", u]
            if args.full_map:
                cmd.append("--full-map")
            subprocess.run(cmd, check=True)
        return
    res = Path(args.results)
    plots = res / ("plots_combo" if args.unit == "combo" else "plots")
    plots.mkdir(parents=True, exist_ok=True)

    runs = []                # (map_label, mode, algo, seed, metrics, dir)
    for mfile in res.glob("*/*/*/seed_*/robot_metrics.json"):
        m = json.loads(mfile.read_text())
        rt = m.get("route", "default")
        label = m["map"] if rt == "default" else f"{m['map']}[{rt}]"
        tk = m.get("task")
        if tk:
            label += f"[{tk}]"
        gpl = m.get("global_planner", "fixed")
        if args.unit == "combo":
            if gpl != "fixed":
                m["algorithm"] = f"{gpl}+{m['algorithm']}"
        elif gpl != "fixed":
            label += f"{{{gpl}}}"
        m["_label"] = label
        runs.append((label, m["mode"], m["algorithm"], int(m["seed"]),
                     m, mfile.parent))
    if not runs:
        print("no results found")
        return
    specs = {}
    missing = set()
    for _, _, _, _, m, _ in runs:
        lbl = m["_label"]
        if lbl in specs or m["map"] in missing:
            continue
        spec_path = Path(args.maps_dir) / m["map"] / "map_spec.json"
        if not spec_path.exists():
            missing.add(m["map"])
            continue
        specs[lbl] = json.loads(spec_path.read_text())
    if missing:
        print(f"note: skipping results for deleted map(s) {sorted(missing)} "
              f"(map_spec.json not found under {args.maps_dir}/)")
        runs = [r for r in runs if r[4]["map"] not in missing]

    def fig_size(spec):
        x0, y0, x1, y1 = spec["extent"]
        w = 12.0
        return (w, max(3.4, w * (y1 - y0 + 12) / (x1 - x0 + 12) + 1.2))

    # ---- 1) same seed, all algorithms ------------------------------------
    by_scenario = defaultdict(list)
    for mp, mode, algo, seed, m, d in runs:
        by_scenario[(mp, mode, seed)].append((algo, m, d))
    for (mp, mode, seed), items in sorted(by_scenario.items()):
        if len(items) < 2:
            continue
        spec = specs[mp]
        items = sorted(items, key=lambda it: unit_key(it[0]))
        title = (f"{mp} | mode={mode} | seed={seed} -- all algorithms "
                 f"(x collision, o goal, s timeout)")
        if args.full_map:
            fig, ax = plt.subplots(figsize=fig_size(spec))
            draw_map(ax, spec)
            for algo, m, d in items:
                xs, ys = read_trace(d / "robot_trace.csv")
                ax.plot(xs, ys, lw=1.5, color=unit_color(algo),
                        ls=unit_ls(algo), alpha=0.9, zorder=7, label=algo)
                outcome_marker(ax, m, xs, ys, unit_color(algo))
            start_goal(ax, items[0][1])
            ax.set_title(title)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=min(6, len(items) + 2), fontsize=8)
        else:
            legs = legs_for(items[0][1], spec)
            fig, axes = strip_axes(title, legs, spec)
            for algo, m, d in items:
                rows = read_trace_rows(d / "robot_trace.csv")
                plot_strip_trace(axes, legs, rows, unit_color(algo),
                                 label=algo)
                strip_outcome(axes, legs, rows, m, unit_color(algo))
            strip_start_goal(axes, legs)
            handles = [Line2D([0], [0], color=unit_color(a),
                              ls=unit_ls(a), lw=2, label=a)
                       for a, _, _ in items]
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(6, len(items)), fontsize=8,
                       bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout()
        fig.savefig(plots / f"overlay_seed{seed}_{mp}_{mode}.png",
                    dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    # ---- 2) one algorithm, all seeds -------------------------------------
    by_algo = defaultdict(list)
    for mp, mode, algo, seed, m, d in runs:
        by_algo[(mp, mode, algo)].append((seed, m, d))
    seed_cmap = plt.get_cmap("viridis")
    for (mp, mode, algo), items in sorted(by_algo.items()):
        spec = specs[mp]
        base = unit_color(algo)
        items = sorted(items, key=lambda it: it[0])
        succ = sum(1 for _, m, _ in items if m.get("success"))
        title = (f"{algo.upper()} on {mp} | mode={mode} | "
                 f"{succ}/{len(items)} success across seeds "
                 f"(x collision, o goal, s timeout)")
        if len(items) >= 3:
            # supervisor protocol: with many seeds, do NOT overplot
            # polylines -- the median + quantile envelope IS the figure.
            # This test used to sit AFTER the per-seed figure was built and
            # laid out (tight_layout is the expensive part), only to throw
            # that figure away; >=3 seeds is the normal case, so the whole
            # per-seed figure was wasted on nearly every group.  The traces
            # are parsed once here and handed to both consumers below.
            traces = [read_trace(d / "robot_trace.csv")
                      for _sd, _m2, d in items]
            try:
                ok = envelope_figure(spec, items, algo, mp, mode, base,
                                     args.dpi,
                                     plots / f"envelope_{algo}_{mp}_"
                                             f"{mode}.png",
                                     traces=traces)
            except Exception as exc:
                ok = False
                print(f"envelope {algo}/{mp}/{mode} skipped: {exc}")
            if not ok:      # coverage too thin -> fall back to overlay
                fig2, ax2 = plt.subplots(figsize=fig_size(spec))
                draw_map(ax2, spec)
                for (xs2, ys2) in traces:
                    ax2.plot(xs2, ys2, color=base, lw=1.1, alpha=0.6,
                             zorder=7)
                ax2.set_title(f"{algo.upper()} on {mp} | {mode} | "
                              f"{len(items)} seeds (envelope coverage "
                              f"too thin; overlay fallback)")
                fig2.savefig(plots / f"paths_{algo}_{mp}_{mode}.png",
                             dpi=args.dpi, bbox_inches="tight")
                plt.close(fig2)
            continue
        if args.full_map:
            fig, ax = plt.subplots(figsize=fig_size(spec))
            draw_map(ax, spec)
            for seed, m, d in items:
                xs, ys = read_trace(d / "robot_trace.csv")
                ax.plot(xs, ys, lw=1.3, alpha=0.85, zorder=7,
                        label=f"seed {seed}")
                outcome_marker(ax, m, xs, ys, base)
            start_goal(ax, items[0][1])
            ax.set_title(title)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=min(8, len(items) + 2), fontsize=8)
        else:
            legs = legs_for(items[0][1], spec)
            fig, axes = strip_axes(title, legs, spec)
            for k, (seed, m, d) in enumerate(items):
                c = seed_cmap(0.15 + 0.7 * k / max(len(items) - 1, 1))
                rows = read_trace_rows(d / "robot_trace.csv")
                plot_strip_trace(axes, legs, rows, c, label=f"seed {seed}")
                strip_outcome(axes, legs, rows, m, c)
            strip_start_goal(axes, legs)
            handles = [Line2D([0], [0],
                              color=seed_cmap(0.15 + 0.7 * k
                                              / max(len(items) - 1, 1)),
                              lw=2, label=f"seed {sd}")
                       for k, (sd, _, _) in enumerate(items)]
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(8, len(items)), fontsize=8,
                       bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout()
        fig.savefig(plots / f"paths_{algo}_{mp}_{mode}.png",
                    dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    # ---- 3) metric bar charts per (map, mode) ----------------------------
    by_mm = defaultdict(lambda: defaultdict(list))
    for mp, mode, algo, seed, m, d in runs:
        by_mm[(mp, mode)][algo].append(m)
    for (mp, mode), algod in sorted(by_mm.items()):
        algos = sorted(algod, key=unit_key)
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
        for ax, (key, label, is_rate) in zip(axes.flat, METRIC_DEFS):
            vals, errs = [], []
            for a in algos:
                xs = []
                for m in algod[a]:
                    if key not in m:
                        continue
                    v = m[key]
                    if is_rate:
                        xs.append(float(bool(v)))
                        continue
                    # min_pedestrian_distance_m has no value when the run
                    # never saw a pedestrian: the runner writes null (older
                    # runs wrote inf).  Either one has to be dropped, not
                    # averaged -- float(None) raises and inf poisons the mean.
                    if v is None:
                        continue
                    v = float(v)
                    if not math.isfinite(v):
                        continue
                    xs.append(v)
                vals.append(sum(xs) / len(xs) if xs else 0.0)
                n = len(xs)
                if n < 2:
                    errs.append(0.0)
                elif is_rate:
                    # Wilson 95% CI half-width for a proportion
                    z = 1.96
                    ph = vals[-1]
                    den = 1 + z * z / n
                    half = (z * math.sqrt(ph * (1 - ph) / n
                                          + z * z / (4 * n * n))) / den
                    errs.append(half)
                else:
                    mu = vals[-1]
                    sd = (sum((v - mu) ** 2 for v in xs)
                          / (n - 1)) ** 0.5
                    errs.append(1.96 * sd / n ** 0.5)   # 95% CI
            ax.bar(range(len(algos)), vals, yerr=errs, capsize=3,
                   color=[unit_color(a) for a in algos],
                   hatch=[unit_hatch(a) for a in algos],
                   edgecolor="black", linewidth=0.4)
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=8)
            ax.set_title(label, fontsize=10)
            if is_rate:
                ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", lw=0.3, alpha=0.4)
        fig.suptitle(f"{mp} | mode={mode} | seeds per algo: "
                     f"{max(len(v) for v in algod.values())} | "
                     f"bars: mean with 95% CI")
        fig.tight_layout()
        fig.savefig(plots / f"metrics_{mp}_{mode}.png", dpi=args.dpi,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- 3b) occupancy hexbin per (map, mode): where do runs spend time
    import re as _re

    def base_map(lbl):
        return _re.sub(r"\[.*?\]|\{.*?\}", "", lbl).split("__")[0]

    specs = {}
    for _mp in {base_map(r[0]) for r in runs}:
        _sp = Path(args.maps_dir) / _mp / "map_spec.json"
        if _sp.exists():
            specs[_mp] = json.loads(_sp.read_text())
    rundirs = {(r[0], r[1], r[2], r[3]): r[5] for r in runs}

    def fig_size2(spec):
        bx = spec.get("bbox") or [0, 0, 360, 90]
        w = max(bx[2] - bx[0], 40.0)
        h = max(bx[3] - bx[1], 30.0)
        sc = min(12.0 / w, 9.0 / h)
        return (max(7, w * sc), max(4.5, h * sc))

    for (mp, mode), algod in sorted(by_mm.items()):
        spec = specs.get(base_map(mp))
        if spec is None:
            continue
        X, Y = [], []
        for a, ms in algod.items():
            for m in ms:
                d = rundirs.get((mp, mode, a, m.get("seed")))
                if d is None:
                    continue
                xs, ys = read_trace(d / "robot_trace.csv")
                X.extend(xs)
                Y.extend(ys)
        if len(X) < 200:
            continue
        xr = max(X) - min(X)
        vy0, vy1 = min(Y) - 3.0, max(Y) + 3.0
        fig, ax = plt.subplots(
            figsize=(13, max(4.2, min(8.0,
                                      2.4 * 13 * (vy1 - vy0) / (xr + 10)))))
        draw_map(ax, spec)
        hb = ax.hexbin(X, Y, gridsize=(120, 14), cmap="inferno", mincnt=1,
                       bins="log", alpha=0.95, zorder=6, linewidths=0)
        fig.colorbar(hb, ax=ax, label="robot presence (samples, log scale)")
        ax.set_xlim(min(X) - 5, max(X) + 5)
        ax.set_ylim(vy0, vy1)
        ax.set_aspect("auto")   # vertical exaggeration for the thin band
        ax.set_title(f"{mp} | mode={mode} | occupancy over all "
                     f"units & seeds (y stretched)")
        fig.tight_layout()
        fig.savefig(plots / f"occupancy_{mp}_{mode}.png", dpi=args.dpi,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- 3c) campus figure (OSM maps): time-coded LineCollection
    from matplotlib.collections import LineCollection
    import numpy as _np
    for (mp, mode), algod in sorted(by_mm.items()):
        base_mp = base_map(mp)
        spec = specs.get(base_mp)
        if spec is None or not spec.get("osm"):
            continue
        for a, ms in algod.items():
            m = ms[0]
            d = rundirs.get((mp, mode, a, m.get("seed")))
            if d is None:
                continue
            import csv as _csv
            with open(d / "robot_trace.csv") as fh:
                rd = list(_csv.DictReader(fh))
            if len(rd) < 10:
                continue
            pts = _np.array([[float(r["x"]), float(r["y"])]
                             for r in rd])
            tt = _np.array([float(r["t"]) for r in rd])
            segs = _np.stack([pts[:-1], pts[1:]], axis=1)
            fig, ax = plt.subplots(figsize=fig_size2(spec))
            draw_map(ax, spec)
            lc = LineCollection(segs, cmap="viridis", alpha=0.9,
                                linewidths=2.2, zorder=8)
            lc.set_array(tt[:-1])
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, label="time / s")
            ax.set_title(f"{a.upper()} on {base_mp} | {mode} | "
                         f"seed {m.get('seed')} | time-coded trajectory")
            fig.tight_layout()
            fig.savefig(plots / f"campus_{a}_{mp}_{mode}.png",
                        dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    # ---- 4) overall success rate -----------------------------------------
    by_map = defaultdict(lambda: defaultdict(list))
    for mp, mode, algo, seed, m, d in runs:
        by_map[mp][algo].append(1.0 if m.get("success") else 0.0)
    maps = sorted(by_map)
    algos = sorted({a for mp in maps for a in by_map[mp]},
                   key=unit_key)
    fig, ax = plt.subplots(
        figsize=(1.6 + max(2.2, 0.45 * len(algos)) * len(maps), 4.8))
    width = 0.8 / max(len(algos), 1)
    for i, a in enumerate(algos):
        xs = [j + i * width for j in range(len(maps))]
        ys, es = [], []
        for mp in maps:
            v = by_map[mp].get(a, [])
            if not v:
                ys.append(0.0)
                es.append(0.0)
                continue
            n = len(v)
            ph = sum(v) / n
            z = 1.96
            den = 1 + z * z / n
            half = (z * math.sqrt(ph * (1 - ph) / n
                                  + z * z / (4 * n * n))) / den
            ys.append(ph)
            es.append(half)
        ax.bar(xs, ys, width=width, color=unit_color(a), label=a,
               hatch=unit_hatch(a), edgecolor="black", linewidth=0.5,
               yerr=es, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(maps))])
    ax.set_xticklabels(maps)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate (all modes & seeds; Wilson 95% CI)")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=min(4, len(algos)), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "success_overall.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"plots -> {plots}")
    print("models & failure taxonomy (GLMM/LMM, odds ratios, "
          "ranking bootstrap): python analysis/stats_models.py "
          f"--results {args.results}")


if __name__ == "__main__":
    main()
