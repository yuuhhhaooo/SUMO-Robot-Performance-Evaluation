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
_BOLD = {"font.weight": "bold", "axes.titleweight": "bold",
         "axes.labelweight": "bold", "figure.titleweight": "bold"}
matplotlib.rcParams.update(_BOLD)   # every title/label/tick/legend in bold
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

import numpy as np  # noqa: E402

_NAME_MODE = "short"          # --names full: written-out algorithm names
_LAYOUT = "page"              # --layout twocol: final-size paper figures
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
    from algo_names import global_display as _global_disp    # noqa: E402
except ImportError:
    def _global_disp(u):
        return _algo_short(u)


def disp_unit_short(u):
    """Short names regardless of --names: for x-axis-dense charts."""
    u = str(u)
    if "+" in u:
        g, a = u.split("+", 1)
        return f"{_global_disp(g)} + {_algo_short(a)}"
    return _algo_short(u)


def disp_unit(u):
    """Display name: plain local algorithm or 'global+local' combination.
    File names keep the raw ids on purpose (stable, greppable)."""
    u = str(u)
    if "+" in u:
        g, a = u.split("+", 1)
        return f"{_global_disp(g)} + {_algo_display(a)}"
    return _algo_display(u)


# paired-family row order for the success heat map (single source; anything
# not listed is appended after, sorted)
FAMILY_ORDER = [("sarl", "sarl_upstream"), ("cadrl", "cadrl_upstream"),
                ("lstm_rl", "lstm_rl_upstream"), ("teb", "teb_upstream"),
                ("mpc", "mpc_dompc"), ("orca", "orca_heuristic"),
                ("crowdnav_dsrnn", "crowdnav_attngraph"),
                ("dwa",), ("astar",), ("dijkstra",), ("rrt",)]

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
    ("success", "Success Rate", True),
    ("sim_time_s", "Time to Finish [s]", False),
    ("path_length_m", "Path Length [m]", False),
    ("min_pedestrian_distance_m", "Min. Pedestrian Distance [m]", False),
    ("time_waiting_at_light_s", "Waiting at Lights [s]", False),
    ("collision", "Collision Rate", True),
]


# ---- real-map background from the SUMO net --------------------------------
# The rect-based map_spec is a faithful drawing of the SYNTHETIC maps but only
# a cartoon of an OSM-imported map: real streets are not axis-aligned
# rectangles. For maps listed in --net-bg-maps the background is rendered
# from the actual net.xml lane geometry (metre-true buffered polygons), in
# the same coordinate frame as robot_trace.csv, with the same palette as the
# rect maps so the two styles read alike.
# muted publication palette for the real-map background: the data must pop,
# the map must whisper (the vivid palette stays for the synthetic rect maps)
NET_BG, NET_ROAD = "#f8f8f6", "#e3e3e3"
NET_WALK, NET_CROSS = "#cbc5b8", "#ffffff"
NET_WALK_EDGE = "#a9a294"          # hairline: keeps thin footpaths crisp
_NET_CACHE: dict = {}
_TILE_STATE: dict = {"down": False}
_SPREAD_INDEX: list = []          # per-envelope divergence stats -> csv
_SPREAD_CURVES: dict = {}         # (cell, mode) -> [(unit, s, spread, color)]
                                  # for the cross-unit divergence comparison


def _load_net_polys(net_path: Path):
    import xml.etree.ElementTree as ET
    from shapely.geometry import LineString, Polygon
    polys = []                                  # (xy array, colour, zorder)
    root = ET.parse(net_path).getroot()
    bounds, offset, proj4 = None, (0.0, 0.0), None
    loc = root.find("location")
    if loc is not None:
        if loc.get("convBoundary"):
            b = [float(v) for v in loc.get("convBoundary").split(",")]
            if len(b) == 4:
                bounds = b
        if loc.get("netOffset"):
            o = [float(v) for v in loc.get("netOffset").split(",")]
            if len(o) == 2:
                offset = (o[0], o[1])
        proj4 = loc.get("projParameter")

    def _shape(s):
        return [tuple(float(v) for v in p.split(",")[:2])
                for p in s.strip().split()]

    def _add(geom, col, z, edge=None):
        for g in getattr(geom, "geoms", [geom]):
            try:
                polys.append((np.asarray(g.exterior.coords), col, z, edge))
            except Exception:
                pass

    for edge in root.iter("edge"):
        fn = edge.get("function", "")
        e_allow = edge.get("allow") or ""
        for lane in edge.iter("lane"):
            s = lane.get("shape")
            if not s:
                continue
            pts = _shape(s)
            if len(pts) < 2:
                continue
            if fn == "walkingarea":
                try:
                    _add(Polygon(pts).buffer(0), NET_WALK, 3,
                         NET_WALK_EDGE)
                except Exception:
                    pass
                continue
            width = float(lane.get("width") or 3.2)
            allow = set((lane.get("allow") or e_allow).split())
            try:
                geom = LineString(pts).buffer(max(width, 0.8) / 2.0,
                                              cap_style=2, join_style=2)
            except Exception:
                continue
            if fn == "crossing":
                _add(geom, NET_CROSS, 4, "#bdbdbd")
            elif allow and allow <= {"pedestrian"}:
                _add(geom, NET_WALK, 3, NET_WALK_EDGE)   # sidewalk
            else:
                _add(geom, NET_ROAD, 2)         # carriageway (incl. mixed)
    if bounds is None and polys:
        allxy = np.vstack([p[0] for p in polys])
        bounds = [allxy[:, 0].min(), allxy[:, 1].min(),
                  allxy[:, 0].max(), allxy[:, 1].max()]
    return polys, bounds, offset, proj4


def _tile_background(ax, spec):
    """Real OSM basemap tiles (supervisor: 'contextily for the basemap'),
    registered into net coordinates via the net's own projParameter +
    netOffset, so traces plot unchanged. Returns False on any failure
    (offline, missing deps, unprojected synthetic net) -- caller falls
    back to the muted net-geometry drawing."""
    geo = spec.get("_net_geo")
    if not geo or _TILE_STATE.get("down"):
        return False
    (dx, dy), proj4, bounds = geo
    if not proj4 or proj4.strip() in ("!", ""):
        return False                    # synthetic net: nothing to register
    try:
        import contextily as ctx
        from pyproj import Proj, Transformer
    except ImportError:
        print("note: real basemap needs 'pip install contextily pyproj'; "
              "drawing net geometry instead")
        return False
    try:
        p = Proj(proj4)
        x0, y0, x1, y1 = bounds
        lon0, lat0 = p(x0 - dx, y0 - dy, inverse=True)
        lon1, lat1 = p(x1 - dx, y1 - dy, inverse=True)
        w, e = sorted((lon0, lon1))
        s, n = sorted((lat0, lat1))
        try:
            src = ctx.providers.CartoDB.Positron     # light, data-friendly
        except Exception:
            src = None
        img, ext = ctx.bounds2img(w, s, e, n, ll=True,
                                  **({"source": src} if src else {}))
        # tiles snap outwards: map the ACTUAL mercator extent back into
        # net coordinates, or the image lands a few metres off
        t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lo0, la0 = t.transform(ext[0], ext[2])
        lo1, la1 = t.transform(ext[1], ext[3])
        nx0, ny0 = p(lo0, la0)
        nx1, ny1 = p(lo1, la1)
        ax.imshow(img, extent=[nx0 + dx, nx1 + dx, ny0 + dy, ny1 + dy],
                  zorder=0, interpolation="bilinear")
        return True
    except Exception as exc:
        _TILE_STATE["down"] = True          # try once per run, not per figure
        print(f"note: basemap tiles unavailable ({type(exc).__name__}: "
              f"{exc}); drawing net geometry instead")
        return False


def attach_net_background(spec, mp, maps_dir, net_maps, basemap="auto"):
    """Cache-parse maps/<mp>/*.net.xml into spec['_net_polys'] when asked."""
    if basemap == "rect" or net_maps == "none" \
            or (net_maps != "all" and mp not in net_maps.split(",")):
        return
    if mp not in _NET_CACHE:
        cand = sorted(Path(maps_dir, mp).glob("*.net.xml"))
        _NET_CACHE[mp] = _load_net_polys(cand[0]) if cand else None
        if _NET_CACHE[mp] is None:
            print(f"note: net background requested for {mp} but no "
                  f"*.net.xml under {maps_dir}/{mp}; using rect map_spec")
    if _NET_CACHE[mp]:
        polys, bounds, offset, proj4 = _NET_CACHE[mp]
        spec["_net_polys"] = (polys, bounds)
        spec["_net_geo"] = (offset, proj4, bounds)
        spec["_basemap"] = basemap


def draw_map(ax, spec):
    net = spec.get("_net_polys")
    if net:
        polys, bounds = net
        x0, y0, x1, y1 = bounds or spec.get("extent", [0, 0, 100, 100])
        style = spec.get("_basemap", "auto")
        tiled = style in ("auto", "tiles") and _tile_background(ax, spec)
        if not tiled:
            ax.add_patch(Rectangle((x0 - 6, y0 - 6), x1 - x0 + 12,
                                   y1 - y0 + 12, color=NET_BG, zorder=0))
            for xy, col, z, edge in polys:
                ax.fill(xy[:, 0], xy[:, 1], color=col, zorder=z,
                        lw=0.25 if edge else 0, edgecolor=edge)
        wps = (spec.get("robot") or {}).get("waypoints") or []
        # the spec's demo route is UNRELATED to task episodes -- drawing it
        # dashed misled a whole reading session; episode routes are drawn
        # by each figure from robot_metrics.json instead
        if wps and spec.get("_draw_default_route", False):
            ax.plot([w[0] for w in wps], [w[1] for w in wps], ls="--",
                    lw=0.9, color="black", alpha=0.55, zorder=5)
        ax.set_xlim(x0 - 6, x1 + 6)
        ax.set_ylim(y0 - 6, y1 + 6)
        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        return
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
    """Trace rows can be truncated mid-write when a run is killed; skip
    any row that does not parse instead of aborting the whole plot run."""
    xs, ys = [], []
    if p.exists():
        with p.open() as f:
            for row in csv.DictReader(f):
                try:
                    x, y = float(row["x"]), float(row["y"])
                except (TypeError, ValueError, KeyError):
                    continue
                xs.append(x)
                ys.append(y)
    return xs, ys


def read_trace_rows(p: Path):
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                out.append((float(r["x"]), float(r["y"]),
                            int(r.get("leg") or 0)))
            except (TypeError, ValueError, KeyError):
                continue
    return out


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
        ax.set_ylabel(f"Leg {i + 1}" + "\nAcross Band / m", fontsize=8)
        ax.set_xlim(-3, leg["len"] + 3)
        ax.set_ylim(-0.2, bw + 0.2)
        ax.set_title(f"Leg {i + 1}: ({w0[0]:.0f},{w0[1]:.0f}) -> "
                     f"({w1[0]:.0f},{w1[1]:.0f})   [Hatched = Signalised "
                     f"Crossing]", fontsize=8, loc="left")
        ax.grid(True, axis="x", lw=0.3, alpha=0.35)
    axes[-1].set_xlabel("Distance Along Leg / m")
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
               edgecolor="black", zorder=9, label="Start")
    ax.scatter([gx], [gy], marker="*", s=210, facecolor="gold",
               edgecolor="black", zorder=9, label="Goal")




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


def _project_traj(xs, ys, W, S, T, L, quality=None):
    """Project a trajectory onto the planned route as (arc length s,
    lateral offset d).

    Global nearest-segment projection is ill-posed on routes that pass
    near themselves (RRT tours on real maps fold back constantly): the
    nearest segment then jumps between far-apart s values and the
    re-embedded median cuts across blocks. Cure: a CONTINUITY PRIOR --
    after locking on, each point may only project onto segments within a
    forward/backward window of the previous s, sized by the distance the
    robot actually moved. `quality`, when given a dict, receives
    diagnostics (max |d|, fraction of window escapes) so the caller can
    refuse to draw a still-pathological cell."""
    import numpy as np
    P = np.stack([xs, ys], axis=1)
    ns = len(P)
    s_out = np.empty(ns); d_out = np.empty(ns)
    seg_lo, seg_hi = S[:-1], S[1:]
    s_prev = None
    escapes = 0
    for i, q in enumerate(P):
        rel = q[None, :] - W[:-1]
        t = np.clip((rel * T).sum(1) / L, 0.0, 1.0)
        proj = W[:-1] + (t * L)[:, None] * T
        dd = np.hypot(*(q - proj).T)
        if s_prev is None:
            j = int(dd.argmin())
        else:
            step = float(np.hypot(*(P[i] - P[i - 1]))) if i else 0.0
            lo = s_prev - 12.0                 # small backtracks allowed
            hi = s_prev + 3.0 * step + 8.0     # generous forward window
            mask = (seg_hi >= lo) & (seg_lo <= hi)
            if not mask.any():
                mask[:] = True
            dd_w = np.where(mask, dd, np.inf)
            j = int(dd_w.argmin())
            # if even the windowed best is far off the route, the robot
            # genuinely left the corridor -- fall back to global nearest
            # once, and count the escape for the quality gate
            if dd[j] > 20.0:
                jg = int(dd.argmin())
                if dd[jg] < dd[j] - 5.0:
                    j = jg
                    escapes += 1
        s_out[i] = S[j] + t[j] * L[j]
        n = np.array([-T[j, 1], T[j, 0]])
        d_out[i] = float((q - proj[j]) @ n)
        s_prev = s_out[i]
    if isinstance(quality, dict):
        quality["max_abs_d"] = float(np.max(np.abs(d_out))) if ns else 0.0
        quality["escape_frac"] = escapes / max(ns, 1)
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
    # a single-route envelope presumes ONE shared route; stochastic
    # globals (RRT) replan per episode, so check the other seeds' routes
    # against seed 0's before projecting anything onto it
    import numpy as _np

    def _route_dev(w2):
        P = _np.asarray(w2, float)
        rel = P[:, None, :] - W[None, :-1, :]
        t = _np.clip((rel * T[None]).sum(-1) / L[None], 0.0, 1.0)
        proj = W[None, :-1, :] + t[..., None] * (L[:, None] * T)[None]
        return float(_np.hypot(*(P[:, None, :] - proj)
                               .transpose(2, 0, 1)).min(1).max())
    hetero = sum(1 for _s, m, _d in items[1:]
                 if len(m.get("waypoints") or []) >= 2
                 and _route_dev(m["waypoints"]) > 15.0)
    if hetero > len(items) // 2:
        print(f"  note: envelope skipped for {algo} on {mp} -- seeds "
              f"follow materially different global routes "
              f"({hetero}/{len(items) - 1} deviate >15 m from seed 0's; "
              f"stochastic replanning), a single-route envelope is "
              f"undefined; drawing the overlay instead")
        return False
    grid = np.linspace(0.0, S[-1], 140)
    D = np.full((len(items), len(grid)), np.nan)
    bad = 0
    for k, (_seed, _metrics, d) in enumerate(items):
        if traces is not None:
            xs, ys = traces[k]
        else:
            xs, ys = read_trace(d / "robot_trace.csv")
        if len(xs) < 3:
            continue
        q = {}
        sv, dv = _project_traj(np.asarray(xs), np.asarray(ys), W, S, T, L,
                               quality=q)
        if q.get("max_abs_d", 0) > 30.0 or q.get("escape_frac", 0) > 0.10:
            bad += 1              # projection still ill-posed for this run
            continue
        mask = grid <= sv[-1] + 1e-6
        D[k, mask] = np.interp(grid[mask], sv, dv)
    if bad > len(items) // 2:
        print(f"  note: envelope skipped for {algo} on {mp} -- arc-length projection onto the shared route is "
              f"ill-posed (self-overlap and/or off-route tours) for "
              f"{bad}/{len(items)} runs; drawing the overlay instead")
        return False
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
    real_bg = bool(spec.get("_net_polys"))
    fig_h = max(2.8, min(12.0 if real_bg else 9.0,
                         13.0 * (vy1 - vy0) / (vx1 - vx0) + 1.2))
    if _LAYOUT == "twocol":
        fig_w = 7.05
        map_h = max(3.0, min(6.6, fig_w * (vy1 - vy0)
                             / max(vx1 - vx0, 1e-9) + 0.4))
        strip_h = 1.05
    else:
        fig_w, map_h, strip_h = 13.0, max(fig_h, 4.0), 1.35
    fig = plt.figure(figsize=(fig_w, map_h + strip_h + 0.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[map_h, strip_h],
                          hspace=0.34)
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1])
    # everything past this point must not leak the figure: main() catches
    # exceptions from this function and falls back to an overlay, so an
    # un-closed figure here would accumulate for the whole run.
    try:
        draw_map(ax, spec)
        ax.set_xlim(vx0, vx1)
        ax.set_ylim(vy0, vy1)
        # thin synthetic corridors are vertically exaggerated on purpose;
        # a real map must keep true geometry or the streets look wrong
        ax.set_aspect("equal" if real_bg else "auto")
        ax.plot(*np.asarray(
            LineString(wps).simplify(0.3).coords).T,
            color="0.35", lw=1.0, ls="--", zorder=6, label="Planned Route")
        lo = base + nvec * q10[:, None]
        hi = base + nvec * q90[:, None]
        band = np.vstack([lo[keep], hi[keep][::-1]])
        ax.fill(band[:, 0], band[:, 1], color=color, alpha=0.25, zorder=6,
                label="10-90% Envelope")
        mid = base + nvec * med[:, None]
        mline = np.asarray(LineString(mid[keep]).simplify(0.15).coords)
        ax.plot(mline[:, 0], mline[:, 1], color=color, lw=2.2, zorder=8,
                label="Median Path")
        succ = sum(1 for _, m, _ in items if m.get("success"))
        if _LAYOUT == "twocol":
            ax.set_title(f"{disp_unit(algo)} -- {mp}", fontsize=10,
                         pad=20)
            ax.text(1.0, 1.002, f"median + 10-90% envelope, "
                    f"{len(items)} seeds ({succ} success)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="0.35")
            ax.tick_params(labelsize=8)
            ax.set_xlabel("x / m", fontsize=9)
            ax.set_ylabel("y / m", fontsize=9)
        else:
            ax.set_title(f"{disp_unit(algo)} on {mp} | mode={mode} | "
                         f"Median + 10-90% Envelope Over {len(items)} "
                         f"Seeds ({succ} Success"
                         + ("" if real_bg else ", y stretched") + ")")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                  ncol=3, fontsize=7.5 if _LAYOUT == "twocol" else 8)
        # divergence profile: WHERE along the route the seeds start to
        # disagree -- flat near zero = agreement, rising = divergence onset
        spread = q90 - q10
        axp.fill_between(grid[keep], 0, spread[keep], color=color,
                         alpha=0.30, lw=0)
        axp.plot(grid[keep], spread[keep], color=color, lw=1.4)
        axp.set_xlim(0, S[-1])
        axp.set_ylim(0, max(1.0, float(np.nanmax(spread[keep])) * 1.15))
        axp.set_xlabel("Distance Along the Planned Route [m]",
                       fontsize=8.5 if _LAYOUT == "twocol" else 9)
        axp.set_ylabel("10-90% Lateral\nSpread [m]",
                       fontsize=8 if _LAYOUT == "twocol" else 8.5)
        axp.tick_params(labelsize=7.5 if _LAYOUT == "twocol" else 9)
        axp.grid(lw=0.3, alpha=0.35)
        axp.set_axisbelow(True)
        for _s in ("top", "right"):
            axp.spines[_s].set_visible(False)
        sp = spread[keep]
        _SPREAD_CURVES.setdefault((mp, mode), []).append(
            (algo, grid[keep].copy(), sp.copy(), color))
        _SPREAD_INDEX.append({
            "cell": mp, "mode": mode, "unit": algo,
            "n_seeds": len(items), "route_m": round(float(S[-1]), 1),
            "coverage": round(float(keep.sum()) / len(grid), 2),
            "mean_spread_m": round(float(np.nanmean(sp)), 2),
            "max_spread_m": round(float(np.nanmax(sp)), 2),
            "s_at_max_m": round(float(grid[keep][int(np.nanargmax(sp))]), 1),
            "file": out_png.name})
        _savefig(fig, out_png, dpi)
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
    ap.add_argument("--max-strip-legs", type=int, default=6,
                    help="route-strip view only up to this many legs; "
                         "beyond it (real-map tours) fall back to the "
                         "whole-map view automatically")
    ap.add_argument("--net-bg-maps", default="map5_ucl",
                    help="comma list of maps whose background is drawn "
                         "from the SUMO net.xml instead of the rect "
                         "map_spec ('all' / 'none'); default: map5_ucl")
    ap.add_argument("--basemap", choices=["auto", "tiles", "net", "rect"],
                    default="net",
                    help="background for --net-bg-maps: real OSM tiles via "
                         "contextily (registered through the net's "
                         "projParameter), net lane geometry, or the rect "
                         "map_spec; auto = tiles when available, else net")
    ap.add_argument("--only-cell", default="*",
                    help="fnmatch filter on the map[task] label, e.g. "
                         "'map5_ucl*t03*' -- regenerate one exemplar at "
                         "print dpi without re-running everything")
    ap.add_argument("--only-unit", default="*",
                    help="fnmatch filter on the unit, e.g. '*crowdnav*' "
                         "or 'rrt+*'")
    ap.add_argument("--layout", choices=["page", "twocol"], default="page",
                    help="twocol: envelope/occupancy/campus designed at "
                         "IEEE textwidth with final point sizes -- "
                         "include 1:1, never rescale")
    ap.add_argument("--names", choices=["short", "full"], default="short",
                    help="axis labels: standardized abbreviations (short) "
                         "or written-out algorithm names (full); the "
                         "6-metric bar chart keeps short names either way")
    ap.add_argument("--pdf", action="store_true",
                    help="also write a vector .pdf next to every .png")
    ap.add_argument("--unit", choices=["algorithm", "combo", "both"],
                    default="both",
                    help="compare local algorithms (per global planner), "
                         "global+local combinations, or both (default)")
    args = ap.parse_args()
    global _NAME_MODE, _LAYOUT, _SAVE_PDF
    _NAME_MODE = args.names
    _LAYOUT = args.layout
    _SAVE_PDF = args.pdf
    if args.layout == "twocol":
        # final-size two-column figures: bump default fonts one notch
        plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11,
                             "axes.labelsize": 10, "xtick.labelsize": 9,
                             "ytick.labelsize": 9, "legend.fontsize": 8.5})
    if args.unit == "both":
        import subprocess
        for u in ("algorithm", "combo"):
            cmd = [sys.executable, __file__, "--results", args.results,
                   "--maps-dir", args.maps_dir, "--dpi", str(args.dpi),
                   "--max-strip-legs", str(args.max_strip_legs),
                   "--net-bg-maps", args.net_bg_maps,
                   "--basemap", args.basemap,
                   "--names", args.names,
                   "--layout", args.layout,
                   "--only-cell", args.only_cell,
                   "--only-unit", args.only_unit,
                   "--unit", u]
            if args.pdf:
                cmd.append("--pdf")
            if args.full_map:
                cmd.append("--full-map")
            subprocess.run(cmd, check=True)
        return
    res = Path(args.results)
    plots = res / ("plots_combo" if args.unit == "combo" else "plots")
    plots.mkdir(parents=True, exist_ok=True)

    runs = []                # (map_label, mode, algo, seed, metrics, dir)
    noplan_runs = []         # global_plan_failed: stats yes, trajectories no
    INFRA_REASONS = {"sumo_crash"}   # global_plan_failed now counts AGAINST the combination (user ruling): the global half failing to route is a failure of the deployed stack, reported as its own category
    infra_count = 0
    noplan_count = 0
    for mfile in res.glob("*/*/*/seed_*/robot_metrics.json"):
        m = json.loads(mfile.read_text())
        # infrastructure outcomes are measurements of the rig, not of an
        # algorithm: keep them out of every figure (same convention as
        # quicklook.py / stats_models.py), with visible accounting
        tr0 = str(m.get("termination_reason", "")).split(":")[0]
        if tr0 in INFRA_REASONS:
            infra_count += 1
            continue
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
        if tr0 == "global_plan_failed":
            # counts against the combination (same ruling as
            # stats_models.py), but has no trajectory: keep it for the
            # success/metric aggregations, never for trajectory figures.
            # Dropping these runs entirely silently INFLATED the plotted
            # success of RRT-global combinations relative to the stats.
            noplan_count += 1
            noplan_runs.append((label, m["mode"], m["algorithm"],
                                int(m["seed"]), m, None))
            continue
        runs.append((label, m["mode"], m["algorithm"], int(m["seed"]),
                     m, mfile.parent))
    if infra_count:
        print(f"{infra_count} infrastructure episodes (termination_reason "
              f"in {sorted(INFRA_REASONS)}) EXCLUDED from all plots")
    if noplan_count:
        print(f"{noplan_count} global_plan_failed episodes: counted as "
              f"failures in the success/metric figures, skipped in "
              f"trajectory plots (no trajectory exists)")
    if args.only_cell != "*" or args.only_unit != "*":
        from fnmatch import fnmatch
        before = len(runs)
        runs = [r for r in runs if fnmatch(r[0], args.only_cell)
                and fnmatch(str(r[2]), args.only_unit)]
        noplan_runs = [r for r in noplan_runs
                       if fnmatch(r[0], args.only_cell)
                       and fnmatch(str(r[2]), args.only_unit)]
        print(f"filter: {len(runs)}/{before} runs match "
              f"--only-cell '{args.only_cell}' --only-unit "
              f"'{args.only_unit}'")
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
        attach_net_background(specs[lbl], m["map"], args.maps_dir,
                              args.net_bg_maps, args.basemap)
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
        title = (f"{mp} | mode={mode} | seed={seed} -- All Algorithms "
                 f"(x Collision, o Goal, s Timeout)")
        legs = legs_for(items[0][1], spec)
        # real-map tours have dozens of waypoints: one strip subplot per leg
        # stops being a figure -- fall back to the whole-map view
        use_full = args.full_map or len(legs) > args.max_strip_legs
        if use_full:
            fig, ax = plt.subplots(figsize=fig_size(spec))
            draw_map(ax, spec)
            for algo, m, d in items:
                xs, ys = read_trace(d / "robot_trace.csv")
                rt = m.get("waypoints") or []
                if len(rt) >= 2:      # THIS episode's planned route
                    ax.plot([w[0] for w in rt], [w[1] for w in rt],
                            ls="--", lw=0.8, color=unit_color(algo),
                            alpha=0.45, zorder=5)
                ax.plot(xs, ys, lw=1.5, color=unit_color(algo),
                        ls=unit_ls(algo), alpha=0.9, zorder=7,
                        label=disp_unit(algo))
                outcome_marker(ax, m, xs, ys, unit_color(algo))
            start_goal(ax, items[0][1])
            ax.set_title(title)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=min(6, len(items) + 2), fontsize=8)
        else:
            fig, axes = strip_axes(title, legs, spec)
            for algo, m, d in items:
                rows = read_trace_rows(d / "robot_trace.csv")
                plot_strip_trace(axes, legs, rows, unit_color(algo),
                                 label=disp_unit(algo))
                strip_outcome(axes, legs, rows, m, unit_color(algo))
            strip_start_goal(axes, legs)
            handles = [Line2D([0], [0], color=unit_color(a),
                              ls=unit_ls(a), lw=2, label=disp_unit(a))
                       for a, _, _ in items]
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(6, len(items)), fontsize=8,
                       bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout()
        _savefig(fig, plots / f"overlay_seed{seed}_{mp}_{mode}.png", args.dpi)
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
        title = (f"{disp_unit(algo)} on {mp} | mode={mode} | "
                 f"{succ}/{len(items)} Success Across Seeds "
                 f"(x Collision, o Goal, s Timeout)")
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
            if not ok:      # routes differ (stochastic global) or coverage
                # too thin -> per-seed overlay: one colour per seed, with
                # THAT seed's planned route dashed in the same colour, so
                # the route fan of a stochastic global is visible
                fig2, ax2 = plt.subplots(figsize=fig_size(spec))
                draw_map(ax2, spec)
                for k, ((sd2, m2, _d2), (xs2, ys2)) in enumerate(
                        zip(items, traces)):
                    c2 = seed_cmap(0.10 + 0.8 * k
                                   / max(len(items) - 1, 1))
                    rt2 = m2.get("waypoints") or []
                    if len(rt2) >= 2:
                        ax2.plot([w[0] for w in rt2],
                                 [w[1] for w in rt2], ls="--", lw=0.7,
                                 color=c2, alpha=0.45, zorder=5)
                    ax2.plot(xs2, ys2, color=c2, lw=1.4, alpha=0.9,
                             zorder=7, label=f"Seed {sd2}")
                    outcome_marker(ax2, m2, xs2, ys2, c2)
                start_goal(ax2, items[0][1])
                ax2.set_title(f"{disp_unit(algo)} on {mp} | {mode} | "
                              f"One Colour per Seed (Dashed = That "
                              f"Seed's Planned Route)")
                ax2.legend(loc="upper center",
                           bbox_to_anchor=(0.5, -0.14),
                           ncol=min(6, len(items)), fontsize=8)
                _savefig(fig2, plots / f"paths_{algo}_{mp}_{mode}.png", args.dpi)
                plt.close(fig2)
            continue
        legs = legs_for(items[0][1], spec)
        use_full = args.full_map or len(legs) > args.max_strip_legs
        if use_full:
            fig, ax = plt.subplots(figsize=fig_size(spec))
            draw_map(ax, spec)
            for seed, m, d in items:
                xs, ys = read_trace(d / "robot_trace.csv")
                rt = m.get("waypoints") or []
                if len(rt) >= 2:      # per-seed planned route: a FAN when
                    ax.plot([w[0] for w in rt],   # the global replans
                            [w[1] for w in rt], ls="--", lw=0.7,
                            color="0.25", alpha=0.35, zorder=5)
                ax.plot(xs, ys, lw=1.3, alpha=0.85, zorder=7,
                        label=f"Seed {seed}")
                outcome_marker(ax, m, xs, ys, base)
            start_goal(ax, items[0][1])
            ax.set_title(title)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                      ncol=min(8, len(items) + 2), fontsize=8)
        else:
            fig, axes = strip_axes(title, legs, spec)
            for k, (seed, m, d) in enumerate(items):
                c = seed_cmap(0.15 + 0.7 * k / max(len(items) - 1, 1))
                rows = read_trace_rows(d / "robot_trace.csv")
                plot_strip_trace(axes, legs, rows, c, label=f"Seed {seed}")
                strip_outcome(axes, legs, rows, m, c)
            strip_start_goal(axes, legs)
            handles = [Line2D([0], [0],
                              color=seed_cmap(0.15 + 0.7 * k
                                              / max(len(items) - 1, 1)),
                              lw=2, label=f"Seed {sd}")
                       for k, (sd, _, _) in enumerate(items)]
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(8, len(items)), fontsize=8,
                       bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout()
        _savefig(fig, plots / f"paths_{algo}_{mp}_{mode}.png", args.dpi)
        plt.close(fig)

    # ---- 3) metric bar charts per (map, mode) ----------------------------
    # noplan runs join every statistic here (success 0, no behavioural
    # values); the trajectory sections above never saw them
    by_mm = defaultdict(lambda: defaultdict(list))
    for mp, mode, algo, seed, m, d in runs + noplan_runs:
        by_mm[(mp, mode)][algo].append(m)
    for (mp, mode), algod in sorted(by_mm.items()):
        algos = sorted(algod, key=unit_key)
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
        for ax, (key, label, is_rate) in zip(axes.flat, METRIC_DEFS):
            vals, errs, ns = [], [], []
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
                    # "time to finish" and "path length" are only defined for
                    # runs that actually finished. Averaging failed runs mixes
                    # truncated collisions and --max-time caps into the bar and
                    # INVERTS the ranking relative to the mixed models, which
                    # restrict to successes: on map3_grid the two fastest-
                    # looking bars were the two algorithms with zero successes.
                    if key in ("sim_time_s", "path_length_m") \
                            and not m.get("success"):
                        continue
                    v = float(v)
                    if not math.isfinite(v):
                        continue
                    xs.append(v)
                # NaN, not 0.0: matplotlib omits the bar entirely. A 0.0 here
                # reads as "the robot was touching a pedestrian" when the truth
                # is "no usable sample".
                vals.append(sum(xs) / len(xs) if xs else float("nan"))
                ns.append(len(xs))
                n = len(xs)
                if n < 2:
                    errs.append(0.0)
                elif is_rate:
                    # Wilson 95% interval. The half-width must be centred on
                    # the Wilson CENTRE, not on the raw proportion -- centring
                    # on p-hat produced intervals reaching negative success
                    # rates for 0/n cells and roughly half the true width.
                    z = 1.96
                    ph = vals[-1]
                    den = 1 + z * z / n
                    centre = (ph + z * z / (2 * n)) / den
                    half = (z * math.sqrt(ph * (1 - ph) / n
                                          + z * z / (4 * n * n))) / den
                    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
                    errs.append((max(0.0, ph - lo), max(0.0, hi - ph)))
                else:
                    mu = vals[-1]
                    sd = (sum((v - mu) ** 2 for v in xs)
                          / (n - 1)) ** 0.5
                    errs.append(1.96 * sd / n ** 0.5)   # 95% CI
            # errs mixes scalars and (lo, hi) pairs; normalise to a 2xN array
            lo_err = [e[0] if isinstance(e, tuple) else e for e in errs]
            hi_err = [e[1] if isinstance(e, tuple) else e for e in errs]
            ax.bar(range(len(algos)), vals, yerr=[lo_err, hi_err], capsize=3,
                   color=[unit_color(a) for a in algos],
                   hatch=[unit_hatch(a) for a in algos],
                   edgecolor="black", linewidth=0.4)
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels([disp_unit_short(a) for a in algos],
                               rotation=45, ha="right", fontsize=8)
            ax.set_title(label, fontsize=10)
            if is_rate:
                ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", lw=0.3, alpha=0.4)
        fig.suptitle(f"{mp} | mode={mode} | Seeds per Algorithm: "
                     f"{max(len(v) for v in algod.values())} | "
                     f"Bars: Mean With 95% CI")
        fig.tight_layout()
        _savefig(fig, plots / f"metrics_{mp}_{mode}.png", args.dpi)
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
            attach_net_background(specs[_mp], _mp, args.maps_dir,
                                  args.net_bg_maps, args.basemap)
    rundirs = {(r[0], r[1], r[2], r[3]): r[5] for r in runs}

    def fig_size2(spec):
        bx = spec.get("bbox") or spec.get("extent") or [0, 0, 360, 90]
        w = max(bx[2] - bx[0], 40.0)
        h = max(bx[3] - bx[1], 30.0)
        sc = min(12.0 / w, 9.0 / h)
        return (max(7, w * sc), max(4.5, h * sc))

    _mapcov: dict = {}     # (base_map, mode) -> pooled points, coverage
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
        # accumulate for the per-map COVERAGE exhibit (rendered after this
        # loop): not an analysis figure -- it shows how much of the map the
        # sampled tasks exercise. Reuses the in-memory traces, no re-read.
        if X:
            acc = _mapcov.setdefault(
                (base_map(mp), mode),
                {"X": [], "Y": [], "sg": [], "cells": 0})
            acc["X"].append(np.asarray(X))
            acc["Y"].append(np.asarray(Y))
            acc["cells"] += 1
            m0a = next(iter(algod.values()))[0]
            wpsa = m0a.get("waypoints") or []
            if len(wpsa) >= 2:
                acc["sg"].append((tuple(wpsa[0]), tuple(wpsa[-1])))
        if len(X) < 200:
            continue
        xr = max(X) - min(X)
        vy0, vy1 = min(Y) - 3.0, max(Y) + 3.0
        real_bg = bool(spec.get("_net_polys"))
        W = 7.05 if _LAYOUT == "twocol" else 13.0
        fig, ax = plt.subplots(
            figsize=(W, max(3.4, min((6.8 if _LAYOUT == "twocol" else
                                      11.0) if real_bg else 8.0,
                                     (1.0 if real_bg else 2.4) * W
                                     * (vy1 - vy0) / (xr + 10)))))
        draw_map(ax, spec)
        # isotropic ~1.5 m hexes: a single int gridsize keeps hexagons
        # regular on any map shape (the old (120, 14) tuple was tuned for
        # thin synthetic corridors and smears square real maps)
        gs = int(np.clip(xr / 1.5, 40, 240))
        hb = ax.hexbin(X, Y, gridsize=gs, cmap="inferno", mincnt=1,
                       bins="log", alpha=0.95, zorder=6, linewidths=0)
        cb = fig.colorbar(hb, ax=ax,
                          label="Time Spent (Fixed-dt Samples "
                                "per Cell, Log)")
        # overlay THIS task's planned route (draw_map only knows the
        # spec's default route, which is a different task)
        m0 = next(iter(algod.values()))[0]
        wps0 = m0.get("waypoints") or []
        if wps0:
            ax.plot([w[0] for w in wps0], [w[1] for w in wps0], ls="--",
                    lw=1.2, color="white", alpha=0.85, zorder=8)
            start_goal(ax, m0)
        ax.set_xlim(min(X) - 5, max(X) + 5)
        ax.set_ylim(vy0, vy1)
        ax.set_aspect("equal" if real_bg else "auto")
        us = sorted(algod)
        who = (disp_unit(us[0]) if len(us) == 1
               else f"{len(us)} units")
        if _LAYOUT == "twocol":
            ax.set_title(f"{mp} -- Time-Occupancy ({who})", fontsize=10, pad=14)
            ax.tick_params(labelsize=8)
            cb.set_label("Time Spent (Fixed-dt Samples per Cell, Log)",
                         fontsize=8.5)
            cb.ax.tick_params(labelsize=7.5)
        else:
            ax.set_title(f"{mp} | mode={mode} | Time-Occupancy, {who} "
                         f"x All Seeds"
                         + ("" if real_bg else " (y stretched)"))
        fig.tight_layout()
        _savefig(fig, plots / f"occupancy_{mp}_{mode}.png", args.dpi)
        plt.close(fig)

    # ---- 3b2) per-map task-coverage exhibit: all task cells pooled.
    # Answers "how much of the map do the sampled tasks exercise?", for
    # the Methodology/overview -- NOT for comparing algorithms (per-task
    # occupancy figures above stay the analysis workhorses).
    for (bm, mode), acc in sorted(_mapcov.items()):
        spec = specs.get(bm)
        if spec is None or acc["cells"] < 2:
            continue           # single cell: the per-cell figure covers it
        Xa = np.concatenate(acc["X"])
        Ya = np.concatenate(acc["Y"])
        if len(Xa) < 200:
            continue
        xr = float(Xa.max() - Xa.min())
        vy0, vy1 = float(Ya.min()) - 3.0, float(Ya.max()) + 3.0
        real_bg = bool(spec.get("_net_polys"))
        W = 7.05 if _LAYOUT == "twocol" else 13.0
        fig, ax = plt.subplots(
            figsize=(W, max(3.4, min((6.8 if _LAYOUT == "twocol" else
                                      11.0) if real_bg else 8.0,
                                     (1.0 if real_bg else 2.4) * W
                                     * (vy1 - vy0) / (xr + 10)))))
        draw_map(ax, spec)
        gs = int(np.clip(xr / 1.5, 40, 240))
        hb = ax.hexbin(Xa, Ya, gridsize=gs, cmap="inferno", mincnt=1,
                       bins="log", alpha=0.95, zorder=6, linewidths=0)
        cb = fig.colorbar(hb, ax=ax,
                          label="Time Spent (Fixed-dt Samples "
                                "per Cell, Log)")
        for (s0, g0) in acc["sg"]:
            ax.scatter([s0[0]], [s0[1]], marker="o", s=34,
                       facecolor="white", edgecolor="black",
                       linewidths=0.6, zorder=9)
            ax.scatter([g0[0]], [g0[1]], marker="*", s=95,
                       facecolor="gold", edgecolor="black",
                       linewidths=0.5, zorder=9)
        ax.set_xlim(float(Xa.min()) - 5, float(Xa.max()) + 5)
        ax.set_ylim(vy0, vy1)
        ax.set_aspect("equal" if real_bg else "auto")
        if _LAYOUT == "twocol":
            ax.set_title(f"{bm} -- Coverage of the Sampled Tasks",
                         fontsize=10, pad=14)
            ax.tick_params(labelsize=8)
            cb.set_label("Time Spent (Fixed-dt Samples per Cell, Log)",
                         fontsize=8.5)
            cb.ax.tick_params(labelsize=7.5)
        else:
            ax.set_title(f"{bm} | mode={mode} | Coverage Exhibit: "
                         f"{acc['cells']} Task Cells Pooled "
                         f"(Circles = Starts, Stars = Goals)")
        fig.tight_layout()
        _savefig(fig, plots / f"coverage_{bm}_{mode}.png", args.dpi)
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
            # ms may now contain noplan episodes (no run dir): take the
            # first seed that actually has a trajectory
            m, d = None, None
            for cand in ms:
                dd = rundirs.get((mp, mode, a, cand.get("seed")))
                if dd is not None:
                    m, d = cand, dd
                    break
            if d is None:
                continue
            import csv as _csv
            # traces are the bulk of a 10k-run tree and are routinely pruned
            # or partially copied off a cluster; every other reader in this
            # file guards, and an unguarded open here killed the whole
            # plotting stage (and, via --unit both + check=True, its parent)
            tf = d / "robot_trace.csv"
            if not tf.exists():
                continue
            with open(tf) as fh:
                rd = list(_csv.DictReader(fh))
            if len(rd) < 10:
                continue
            vals = []
            for r in rd:            # skip truncated rows (killed writer)
                try:
                    vals.append((float(r["x"]), float(r["y"]),
                                 float(r["t"])))
                except (TypeError, ValueError, KeyError):
                    continue
            if len(vals) < 10:
                continue
            pts = _np.array([[v[0], v[1]] for v in vals])
            tt = _np.array([v[2] for v in vals])
            segs = _np.stack([pts[:-1], pts[1:]], axis=1)
            fig, ax = plt.subplots(figsize=fig_size2(spec))
            draw_map(ax, spec)
            lc = LineCollection(segs, cmap="viridis", alpha=0.9,
                                linewidths=2.2, zorder=8)
            lc.set_array(tt[:-1])
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, label="Time / s")
            if _LAYOUT == "twocol":
                ax.set_title(f"{disp_unit(a)} -- {base_mp}", fontsize=10,
                             pad=18)
                ax.text(1.0, 1.002, f"seed {m.get('seed')}, time-coded",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color="0.35")
                ax.tick_params(labelsize=8)
            else:
                ax.set_title(f"{disp_unit(a)} on {base_mp} | {mode} | "
                             f"Seed {m.get('seed')} | "
                             f"Time-Coded Trajectory")
            fig.tight_layout()
            _savefig(fig, plots / f"campus_{a}_{mp}_{mode}.png", args.dpi)
            plt.close(fig)

    # ---- 4) overall success rate -----------------------------------------
    # The old grouped-bar wall scaled its width with units x map-cells and in
    # combo mode left the printable page by an order of magnitude. Replaced
    # by an annotated heat map: rows = local algorithms in paired-family
    # order (upstream next to its counterpart), columns = base maps (tasks &
    # seeds pooled); combo unit splits into one panel per global planner so
    # all 54 combinations stay on one readable page.
    by_map = defaultdict(lambda: defaultdict(list))
    for mp, mode, algo, seed, m, d in runs + noplan_runs:
        by_map[base_map(mp)][algo].append(1.0 if m.get("success") else 0.0)
    maps = sorted(by_map)
    units = sorted({a for mp in maps for a in by_map[mp]}, key=unit_key)
    combo = any("+" in u for u in units)
    locals_ = sorted({unit_local(u) for u in units})
    gpls = sorted({unit_gpl(u) for u in units if unit_gpl(u)},
                  key=lambda x: (["astar", "dijkstra", "rrt"].index(x)
                                 if x in ("astar", "dijkstra", "rrt")
                                 else 99, x)) if combo else [""]
    rows = [a for fam in FAMILY_ORDER for a in fam if a in locals_]
    rows += sorted(set(locals_) - set(rows))
    seps, yy = [], 0
    for fam in FAMILY_ORDER:
        k = sum(1 for a in fam if a in locals_)
        if k:
            yy += k
            seps.append(yy - 0.5)
    seps = seps[:-1]
    net_set = (set(maps) if args.net_bg_maps == "all"
               else set() if args.net_bg_maps == "none"
               else set(args.net_bg_maps.split(",")))

    fig, axes = plt.subplots(
        1, len(gpls), sharey=True,
        figsize=(1.9 + 2.35 * len(gpls) * max(len(maps), 1) / 5,
                 2.2 + 0.28 * len(rows)))
    axes = np.atleast_1d(axes)
    im = None
    for ax, gp in zip(axes, gpls):
        M = np.full((len(rows), len(maps)), np.nan)
        for i, a in enumerate(rows):
            for j, mp in enumerate(maps):
                u = f"{gp}+{a}" if gp else a
                v = by_map[mp].get(u, [])
                if v:
                    M[i, j] = sum(v) / len(v)
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(maps)):
                if not math.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center",
                            va="center", fontsize=6.8)
        if seps:
            ax.hlines(seps, -0.5, len(maps) - 0.5, color="white", lw=2.4)
        ax.set_xticks(range(len(maps)),
                      [mp + (" (real)" if mp in net_set else "")
                       for mp in maps],
                      fontsize=7.5, rotation=30, ha="right")
        for j, mp in enumerate(maps):
            if mp in net_set:
                ax.get_xticklabels()[j].set_fontweight("bold")
                if j > 0:
                    ax.axvline(j - 0.5, color="black", lw=1.6)
        if gp:
            ax.set_title(f"Global: {_algo_display(gp)}", fontsize=10)
    axes[0].set_yticks(range(len(rows)),
                       [_algo_display(a) for a in rows], fontsize=8.5)
    fig.colorbar(im, ax=list(axes), shrink=0.72, label="Success Rate",
                 pad=0.015)
    fig.suptitle("Success Rate per Map (Tasks & Seeds Pooled)"
                 + (" -- One Panel per Global Planner" if combo else ""),
                 fontsize=10, y=0.995)
    _savefig(fig, plots / "success_overall.png", args.dpi)
    plt.close(fig)
    if _SPREAD_INDEX:
        import csv as _csv
        idx = plots / "envelope_spread_index.csv"
        with open(idx, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(_SPREAD_INDEX[0]))
            w.writeheader()
            w.writerows(sorted(_SPREAD_INDEX,
                               key=lambda r: -r["mean_spread_m"]))
        print(f"exemplar index -> {idx}  "
              f"({len(_SPREAD_INDEX)} envelopes, sorted by mean spread)")
    # ---- cross-unit divergence comparison: all units' spread curves for
    # one cell on shared axes. Every curve in thin grey; the widest and the
    # narrowest (by mean spread) highlighted -- "who diverges where" in one
    # frame instead of one bottom strip per envelope figure.
    def _pretty_cell(lbl, mode):
        """'map3_grid[t03]{rrt}' -> 'map3_grid, task t03, global RRT'."""
        g = _re.search(r"\{(.+?)\}", lbl)
        parts = _re.findall(r"\[(.+?)\]", lbl)
        s = _re.sub(r"\[.*?\]|\{.*?\}", "", lbl)
        if parts:
            s += ", task " + "/".join(parts)
        if g:
            s += f", global {_algo_short(g.group(1))}"
        if mode and mode != "all":
            s += f", mode {mode}"
        return s

    for (mp, mode), curves in sorted(_SPREAD_CURVES.items()):
        if len(curves) < 2:
            continue
        order = sorted(curves, key=lambda c: float(np.nanmean(c[2])))
        n_hi = min(2, len(order))
        # with <=4 curves the head and tail slices overlap: dedupe by
        # identity so no curve is drawn (or listed in the legend) twice
        picks = []
        for c in order[:n_hi] + order[-n_hi:]:
            if not any(c is p for p in picks):
                picks.append(c)
        marked = {id(c) for c in picks}
        fig, ax = plt.subplots(
            figsize=(7.05, 2.8) if _LAYOUT == "twocol" else (11.0, 3.6))
        for c in curves:
            if id(c) in marked:
                continue
            _unit, s, sp, _c = c
            ax.plot(s, sp, color="0.78", lw=0.8, zorder=2)
        # unit_color collides across highlighted units (e.g. teb and
        # sarl_upstream share a hue), so highlights get a fixed
        # colour-blind-safe palette instead
        _HI = ["#009E73", "#56B4E9", "#D55E00", "#CC79A7"]
        for i, c in enumerate(picks):
            unit, s, sp, _col = c
            ax.plot(s, sp, color=_HI[i % len(_HI)], lw=1.9, zorder=4,
                    label=f"{disp_unit_short(unit)} "
                          f"(mean {np.nanmean(sp):.1f} m)")
        n_grey = len(curves) - len(picks)
        if n_grey > 0:
            ax.plot([], [], color="0.78", lw=1.2,
                    label=f"{n_grey} other planners")
        ax.set_xlabel("Distance Along the Planned Route [m]", fontsize=10)
        ax.set_ylabel("Trajectory Spread\nAcross Seeds [m]", fontsize=10)
        ax.set_title(f"Where Trajectories Diverge: "
                     f"{_pretty_cell(mp, mode)}",
                     fontsize=11, pad=20)
        ax.text(1.0, 1.02, "spread = 10–90% band of lateral offsets, "
                f"{len(curves)} planners; widest and narrowest highlighted",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="0.35")
        ax.tick_params(labelsize=9)
        ax.margins(x=0.01)
        ax.set_ylim(bottom=0)
        ax.grid(lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        for _s in ("top", "right"):
            ax.spines[_s].set_visible(False)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        fig.tight_layout()
        _savefig(fig, plots / f"divergence_compare_{mp}_{mode}.png", args.dpi)
        plt.close(fig)
    print(f"plots -> {plots}")
    print("models & failure taxonomy (GLMM/LMM, odds ratios, "
          "ranking bootstrap): python analysis/stats_models.py "
          f"--results {args.results}")


if __name__ == "__main__":
    main()
