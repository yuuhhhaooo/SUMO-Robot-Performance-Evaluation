#!/usr/bin/env python3
"""Run ONE (map, mode, algorithm, seed) benchmark episode on the v7 native maps.

Architecture (v7):
  * the robot is a red POI whose kinematics are integrated HERE -- pedestrians
    cannot see or react to it (fairness), planner files are used UNCHANGED
  * the global route is the map_spec waypoint list (overridable); each
    axis-aligned leg is presented to the planner as a straight local sidewalk
    (x along travel in [0, leg_len], y across the band in [0, band_width])
  * the robot obeys the REAL traffic lights through NativeSignalGate: while
    the next crossing on its way is red it holds in the wait strip
    (time_waiting_at_light_s), identically for every algorithm
  * collision = distance < max(0.42, r_robot + r_ped), 3 s spawn grace
  * the SEED drives pedestrian density: per-road flows are sampled from
    [--flow-min, --flow-max] and the crossing flow from
    [--crossing-flow-min, --crossing-flow-max]

Outputs under <out-root>/<map>/<mode>/<algorithm>/seed_<k>/:
    robot_trace.csv   robot_metrics.json   scenario.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # sim/
REPO = ROOT.parent                          # repository root
sys.path.insert(0, str(ROOT))

from benchmark_adapters import build_planner, leg_config, RobotState, Obstacle  # noqa: E402
from native_signal_gate import NativeSignalGate  # noqa: E402

SPAWN_GRACE = 3.0
PEDESTRIAN_R = 0.15         # SUMO DEFAULT_PEDTYPE half-width [m]
COLLIDE_FLOOR = 0.42        # protocol floor, kept for bit-compatibility
COLLIDE_R = 0.42            # max(COLLIDE_FLOOR, r_robot + r_ped); recomputed
                            # per run from --robot-radius (see collide_r below)


def collision_radius(robot_radius: float) -> float:
    """Centre-distance below which a robot-pedestrian contact is a collision.

    The historical constant 0.42 is max(0.42, 0.25 + 0.15), so the shipped
    default --robot-radius 0.25 reproduces it exactly. A larger robot widens
    the threshold, which is the point of making the radius a real parameter.
    """
    return max(COLLIDE_FLOOR, float(robot_radius) + PEDESTRIAN_R)


SOCIAL_R = 0.85
HARD_SPEED_CAP = 1.6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True)
    p.add_argument("--mode", default="mixed",
                   choices=["same", "opposite", "mixed", "static", "all"])
    p.add_argument("--algorithm", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-root", default="results")
    p.add_argument("--flow-min", type=float, default=80.0)
    p.add_argument("--flow-max", type=float, default=350.0)
    p.add_argument("--crossing-flow-min", type=float, default=60.0)
    p.add_argument("--crossing-flow-max", type=float, default=160.0)
    p.add_argument("--speed-min", type=float, default=0.80)
    p.add_argument("--speed-max", type=float, default=1.60)
    p.add_argument("--veh-scale", type=float, default=1.0)
    p.add_argument("--ped-period", type=float, default=0.0,
                   help="OSM maps: OPTIONAL background randomTrips "
                        "pedestrians network-wide (one every N s; 0=off). "
                        "The directional mode crowd along the route is "
                        "always generated and controlled by --flow-min/max")
    p.add_argument("--ped-period-min", type=float, default=None)
    p.add_argument("--ped-period-max", type=float, default=None,
                   help="OSM maps: sample the period per seed from "
                        "[min,max] so seeds also vary crowd density")
    p.add_argument("--veh-period", type=float, default=6.0,
                   help="OSM maps: randomTrips vehicle period (0 = none)")
    p.add_argument("--veh-period-min", type=float, default=None)
    p.add_argument("--veh-period-max", type=float, default=None,
                   help="OSM maps: sample the vehicle period per seed from "
                        "[min,max] so seeds also vary car traffic")
    p.add_argument("--max-time", type=float, default=900.0)
    p.add_argument("--step-length", type=float, default=0.5)
    p.add_argument("--sensor-range", type=float, default=12.0)
    p.add_argument("--goal-tol", type=float, default=1.0)
    p.add_argument("--leg-switch-dist", type=float, default=0.8)
    p.add_argument("--route", default=None,
                   help="named route preset from map_spec.json routes{}, "
                        "e.g. map4_london: path1 | path2")
    p.add_argument("--waypoints", default=None,
                   help="Route waypoints 'x1,y1;x2,y2;...' (straight legs, "
                        "any direction)")
    p.add_argument("--global-rrt-params", default=None,
                   help="JSON overriding GLOBAL_RRT_PARAMS (protocol runs "
                        "do NOT use this; sensitivity/tuning hook only)")
    p.add_argument("--task-file", default=None,
                   help="alternative task list (e.g. configs/"
                        "tuning_tasks_<map>.json for held-out tuning "
                        "tasks); default configs/tasks_<map>.json")
    p.add_argument("--task", default=None,
                   help="task ID from configs/tasks_<map>.json (Option B "
                        "protocol); sets start/goal and implies a "
                        "planned route (global planner defaults to "
                        "dijkstra when 'fixed')")
    p.add_argument("--params-file", default=None,
                   help="JSON of tuned planner parameters (from tune.py); "
                        "applied to every per-leg planner instance")
    p.add_argument("--robot-radius", type=float, default=0.25,
                   help="physical robot radius [m]. Drives the collision "
                        "threshold (max(0.42, r_robot + r_ped)), the planners' "
                        "clearance, the JuPedSim agent radius under "
                        "--robot-in-jps, and the GUI marker. Default 0.25 is "
                        "the historical value; a sidewalk delivery robot is "
                        "closer to 0.30-0.35.")
    p.add_argument("--robot-height", type=float, default=1.0,
                   help="physical robot height [m]. Recorded in the metrics "
                        "and used for the GUI marker; the dynamics are 2-D, "
                        "so height does not enter the collision test.")
    p.add_argument("--jps-model", default="collision_free_speed",
                   choices=["collision_free_speed", "social_force",
                            "anticipation_velocity"],
                   help="JuPedSim operational model (--reactive-peds "
                        "jupedsim). collision_free_speed is the one SUMO "
                        "documents as extensively tested.")
    p.add_argument("--robot-in-jps", action="store_true",
                   help="inject the robot into JuPedSim as a direct-steered "
                        "agent so pedestrians SEE and react to it, with its "
                        "real radius. Off by default, which preserves the "
                        "fairness rule that the robot does all the avoiding. "
                        "NOTE: a direct steering stage bypasses JuPedSim's "
                        "strategic/tactical levels but NOT its operational "
                        "one, so JuPedSim also nudges the robot; the residual "
                        "is reported as jps_robot_track_err_*.")
    p.add_argument("--reactive-peds", choices=["off", "sfm", "jupedsim"],
                   default="off",
                   help="reactive pedestrian layer: sfm = pedestrians in an "
                        "interaction bubble around the robot are driven by "
                        "a Social Force Model that includes the robot as a "
                        "repulsive agent (supervisor-required for social-"
                        "navigation claims). off = legacy striping-only.")
    p.add_argument("--strict-sidewalk", choices=["auto", "on", "off"],
                   default="auto",
                   help="hard-constrain the robot to the real walkable "
                        "surface (sidewalk lanes + walkingareas + "
                        "crossings). auto = on for OSM-imported maps")
    p.add_argument("--global-planner",
                   choices=["fixed", "dijkstra", "astar", "rrt"],
                   default=None,
                   help="global planning factor: fixed = use the map's "
                        "stored waypoints verbatim; dijkstra/astar/rrt = "
                        "plan the route on the walkable graph/workspace "
                        "between the given start[/vias]/goal. Implies "
                        "--auto-route for the non-fixed choices.")
    p.add_argument("--auto-route", action="store_true",
                   help="treat --waypoints as start[;vias];goal and trace "
                        "the actual pedestrian network between them "
                        "(follows curved OSM sidewalks automatically)")
    p.add_argument("--reverse", action="store_true",
                   help="Run the waypoint route backwards (goal -> start)")
    p.add_argument("--model-dir", default=str(ROOT / "planners" / "models"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--gui", "--sumo-gui", dest="gui", action="store_true")
    p.add_argument("--delay", type=int, default=30,
                   help="sumo-gui playback delay in ms (only with --gui); "
                        "bigger = slower/easier to watch")
    p.add_argument("--keep-demand", action="store_true")
    p.add_argument("--robot-as-person", action="store_true",
                   help="embody the robot as a SUMO person instead of a POI: "
                        "pedestrians will SEE and REACT to it (this breaks "
                        "the default fairness rule where the robot does all "
                        "the avoiding -- use only as an explicit experimental "
                        "condition, never mixed with POI runs)")
    p.add_argument("--demand", default=None,
                   help="use this .rou.xml instead of generating mode demand "
                        "(default for OSM-imported maps: the map's base file)")
    return p.parse_args()


# ---------------------------------------------------------------- geometry
def in_rect(x, y, r, eps=0.0):
    return r[0] - eps <= x <= r[2] + eps and r[1] - eps <= y <= r[3] + eps


def observe_persons(traci, tc, subscribed, rx, ry, sensor_range, frame,
                    obstacle_cls, robot_id="robot0"):
    """One-round-trip pedestrian observation.

    Returns (obstacles_in_leg_local_frame, min_distance_to_any_person).

    This replaces a loop that issued one blocking TraCI round-trip per person
    for the position plus two more for every person inside the sensor range
    (1 + N + 2M exchanges per step, ~1.2M per run). Persons are subscribed
    once on arrival; the whole population is then read back in a single call.

    Semantics are identical to the per-call version by construction:
      * iteration follows getIDList() order, so the obstacle list order -- and
        any planner tie-breaking that depends on it -- is unchanged;
      * `step_min` is still taken over EVERY person, not just in-range ones;
      * subscription results only become available the step AFTER subscribe(),
        so a person is read directly on the step it first appears rather than
        being invisible for one step.

    `subscribed` is mutated in place and must persist across steps.
    """
    ped_ids = traci.person.getIDList()
    for pid in ped_ids:
        if pid not in subscribed:
            traci.person.subscribe(
                pid, [tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ANGLE])
            subscribed.add(pid)
    if len(subscribed) > len(ped_ids):          # prune departed persons
        subscribed.intersection_update(ped_ids)
    sub = traci.person.getAllSubscriptionResults()

    obstacles = []
    step_min = float("inf")
    for pid in ped_ids:
        if pid == robot_id:
            continue
        rec = sub.get(pid)
        if rec is None or tc.VAR_POSITION not in rec:
            px, py = traci.person.getPosition(pid)
        else:
            px, py = rec[tc.VAR_POSITION]
        d = math.hypot(px - rx, py - ry)
        if d < step_min:
            step_min = d
        if d <= sensor_range:
            if rec is None or tc.VAR_SPEED not in rec:
                sp = traci.person.getSpeed(pid)
                ang = math.radians(traci.person.getAngle(pid))
            else:
                sp = rec[tc.VAR_SPEED]
                ang = math.radians(rec[tc.VAR_ANGLE])
            wvx, wvy = sp * math.sin(ang), sp * math.cos(ang)
            lx, ly = frame.to_local(px, py)
            lvx, lvy = frame.vel_to_local(wvx, wvy)
            obstacles.append(obstacle_cls(pid, lx, ly, lvx, lvy))
    return obstacles, step_min


def _rdp(points, eps):
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return list(points)
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1e-9
    dmax, imax = -1.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / L
        if d > dmax:
            dmax, imax = d, i
    if dmax <= eps:
        return [points[0], points[-1]]
    left = _rdp(points[: imax + 1], eps)
    return left[:-1] + _rdp(points[imax:], eps)


# Global-RRT protocol constants (disclosed in the paper's appendix).
# FIXED during the protocol -- the global planner is a manipulated
# condition, not a system under test -- but exposed here as named,
# overridable parameters (--global-rrt-params) so the protocol table,
# sensitivity analyses, and any future tuning decision all read/write
# ONE place. Defaults reproduce the published routes bit-for-bit.
GLOBAL_RRT_PARAMS = {
    "step_m": 8.0,               # steer step length
    "goal_bias": 0.15,           # P(sample = goal)
    "corridor_sample": 0.7,      # P(sample on a sidewalk centreline)
    "lateral_jitter_m": 0.8,     # +/- jitter around the centreline
    "piece_buffer_m": 1.2,       # walkable half-width around pieces
    "warea_buffer_m": 3.0,       # tolerant junction buffer
    "max_iters": 40000,          # per attempt (env RRT_MAX_ITERS overrides)
    "restarts": 3,               # deterministic restarts per segment
    "shortcut_iters": 400,       # walkable-constrained smoothing
    "snap_inset_m": 0.25,        # endpoint snap onto eroded union
}


class GlobalPlanFailure(Exception):
    """Raised when the global planner cannot produce a route; recorded as a
    per-run outcome (termination_reason=global_plan_failed), never a crash."""


def _rrt_route(net_file, pts, pieces, warea, eps, rng_seed, return_edges):
    """Global RRT over the walkable workspace: corridor-informed sampling
    (points drawn on sidewalk/crossing centrelines with lateral jitter, plus
    goal bias), shortcut smoothing constrained to the walkable union, and up
    to three deterministic restarts. Stochastic by design: the produced
    route is part of the seeded experimental unit."""
    import random as _random
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union, nearest_points as _np
    from shapely.prepared import prep

    polys = [LineString(pc["pts"]).buffer(GLOBAL_RRT_PARAMS["piece_buffer_m"])
             for pc in pieces]
    polys += [g.buffer(GLOBAL_RRT_PARAMS["warea_buffer_m"]) for g in warea]   # tolerant: some netconvert
    union = unary_union(polys)                # versions displace walkingareas
    up = prep(union)
    union_in = union.buffer(-GLOBAL_RRT_PARAMS["snap_inset_m"])
    if union_in.is_empty:
        union_in = union
    upi = prep(union_in)

    def free_seg(pq, qq):
        return up.covers(LineString([pq, qq]))

    def _snap(pt):
        P = Point(pt)
        if upi.covers(P):
            return tuple(pt)
        q = _np(union_in, P)[0]
        return (q.x, q.y)

    STEP = GLOBAL_RRT_PARAMS["step_m"]

    def _grow(a, b, seed):
        rng = _random.Random(seed)
        CELL = 14.0
        grid = {}
        tree = {a: None}

        def _add(pt):
            grid.setdefault((int(pt[0] // CELL), int(pt[1] // CELL)),
                            []).append(pt)

        def _ring(r):
            """Cells at Chebyshev radius r, in the SAME lexicographic (dx, dy)
            order the original full-square scan visited them.

            The original iterated the whole (2r+1)^2 square and skipped every
            cell with max(|dx|,|dy|) != r, costing ~(4/3)R^3 probes to reach
            radius R instead of ~4R^2 -- measured at 878k inner iterations for
            a 1200 m query, and reported as ~98.7% of global-RRT runtime.
            Emitting only the perimeter is O(r) per ring. Order is preserved
            exactly, so the strict `d2 < bd` comparison below still keeps the
            same node when two candidates tie, and routes stay bit-identical.
            """
            if r == 0:
                yield (0, 0)
                return
            for dx in range(-r, r + 1):
                if abs(dx) == r:                 # full column
                    for dy in range(-r, r + 1):
                        yield (dx, dy)
                else:                            # only the two edge rows
                    yield (dx, -r)
                    yield (dx, r)

        def _nearest(q):
            kx, ky = int(q[0] // CELL), int(q[1] // CELL)
            best, bd = None, float("inf")
            r = 0
            while r < 300:
                for dx, dy in _ring(r):
                    for n in grid.get((kx + dx, ky + dy), ()):
                        d2 = (n[0] - q[0]) ** 2 + (n[1] - q[1]) ** 2
                        if d2 < bd:
                            bd, best = d2, n
                if best is not None and bd <= (r * CELL) ** 2:
                    return best
                r += 1
            return best

        def _sample():
            if rng.random() < GLOBAL_RRT_PARAMS["corridor_sample"]:
                pc = rng.choice(pieces)
                P = pc["pts"]
                i = rng.randrange(len(P) - 1)
                t = rng.random()
                x = P[i][0] + t * (P[i + 1][0] - P[i][0])
                y = P[i][1] + t * (P[i + 1][1] - P[i][1])
                j = GLOBAL_RRT_PARAMS["lateral_jitter_m"]
                return (x + rng.uniform(-j, j),
                        y + rng.uniform(-j, j))
            m = union.bounds
            return (rng.uniform(m[0], m[2]), rng.uniform(m[1], m[3]))

        _add(a)
        import os as _os
        _max_it = int(_os.environ.get("RRT_MAX_ITERS",
                                      str(GLOBAL_RRT_PARAMS["max_iters"])))
        for _it in range(_max_it):
            q = b if rng.random() < GLOBAL_RRT_PARAMS["goal_bias"] else _sample()
            near = _nearest(q)
            if near is None:
                continue
            d = math.hypot(q[0] - near[0], q[1] - near[1])
            if d < 1e-6:
                continue
            step = min(STEP, d)
            new = (near[0] + (q[0] - near[0]) / d * step,
                   near[1] + (q[1] - near[1]) / d * step)
            if not up.covers(Point(new)) or not free_seg(near, new):
                continue
            tree[new] = near
            _add(new)
            if math.hypot(new[0] - b[0], new[1] - b[1]) < STEP and \
                    free_seg(new, b):
                tree[b] = new
                path = [b]
                n = b
                while tree[n] is not None:
                    n = tree[n]
                    path.append(n)
                path.reverse()
                for _ in range(GLOBAL_RRT_PARAMS["shortcut_iters"]):
                    if len(path) < 3:
                        break
                    i = rng.randrange(0, len(path) - 2)
                    j = rng.randrange(i + 2, len(path))
                    if free_seg(path[i], path[j]):
                        path = path[:i + 1] + path[j:]
                return path
        import os as _os
        if _os.environ.get("RRT_DEBUG"):
            _all = [n for c in grid.values() for n in c]
            _dg = min(math.hypot(n[0] - b[0], n[1] - b[1]) for n in _all)
            print(f"RRT_DEBUG attempt-fail nodes={len(_all)} "
                  f"d2goal={_dg:.1f}")
        return None

    out_all = [tuple(pts[0])]
    for a0, b0 in zip(pts[:-1], pts[1:]):
        a, b = _snap(a0), _snap(b0)
        path = None
        for attempt in range(GLOBAL_RRT_PARAMS["restarts"]):
            path = _grow(a, b, (rng_seed << 8) + 97 * attempt)
            if path is not None:
                break
        if path is None:
            raise GlobalPlanFailure(
                f"rrt: no path {a0} -> {b0} after 3 restarts "
                f"(seed {rng_seed})")
        out_all += path[1:]
        if math.hypot(b0[0] - b[0], b0[1] - b[1]) > 0.3:
            out_all.append(tuple(b0))
    out = _rdp(out_all, eps)
    clean = [out[0]]
    for q in out[1:]:
        if math.hypot(q[0] - clean[-1][0], q[1] - clean[-1][1]) > 0.8:
            clean.append(q)
    if return_edges:
        edges = []
        for w in clean:
            best = min(pieces, key=lambda pc: min(
                (w[0] - px) ** 2 + (w[1] - py) ** 2 for px, py in pc["pts"]))
            e = best.get("edge")
            if e and not best.get("crossing") and \
                    (not edges or edges[-1] != e):
                edges.append(e)
        return clean, edges
    return clean


def build_walk_graph(net_file, pts=None):
    """Build the walkable graph ONCE (whole map when pts is None) so many
    routing queries can reuse it -- used by the task sampler. Returns an
    opaque tuple consumed by auto_route(..., _graph=...)."""
    big = [(-1e9, -1e9), (1e9, 1e9)] if pts is None else pts
    return auto_route(net_file, big, _build_only=True)


def auto_route(net_file, pts, eps=0.6, return_edges=False,
               method="dijkstra", rng_seed=0, _graph=None,
               _build_only=False):
    """Route THROUGH the walkable geometry itself: sidewalk/ped lanes and
    crossings are graph edges, walkingarea polygons are the connectors --
    so junction bridges follow walkingarea -> crossing -> walkingarea and the
    produced polyline lies on the walkable surface metre by metre."""
    try:
        from shapely.geometry import Polygon, Point
        from shapely.prepared import prep
    except ImportError:
        sys.exit("--auto-route needs shapely:  pip install shapely")
    import heapq
    import xml.etree.ElementTree as ET

    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    M = 320.0
    bx0, bx1 = min(xs) - M, max(xs) + M
    by0, by1 = min(ys) - M, max(ys) + M

    if _graph is not None:
        pieces, warea, base_nodes, base_adj, ends = _graph
        nodes = list(base_nodes)
        adj = [list(x) for x in base_adj]
        _skip_build = True
    else:
        _skip_build = False
    pieces, warea = [], []
    for _ev, edge in ET.iterparse(str(net_file)):
        if edge.tag != "edge":
            continue
        func = edge.get("function", "normal")
        for lane in edge.iter("lane"):
            shape = lane.get("shape")
            allow = lane.get("allow") or ""
            if not shape:
                continue
            P = [tuple(map(float, q.split(","))) for q in shape.split()]
            if len(P) < 2 or not any(bx0 <= px <= bx1 and by0 <= py <= by1
                                     for px, py in P):
                continue
            if func == "walkingarea":
                try:
                    g = Polygon(P)
                    if not g.is_valid:
                        g = g.buffer(0)
                    if g.is_empty or g.area < 0.05:
                        from shapely.geometry import MultiPoint
                        g = MultiPoint(P).convex_hull.buffer(0.2)
                    warea.append(g)
                except Exception:
                    pass
            elif func == "crossing" or (func in ("normal", "") and
                                        "pedestrian" in allow):
                L = sum(math.hypot(P[i + 1][0] - P[i][0],
                                   P[i + 1][1] - P[i][1])
                        for i in range(len(P) - 1))
                if L > 0.3:
                    pieces.append({"pts": P, "len": L,
                                   "edge": edge.get("id"),
                                   "crossing": func == "crossing"})
        edge.clear()
    if not pieces:
        sys.exit("--auto-route: no walkable geometry found")

    # graph: nodes = piece endpoints (+ virtual start/goal); edges = pieces,
    # walkingarea cliques, and near-touch endpoint pairs
    if method == "rrt":            # dispatch BEFORE the graph build
        return _rrt_route(net_file, pts, pieces, warea, eps, rng_seed,
                          return_edges)

    if not _skip_build:
        nodes, adj = [], []

    def add_node(xy):
        nodes.append(xy)
        adj.append([])
        return len(nodes) - 1

    def link(i, j, cost, path_pts, meta=None):
        adj[i].append((j, cost, path_pts, meta))
        adj[j].append((i, cost, list(reversed(path_pts)), meta))

    if not _skip_build:
        ends = []
        for pc in pieces:
            n0 = add_node(pc["pts"][0])
            n1 = add_node(pc["pts"][-1])
            link(n0, n1, pc["len"], pc["pts"],
                 (pc.get("edge"), pc.get("crossing", False)))
            ends += [n0, n1]

        from shapely.geometry import LineString as _LS
        from shapely.strtree import STRtree
        WA_TOL = 3.5     # some netconvert versions place walkingareas a few
        # metres off the lane endpoints -- be tolerant.
        #
        # The membership test used to scan EVERY endpoint for EVERY
        # walkingarea: O(|warea| x |ends|) prepared-covers calls, ~3.8M on
        # map5_ucl and ~45 s of the ~37 s-per-route build. The STRtree narrows
        # each walkingarea to its bounding-box candidates first. The
        # gp.covers() predicate below is UNCHANGED, and candidates are sorted
        # back into `ends` order, so `members` -- and therefore the order in
        # which link() appends to the adjacency lists, and therefore Dijkstra's
        # tie-breaking among equal-cost routes -- is bit-identical.
        _end_pts = [Point(nodes[n]) for n in ends]
        _tree = STRtree(_end_pts) if _end_pts else None
        for g in warea:
            gb = g.buffer(WA_TOL)
            gp = prep(gb)
            _cand = sorted(int(i) for i in _tree.query(gb)) if _tree else []
            members = [ends[i] for i in _cand if gp.covers(_end_pts[i])]
            for ii in range(len(members)):
                for jj in range(ii + 1, len(members)):
                    a_, b_ = nodes[members[ii]], nodes[members[jj]]
                    d = math.hypot(a_[0] - b_[0], a_[1] - b_[1])
                    # the straight connector must stay INSIDE the walking area
                    # (plaza-sized areas can be concave)
                    if d < 2.5 or gp.covers(_LS([a_, b_])):
                        link(members[ii], members[jj], d, [a_, b_])
        # near-touch endpoints (consecutive segments of one street)
        cell = {}
        for n in ends:
            key = (int(nodes[n][0] // 3), int(nodes[n][1] // 3))
            cell.setdefault(key, []).append(n)
        for n in ends:
            kx, ky = int(nodes[n][0] // 3), int(nodes[n][1] // 3)
            for dx_ in (-1, 0, 1):
                for dy_ in (-1, 0, 1):
                    for m in cell.get((kx + dx_, ky + dy_), ()):
                        if m <= n:
                            continue
                        d = math.hypot(nodes[n][0] - nodes[m][0],
                                       nodes[n][1] - nodes[m][1])
                        if d < 1.2:
                            link(n, m, d, [nodes[n], nodes[m]])

    if _build_only:
        return (pieces, warea, list(nodes), [list(x) for x in adj],
                list(ends))

    def attach(xy):
        """virtual node on the nearest piece (split at the projection)."""
        best = None
        for pc in pieces:
            P = pc["pts"]
            acc = 0.0
            for i in range(len(P) - 1):
                ax, ay = P[i]
                bx_, by_ = P[i + 1]
                ex, ey = bx_ - ax, by_ - ay
                L2 = ex * ex + ey * ey
                t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, (
                    (xy[0] - ax) * ex + (xy[1] - ay) * ey) / L2))
                qx, qy = ax + t * ex, ay + t * ey
                d = math.hypot(xy[0] - qx, xy[1] - qy)
                seg = math.sqrt(L2)
                if best is None or d < best[0]:
                    best = (d, pc, acc + t * seg, (qx, qy), i, t)
                acc += seg
        d, pc, s_along, q, i, t = best
        v = add_node(q)
        # connect to the piece's two endpoint nodes with partial paths
        P = pc["pts"]
        n0 = next(n for n in ends if nodes[n] == P[0])
        n1 = next(n for n in ends if nodes[n] == P[-1])
        pre = P[:i + 1] + [q]
        post = [q] + P[i + 1:]
        link(v, n0, s_along, list(reversed(pre)),
             (pc.get("edge"), pc.get("crossing", False)))
        link(v, n1, pc["len"] - s_along, post,
             (pc.get("edge"), pc.get("crossing", False)))
        return v

    out_pts = []
    route_edges = []
    for a, b in zip(pts[:-1], pts[1:]):
        va, vb = attach(tuple(a)), attach(tuple(b))
        gx, gy = nodes[vb]

        def _h(n):
            if method != "astar":
                return 0.0
            px, py = nodes[n]
            return math.hypot(px - gx, py - gy)

        dist = {va: 0.0}
        prev = {}
        pq = [(_h(va), va)]
        while pq:
            f, n = heapq.heappop(pq)
            if n == vb:
                break
            d = dist.get(n, float("inf"))
            if f - _h(n) > d + 1e-9:
                continue
            for m, w, pp, meta in adj[n]:
                nd = d + w
                if nd < dist.get(m, float("inf")):
                    dist[m] = nd
                    prev[m] = (n, pp, meta)
                    heapq.heappush(pq, (nd + _h(m), m))
        if vb not in prev and va != vb:
            sys.exit(f"--auto-route: no walkable path {a} -> {b}")
        chain = []
        n = vb
        while n != va:
            n, pp, meta = prev[n]
            chain.append((pp, meta))
        for pp, meta in reversed(chain):
            out_pts += pp if not out_pts else pp[1:]
            if meta and meta[0] and not meta[1]:      # normal ped edge
                if not route_edges or route_edges[-1] != meta[0]:
                    route_edges.append(meta[0])
    out = _rdp([tuple(a) for a in [pts[0]]] + out_pts + [tuple(pts[-1])], eps)
    clean = [out[0]]
    for q in out[1:]:
        if math.hypot(q[0] - clean[-1][0], q[1] - clean[-1][1]) > 0.8:
            clean.append(q)
    if len(clean) < 2:
        clean = [tuple(pts[0]), tuple(pts[-1])]
    if return_edges:
        return clean, route_edges
    return clean


def make_legs(waypoints, spec):
    """Straight legs in ANY direction (axis-aligned is a special case)."""
    legs = []
    for w0, w1 in zip(waypoints[:-1], waypoints[1:]):
        dx, dy = w1[0] - w0[0], w1[1] - w0[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        nvx, nvy = -uy, ux
        W, off = 2.0, 1.0
        if abs(ux) > 0.999 or abs(uy) > 0.999:      # axis-aligned: use spec
            axis = "h" if abs(ux) > abs(uy) else "v"
            for sw in spec.get("sidewalks", []):
                r = sw["rect"]
                if not in_rect(w0[0], w0[1], r, eps=0.05):
                    continue
                elong_h = (r[2] - r[0]) >= (r[3] - r[1])
                if (axis == "h") == elong_h:
                    lo, hi = (r[1], r[3]) if axis == "h" else (r[0], r[2])
                    W = hi - lo
                    lat0 = w0[1] if axis == "h" else w0[0]
                    nlat = nvy if axis == "h" else nvx
                    off = (lat0 - lo) if nlat > 0 else (hi - lat0)
                    break
        legs.append({"w0": tuple(w0), "w1": tuple(w1), "len": length,
                     "d": (ux, uy), "n": (nvx, nvy), "W": W, "off": off})
    return legs


class Frame:
    """World <-> leg-local transform (any straight leg direction)."""

    def __init__(self, leg):
        self.leg = leg

    def to_local(self, x, y):
        L = self.leg
        rx, ry = x - L["w0"][0], y - L["w0"][1]
        return (rx * L["d"][0] + ry * L["d"][1],
                rx * L["n"][0] + ry * L["n"][1] + L["off"])

    def to_world(self, lx, ly):
        L = self.leg
        a, b = lx, ly - L["off"]
        return (L["w0"][0] + a * L["d"][0] + b * L["n"][0],
                L["w0"][1] + a * L["d"][1] + b * L["n"][1])

    def vel_to_local(self, vx, vy):
        L = self.leg
        return (vx * L["d"][0] + vy * L["d"][1],
                vx * L["n"][0] + vy * L["n"][1])

    def vel_to_world(self, vxl, vyl):
        L = self.leg
        return (vxl * L["d"][0] + vyl * L["n"][0],
                vxl * L["d"][1] + vyl * L["n"][1])


def load_walkable(net_file, waypoints, buffer_m=20.0):
    """Union of every pedestrian-walkable surface near the route:
    sidewalk / ped lanes + crossings (buffered centrelines) + walkingareas
    (native polygons)."""
    try:
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError:
        sys.exit("--strict-sidewalk needs shapely:  pip install shapely")
    import xml.etree.ElementTree as ET
    corridor = LineString(waypoints).buffer(buffer_m)
    polys = []
    for _ev, edge in ET.iterparse(str(net_file)):
        if edge.tag != "edge":
            continue
        func = edge.get("function", "normal")
        for lane in edge.iter("lane"):
            allow = lane.get("allow") or ""
            shape = lane.get("shape")
            if not shape:
                continue
            pts = [tuple(map(float, q.split(","))) for q in shape.split()]
            if len(pts) < 2:
                continue
            try:
                if func == "walkingarea":
                    g = Polygon(pts)
                    if not g.is_valid:
                        g = g.buffer(0)
                    if g.is_empty or g.area < 0.05:
                        from shapely.geometry import MultiPoint
                        g = MultiPoint(pts).convex_hull.buffer(0.2)
                elif func == "crossing" or (func in ("normal", "") and
                                            "pedestrian" in allow):
                    g = LineString(pts).buffer(
                        float(lane.get("width", "2.0")) / 2.0 + 0.05)
                else:
                    continue
            except Exception:
                continue
            if g.intersects(corridor):
                polys.append(g)
        edge.clear()
    if not polys:
        return None, None
    union = unary_union(polys)
    return union, prep(union)


def crossing_ahead(pos, v_world, leg, gate, frame=None, look=2.4):
    """Nearest red crossing the robot is about to enter along this leg."""
    fr = frame or Frame(leg)
    x, y = pos
    lx, _ly = fr.to_local(x, y)
    v_along, _ = fr.vel_to_local(*v_world)
    for z in gate.zone_states():
        r = z["rect"]
        if in_rect(x, y, r):
            return None                     # already inside; keep going
        corners = [fr.to_local(r[0], r[1]), fr.to_local(r[0], r[3]),
                   fr.to_local(r[2], r[1]), fr.to_local(r[2], r[3])]
        lys = [c[1] for c in corners]
        if max(lys) < -0.3 or min(lys) > leg["W"] + 0.3:
            continue
        front = min(c[0] for c in corners)
        gap = front - lx
        if -0.05 <= gap <= look and (v_along > 0.02 or gap <= 0.6):
            if not z["green"]:
                return {"zone": z, "gap": max(gap, 0.0)}
    return None


def main():
    args = parse_args()
    map_dir = REPO / "maps" / args.map
    spec = json.loads((map_dir / "map_spec.json").read_text())

    if args.waypoints:
        wps = [tuple(map(float, t.split(",")))
               for t in args.waypoints.split(";") if t.strip()]
    elif args.route:
        routes = spec.get("routes", {})
        if args.route not in routes:
            sys.exit(f"--route '{args.route}' not in map_spec; available: "
                     f"{sorted(routes) or ['(none)']}")
        wps = [tuple(w) for w in routes[args.route]]
    else:
        wps = [tuple(w) for w in spec["robot"]["waypoints"]]
    if args.reverse:
        wps = list(reversed(wps))
    _tuned_params = None
    if args.params_file:
        import json as _json
        _tuned_params = _json.loads(Path(args.params_file).read_text())
    if args.global_rrt_params:
        GLOBAL_RRT_PARAMS.update(
            json.loads(Path(args.global_rrt_params).read_text()))
        print(f"global-rrt params overridden: {GLOBAL_RRT_PARAMS}")
    task_id = None
    if args.task:
        tfile = (Path(args.task_file) if args.task_file
                 else REPO / "configs" / f"tasks_{args.map}.json")
        if not tfile.exists():
            sys.exit(f"--task given but {tfile} not found "
                     f"(run sim/sample_tasks.py first)")
        tspec = json.loads(tfile.read_text())
        tmap = {t["id"]: t for t in tspec["tasks"]}
        if args.task not in tmap:
            sys.exit(f"task {args.task} not in {tfile.name}; "
                     f"available: {sorted(tmap)}")
        task_id = args.task
        t = tmap[task_id]
        wps = [tuple(t["start"]), tuple(t["goal"])]
        if args.global_planner is None and not args.auto_route:
            args.auto_route = True     # tasks always route on the network

    gp = args.global_planner
    if gp is None:
        gp = "dijkstra" if args.auto_route else "fixed"
    osm_route_edges = None
    plan_t0 = time.time()
    plan_failed = None
    if gp != "fixed":
        try:
            wps, osm_route_edges = auto_route(
                map_dir / f"{args.map}.net.xml", wps, return_edges=True,
                method=gp, rng_seed=args.seed)
            print(f"global-planner {gp}: {len(wps)} waypoints along the "
                  f"pedestrian network")
        except GlobalPlanFailure as exc:
            plan_failed = str(exc)
            print(f"global-planner {gp} FAILED: {exc}")
    global_plan_time = round(time.time() - plan_t0, 1)
    if plan_failed is not None:
        route_name = ("custom" if args.waypoints else
                      (args.route or "default"))
        map_label = (args.map if route_name == "default"
                     else f"{args.map}__{route_name}")
        if task_id:
            map_label += f"__{task_id}"
        # every non-default global-planner level gets its own directory.
        # 'dijkstra' must NOT share the bare label with 'fixed': a sweep
        # crossing both levels silently overwrote one with the other.
        if gp != "fixed":
            map_label += f"__g-{gp}"
        run_dir = (Path(args.out_root) / map_label / args.mode
                   / args.algorithm / f"seed_{args.seed}")
        run_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: SUMO was never started on this path, so there is no SFM layer
        # to query -- the sfm_* fields are emitted as their "layer idle"
        # values purely to keep the metrics schema identical across runs.
        metrics = {
            "map": args.map, "route": route_name, "global_planner": gp,
            "task": task_id,
            "global_rrt_params": (dict(GLOBAL_RRT_PARAMS) if gp == "rrt"
                                  else None),
            "reactive_peds": args.reactive_peds,
            "params_file": (args.params_file or None),
            "sfm_controlled_steps": 0,
            "sfm_capture_events": 0,
            "mode": args.mode, "algorithm": args.algorithm,
            "seed": args.seed, "success": False, "collision": False,
            "termination_reason": "global_plan_failed",
            "global_plan_time_s": global_plan_time,
            "global_plan_error": plan_failed,
            "path_length_m": 0.0, "sim_time_s": 0.0,
            "time_waiting_at_light_s": 0.0,
            # null, not inf: json.dumps(inf) emits bare `Infinity`, which is
            # not valid JSON and breaks strict parsers downstream
            "min_pedestrian_distance_m": None,
            "avg_speed_mps": 0.0, "num_legs": 0,
            "waypoints": [list(w) for w in wps],
        }
        (run_dir / "robot_metrics.json").write_text(
            json.dumps(metrics, indent=2))
        (run_dir / "robot_trace.csv").write_text(
            "t,x,y,vx,vy,leg,held_at_light,min_ped_dist\n")
        print(json.dumps(metrics))
        return
    legs = make_legs(wps, spec)
    goal_xy = wps[-1]

    route_name = ("custom" if args.waypoints else
                  (args.route or "default"))
    map_label = (args.map if route_name == "default"
                 else f"{args.map}__{route_name}")
    if task_id:
        map_label += f"__{task_id}"
    # every non-default global-planner level gets its own directory. This MUST
    # match the plan-failure path above and benchmark_batch's _run_dir: while
    # 'dijkstra' shared the bare label with 'fixed', a sweep crossing both
    # levels wrote them to one directory (silently losing the 'fixed' run) and
    # the batch then reported "runner exited 0 but wrote no metrics" for every
    # dijkstra cell -- 3,500 of them on the full protocol design.
    if gp != "fixed":
        map_label += f"__g-{gp}"
    run_dir = (Path(args.out_root) / map_label / args.mode / args.algorithm
               / f"seed_{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    crossing_flow = rng.uniform(args.crossing_flow_min, args.crossing_flow_max)
    mode_demand = bool(spec.get("roads")) and not spec.get("osm")
    osm_ped_period = None
    if args.demand:
        rou = Path(args.demand).resolve()
    elif not mode_demand:
        # OSM import: per-seed demand with the SAME mode semantics as the
        # built-in maps, defined relative to the robot's route direction.
        # Background randomTrips pedestrians only if --ped-period* given.
        if args.ped_period_min is not None and args.ped_period_max is not None:
            osm_ped_period = rng.uniform(args.ped_period_min,
                                         args.ped_period_max)
        elif args.ped_period and args.ped_period > 0:
            osm_ped_period = args.ped_period
        else:
            osm_ped_period = None
        osm_flow = rng.uniform(args.flow_min, args.flow_max)
        if args.veh_period_min is not None and args.veh_period_max is not None:
            osm_veh_period = rng.uniform(args.veh_period_min,
                                         args.veh_period_max)
        else:
            osm_veh_period = args.veh_period
        osm_statics = rng.randint(5, 12) if args.mode in ("static", "all")             else 0
        rou = (run_dir / "demand.rou.xml").resolve()
        try:
            from osm_import import osm_mode_demand
            if osm_route_edges is not None:
                route_edges = osm_route_edges
            else:
                _, route_edges = auto_route(
                    map_dir / f"{args.map}.net.xml", wps, return_edges=True)
            osm_mode_demand(map_dir / f"{args.map}.net.xml", route_edges,
                            args.mode, osm_flow, osm_statics,
                            args.speed_min, args.speed_max,
                            osm_veh_period, rou, args.seed,
                            end=args.max_time + 100.0,
                            bg_ped_period=osm_ped_period)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"mode demand generation failed ({exc}); "
                  f"falling back to the shipped base demand")
            rou = (map_dir / f"{args.map}_base.rou.xml").resolve()
    else:
        rou = (run_dir / "demand.rou.xml").resolve()
    if mode_demand and not args.demand:
        subprocess.run([sys.executable, str(ROOT / "generate_demand.py"),
                    "--spec", str(map_dir / "map_spec.json"),
                    "--mode", args.mode, "--seed", str(args.seed),
                    "--out", str(rou),
                    "--flow-min", str(args.flow_min), "--flow-max", str(args.flow_max),
                    "--speed-min", str(args.speed_min), "--speed-max", str(args.speed_max),
                    "--crossing-flow", f"{crossing_flow:.2f}",
                    "--veh-scale", str(args.veh_scale)],
                       check=True, capture_output=True, text=True)

    import os
    import traci
    import traci.constants as tc
    from sumolib import checkBinary
    binary = checkBinary("sumo-gui" if args.gui else "sumo")
    cmd = [binary, "-c", str(map_dir / f"{args.map}.sumocfg"),
           "--route-files", str(rou), "--step-length", str(args.step_length),
           "--seed", str(args.seed), "--no-warnings", "--no-step-log",
           "--quit-on-end"]
    if args.gui:
        cmd += ["--start", "--delay", str(args.delay)]
    old_cwd = os.getcwd()
    os.chdir(map_dir)
    traci.start(cmd)

    gate = NativeSignalGate(spec, traci)
    # defensive, gate-version-independent: drop crossings whose program NEVER
    # shows green (netconvert artefacts at complex OSM junctions) -- holding
    # for them would trap the robot until max_time. Idempotent: if the gate
    # already filtered them, this removes nothing.
    try:
        _served = []
        for _c in gate.crossings:
            _lg = gate._logic[_c["tls"]]
            _li = _c["linkIndex"]
            if any(_li < len(_ph.state) and _ph.state[_li] in "Gg"
                   for _ph in _lg.phases):
                _served.append(_c)
        if len(_served) < len(gate.crossings):
            print(f"note: {len(gate.crossings) - len(_served)} never-green "
                  f"crossings treated as unsignalised")
        gate.crossings = _served
    except Exception:
        pass
    x, y = wps[0]
    collide_r = collision_radius(args.robot_radius)
    if abs(collide_r - COLLIDE_R) > 1e-9:
        print(f"robot radius {args.robot_radius:.3f} m -> collision "
              f"threshold {collide_r:.3f} m (default {COLLIDE_R:.2f})")

    def nearest_sidewalk_edge(px, py):
        best, bd = None, 1e18
        for sw in spec.get("sidewalks", []):
            r = sw["rect"]
            d = (max(r[0] - px, 0, px - r[2]) ** 2 +
                 max(r[1] - py, 0, py - r[3]) ** 2)
            if d < bd:
                bd, best = d, sw["edge"]
        return best

    def spawn_robot():
        if args.robot_as_person:
            e = nearest_sidewalk_edge(x, y)
            traci.person.add("robot0", e, pos=0.5, depart=traci.simulation
                             .getTime())
            traci.person.appendWalkingStage("robot0", [e], arrivalPos=1.0)
            traci.person.setColor("robot0", (255, 40, 40, 255))
            traci.person.setWidth("robot0", 2.0 * args.robot_radius)
        else:
            traci.poi.add("robot0", x, y, (255, 40, 40, 255),
                          poiType="robot", layer=40,
                          width=2.0 * args.robot_radius,
                          height=2.0 * args.robot_radius)

    spawn_robot()

    dt = args.step_length
    strict = (args.strict_sidewalk == "on" or
              (args.strict_sidewalk == "auto" and spec.get("osm")))
    walk_union = walk_prep = None
    if strict:
        walk_union, walk_prep = load_walkable(
            map_dir / f"{args.map}.net.xml", wps)
        if walk_union is None:
            strict = False
        else:
            # the router guarantees the waypoint polyline is walkable (with
            # tolerant junction connectors); union in a 1.2 m corridor
            # around it so the strict layer never fights the sanctioned
            # route where this net's walkingareas are geometrically off
            from shapely.geometry import LineString as _LS2
            from shapely.ops import unary_union as _uu
            from shapely.prepared import prep as _prep
            walk_union = _uu([walk_union,
                              _LS2([tuple(w) for w in wps]).buffer(1.2)])
            walk_prep = _prep(walk_union)
            from shapely.geometry import Point
            from shapely.ops import nearest_points
            p0 = Point(x, y)
            if not walk_prep.covers(p0):
                q, _ = nearest_points(walk_union, p0)
                x, y = q.x, q.y
            print("strict-sidewalk: robot confined to the walkable surface")
    walk_clamped = 0
    model_dir = Path(args.model_dir)
    leg_i = 0
    frame = Frame(legs[0])
    cfg = leg_config(legs[0]["len"], legs[0]["W"], dt, args.max_time)
    planner = build_planner(args.algorithm, cfg, args.seed, model_dir, args.device, params=_tuned_params)

    t = 0.0
    vx = vy = 0.0
    subscribed: set = set()        # persons with a live TraCI subscription
    frozen_since = None            # stall watchdog (not counting light holds)
    path_len = 0.0
    min_ped = float("inf")
    close_steps = 0
    wait_light = 0.0
    held_prev = False
    rows = []
    sfm = None
    if args.reactive_peds != "off" and args.mode == "static":
        print(f"reactive-peds: static mode has no walking pedestrians to "
              f"capture -- {args.reactive_peds} layer idle by construction")
    elif args.reactive_peds != "off":
        su, sp_ = walk_union, walk_prep
        if su is None:
            su, sp_ = load_walkable(map_dir / f"{args.map}.net.xml", wps,
                                    buffer_m=30.0)
        if args.reactive_peds == "sfm":
            from social_pedestrians import SocialForceLayer
            sfm = SocialForceLayer(traci, su, sp_,
                                   net_file=map_dir / f"{args.map}.net.xml")
        else:
            from jupedsim_pedestrians import JuPedSimLayer
            try:
                sfm = JuPedSimLayer(
                    traci, su, sp_,
                    net_file=map_dir / f"{args.map}.net.xml",
                    route_pts=wps, model=args.jps_model,
                    robot_radius=args.robot_radius,
                    robot_in_jps=args.robot_in_jps,
                    seed=args.seed)
            except ImportError as exc:
                sys.exit(f"--reactive-peds jupedsim needs the jupedsim "
                         f"package: pip install jupedsim==1.4.2  ({exc})")

    success = False
    reason = "max_time"
    _statics_scattered = False

    def _scatter_statics():
        """One-time lateral scatter of standing pedestrians: SUMO ignores
        departPosLat for persons, so all statics spawn on the same
        stripe (single file). Spread them across the sidewalk width with
        a seed-driven offset along their local normal, kept on the
        walkable surface."""
        import random as _rnd
        stands = [q for q in traci.person.getIDList()
                  if q.startswith("stand_")]
        if not stands:
            return
        su_, sp2 = walk_union, walk_prep
        if su_ is None:
            su_, sp2 = load_walkable(map_dir / f"{args.map}.net.xml",
                                     wps, buffer_m=60.0)
        from shapely.geometry import Point as _P
        rr = _rnd.Random(args.seed + 77)
        # stratified columns: statics on the SAME sidewalk edge alternate
        # between the two walking columns (one offset band per column),
        # so every sidewalk with >=2 statics has BOTH columns occupied
        by_edge = {}
        for q in stands:
            try:
                by_edge.setdefault(traci.person.getRoadID(q),
                                   []).append(q)
            except traci.exceptions.TraCIException:
                by_edge.setdefault("?", []).append(q)
        for _e, group in sorted(by_edge.items()):
            for k, q in enumerate(sorted(group)):
                try:
                    x, y = traci.person.getPosition(q)
                    ang = math.radians(traci.person.getAngle(q))
                    nx_, ny_ = math.cos(ang), -math.sin(ang)
                    # probe the walkable cross-section [lo, hi] along
                    # the normal, then place columns at absolute 28% /
                    # 72% of that section (alternating) -- robust to
                    # spawn stripes sitting at either sidewalk boundary
                    lo = hi = 0.0
                    if sp2 is not None:
                        while lo > -2.0 and sp2.covers(
                                _P(x + nx_ * (lo - 0.1),
                                   y + ny_ * (lo - 0.1))):
                            lo -= 0.1
                        while hi < 2.0 and sp2.covers(
                                _P(x + nx_ * (hi + 0.1),
                                   y + ny_ * (hi + 0.1))):
                            hi += 0.1
                    frac = 0.28 if k % 2 == 0 else 0.72
                    off = lo + frac * (hi - lo) + rr.uniform(-0.08, 0.08)
                    cx, cy = x + nx_ * off, y + ny_ * off
                    if sp2 is None or sp2.covers(_P(cx, cy)):
                        traci.person.moveToXY(q, "", cx, cy,
                                              keepRoute=2)
                except traci.exceptions.TraCIException:
                    pass

    while t < args.max_time:
        traci.simulationStep()
        if not _statics_scattered:
            _scatter_statics()
            _statics_scattered = True
        t = traci.simulation.getTime()
        gate.step(t)

        # --- observe pedestrians (world) -> obstacles (leg-local)
        obstacles, step_min = observe_persons(
            traci, tc, subscribed, x, y, args.sensor_range, frame, Obstacle)
        min_ped = min(min_ped, step_min)
        if step_min < SOCIAL_R:
            close_steps += 1
        if step_min < collide_r and t > SPAWN_GRACE:
            reason = "collision"
            break

        # --- plan in the leg-local frame
        L = legs[leg_i]
        lx, ly = frame.to_local(x, y)
        lvx, lvy = frame.vel_to_local(vx, vy)
        # clamp the planner-visible pose into the leg box: right after a leg
        # switch on an obtuse corner the true projection can fall slightly
        # BEHIND the new leg (lx < 0), which makes boundary-checking planners
        # (DWA & co) reject every candidate and freeze
        slx = min(max(lx, 0.02), L["len"])
        sly = min(max(ly, 0.06), L["W"] - 0.06)
        state = RobotState(x=slx, y=sly, yaw=math.atan2(lvy, lvx),
                           v=math.hypot(lvx, lvy), w=0.0)
        goal_l = frame.to_local(*L["w1"])
        try:
            cvx, cvy, _info = planner.compute_command(state, goal_l,
                                                      obstacles, t)
        except Exception as exc:      # planner crash -> stop episode cleanly
            reason = f"planner_error:{type(exc).__name__}"
            break
        sp = math.hypot(cvx, cvy)
        cap = min(getattr(planner, "cfg", cfg).max_speed
                  if hasattr(planner, "cfg") else cfg.max_speed, HARD_SPEED_CAP)
        if sp > cap:
            cvx, cvy = cvx / sp * cap, cvy / sp * cap
        wvx, wvy = frame.vel_to_world(cvx, cvy)

        # --- native traffic lights: identical rule for every algorithm
        held = False
        blk = crossing_ahead((x, y), (wvx, wvy), L, gate, frame)
        if blk is not None:
            step_room = max(0.0, blk["gap"] - 0.12)
            max_sp = step_room / dt
            spd = math.hypot(wvx, wvy)
            if spd > max_sp:
                if max_sp < 0.05:
                    wvx = wvy = 0.0
                    held = True
                else:
                    wvx, wvy = wvx / spd * max_sp, wvy / spd * max_sp
            if held:
                wait_light += dt
        held_prev = held

        # --- integrate the POI robot in leg-local coordinates
        plx, ply = frame.to_local(x, y)
        pvx, pvy = frame.vel_to_local(wvx, wvy)
        m = 0.08
        nlx = plx + pvx * dt
        nly = min(max(ply + pvy * dt, m), L["W"] - m)
        nx, ny = frame.to_world(nlx, nly)
        if strict:
            from shapely.geometry import Point
            from shapely.ops import nearest_points
            P = Point(nx, ny)
            if not walk_prep.covers(P):
                q, _ = nearest_points(walk_union, P)
                if q.distance(P) <= 1.5:
                    nx, ny = q.x, q.y        # slide along walkable boundary
                else:
                    nx, ny = x, y            # blocked
                walk_clamped += 1
        path_len += math.hypot(nx - x, ny - y)
        x, y = nx, ny
        vx, vy = wvx, wvy
        if args.robot_as_person:
            if "robot0" not in traci.person.getIDList():
                spawn_robot()          # re-add if its walk stage finished
            try:
                traci.person.moveToXY("robot0", "", x, y, keepRoute=2)
            except traci.exceptions.TraCIException:
                spawn_robot()
        else:
            traci.poi.setPosition("robot0", x, y)
        if sfm is not None:
            sfm.step((x, y), (vx, vy), dt)
        rows.append((round(t, 2), round(x, 3), round(y, 3), round(vx, 3),
                     round(vy, 3), leg_i, int(held),
                     round(step_min, 3) if math.isfinite(step_min) else ""))

        # --- stall watchdog: frozen >45 s while NOT waiting at a light
        if math.hypot(vx, vy) < 0.05 and not held:
            if frozen_since is None:
                frozen_since = t
            elif t - frozen_since > 45.0:
                reason = "stalled"
                break
        else:
            frozen_since = None

        # --- leg switching / arrival
        la, _ = frame.to_local(x, y)
        near_end = (la >= L["len"] - args.leg_switch_dist or
                    math.hypot(L["w1"][0] - x, L["w1"][1] - y)
                    <= args.leg_switch_dist)
        if leg_i == len(legs) - 1:
            if math.hypot(goal_xy[0] - x, goal_xy[1] - y) <= args.goal_tol:
                success = True
                reason = "goal"
                break
        elif near_end:
            leg_i += 1
            frame = Frame(legs[leg_i])
            cfg = leg_config(legs[leg_i]["len"], legs[leg_i]["W"],
                             dt, args.max_time)
            planner = build_planner(args.algorithm, cfg, args.seed,
                                    model_dir, args.device, params=_tuned_params)

    try:
        traci.close(False)
    except Exception:
        pass
    os.chdir(old_cwd)

    with (run_dir / "robot_trace.csv").open("w") as f:
        f.write("t,x,y,vx,vy,leg,held_at_light,min_ped_dist\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    metrics = {
        "map": args.map, "route": route_name, "global_planner": gp,
        "task": task_id,
        "global_plan_time_s": global_plan_time,
        "global_rrt_params": (dict(GLOBAL_RRT_PARAMS) if gp == "rrt"
                              else None),
        "reactive_peds": args.reactive_peds,
        "robot_radius_m": args.robot_radius,
        "robot_height_m": args.robot_height,
        "collision_radius_m": round(collide_r, 3),
        "params_file": (args.params_file or None),
        "sfm_controlled_steps": (sfm.controlled_steps if sfm else 0),
        "sfm_capture_events": (sfm.capture_events if sfm else 0),
        **(sfm.ped_metrics() if sfm else {}),
        "mode": args.mode, "algorithm": args.algorithm,
        "seed": args.seed, "success": success, "termination_reason": reason,
        "sim_time_s": round(t, 2), "path_length_m": round(path_len, 2),
        "avg_speed_mps": round(path_len / max(t, 1e-9), 3),
        # null when the episode never saw a pedestrian: json.dumps(inf) emits
        # bare `Infinity`, which is invalid JSON and silently poisons means
        "min_pedestrian_distance_m": (round(min_ped, 3)
                                      if math.isfinite(min_ped) else None),
        "close_encounter_steps": close_steps,
        "collision": reason == "collision",
        "time_waiting_at_light_s": round(wait_light, 1),
        "strict_sidewalk": strict, "walkable_clamped_steps": walk_clamped,
        "robot_as_person": bool(args.robot_as_person),
        "num_legs": len(legs), "waypoints": wps, "goal_tol": args.goal_tol,
    }
    (run_dir / "robot_metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "scenario.json").write_text(json.dumps({
        "osm_ped_period": (round(osm_ped_period, 3)
                           if osm_ped_period is not None else None),
        "osm_mode_flow_ph": (round(osm_flow, 1)
                             if not mode_demand and not args.demand else None),
        "osm_veh_period": (round(osm_veh_period, 2)
                           if not mode_demand and not args.demand else None),
        "crossing_flow": round(crossing_flow, 2),
        "flow_range": [args.flow_min, args.flow_max],
        "speed_range": [args.speed_min, args.speed_max],
        "veh_scale": args.veh_scale, "route_file": str(rou)}, indent=2))
    if not args.keep_demand and not args.demand and \
            rou.parent == run_dir:
        for pth in (rou, rou.with_suffix(".scenario.json")):
            try:
                pth.unlink()
            except OSError:
                pass
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
