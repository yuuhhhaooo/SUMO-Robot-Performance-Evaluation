#!/usr/bin/env python3
"""Stratified start-goal task sampling (supervisor Option B protocol).

Samples N tasks per map on the pedestrian network:
  * candidates are random points ON sidewalk pieces (length-weighted);
  * screened: unreachable pairs (no walkable route) and pairs whose start
    and goal lie on the same edge are rejected;
  * stratified into path-length bins with equal counts per bin (bins over
    [--length-min, --length-max]);
  * per-task GEOMETRY FEATURES are logged for the mixed models:
    path_length_m, n_turns (heading changes > 30 deg), min_sidewalk_width_m
    (narrowest lane along the route), n_signalised_junctions (TLS within
    12 m of the route), n_route_edges.

The sampling seed, the script name and the bin edges are committed next to
the task list (configs/tasks_<map>.json): the list is a REPRODUCIBLE
random draw, not a hand-picked set. Note: pedestrian flow along the route
is a per-RUN covariate (density is sampled per crowd seed and recorded in
each run's metrics/scenario), not a per-task constant.

    python sim/sample_tasks.py --map map5_ucl --n-tasks 10 \
        --length-min 300 --length-max 1100 --bins 3
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
pathlib_Path = Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))


def parse_ped_geometry(net_file):
    pieces, tls = [], []
    for _ev, el in ET.iterparse(str(net_file)):
        if el.tag == "edge":
            func = el.get("function", "normal")
            if func in ("normal", ""):
                for lane in el.iter("lane"):
                    allow = lane.get("allow") or ""
                    shape = lane.get("shape")
                    if "pedestrian" not in allow or not shape:
                        continue
                    P = [tuple(map(float, q.split(",")))
                         for q in shape.split()]
                    if len(P) < 2:
                        continue
                    L = sum(math.hypot(P[i + 1][0] - P[i][0],
                                       P[i + 1][1] - P[i][1])
                            for i in range(len(P) - 1))
                    if L > 2.0:
                        pieces.append({"pts": P, "len": L,
                                       "edge": el.get("id"),
                                       "width": float(lane.get("width",
                                                               "2.0"))})
            el.clear()
        elif el.tag == "junction":
            if el.get("type") == "traffic_light":
                tls.append((float(el.get("x")), float(el.get("y"))))
            el.clear()
    return pieces, tls


def sample_point(pieces, rng):
    tot = sum(p["len"] for p in pieces)
    r = rng.uniform(0, tot)
    for p in pieces:
        if r <= p["len"]:
            P = p["pts"]
            target = rng.uniform(0, p["len"])
            acc = 0.0
            for i in range(len(P) - 1):
                seg = math.hypot(P[i + 1][0] - P[i][0],
                                 P[i + 1][1] - P[i][1])
                if acc + seg >= target:
                    t = (target - acc) / max(seg, 1e-9)
                    return ((P[i][0] + t * (P[i + 1][0] - P[i][0]),
                             P[i][1] + t * (P[i + 1][1] - P[i][1])),
                            p["edge"])
                acc += seg
            return (P[-1], p["edge"])
        r -= p["len"]
    p = pieces[-1]
    return (p["pts"][0], p["edge"])


def poly_len(w):
    return sum(math.hypot(w[i + 1][0] - w[i][0], w[i + 1][1] - w[i][1])
               for i in range(len(w) - 1))


def n_turns(w, thresh_deg=30.0):
    n = 0
    for i in range(1, len(w) - 1):
        a = math.atan2(w[i][1] - w[i - 1][1], w[i][0] - w[i - 1][0])
        b = math.atan2(w[i + 1][1] - w[i][1], w[i + 1][0] - w[i][0])
        d = abs((b - a + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(d) > thresh_deg:
            n += 1
    return n


def dist_pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def tls_near_route(w, tls, radius=12.0):
    n = 0
    for (tx, ty) in tls:
        for i in range(len(w) - 1):
            if dist_pt_seg(tx, ty, w[i][0], w[i][1],
                           w[i + 1][0], w[i + 1][1]) < radius:
                n += 1
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--length-min", type=float, default=300.0)
    ap.add_argument("--length-max", type=float, default=1100.0)
    ap.add_argument("--sampling-seed", type=int, default=20260810)
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--id-prefix", default="t",
                    help="task id prefix (use 'tt' for held-out TUNING "
                         "task lists)")
    ap.add_argument("--out-name", default=None,
                    help="output filename (default tasks_<map>.json; use "
                         "tuning_tasks_<map>.json for the tuning split)")
    ap.add_argument("--avoid", default=None,
                    help="existing task file; candidates whose endpoints "
                         "sit near any of its tasks are rejected "
                         "(keeps tuning and evaluation splits disjoint)")
    args = ap.parse_args()

    net = REPO / "maps" / args.map / f"{args.map}.net.xml"
    if not net.exists():
        sys.exit(f"net not found: {net}")
    from benchmark_runner import auto_route, build_walk_graph

    avoid = []
    if args.avoid and pathlib_Path(args.avoid).exists():
        av = json.loads(pathlib_Path(args.avoid).read_text())
        avoid = [(tuple(t["start"]), tuple(t["goal"]))
                 for t in av["tasks"]]
    rng = random.Random(args.sampling_seed)
    print("building walkable graph once ...", flush=True)
    G = build_walk_graph(net)
    pieces, tls = parse_ped_geometry(net)
    widths = {p["edge"]: p["width"] for p in pieces}
    edges_b = [args.length_min + k * (args.length_max - args.length_min)
               / args.bins for k in range(args.bins + 1)]
    per_bin = -(-args.n_tasks // args.bins)          # ceil
    bins = [[] for _ in range(args.bins)]
    attempts = rejected_same_edge = rejected_unreach = 0

    while (sum(len(b) for b in bins) < args.n_tasks
           and attempts < args.max_attempts):
        attempts += 1
        (sx, sy), s_edge = sample_point(pieces, rng)
        (gx, gy), g_edge = sample_point(pieces, rng)
        if s_edge == g_edge or s_edge.lstrip("-") == g_edge.lstrip("-"):
            rejected_same_edge += 1
            continue
        if math.hypot(gx - sx, gy - sy) < args.length_min * 0.4:
            continue
        near = False
        for (es, eg) in avoid:
            d1 = (math.hypot(sx-es[0], sy-es[1])
                  + math.hypot(gx-eg[0], gy-eg[1]))
            d2 = (math.hypot(sx-eg[0], sy-eg[1])
                  + math.hypot(gx-es[0], gy-es[1]))
            if min(d1, d2) < 30.0:
                near = True
                break
        if near:
            continue
        try:
            wps, redges = auto_route(net, [(sx, sy), (gx, gy)],
                                     return_edges=True, _graph=G)
        except SystemExit:
            rejected_unreach += 1
            continue
        L = poly_len(wps)
        if not (args.length_min <= L <= args.length_max):
            continue
        b = min(int((L - args.length_min)
                    / max(edges_b[1] - edges_b[0], 1e-9)), args.bins - 1)
        if len(bins[b]) >= per_bin:
            continue
        feat = {
            "path_length_m": round(L, 1),
            "n_turns": n_turns(wps),
            "min_sidewalk_width_m": round(
                min((widths.get(e, 2.0) for e in redges), default=2.0), 2),
            "n_signalised_junctions": tls_near_route(wps, tls),
            "n_route_edges": len(redges),
        }
        bins[b].append({"start": [round(sx, 2), round(sy, 2)],
                        "goal": [round(gx, 2), round(gy, 2)], **feat})

    tasks = []
    for b in bins:
        tasks.extend(b)
    tasks = tasks[:args.n_tasks]
    for i, t in enumerate(tasks, 1):
        t_id = f"{args.id_prefix}{i:02d}"
        tasks[i - 1] = {"id": t_id, **t}
    out = {
        "map": args.map,
        "sampling_script": "sim/sample_tasks.py",
        "sampling_seed": args.sampling_seed,
        "length_bins_m": [round(e, 1) for e in edges_b],
        "per_bin_target": per_bin,
        "screening": {"attempts": attempts,
                      "rejected_same_edge": rejected_same_edge,
                      "rejected_unreachable": rejected_unreach},
        "note": ("Shared crowd seed fixes demand, spawn times, walking "
                 "speeds and appearance -- NOT identical realised "
                 "trajectories once reactive pedestrians interact with "
                 "the robot."),
        "tasks": tasks,
    }
    dest = REPO / "configs" / (args.out_name or f"tasks_{args.map}.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"{args.map}: {len(tasks)} tasks -> {dest}")
    for t in tasks:
        print(f"  {t['id']}: {t['path_length_m']}m turns={t['n_turns']} "
              f"minW={t['min_sidewalk_width_m']} "
              f"tls={t['n_signalised_junctions']}")
    if len(tasks) < args.n_tasks:
        print(f"WARNING: only {len(tasks)}/{args.n_tasks} tasks "
              f"(attempts exhausted; widen length range or raise "
              f"--max-attempts)")


if __name__ == "__main__":
    main()
