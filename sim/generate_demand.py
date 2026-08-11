#!/usr/bin/env python3
"""Pedestrian + vehicle demand for the FULLY NATIVE benchmark maps (Option B).

personTrip-OD style walks on real sidewalks; the pedestrian router sends
perpendicular flows over the signalised crossings, where SUMO's striping
model makes them wait for green NATIVELY.  Per-flow vTypes carry the sampled
walking speed (maxSpeed); departPos is random along the first edge.

EVERY road follows the mode and every walk covers the FULL road length
(inserted 0.3 m from one end, arriving 0.3 m before the other, crossing
every junction on the way).

TWO PEDESTRIAN COLUMNS PER SIDEWALK (v5 semantics on native sidewalks):
each sidewalk carries two column flows "a" and "b"; the sumocfg sets
pedestrian.striping.stripe-width=1.0 so the 2.0 m sidewalk is exactly two
stripes and opposing columns sort onto separate stripes physically.
    same     : both columns west->east on horizontal roads,
               NORTH->SOUTH on vertical roads
    opposite : both columns east->west / south->north
    mixed    : column a walks hi -> lo, column b walks lo -> hi
    static   : standing pedestrians only (scattered on both stripes)
    all      : mixed columns + statics
Rates are PER COLUMN: "main"-role roads sample --flow-min..max (or --flow);
"cross"-role roads use --crossing-flow.  --road-mode ROAD=MODE overrides
per road.

Vehicles: one from->to flow per road direction (routes straight through the
junctions), scaled by --veh-scale.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--mode", choices=["same", "opposite", "mixed", "static", "all"],
                   required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--flow", type=float, default=None)
    p.add_argument("--speed-min", type=float, default=0.80)
    p.add_argument("--speed-max", type=float, default=1.60)
    p.add_argument("--flow-min", type=float, default=80.0)
    p.add_argument("--flow-max", type=float, default=350.0)
    p.add_argument("--crossing-flow", type=float, default=140.0)
    p.add_argument("--road-mode", action="append", default=[], metavar="ROAD=MODE")
    p.add_argument("--static-min", type=int, default=4)
    p.add_argument("--static-max", type=int, default=14)
    p.add_argument("--veh-scale", type=float, default=1.0)
    p.add_argument("--begin", type=float, default=0.0)
    p.add_argument("--end", type=float, default=36000.0)
    return p


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    spec = json.loads(Path(args.spec).read_text())
    if args.speed is not None:
        args.speed_min = args.speed_max = args.speed
    if args.flow is not None:
        args.flow_min = args.flow_max = args.flow

    overrides = {}
    for rm in args.road_mode:
        rid, _, mode = rm.partition("=")
        overrides[rid] = mode

    vtypes, items = [], []
    meta = {"map": spec["name"], "mode": args.mode, "seed": args.seed,
            "flows": [], "static": [], "vehicles": []}
    fid_n = [0]

    edge_len = {s_["edge"]: s_["length"] for s_ in spec["sidewalks"]}
    MARGIN = 0.3

    def walk_flow(route_edges, side, speed, per_hour, backward):
        """Full-length walk: road end -> road end.

        `backward` = the edges are traversed against their direction
        (positions are measured along each edge's own geometry)."""
        if per_hour <= 0 or not route_edges:
            return
        first_l = edge_len.get(route_edges[0], 10.0)
        last_l = edge_len.get(route_edges[-1], 10.0)
        dep = (first_l - MARGIN) if backward else MARGIN
        arr = MARGIN if backward else (last_l - MARGIN)
        fid = f"pf{fid_n[0]}"
        fid_n[0] += 1
        vt = f"t_{fid}"
        vtypes.append(f'    <vType id="{vt}" vClass="pedestrian" '
                      f'maxSpeed="{speed:.3f}" width="0.64" length="0.35"/>')
        items.append((1.0,
            f'    <personFlow id="{fid}" type="{vt}" begin="{args.begin:.2f}" '
            f'end="{args.end:.2f}" period="exp({per_hour / 3600.0:.6f})" '
            f'departPos="{dep:.2f}">\n'
            f'        <walk edges="{" ".join(route_edges)}" '
            f'arrivalPos="{arr:.2f}"/>\n'
            f'    </personFlow>'))
        meta["flows"].append({"id": fid, "edges": route_edges[:1] + ["..."],
                              "side": side, "speed": round(speed, 3),
                              "per_hour": round(per_hour, 2),
                              "backward": backward})

    def road_mode(r):
        return overrides.get(r["id"], args.mode)

    roads = {r["id"]: r for r in spec["roads"]}
    for r in spec["roads"]:
        f_route = r["routes"]["f"]                     # lo -> hi
        r_route = r["routes"]["r"]                     # hi -> lo
        # full-length routes per sidewalk side: +1 = lo->hi, -1 = hi->lo
        routes_by_dir = {
            "f": {+1: (f_route, False), -1: (list(reversed(f_route)), True)},
            "r": {+1: (list(reversed(r_route)), True), -1: (r_route, False)},
        }
        mode = road_mode(r)
        # v5 two-column semantics on each sidewalk; vertical roads flip:
        # same = north->south (hi->lo), opposite = south->north
        s_ = +1 if r["axis"] == "h" else -1
        col_dirs = {"same": {"a": s_, "b": s_},
                    "opposite": {"a": -s_, "b": -s_},
                    "mixed": {"a": -s_, "b": s_},
                    "all": {"a": -s_, "b": s_}}
        sp = lambda: rng.uniform(args.speed_min, args.speed_max)
        if mode in col_dirs:
            for side in ("f", "r"):
                for col, d in col_dirs[mode].items():
                    per = (args.crossing_flow if r["role"] == "cross"
                           else rng.uniform(args.flow_min, args.flow_max))
                    route, bw = routes_by_dir[side][d]
                    walk_flow(route, f"{side}.{col}", sp(), per, bw)

        if mode in ("static", "all"):
            n = rng.randint(args.static_min, args.static_max)
            cands = [s for s in spec["sidewalks"]
                     if s["edge"] in f_route + r_route]
            rng.shuffle(cands)
            for i in range(n):
                # round-robin across sidewalk sides/columns so statics
                # spread over BOTH columns instead of piling on one
                s = cands[i % len(cands)]
                x0, y0, x1, y1 = s["rect"]
                length = max(x1 - x0, y1 - y0)
                pos = rng.uniform(2.0, max(2.5, length - 2.0))
                pid = f"stand_{r['id']}_{i}"
                items.append((0.5,
                    f'    <person id="{pid}" '
                    f'depart="0.00" departPos="{pos:.2f}">\n'
                    f'        <walk edges="{s["edge"]}" '
                    f'arrivalPos="{min(length - 0.5, pos + 1.0):.2f}" '
                    f'speed="0.00002"/>\n'
                    f'    </person>'))
                meta["static"].append({"id": pid, "edge": s["edge"],
                                       "pos": round(pos, 2)})

    if args.veh_scale > 0:
        for r in spec["roads"]:
            vph = r["veh_per_hour"] * args.veh_scale
            if vph <= 0:
                continue
            for dtag in ("f", "r"):
                segs = r["routes"][dtag]
                items.append((0.0,
                    f'    <flow id="veh_{r["id"]}_{dtag}" begin="0.00" '
                    f'end="{args.end:.2f}" vehsPerHour="{vph:.2f}" '
                    f'from="{segs[0]}" to="{segs[-1]}" '
                    f'departLane="best" departSpeed="max"/>'))
                meta["vehicles"].append({"road": r["id"], "dir": dtag,
                                         "vehs_per_hour": round(vph, 2)})

    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n'
           f"    <!-- native demand: map={spec['name']} mode={args.mode} "
           f"seed={args.seed} -->\n"
           + "\n".join(vtypes) + "\n"
           + "\n".join(x for _, x in sorted(items, key=lambda kv: kv[0]))
           + "\n</routes>\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    out.with_suffix(".scenario.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({len(meta['flows'])} walk flows, "
          f"{len(meta['static'])} statics, {len(meta['vehicles'])} veh flows)")


if __name__ == "__main__":
    main()
