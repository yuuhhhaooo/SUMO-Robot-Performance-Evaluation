#!/usr/bin/env python3
"""Import an OpenStreetMap extract as a benchmark map directory.

python osm_import.py --osm area.osm --name map5_city \
    [--start "x,y" --goal "x,y"] [--ped-period 1.5] [--keep-vehicles]

Produces maps/<name>/ with:
  <name>.net.xml    netconvert build: sidewalks + zebra crossings guessed,
                    OSM traffic signals kept (same green/yellow/clearance
                    timing family as the hand-built maps)
  <name>_base.rou.xml   pedestrian (and vehicle) background demand via
                        SUMO randomTrips
  <name>.sumocfg, view.xml, map_spec.json

The spec is extracted generically from the built net: sidewalk lane rects
(+length), signalised crossings with tls id + linkIndex + wait strips, tls
list -- so NativeSignalGate and benchmark_runner.py work.

LIMITS (honest ones):
  * generate_demand.py's mode system (same/opposite/mixed/...) needs the
    hand-built axis-aligned road spec; OSM maps use randomTrips demand
    instead (regenerate with tools/randomTrips.py for other densities).
  * benchmark_runner legs must be AXIS-ALIGNED: pass --waypoints along
    streets that run ~east-west / ~north-south (grid-like areas). Curved or
    diagonal streets are not supported by the leg-local transform.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from sumolib import checkBinary

ROOT = Path(__file__).resolve().parent      # sim/
REPO = ROOT.parent
WAIT_STRIP = 1.8


def run(cmd, **kw):
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, cmd))}\n{res.stderr}")
    return res


def netconvert_osm(osm: Path, out_net: Path, keep_vehicles: bool,
                   lefthand: bool = False, crossing_speed: float = 20.0):
    cmd = [checkBinary("netconvert"), "--osm-files", str(osm),
           "-o", str(out_net),
           "--geometry.remove", "--junctions.join", "--roundabouts.guess",
           "--ramps.guess", "--remove-edges.isolated",
           "--tls.guess-signals", "--tls.discard-simple", "--tls.join",
           "--crossings.guess", "--sidewalks.guess",
           "--crossings.guess.speed-threshold", str(crossing_speed),
           "--sidewalks.guess.max-speed", str(crossing_speed),
           "--tls.green.time", "22", "--tls.yellow.time", "3",
           "--tls.crossing-clearance.time", "5", "--tls.crossing-min.time", "8",
           "--default.sidewalk-width", "2.00", "--no-turnarounds"]
    if lefthand:
        cmd += ["--lefthand", "true"]
    if not keep_vehicles:
        pass  # vehicles stay routable; demand decides what actually drives
    res = run(cmd)
    if res.stderr.strip():
        print(f"[netconvert notes] {res.stderr.strip()[:400]}")


def lane_rect(shape: str, width: float):
    pts = [tuple(map(float, p.split(","))) for p in shape.split()]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w2 = width / 2.0
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    if dx >= dy:      # ~horizontal
        return [round(min(xs), 2), round(min(ys) - w2, 2),
                round(max(xs), 2), round(max(ys) + w2, 2)], "h"
    return [round(min(xs) - w2, 2), round(min(ys), 2),
            round(max(xs) + w2, 2), round(max(ys), 2)], "v"


def extract_spec(name: str, net_path: Path, start, goal):
    tree = ET.parse(net_path)
    root = tree.getroot()
    loc = root.find("location")
    x0, y0, x1, y1 = map(float, loc.get("convBoundary").split(","))

    sidewalks, crossings, cross_by_id = [], [], {}
    for edge in root.iter("edge"):
        func = edge.get("function", "normal")
        if func == "normal":
            for lane in edge.iter("lane"):
                if "pedestrian" in (lane.get("allow") or ""):
                    rect, _ = lane_rect(lane.get("shape"),
                                        float(lane.get("width", "2.0")))
                    sidewalks.append({"edge": edge.get("id"),
                                      "lane": lane.get("id"),
                                      "length": round(float(lane.get("length")), 2),
                                      "rect": rect})
        elif func == "crossing":
            lane = edge.find("lane")
            rect, axis = lane_rect(lane.get("shape"),
                                   float(lane.get("width", "4.0")))
            cr = {"id": edge.get("id"), "rect": rect, "axis": axis,
                  "length": round(float(lane.get("length")), 2),
                  "width": float(lane.get("width", "4.0")),
                  "tls": None, "linkIndex": None}
            crossings.append(cr)
            cross_by_id[edge.get("id")] = cr
    for con in root.iter("connection"):
        to = con.get("to")
        if to in cross_by_id and con.get("tl") is not None:
            cross_by_id[to]["tls"] = con.get("tl")
            cross_by_id[to]["linkIndex"] = int(con.get("linkIndex"))
    for cr in crossings:
        a, b, c, d = cr["rect"]
        if cr["axis"] == "v":
            cr["strips"] = {"N": [a, round(b - WAIT_STRIP, 2), c, b],
                            "S": [a, d, c, round(d + WAIT_STRIP, 2)]}
        else:
            cr["strips"] = {"E": [round(a - WAIT_STRIP, 2), b, a, d],
                            "W": [c, b, round(c + WAIT_STRIP, 2), d]}

    if start is None and sidewalks:
        # placeholder: two far-apart sidewalk rect centres
        def centre(s):
            r = s["rect"]
            return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        pts = [centre(s) for s in sidewalks]
        import math
        best = max(((p, q) for p in pts for q in pts),
                   key=lambda pq: math.hypot(pq[0][0] - pq[1][0],
                                             pq[0][1] - pq[1][1]))
        start, goal = best

    spec = {
        "name": name, "title": f"{name} (OSM import)",
        "extent": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "signal_backend": "native_full", "osm": True,
        "roads": [],                      # mode-demand not applicable
        "sidewalks": sidewalks,
        "walkable_rects": [s["rect"] for s in sidewalks],
        "crossings": crossings,
        "tls": sorted({c["tls"] for c in crossings if c["tls"]}),
        "plazas": [],
        "robot": {"start": list(start), "goal": list(goal),
                  "waypoints": [list(start), list(goal)]},
        "routes": {"default": [list(start), list(goal)]},
        "signal_params": {"wait_strip": WAIT_STRIP},
    }
    return spec


def random_trips(net: Path, out_rou: Path, ped_period: float,
                 veh_period: float, seed: int, end: float):
    import sumolib
    sp = Path(sumolib.__file__).resolve()
    cands = [Path(os.environ.get("SUMO_HOME", "/nonexistent")) / "tools",
             sp.parents[1] / "sumo" / "tools",       # pip eclipse-sumo layout
             sp.parents[1] / "tools",
             sp.parents[2] / "tools"]
    tools = next((c for c in cands if (c / "randomTrips.py").exists()), None)
    if tools is None:
        sys.exit("randomTrips.py not found -- set SUMO_HOME to your SUMO "
                 "installation")
    rt = tools / "randomTrips.py"
    parts = []
    if ped_period > 0:
        ped = out_rou.with_suffix(".ped.rou.xml")
        ped_val = Path(str(ped) + ".val.rou.xml")   # per-run: kills the
        # Windows race on randomTrips' default CWD-shared routes.rou.xml
        run([sys.executable, str(rt), "-n", str(net), "-o", str(ped),
             "-r", str(ped_val),
             "--pedestrians", "--period", str(ped_period), "--seed", str(seed),
             "--begin", "0", "--end", str(end), "--max-distance", "800",
             "--validate"])
        ped_val.unlink(missing_ok=True)
        parts.append(ped)
    if veh_period > 0:
        veh = out_rou.with_suffix(".veh.rou.xml")
        veh_val = Path(str(veh) + ".val.rou.xml")   # per-run, same reason
        run([sys.executable, str(rt), "-n", str(net), "-o", str(veh),
             "-r", str(veh_val),
             "--period", str(veh_period), "--seed", str(seed + 7),
             "--begin", "0", "--end", str(end), "--validate"])
        veh_val.unlink(missing_ok=True)
        parts.append(veh)
    # merge into ONE file SORTED BY DEPART TIME -- sumo streams route files
    # in order and silently discards entries whose depart lies in the past,
    # so a naive "peds block then cars block" concat loses every vehicle
    import xml.etree.ElementTree as _ET
    vtypes, items = [], []
    for pth in parts:
        root = _ET.parse(pth).getroot()
        for el in list(root):
            if el.tag == "vType":
                vtypes.append(el)
            else:
                items.append((float(el.get("depart", "0")), len(items), el))
        pth.unlink(missing_ok=True)
    items.sort(key=lambda t: (t[0], t[1]))
    routes = _ET.Element("routes")
    for el in vtypes:
        routes.append(el)
    for _, _, el in items:
        routes.append(el)
    _ET.indent(routes, space="    ")
    _ET.ElementTree(routes).write(out_rou, encoding="UTF-8",
                                  xml_declaration=True)


def _sorted_routes_write(elements, out_rou):
    """vTypes first, then everything sorted by depart/begin."""
    import xml.etree.ElementTree as _ET
    vtypes = [e for e in elements if e.tag == "vType"]
    items = [(float(e.get("depart", e.get("begin", "0"))), i, e)
             for i, e in enumerate(elements) if e.tag != "vType"]
    items.sort(key=lambda t: (t[0], t[1]))
    routes = _ET.Element("routes")
    for e in vtypes:
        routes.append(e)
    for _, _, e in items:
        routes.append(e)
    _ET.indent(routes, space="    ")
    _ET.ElementTree(routes).write(out_rou, encoding="UTF-8",
                                  xml_declaration=True)


def osm_mode_demand(net_file, route_edges, mode, flow_ph, statics_n,
                    speed_min, speed_max, veh_period, out_rou, seed,
                    end=3600.0, bg_ped_period=None):
    """Directional pedestrian demand ALONG THE ROBOT'S ROUTE for OSM maps.

    Mirrors the built-in maps' mode semantics, defined relative to the
    robot's direction of travel: same / opposite / mixed / static / all.
    flow_ph = pedestrians per hour PER DIRECTION; statics_n standing
    pedestrians on the route (static/all); vehicles via randomTrips;
    optional randomTrips background pedestrians (bg_ped_period)."""
    import random as _random
    import xml.etree.ElementTree as _ET
    rng = _random.Random(seed)
    if len(route_edges) < 2:
        sys.exit("osm_mode_demand: route has fewer than 2 edges")
    # anchor chain: first, ~1/3, ~2/3, last (dedup, keep order)
    idx = sorted({0, len(route_edges) // 3, (2 * len(route_edges)) // 3,
                  len(route_edges) - 1})
    anchors = [route_edges[i] for i in idx]
    els = []

    def add_dir_flows(tag, edge_seq, per_hour):
        """Individual persons, entering at a RANDOM point along the route
        (random segment + random departPos) so the corridor is at steady-
        state density immediately -- mirrors the built-in maps' departPos
        random behaviour."""
        for k in range(3):      # 3 walking-speed classes
            spd = round(rng.uniform(speed_min, speed_max), 3)
            els.append(_ET.Element("vType", id=f"pt_{tag}_{k}",
                                   vClass="pedestrian", maxSpeed=str(spd)))
        if per_hour <= 0.05:
            return
        n = int(round(per_hour * end / 3600.0))
        for i in range(n):
            t = rng.uniform(0.0, end)
            j = rng.randrange(0, len(edge_seq) - 1)
            pe = _ET.Element("person", id=f"{tag}_{i}",
                            depart=f"{t:.2f}", departPos="random",
                            type=f"pt_{tag}_{rng.randrange(3)}")
            for a, b in zip(edge_seq[j:-1], edge_seq[j + 1:]):
                _ET.SubElement(pe, "personTrip",
                               attrib={"from": a, "to": b})
            els.append(pe)

    if mode in ("same", "mixed", "all"):
        add_dir_flows("fwd", anchors,
                      flow_ph if mode == "same" else flow_ph / 2.0)
    if mode in ("opposite", "mixed", "all"):
        add_dir_flows("bwd", list(reversed(anchors)),
                      flow_ph if mode == "opposite" else flow_ph / 2.0)
    if mode in ("static", "all") and statics_n > 0:
        # standing pedestrians at random positions on the route edges
        lens = {}
        for _ev, edge in __import__("xml.etree.ElementTree", fromlist=["x"])                 .iterparse(str(net_file)):
            if edge.tag != "edge":
                continue
            if edge.get("id") in set(route_edges):
                for lane in edge.iter("lane"):
                    if "pedestrian" in (lane.get("allow") or ""):
                        lens[edge.get("id")] = float(lane.get("length"))
            edge.clear()
        for k in range(statics_n):
            e = rng.choice(route_edges)
            L = lens.get(e, 20.0)
            pos = round(rng.uniform(0.15, 0.85) * L, 2)
            pe = _ET.Element("person", id=f"static_{k}",
                            depart=f"{rng.uniform(0, 30):.2f}",
                            departPos=str(pos))
            _ET.SubElement(pe, "stop", edge=e, endPos=str(pos),
                           duration=f"{end:.0f}")
            els.append(pe)
    # vehicles (+ optional background peds) via randomTrips, merged in
    if (veh_period and veh_period > 0) or bg_ped_period:
        tmp = out_rou.with_suffix(".bgtmp.rou.xml")
        random_trips(Path(net_file), tmp,
                     bg_ped_period if bg_ped_period else 0.0,
                     veh_period if veh_period else 0.0, seed + 13, end)
        root = _ET.parse(tmp).getroot()
        els.extend(list(root))
        tmp.unlink(missing_ok=True)
    _sorted_routes_write(els, out_rou)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--osm", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--start", default=None, help='"x,y" in net coordinates')
    p.add_argument("--goal", default=None)
    p.add_argument("--ped-period", type=float, default=1.5,
                   help="randomTrips: one pedestrian every N seconds (0=off)")
    p.add_argument("--veh-period", type=float, default=6.0,
                   help="randomTrips: one vehicle every N seconds (0=off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--end", type=float, default=3600.0)
    p.add_argument("--keep-vehicles", action="store_true")
    p.add_argument("--lefthand", action="store_true",
                   help="left-hand traffic (UK, Japan, ...)")
    p.add_argument("--crossing-speed-threshold", type=float, default=20.0,
                   help="guess zebra crossings on roads up to this speed "
                        "[m/s]; default 20 covers 40 mph urban roads "
                        "(netconvert's default 13.89 leaves footpaths at "
                        "bigger roads dead-ended)")
    args = p.parse_args()

    d = REPO / "maps" / args.name
    d.mkdir(parents=True, exist_ok=True)
    net = d / f"{args.name}.net.xml"
    netconvert_osm(Path(args.osm), net, args.keep_vehicles,
                   args.lefthand, args.crossing_speed_threshold)

    start = tuple(map(float, args.start.split(","))) if args.start else None
    goal = tuple(map(float, args.goal.split(","))) if args.goal else None
    spec = extract_spec(args.name, net, start, goal)
    (d / "map_spec.json").write_text(json.dumps(spec, indent=2))

    rou = d / f"{args.name}_base.rou.xml"
    random_trips(net, rou, args.ped_period, args.veh_period, args.seed,
                 args.end)
    (d / f"{args.name}.sumocfg").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{args.name}.net.xml"/>
        <route-files value="{args.name}_base.rou.xml"/>
        <gui-settings-file value="view.xml"/>
    </input>
    <time><begin value="0"/><end value="36000"/><step-length value="0.5"/></time>
    <processing><pedestrian.model value="striping"/><pedestrian.striping.stripe-width value="1.00"/></processing>
</configuration>
""")
    (d / "view.xml").write_text((ROOT / "BenchView.xml").read_text())
    sig = sum(1 for c in spec["crossings"] if c["tls"])
    print(f"{args.name}: {len(spec['tls'])} TLS, {len(spec['crossings'])} "
          f"crossings ({sig} signalised), {len(spec['sidewalks'])} sidewalk "
          f"lanes -> {d}")
    print(f"robot placeholder start/goal: {spec['robot']['start']} -> "
          f"{spec['robot']['goal']}  (set your own with --start/--goal or "
          f"benchmark_runner --waypoints)")
    print(f"view:  sumo-gui -c {d / (args.name + '.sumocfg')}")


if __name__ == "__main__":
    main()
