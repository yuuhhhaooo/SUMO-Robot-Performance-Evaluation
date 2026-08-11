#!/usr/bin/env python3
"""Option B builder -- FULLY NATIVE SUMO benchmark maps (v7 architecture).

Standard SUMO pipeline: .nod.xml (traffic_light nodes) + .edg.xml
(sidewalkWidth=2.00) -> netconvert --crossings.guess -> real junctions,
native tlLogic with vehicle AND pedestrian phases, signalised zebra
crossings, walkingareas.  Vehicles and pedestrians obey the lights NATIVELY
(no TraCI controller at all).

Consequences handled here (matching the reference design):
  * pedestrianised roads (King's Mews, Garden Row) are real roads with
    sidewalks but zero vehicle flows -> their junctions with vehicular roads
    are ordinary TLS junctions (map4 ends up with 8 TLS)
  * T-junctions get a 4th-arm alley stub so netconvert emits a standard
    alternating program and crossings exist on all sides
  * the ROBOT is a red POI marker, NOT a SUMO person: pedestrians cannot
    react to it (fairness) and it avoids the SUMO 1.27 moveToXY-into-
    walkingarea segfault.  Robot kinematics belong to the benchmark runner;
    light compliance goes through native_signal_gate.NativeSignalGate.

map_spec.json per map: extent, roads (+routes for vehicle flows), sidewalk
walkable rects, signalised crossings (tls + linkIndex + rect + wait strips),
robot start/goal/waypoints, tls list.
"""
from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from sumolib import checkBinary

ROOT = Path(__file__).resolve().parent      # sim/
REPO = ROOT.parent
MAPS_DIR = REPO / "maps"
SPEED = 13.89
SIDEWALK_W = 2.00
STUB = 14.0
WAIT_STRIP = 1.8

# road: {id, axis, c, lo, hi, veh, role}  role: main | cross
MAPS = {
    "map1_straight": {
        "title": "Map 1 - straight road (native)",
        "extent": (0.0, -8.0, 300.0, 8.0),
        "roads": [
            {"id": "main", "axis": "h", "c": 0.0, "lo": 0.0, "hi": 300.0,
             "veh": 600.0, "role": "main"},
        ],
        "robot": {"start": (2.0, 4.2), "goal": (298.0, 4.2),
                  "waypoints": [(2.0, 4.2), (298.0, 4.2)]},
    },
    "map2_crossing": {
        "title": "Map 2 - single crossing (native)",
        "extent": (0.0, -80.0, 300.0, 80.0),
        "roads": [
            {"id": "main", "axis": "h", "c": 0.0, "lo": 0.0, "hi": 300.0,
             "veh": 600.0, "role": "main"},
            {"id": "cross", "axis": "v", "c": 150.0, "lo": -80.0, "hi": 80.0,
             "veh": 120.0, "role": "cross"},
        ],
        "robot": {"start": (2.0, 4.2), "goal": (298.0, 4.2),
                  "waypoints": [(2.0, 4.2), (298.0, 4.2)]},
    },
    "map3_grid": {
        "title": "Map 3 - 2x2 street grid (native)",
        "extent": (0.0, 0.0, 300.0, 200.0),
        "roads": [
            {"id": "A", "axis": "h", "c": 50.0, "lo": 0.0, "hi": 300.0,
             "veh": 400.0, "role": "main"},
            {"id": "B", "axis": "h", "c": 150.0, "lo": 0.0, "hi": 300.0,
             "veh": 400.0, "role": "main"},
            {"id": "V1", "axis": "v", "c": 100.0, "lo": 0.0, "hi": 200.0,
             "veh": 100.0, "role": "cross"},
            {"id": "V2", "axis": "v", "c": 200.0, "lo": 0.0, "hi": 200.0,
             "veh": 100.0, "role": "cross"},
        ],
        "robot": {"start": (2.0, 54.2), "goal": (298.0, 154.2),
                  "waypoints": [(2.0, 54.2), (104.2, 54.2),
                                (104.2, 154.2), (298.0, 154.2)]},
    },
    "map4_london": {
        "title": "Map 4 - stylised London block (native)",
        "extent": (0.0, 0.0, 360.0, 260.0),
        "roads": [
            {"id": "high",   "axis": "h", "c": 40.0,  "lo": 0.0,  "hi": 360.0,
             "veh": 500.0, "role": "main"},
            {"id": "market", "axis": "h", "c": 140.0, "lo": 70.0, "hi": 360.0,
             "veh": 200.0, "role": "main"},
            {"id": "garden", "axis": "h", "c": 220.0, "lo": 70.0, "hi": 300.0,
             "veh": 0.0, "role": "main"},
            {"id": "church", "axis": "v", "c": 70.0,  "lo": 0.0,  "hi": 260.0,
             "veh": 150.0, "role": "cross"},
            {"id": "mews",   "axis": "v", "c": 180.0, "lo": 40.0, "hi": 140.0,
             "veh": 0.0, "role": "cross"},
            {"id": "bridge", "axis": "v", "c": 300.0, "lo": 40.0, "hi": 260.0,
             "veh": 150.0, "role": "cross"},
        ],
        "plazas": [{"id": "queen_square", "rect": (232.0, 147.0, 291.0, 213.0)}],
        "robot": {"start": (2.0, 44.2), "goal": (358.0, 144.2),
                  "waypoints": [(2.0, 44.2), (295.4, 44.2),
                                (295.4, 144.2), (358.0, 144.2)]},
        "alt_routes": {
            # path1 = the original default itinerary, addressable by name
            "path1": [(2.0, 44.2), (295.4, 44.2), (295.4, 144.2),
                      (358.0, 144.2)],
            # path2: east on High St -> up the middle (King's Mews) ->
            # RIGHT at the market T-junction -> LEFT at Bridge St -> up
            "path2": [(2.0, 44.2), (175.8, 44.2), (175.8, 135.8),
                      (295.8, 135.8), (295.8, 258.0)],
        },
    },
}


def build_nodes_edges(m):
    roads = {r["id"]: r for r in m["roads"]}
    # junctions: every intersection of two perpendicular roads
    junctions = []
    for hid, H in roads.items():
        if H["axis"] != "h":
            continue
        for vid, Vr in roads.items():
            if Vr["axis"] != "v":
                continue
            if (H["lo"] - 0.5 <= Vr["c"] <= H["hi"] + 0.5 and
                    Vr["lo"] - 0.5 <= H["c"] <= Vr["hi"] + 0.5):
                junctions.append({"id": f"J_{hid}_{vid}", "x": Vr["c"],
                                  "y": H["c"], "h": hid, "v": vid})
    nodes, edges, routes = {}, [], {}
    for rid, road in roads.items():
        pts = {road["lo"]: f"E_{rid}_0", road["hi"]: f"E_{rid}_1"}
        for j in junctions:
            if rid in (j["h"], j["v"]):
                pts[j["x"] if road["axis"] == "h" else j["y"]] = j["id"]
        seq = []
        for along in sorted(pts):
            nid = pts[along]
            xy = ((along, road["c"]) if road["axis"] == "h"
                  else (road["c"], along))
            nodes[nid] = xy
            seq.append(nid)
        f_segs, r_segs = [], []
        for k in range(len(seq) - 1):
            f_segs.append((f"{rid}_f{k}", seq[k], seq[k + 1]))
            r_segs.append((f"{rid}_r{k}", seq[k + 1], seq[k]))
        r_segs.reverse()
        edges += f_segs + r_segs
        routes[rid] = {"f": [e for e, _, _ in f_segs],
                       "r": [e for e, _, _ in r_segs]}
    # alley stubs for T junctions
    for j in junctions:
        H, Vr = roads[j["h"]], roads[j["v"]]
        arms = {"W": H["lo"] < j["x"] - 1.0, "E": H["hi"] > j["x"] + 1.0,
                "S": Vr["lo"] < j["y"] - 1.0, "N": Vr["hi"] > j["y"] + 1.0}
        for d, present in arms.items():
            if present:
                continue
            dx, dy = {"W": (-STUB, 0), "E": (STUB, 0),
                      "S": (0, -STUB), "N": (0, STUB)}[d]
            sn = f"S_{j['id']}_{d}"
            nodes[sn] = (j["x"] + dx, j["y"] + dy)
            edges.append((f"stub_{j['id']}_{d}_in", sn, j["id"]))
            edges.append((f"stub_{j['id']}_{d}_out", j["id"], sn))
    tls = {j["id"] for j in junctions}
    return nodes, edges, routes, junctions, tls


def netconvert(name, nodes, edges, tls, out_net, tmp):
    nod = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    for nid, (x, y) in sorted(nodes.items()):
        typ = "traffic_light" if nid in tls else "priority"
        nod.append(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}" type="{typ}"/>')
    nod.append("</nodes>")
    edg = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for eid, a, b in edges:
        edg.append(f'    <edge id="{eid}" from="{a}" to="{b}" numLanes="1" '
                   f'speed="{SPEED}" sidewalkWidth="{SIDEWALK_W:.2f}"/>')
    edg.append("</edges>")
    (tmp / "n.nod.xml").write_text("\n".join(nod) + "\n")
    (tmp / "n.edg.xml").write_text("\n".join(edg) + "\n")
    res = subprocess.run(
        [checkBinary("netconvert"),
         "-n", str(tmp / "n.nod.xml"), "-e", str(tmp / "n.edg.xml"),
         "--crossings.guess", "--no-turnarounds",
         "--offset.disable-normalization", "true",
         "--tls.green.time", "22", "--tls.yellow.time", "3",
         "--tls.crossing-clearance.time", "5", "--tls.crossing-min.time", "8",
         "-o", str(out_net)], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"netconvert failed for {name}:\n{res.stderr}")
    if res.stderr.strip():
        print(f"  [netconvert notes] {res.stderr.strip()[:300]}")


def extract_spec(m, name, net_path, routes, junctions):
    """Pull walkable sidewalks + signalised crossings out of the built net."""
    tree = ET.parse(net_path)
    root = tree.getroot()

    def lane_rect(shape, width):
        pts = [tuple(map(float, p.split(","))) for p in shape.split()]
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        w2 = width / 2.0
        if abs(x1 - x0) >= abs(y1 - y0):     # horizontal
            return [round(min(x0, x1), 2), round(min(y0, y1) - w2, 2),
                    round(max(x0, x1), 2), round(max(y0, y1) + w2, 2)]
        return [round(min(x0, x1) - w2, 2), round(min(y0, y1), 2),
                round(max(x0, x1) + w2, 2), round(max(y0, y1), 2)]

    sidewalks, crossings, cross_by_id = [], [], {}
    for edge in root.iter("edge"):
        func = edge.get("function", "normal")
        if func == "normal":
            for lane in edge.iter("lane"):
                if "pedestrian" in (lane.get("allow") or ""):
                    sidewalks.append({
                        "edge": edge.get("id"), "lane": lane.get("id"),
                        "length": round(float(lane.get("length")), 2),
                        "rect": lane_rect(lane.get("shape"),
                                          float(lane.get("width", "2.0")))})
        elif func == "crossing":
            lane = edge.find("lane")
            width = float(lane.get("width", "4.0"))
            rect = lane_rect(lane.get("shape"), width)
            cr = {"id": edge.get("id"), "rect": rect,
                  "axis": "v" if (rect[3] - rect[1]) > (rect[2] - rect[0])
                  else "h",
                  "length": float(lane.get("length")), "width": width,
                  "tls": None, "linkIndex": None}
            crossings.append(cr)
            cross_by_id[edge.get("id")] = cr

    for con in root.iter("connection"):
        to = con.get("to")
        if to in cross_by_id and con.get("tl") is not None:
            cross_by_id[to]["tls"] = con.get("tl")
            cross_by_id[to]["linkIndex"] = int(con.get("linkIndex"))

    # wait strips + crossed road (nearest junction's arm direction)
    for cr in crossings:
        x0, y0, x1, y1 = cr["rect"]
        if cr["axis"] == "v":            # pedestrians walk vertically
            cr["strips"] = {"N": [x0, round(y0 - WAIT_STRIP, 2), x1, y0],
                            "S": [x0, y1, x1, round(y1 + WAIT_STRIP, 2)]}
        else:
            cr["strips"] = {"E": [round(x0 - WAIT_STRIP, 2), y0, x0, y1],
                            "W": [x1, y0, round(x1 + WAIT_STRIP, 2), y1]}
        jid = cr["tls"]
        j = next((jj for jj in junctions if jj["id"] == jid), None)
        if j:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            cr["road"] = j["h"] if cr["axis"] == "v" else j["v"]
            cr["junction"] = jid

    spec = {
        "name": name, "title": m["title"], "extent": list(m["extent"]),
        "signal_backend": "native_full",
        "roads": [{**{k: r[k] for k in
                      ("id", "axis", "c", "lo", "hi", "role")},
                   "veh_per_hour": r["veh"], "routes": routes[r["id"]]}
                  for r in m["roads"]],
        "sidewalks": sidewalks,
        "walkable_rects": [s["rect"] for s in sidewalks],
        "crossings": crossings,
        "tls": sorted({c["tls"] for c in crossings if c["tls"]}),
        "plazas": m.get("plazas", []),
        "robot": {"start": list(m["robot"]["start"]),
                  "goal": list(m["robot"]["goal"]),
                  "waypoints": [list(w) for w in m["robot"]["waypoints"]]},
        "routes": {"default": [list(w) for w in m["robot"]["waypoints"]],
                   **{k: [list(w) for w in v]
                      for k, v in m.get("alt_routes", {}).items()}},
        "signal_params": {"wait_strip": WAIT_STRIP},
    }
    return spec


def write_cfg(name, path):
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{name}.net.xml"/>
        <route-files value="{name}_base.rou.xml"/>
        <gui-settings-file value="view.xml"/>
    </input>
    <time><begin value="0"/><end value="36000"/><step-length value="0.5"/></time>
    <processing><pedestrian.model value="striping"/><pedestrian.striping.stripe-width value="1.00"/></processing>
</configuration>
""")


def main():
    view = (ROOT / "BenchView.xml").read_text()
    tmp = ROOT / "_ncbuild"
    tmp.mkdir(exist_ok=True)
    for name, m in MAPS.items():
        d = MAPS_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        nodes, edges, routes, junctions, tls = build_nodes_edges(m)
        net_path = d / f"{name}.net.xml"
        netconvert(name, nodes, edges, tls, net_path, tmp)
        spec = extract_spec(m, name, net_path, routes, junctions)
        (d / "map_spec.json").write_text(json.dumps(spec, indent=2))
        write_cfg(name, d / f"{name}.sumocfg")
        (d / "view.xml").write_text(view)
        sig = sum(1 for c in spec["crossings"] if c["tls"])
        print(f"{name}: {len(tls)} TLS, {len(spec['crossings'])} crossings "
              f"({sig} signalised), {len(spec['sidewalks'])} sidewalk lanes -> {d}")


if __name__ == "__main__":
    main()
