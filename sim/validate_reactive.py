#!/usr/bin/env python3
"""Validation: do pedestrians react to the robot?

A robot STANDS mid-sidewalk in a same-direction pedestrian stream
(map2_crossing, fixed flow, fixed seed). We compare striping-only
(--layer off) against the Social Force bubble (--layer sfm):

  * pass-throughs: pedestrians whose centre came closer than 0.30 m
    (physically walking through the robot)
  * min gap per passing pedestrian, and its distribution
  * mean |lateral offset| at the robot's x (deflection evidence)

Usage:  python sim/validate_reactive.py --layer off
        python sim/validate_reactive.py --layer sfm
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["off", "sfm"], required=True)
    ap.add_argument("--mode", default="same",
                    choices=["same", "opposite", "mixed"])
    ap.add_argument("--flow", type=float, default=260.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sim-time", type=float, default=300.0)
    args = ap.parse_args()

    map_dir = REPO / "maps" / "map2_crossing"
    spec = json.loads((map_dir / "map_spec.json").read_text())
    wps = spec["robot"]["waypoints"]
    # robot stands at the midpoint of the first leg, on the sidewalk
    (x0, y0), (x1, y1) = wps[0], wps[1]
    # stand MID-BLOCK (30% along the leg), away from the junction core:
    # the reactive layer's coverage is the sidewalk segment; junction
    # cores are excluded from SFM control for SUMO stability
    rx, ry = x0 + 0.3 * (x1 - x0), y0 + 0.3 * (y1 - y0)

    tmp = Path(tempfile.mkdtemp())
    rou = tmp / "demand.rou.xml"
    subprocess.run([sys.executable, str(ROOT / "generate_demand.py"),
                    "--spec", str(map_dir / "map_spec.json"),
                    "--mode", args.mode, "--out", str(rou),
                    "--seed", str(args.seed),
                    "--flow-min", str(args.flow),
                    "--flow-max", str(args.flow),
                    "--veh-scale", "0",
                    "--end", str(args.sim_time + 60)], check=True)

    traci.start([checkBinary("sumo"),
                 "-n", str(map_dir / "map2_crossing.net.xml"),
                 "-r", str(rou), "--step-length", "0.5",
                 "--no-step-log", "--no-warnings",
                 "--seed", str(args.seed)])

    # robot as remote-controlled person so striping COULD see it (it does
    # not -- that is the point) and so both layers share one embodiment
    edge0 = None
    for e in traci.edge.getIDList():
        if not e.startswith(":"):
            edge0 = e
            break
    # probe: find the modal walking line near x=rx, then stand THERE
    ys = []
    t = 0.0
    while t < 150.0:
        traci.simulationStep()
        t = traci.simulation.getTime()
        for pid in traci.person.getIDList():
            px, py = traci.person.getPosition(pid)
            if abs(px - rx) < 3.0 and abs(py - ry) < 5.0:
                ys.append(py)
    if ys:
        # modal 0.3 m bin centre: stand ON the busiest walking line,
        # not between stripes (median of a bimodal is a trap)
        from collections import Counter
        bins = Counter(round(y / 0.3) for y in ys)
        ry = bins.most_common(1)[0][0] * 0.3
    print(f"robot standing at ({rx:.1f}, {ry:.2f}) "
          f"(probe n={len(ys)})", file=sys.stderr)
    traci.person.add("robot0", edge0, 0.0)
    traci.person.appendWalkingStage("robot0", [edge0], 1.0)
    traci.simulationStep()
    traci.person.moveToXY("robot0", "", rx, ry, keepRoute=2)

    sfm = None
    if args.layer == "sfm":
        from social_pedestrians import SocialForceLayer
        from benchmark_runner import load_walkable
        su, sp_ = load_walkable(map_dir / "map2_crossing.net.xml",
                                wps, buffer_m=30.0)
        sfm = SocialForceLayer(traci, su, sp_,
                       net_file=map_dir / 'map2_crossing.net.xml')

    min_gap = {}
    lateral_at_robot = []
    t0 = traci.simulation.getTime()
    t = t0
    while t < t0 + args.sim_time:
        traci.simulationStep()
        t = traci.simulation.getTime()
        try:
            traci.person.moveToXY("robot0", "", rx, ry, keepRoute=2)
        except traci.exceptions.TraCIException:
            pass
        for pid in traci.person.getIDList():
            if pid == "robot0":
                continue
            px, py = traci.person.getPosition(pid)
            d = math.hypot(px - rx, py - ry)
            if d < min_gap.get(pid, 1e9):
                min_gap[pid] = d
            if abs(px - rx) < 0.5 and abs(py - ry) < 6.0:
                lateral_at_robot.append(abs(py - ry))
        if sfm is not None:
            sfm.step((rx, ry), (0.0, 0.0), 0.5)
    traci.close()

    passers = {p: g for p, g in min_gap.items() if g < 6.0}
    thru = sum(1 for g in passers.values() if g < 0.30)
    gaps = sorted(passers.values())
    med = gaps[len(gaps) // 2] if gaps else float("nan")
    lat = (sum(lateral_at_robot) / len(lateral_at_robot)
           if lateral_at_robot else float("nan"))
    print(json.dumps({
        "layer": args.layer, "flow": args.flow, "seed": args.seed,
        "passing_peds": len(passers),
        "pass_throughs(<0.30m)": thru,
        "median_min_gap_m": round(med, 3),
        "mean_abs_lateral_at_robot_m": round(lat, 3),
        "sfm_captures": (sfm.capture_events if sfm else 0),
    }))


if __name__ == "__main__":
    main()
