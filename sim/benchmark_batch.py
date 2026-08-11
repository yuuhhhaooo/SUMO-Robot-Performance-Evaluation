#!/usr/bin/env python3
"""Sweep algorithms x maps x modes x seeds, aggregate metrics, draw plots.

python benchmark_batch.py --maps map2_crossing map3_grid \
    --modes same mixed --algorithms dwa astar orca sarl \
    --seeds 1 2 3 --out-root results

Seeds drive pedestrian density (flows sampled per seed inside the runner).
Add --waypoints/--reverse to test other start/goal routes (forwarded).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent      # sim/
REPO = ROOT.parent
from benchmark_adapters import ALGORITHMS  # noqa: E402

MAPS = ["map1_straight", "map2_crossing", "map3_grid", "map4_london"]
MODES = ["same", "opposite", "mixed", "static", "all"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps", nargs="+", default=["map2_crossing"],
                   help="any directory under maps/ (built-ins or OSM "
                        "imports)")
    p.add_argument("--modes", nargs="+", default=["mixed"], choices=MODES)
    p.add_argument("--algorithms", nargs="+", default=["dwa"],
                   choices=ALGORITHMS)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="explicit seed list, e.g. --seeds 1 2 3")
    p.add_argument("--start-seed", type=int, default=1)
    p.add_argument("--num-seeds", type=int, default=None,
                   help="run seeds start-seed .. start-seed+num-seeds-1")
    p.add_argument("--out-root", default="results")
    p.add_argument("--max-time", type=float, default=900.0)
    p.add_argument("--flow-min", type=float, default=80.0)
    p.add_argument("--flow-max", type=float, default=350.0)
    p.add_argument("--crossing-flow-min", type=float, default=60.0)
    p.add_argument("--crossing-flow-max", type=float, default=160.0)
    p.add_argument("--veh-scale", type=float, default=1.0)
    p.add_argument("--waypoints", default=None)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="task IDs to cross (or 'all' = every task in each "
                        "map's configs/tasks_<map>.json; maps without a "
                        "task file run their default route once)")
    p.add_argument("--global-planners", nargs="+",
                   choices=["fixed", "dijkstra", "astar", "rrt"],
                   default=["fixed"],
                   help="global-planning factor levels to cross with the "
                        "local planners (--algorithms)")
    p.add_argument("--auto-route", action="store_true",
                   help="forward --auto-route to every run (recommended "
                        "for OSM maps)")
    p.add_argument("--ped-period", type=float, default=None)
    p.add_argument("--ped-period-min", type=float, default=None)
    p.add_argument("--ped-period-max", type=float, default=None)
    p.add_argument("--veh-period", type=float, default=None)
    p.add_argument("--veh-period-min", type=float, default=None)
    p.add_argument("--veh-period-max", type=float, default=None)
    p.add_argument("--routes", "--route", dest="routes", nargs="+",
                   default=["default"],
                   help="one or more named routes; 'default' = the map's "
                        "built-in itinerary. Routes a map doesn't define "
                        "are skipped for that map.")
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--params-file", default=None,
                   help="tuned planner parameters JSON, forwarded to every "
                        "run (use per-algorithm batches: configs/<algo>.json)")
    p.add_argument("--reactive-peds", choices=["off", "sfm"], default=None,
                   help="forwarded to every run")
    p.add_argument("--delay", type=int, default=None,
                   help="sumo-gui delay in ms, forwarded with --gui")
    p.add_argument("--gui", "--sumo-gui", dest="gui", action="store_true",
                   help="open sumo-gui for every run (only sensible for "
                        "small spot-check batches)")
    args = p.parse_args()
    if args.seeds is None:
        n = args.num_seeds if args.num_seeds is not None else 3
        args.seeds = list(range(args.start_seed, args.start_seed + n))

    for mp in args.maps:
        if not (REPO / "maps" / mp).is_dir():
            avail = sorted(d.name for d in (REPO / "maps").iterdir()
                           if d.is_dir())
            sys.exit(f"map '{mp}' not found under maps/; available: {avail}")

    out_root = Path(args.out_root)
    # which named routes each map actually defines
    import json as _json
    map_routes = {}
    for mp in args.maps:
        try:
            spec = _json.loads((REPO / "maps" / mp / "map_spec.json")
                               .read_text())
            map_routes[mp] = set(spec.get("routes", {})) | {"default"}
        except Exception:
            map_routes[mp] = {"default"}
    rows = []
    gps = args.global_planners
    if args.auto_route and gps == ["fixed"]:
        gps = ["dijkstra"]
    map_tasks = {}
    for mp in args.maps:
        tf = REPO / "configs" / f"tasks_{mp}.json"
        if args.tasks and tf.exists():
            ids = [t["id"] for t in
                   _json.loads(tf.read_text())["tasks"]]
            map_tasks[mp] = (ids if "all" in args.tasks
                             else [t for t in args.tasks if t in ids])
        else:
            map_tasks[mp] = [None]
    combos = [(mp, rt, tk, gp, mode, algo, seed)
              for mp, rt, gp, mode, algo, seed
              in itertools.product(args.maps, args.routes, gps, args.modes,
                                   args.algorithms, args.seeds)
              for tk in map_tasks[mp]
              if rt in map_routes[mp]]
    skipped = (len(args.maps) * len(args.routes) * len(gps)
               * len(args.modes) * len(args.algorithms)
               * len(args.seeds)) - len([c for c in combos
                                         if c[2] == map_tasks[c[0]][0]])
    if skipped:
        print(f"(skipping {skipped} combos: route not defined on that map)")
    for i, (mp, rt, tk, gp, mode, algo, seed) in enumerate(combos, 1):
        mlabel = mp if rt == "default" else f"{mp}__{rt}"
        if tk:
            mlabel += f"__{tk}"
        if gp not in ("fixed", "dijkstra"):
            mlabel += f"__g-{gp}"
        run_dir = out_root / mlabel / mode / algo / f"seed_{seed}"
        mfile = run_dir / "robot_metrics.json"
        if args.skip_existing and mfile.exists():
            rows.append(json.loads(mfile.read_text()))
            continue
        rtag = "" if rt == "default" else f" | {rt}"
        ttag = f" | {tk}" if tk else ""
        gtag = "" if gp == "fixed" else f" | g:{gp}"
        print(f"[{i}/{len(combos)}] {mp}{rtag}{ttag}{gtag} | {mode} | "
              f"{algo} | seed {seed}", flush=True)
        cmd = [sys.executable, str(ROOT / "benchmark_runner.py"),
               "--map", mp, "--mode", mode, "--algorithm", algo,
               "--seed", str(seed), "--out-root", str(out_root),
               "--max-time", str(args.max_time),
               "--flow-min", str(args.flow_min), "--flow-max", str(args.flow_max),
               "--crossing-flow-min", str(args.crossing_flow_min),
               "--crossing-flow-max", str(args.crossing_flow_max),
               "--veh-scale", str(args.veh_scale), "--device", args.device]
        if args.waypoints:
            cmd += ["--waypoints", args.waypoints]
        if rt != "default":
            cmd += ["--route", rt]
        if args.reverse:
            cmd += ["--reverse"]
        if args.gui:
            cmd += ["--gui"]
            if args.delay is not None:
                cmd += ["--delay", str(args.delay)]
        if tk:
            cmd += ["--task", tk]
        if gp != "fixed":
            cmd += ["--global-planner", gp]
        if args.params_file:
            pf = (args.params_file.replace("{algo}", algo)
                  .replace("{gp}", gp))
            if not Path(pf).exists():
                sys.exit(f"params file not found for this combo: {pf}")
            cmd += ["--params-file", pf]
            if gp == "rrt":
                sib = Path(pf).with_name(
                    Path(pf).stem + ".globalrrt.json")
                if sib.exists():
                    cmd += ["--global-rrt-params", str(sib)]
        if args.reactive_peds:
            cmd += ["--reactive-peds", args.reactive_peds]
        for flag, val in (("--ped-period", args.ped_period),
                          ("--ped-period-min", args.ped_period_min),
                          ("--ped-period-max", args.ped_period_max),
                          ("--veh-period", args.veh_period),
                          ("--veh-period-min", args.veh_period_min),
                          ("--veh-period-max", args.veh_period_max)):
            if val is not None:
                cmd += [flag, str(val)]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"   !! runner failed ({res.returncode})")
            if args.stop_on_error:
                sys.exit(res.returncode)
            continue
        rows.append(json.loads(mfile.read_text()))

    # aggregate over EVERYTHING under out_root (works incrementally)
    rows = [json.loads(f.read_text())
            for f in out_root.glob("*/*/*/seed_*/robot_metrics.json")]

    # ---------------- aggregate: per-run CSV + per-combo summaries
    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with (out_root / "summary_all.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def agg(sel):
        num = {}
        for k in ("success", "collision", "sim_time_s", "path_length_m",
                  "avg_speed_mps", "min_pedestrian_distance_m",
                  "close_encounter_steps", "time_waiting_at_light_s"):
            vals = [float(r[k]) if not isinstance(r[k], bool) else float(r[k])
                    for r in sel if k in r and r[k] is not None
                    and math.isfinite(float(r[k]))]
            if vals:
                num[f"{k}_mean"] = round(mean(vals), 4)
                num[f"{k}_std"] = round(stdev(vals), 4) if len(vals) > 1 else 0.0
        num["n"] = len(sel)
        return num

    summaries = {}
    keyset = {(r["map"], r.get("route", "default"), r["mode"],
               r["algorithm"]) for r in rows}
    for mp, rt, mode, algo in keyset:
        sel = [r for r in rows
               if (r["map"], r.get("route", "default"), r["mode"],
                   r["algorithm"]) == (mp, rt, mode, algo)]
        tag = mp if rt == "default" else f"{mp}[{rt}]"
        summaries[f"{tag}/{mode}/{algo}"] = agg(sel)
    (out_root / "batch_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True))
    print(f"\nper-run rows -> {out_root/'summary_all.csv'}"
          f"\ncombo summary -> {out_root/'batch_summary.json'}")

    if not args.no_plots and rows:
        subprocess.run([sys.executable,
                        str(REPO / "analysis" / "benchmark_plots.py"),
                        "--results", str(out_root)], check=False)


if __name__ == "__main__":
    main()
