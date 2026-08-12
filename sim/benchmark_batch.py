#!/usr/bin/env python3
"""Sweep algorithms x maps x modes x seeds, aggregate metrics, draw plots.

python benchmark_batch.py --maps map2_crossing map3_grid \
    --modes same mixed --algorithms dwa astar orca sarl \
    --seeds 1 2 3 --out-root results

Seeds drive pedestrian density (flows sampled per seed inside the runner).
Add --waypoints/--reverse to test other start/goal routes (forwarded).

--jobs N runs N runner subprocesses at once (default 1 = sequential). Runs
never share a directory or a SUMO port, so this is a pure wall-clock win:
the full protocol is ~44 CPU-days, i.e. ~1.4 wall-days at --jobs 32.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent      # sim/
REPO = ROOT.parent
from benchmark_adapters import ALGORITHMS  # noqa: E402

MAPS = ["map1_straight", "map2_crossing", "map3_grid", "map4_london"]
MODES = ["same", "opposite", "mixed", "static", "all"]

# metrics averaged into every summary cell
AGG_KEYS = ("success", "collision", "sim_time_s", "path_length_m",
            "avg_speed_mps", "min_pedestrian_distance_m",
            "close_encounter_steps", "time_waiting_at_light_s")

# live child processes, so Ctrl-C does not leave a swarm of sumo behind
_CHILDREN: set[subprocess.Popen] = set()
_CHILDREN_LOCK = threading.Lock()


def _spawn(cmd):
    """Run cmd to completion, return its exit code.

    Equivalent to ``subprocess.run(cmd).returncode`` (same argv, cwd, env and
    inherited stdio); written with an explicit Popen only so the handle can be
    registered in _CHILDREN and killed on Ctrl-C.
    """
    with subprocess.Popen(cmd) as proc:
        with _CHILDREN_LOCK:
            _CHILDREN.add(proc)
        try:
            return proc.wait()
        finally:
            with _CHILDREN_LOCK:
                _CHILDREN.discard(proc)


def _kill_children():
    with _CHILDREN_LOCK:
        procs = list(_CHILDREN)
    for proc in procs:
        try:
            proc.kill()
        except Exception:
            pass


def _num(v):
    """Coerce a metrics value to a finite float, or None if it cannot be.

    Metrics files are written by a long simulation and legitimately contain
    None (metric never observed), NaN/Infinity (json allows both) and the odd
    string; none of those may take down the aggregation of a 10k-run sweep.
    """
    if v is None:
        return None
    # NOTE: do NOT short-circuit on str. The pre-change code did float(r[k]),
    # which accepts a numeric string like "0.912"; rejecting those outright
    # would drop real samples and, when every value in a cell is a string,
    # make <k>_mean/<k>_std/<k>_n vanish entirely and KeyError a downstream
    # reader. Let float() decide, and let non-numeric strings fall through.
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="run this many runner subprocesses concurrently "
                        "(default 1 = the old strictly sequential "
                        "behaviour). Runs write to disjoint directories, so "
                        "N jobs cut wall-clock by ~N up to the core count. "
                        "Console lines from different runs interleave.")
    args = p.parse_args()
    if args.jobs < 1:
        sys.exit("--jobs must be >= 1")
    if args.gui and args.jobs > 1:
        sys.exit("--gui with --jobs > 1 would open one sumo-gui window per "
                 "concurrent run and every one of them blocks on a human; "
                 "use --jobs 1 with --gui, or drop --gui to sweep in "
                 "parallel.")
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
    gps = args.global_planners
    if args.auto_route and gps == ["fixed"]:
        gps = ["dijkstra"]
    routes = args.routes
    if args.waypoints and len(routes) > 1:
        # benchmark_runner ignores --route entirely when --waypoints is given
        # (route_name becomes "custom"), so crossing several routes would
        # enqueue byte-identical runs that all target the SAME run_dir. Under
        # --jobs > 1 that is concurrent writers on one directory; even at
        # --jobs 1 it is duplicated work whose last writer wins.
        print(f"(--waypoints overrides --routes: collapsing "
              f"{routes} to a single 'custom' route)")
        routes = routes[:1]
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
              in itertools.product(args.maps, routes, gps, args.modes,
                                   args.algorithms, args.seeds)
              for tk in map_tasks[mp]
              if rt in map_routes[mp]]
    skipped = (len(args.maps) * len(routes) * len(gps)
               * len(args.modes) * len(args.algorithms)
               * len(args.seeds)) - len([c for c in combos
                                         if c[2] == map_tasks[c[0]][0]])
    if skipped:
        print(f"(skipping {skipped} combos: route not defined on that map)")

    def _run_dir(mp, rt, tk, gp, mode, algo, seed):
        # must mirror benchmark_runner.py's run_dir layout EXACTLY, otherwise
        # the metrics file is looked up at a path the runner never wrote
        rlabel = "custom" if args.waypoints else rt
        mlabel = mp if rlabel == "default" else f"{mp}__{rlabel}"
        if tk:
            mlabel += f"__{tk}"
        if gp != "fixed":
            mlabel += f"__g-{gp}"
        return out_root / mlabel / mode / algo / f"seed_{seed}"

    def _label(mp, rt, tk, gp, mode, algo, seed):
        rtag = "" if rt == "default" else f" | {rt}"
        ttag = f" | {tk}" if tk else ""
        gtag = "" if gp == "fixed" else f" | g:{gp}"
        return (f"{mp}{rtag}{ttag}{gtag} | {mode} | {algo} | seed {seed}")

    def _cmd(mp, rt, tk, gp, mode, algo, seed):
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
        return cmd

    # Decide the work list up front, in the main thread and in combo order:
    # --skip-existing then reads a stable snapshot of what is already on disk
    # (every combo owns a disjoint run_dir, so no run can create another
    # combo's metrics file -- the decision cannot race the workers).
    # Building the commands here also means a bad --params-file aborts before
    # the first run instead of thousands of runs in.
    ntotal = len(combos)
    work = []
    for i, combo in enumerate(combos, 1):
        mfile = _run_dir(*combo) / "robot_metrics.json"
        if args.skip_existing and mfile.exists():
            continue
        work.append((i, _label(*combo), _cmd(*combo), mfile))

    stop = threading.Event()

    def _do_run(i, label, cmd, mfile):
        """Return 0 on success, a non-zero exit code on failure, None if the
        run was skipped because a previous failure stopped the sweep."""
        if stop.is_set():
            return None
        # self-describing: with --jobs > 1 these lines interleave
        print(f"[{i}/{ntotal}] {label}", flush=True)
        rc = _spawn(cmd)
        if rc != 0:
            print(f"[{i}/{ntotal}] !! runner failed ({rc}): {label}",
                  flush=True)
            return _fail(rc or 1)
        # a runner can exit 0 without writing metrics (early-return paths);
        # do not let that kill a multi-thousand-run sweep
        if not mfile.exists():
            print(f"[{i}/{ntotal}] !! runner exited 0 but wrote no metrics: "
                  f"{mfile}", flush=True)
            return _fail(1)
        return 0

    def _fail(rc):
        # raise the stop flag HERE, not back in the main thread: a worker that
        # has just finished a failing run would otherwise pick up the next
        # queued run before the main thread woke up and cancelled it (at
        # --jobs 1 that alone would launch one run more than the old
        # sequential loop did)
        if args.stop_on_error:
            stop.set()
        return rc

    fail_code = 0
    if work:
        ex = ThreadPoolExecutor(max_workers=args.jobs)
        futs = [ex.submit(_do_run, *w) for w in work]
        try:
            # with --jobs 1 the pool runs one task at a time in submission
            # order, so completion order == the old sequential order
            for fut in as_completed(futs):
                rc = fut.result()
                if rc:
                    fail_code = fail_code or rc
                    if args.stop_on_error:
                        stop.set()          # workers not yet started bail out
                        for f in futs:
                            f.cancel()
                        break
        except KeyboardInterrupt:
            stop.set()
            for f in futs:
                f.cancel()
            _kill_children()
            ex.shutdown(wait=True, cancel_futures=True)
            print("\ninterrupted -- cancelled pending runs", flush=True)
            sys.exit(130)
        finally:
            # waits for the runs still in flight: no orphaned children
            ex.shutdown(wait=True, cancel_futures=True)
        if fail_code and args.stop_on_error:
            sys.exit(fail_code)

    # aggregate over EVERYTHING under out_root (works incrementally)
    rows = ([json.loads(f.read_text())
             for f in out_root.glob("*/*/*/seed_*/robot_metrics.json")]
            if out_root.is_dir() else [])

    if not rows and not out_root.is_dir():
        # nothing ran and nothing was ever written there: creating the
        # directory just to drop an empty summary in it is noise, and the
        # old code died with FileNotFoundError instead
        print(f"\nno runs and no results under {out_root} -- "
              f"nothing to aggregate", flush=True)
        return          # exit 0, exactly as the old code did when the
        #                 directory happened to exist and held no rows

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
        for k in AGG_KEYS:
            # _num() drops None / NaN / Infinity / non-numeric strings instead
            # of raising, so one malformed metrics file cannot sink the whole
            # summary; "<k>_n" says how many runs actually fed each mean
            vals = [v for v in (_num(r.get(k)) for r in sel) if v is not None]
            if vals:
                num[f"{k}_mean"] = round(mean(vals), 4)
                num[f"{k}_std"] = round(stdev(vals), 4) if len(vals) > 1 else 0.0
                num[f"{k}_n"] = len(vals)
        num["n"] = len(sel)          # runs in the cell (unchanged meaning)
        return num

    # The summary key MUST include every manipulated factor. It previously
    # keyed on (map, route, mode, algorithm) only, so runs that differ in
    # TASK or GLOBAL PLANNER were averaged into one cell -- e.g. every
    # task of map4_london under g-rrt and g-astar collapsed into a single
    # "map4_london/mixed/dwa" mean.
    # Components are NORMALISED so that values meaning "absent" collapse to one
    # representation before they are keyed. Without this, ("fixed", None) and
    # (None, None) and ("fixed", "") are distinct cells that all render to the
    # same summary string, and whichever is written last silently wins --
    # discarding a whole cell rather than merging it.
    def _cell(r):
        return (r["map"], r.get("route") or "default", r.get("task") or None,
                r.get("global_planner") or "fixed", r["mode"], r["algorithm"])

    # single pass: the old code re-filtered every row for every cell (O(cells
    # x runs) -- ~10.5k rows x ~1k cells on the full protocol)
    cells = defaultdict(list)
    for r in rows:
        cells[_cell(r)].append(r)

    summaries = {}
    _seen_tags = {}
    for key, sel in cells.items():
        mp, rt, tk, gp_, mode, algo = key
        tag = mp if rt == "default" else f"{mp}[{rt}]"
        if tk:
            tag += f"[{tk}]"
        if gp_ != "fixed":
            tag += f"[g:{gp_}]"
        name = f"{tag}/{mode}/{algo}"
        # the rendered name omits defaults, so two distinct cells could in
        # principle collide (e.g. a map literally named "a[b]"). Never let one
        # silently overwrite the other.
        if name in _seen_tags and _seen_tags[name] != key:
            print(f"   !! summary key collision on '{name}': "
                  f"{_seen_tags[name]} vs {key}; disambiguating")
            name = f"{name}#{abs(hash(key)) % 10000:04d}"
        _seen_tags[name] = key
        summaries[name] = agg(sel)
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
