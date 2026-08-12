#!/usr/bin/env python3
"""Run the COMPLETE factorial protocol and prove every cell was produced.

benchmark_batch.py crosses maps x routes x tasks x global planners x modes x
algorithms x seeds, but it takes a single --reactive-peds value and it never
checks afterwards that the tree it produced actually contains every cell of the
design. For a 10,000-run sweep that matters: a cell can go missing because a
runner crashed, because two cells collided on one directory, or because a
resume skipped work it should not have. Silence looks identical to success.

This driver:
  * enumerates the full design, including the reactive-pedestrian factor;
  * runs it through benchmark_batch (one invocation per reactive-peds level,
    which is the only factor the batch does not cross itself);
  * AUDITS the results tree against the enumerated design and reports exactly
    which cells are missing, which are present but failed to produce metrics,
    and which directories exist that the design does not explain;
  * exits non-zero if the tree is incomplete, so a sweep cannot be believed
    finished when it is not.

    # what would the full protocol cost?
    python sim/run_protocol.py --dry-run

    # run it
    python sim/run_protocol.py --jobs 16 --out-root results

    # audit a tree produced earlier, without running anything
    python sim/run_protocol.py --verify-only --out-root results

Run directories come from sim/run_layout.py, the single definition shared with
benchmark_runner and benchmark_batch, so the audit cannot drift out of step
with what the runner writes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import run_layout  # noqa: E402
from benchmark_adapters import ALGORITHMS  # noqa: E402

# The protocol design (README "Full evaluation protocol" + Option B tasks).
DEFAULT_MAPS = ["map1_straight", "map2_crossing", "map3_grid",
                "map4_london", "map5_ucl"]
DEFAULT_GLOBALS = ["dijkstra", "astar", "rrt"]
DEFAULT_MODES = ["mixed"]
DEFAULT_REACTIVE = ["sfm"]


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--maps", nargs="+", default=DEFAULT_MAPS)
    p.add_argument("--routes", nargs="+", default=["default"])
    p.add_argument("--tasks", nargs="+", default=["all"],
                   help="'all' = every task in configs/tasks_<map>.json, "
                        "'none' = the map's default route only")
    p.add_argument("--global-planners", nargs="+", default=DEFAULT_GLOBALS,
                   choices=["fixed", "dijkstra", "astar", "rrt"])
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                   choices=["same", "opposite", "mixed", "static", "all"])
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="default: every registered algorithm")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--reactive-peds", nargs="+", default=DEFAULT_REACTIVE,
                   choices=["off", "sfm", "jupedsim"],
                   help="crossed as a factor; benchmark_batch takes only one "
                        "value, so this driver invokes it once per level")
    p.add_argument("--out-root", default="results")
    p.add_argument("--max-time", type=float, default=3000.0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--params-file", default=None)
    p.add_argument("--minutes-per-run", type=float, default=6.0)
    p.add_argument("--dry-run", action="store_true",
                   help="enumerate the design and print its size; run nothing")
    p.add_argument("--verify-only", action="store_true",
                   help="audit an existing tree against the design")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def map_tasks(map_name: str, requested) -> list:
    """Task ids for a map, or [None] when the map has no task list."""
    if requested == ["none"]:
        return [None]
    tf = REPO / "configs" / f"tasks_{map_name}.json"
    if not tf.exists():
        return [None]
    ids = [t["id"] for t in json.loads(tf.read_text())["tasks"]]
    if requested == ["all"]:
        return ids
    picked = [t for t in requested if t in ids]
    if not picked:
        print(f"  !! none of {requested} are tasks of {map_name} "
              f"(has {ids[:3]}...); using its default route")
        return [None]
    return picked


def map_routes(map_name: str, requested) -> list:
    """Named routes a map actually defines; others are skipped for that map."""
    try:
        spec = json.loads((REPO / "maps" / map_name / "map_spec.json")
                          .read_text())
        have = set(spec.get("routes", {})) | {"default"}
    except Exception:
        have = {"default"}
    return [r for r in requested if r in have]


def enumerate_design(args) -> list:
    """Every cell of the full factorial design."""
    algos = args.algorithms or list(ALGORITHMS)
    cells = []
    for mp in args.maps:
        for rt in map_routes(mp, args.routes):
            for tk in map_tasks(mp, args.tasks):
                for gp, mode, algo, seed, rp in itertools.product(
                        args.global_planners, args.modes, algos,
                        args.seeds, args.reactive_peds):
                    cells.append({"map": mp, "route": rt, "task": tk,
                                  "global_planner": gp, "mode": mode,
                                  "algorithm": algo, "seed": seed,
                                  "reactive_peds": rp})
    return cells


def cell_dir(out_root: Path, c: dict, rp_root: bool = True) -> Path:
    """Where a cell's run lives.

    reactive_peds is not part of run_layout's directory rule (the runner does
    not encode it), so levels of that factor are kept in sibling roots.
    """
    root = out_root / f"peds_{c['reactive_peds']}" if rp_root else out_root
    return run_layout.run_dir(root, c["map"], c["mode"], c["algorithm"],
                              c["seed"], route=c["route"], task=c["task"],
                              global_planner=c["global_planner"])


def audit(args, cells) -> int:
    out_root = Path(args.out_root)
    missing, empty, present = [], [], []
    for c in cells:
        d = cell_dir(out_root, c)
        mf = d / "robot_metrics.json"
        if not mf.exists():
            missing.append(c)
        elif mf.stat().st_size == 0:
            empty.append(c)
        else:
            present.append(c)

    expected_dirs = {cell_dir(out_root, c) for c in cells}
    found_dirs = {p.parent for p in out_root.glob("*/*/*/*/seed_*/robot_metrics.json")}
    found_dirs |= {p.parent for p in out_root.glob("*/*/*/seed_*/robot_metrics.json")}
    unexplained = sorted(found_dirs - expected_dirs)

    n = len(cells)
    print()
    print("=" * 68)
    print(f"COVERAGE AUDIT  ({args.out_root})")
    print("=" * 68)
    print(f"  design cells      {n}")
    print(f"  present           {len(present)}  ({100.0 * len(present) / max(n, 1):.1f}%)")
    print(f"  MISSING           {len(missing)}")
    print(f"  empty metrics     {len(empty)}")
    print(f"  unexplained dirs  {len(unexplained)}")

    if missing:
        by = Counter((c["map"], c["global_planner"], c["algorithm"],
                      c["reactive_peds"]) for c in missing)
        print("\n  missing cells by (map, global, algorithm, peds) "
              "-- top 15:")
        for k, v in by.most_common(15):
            print(f"    {k}  x{v}")
        mp = Path(args.out_root) / "missing_cells.json"
        mp.write_text(json.dumps(missing, indent=2))
        print(f"\n  full list -> {mp}")
    if unexplained:
        print("\n  directories the design does not explain (stale results? "
              "a factor level you forgot to pass?) -- first 10:")
        for d in unexplained[:10]:
            print(f"    {d.relative_to(out_root)}")

    ok = not missing and not empty
    print("\n  RESULT:", "COMPLETE" if ok else "INCOMPLETE")
    print("=" * 68)
    return 0 if ok else 1


def main() -> int:
    args = parse_args()
    cells = enumerate_design(args)
    algos = args.algorithms or list(ALGORITHMS)

    print(f"{'factor':24s}{'levels':>8s}")
    for name, lv in (("maps", len(args.maps)),
                     ("routes (requested)", len(args.routes)),
                     ("tasks / map", len(map_tasks(args.maps[0], args.tasks))),
                     ("global planners", len(args.global_planners)),
                     ("pedestrian modes", len(args.modes)),
                     ("local planners", len(algos)),
                     ("crowd seeds", len(args.seeds)),
                     ("reactive-ped layers", len(args.reactive_peds))):
        print(f"{name:24s}{lv:>8d}")
    cpu_days = len(cells) * args.minutes_per_run / 60 / 24
    print(f"\n  TOTAL RUNS  {len(cells)}")
    print(f"  ~{cpu_days:.1f} CPU-days at {args.minutes_per_run:g} min/run"
          f"  ->  ~{cpu_days / max(args.jobs, 1):.1f} wall-days at "
          f"--jobs {args.jobs}")
    print(f"  algorithms: {', '.join(algos)}")

    if args.dry_run:
        return 0
    if args.verify_only:
        return audit(args, cells)

    # benchmark_batch crosses everything except reactive_peds, so one
    # invocation per level of that factor, into sibling roots.
    for rp in args.reactive_peds:
        root = Path(args.out_root) / f"peds_{rp}"
        root.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "benchmark_batch.py"),
               "--maps", *args.maps,
               "--routes", *args.routes,
               "--global-planners", *args.global_planners,
               "--modes", *args.modes,
               "--algorithms", *algos,
               "--seeds", *[str(s) for s in args.seeds],
               "--reactive-peds", rp,
               "--out-root", str(root),
               "--max-time", str(args.max_time),
               "--jobs", str(args.jobs),
               "--no-plots"]
        if args.tasks != ["none"]:
            cmd += ["--tasks", *args.tasks]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.stop_on_error:
            cmd.append("--stop-on-error")
        if args.params_file:
            cmd += ["--params-file", args.params_file]
        print(f"\n=== reactive-peds {rp} -> {root} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0 and args.stop_on_error:
            print(f"batch failed for reactive-peds={rp} (rc={rc})")
            return rc

    return audit(args, cells)


if __name__ == "__main__":
    raise SystemExit(main())
