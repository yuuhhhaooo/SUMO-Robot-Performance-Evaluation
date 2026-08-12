#!/usr/bin/env python3
"""Factor table + run-count calculator (Wednesday deliverable).

python analysis/factor_table.py --maps map1_straight map2_crossing \
    map3_grid map4_london map5_ucl --tasks 10 --seeds 10 \
    --globals 3 --locals 7 --modes 1 --minutes-per-run 6 --jobs 32

--jobs is the value you intend to pass to sim/benchmark_batch.py --jobs; the
wall-clock line is a projection for that setting, not a measurement.
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--maps", nargs="+", required=True)
ap.add_argument("--tasks", type=int, default=10)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--globals", dest="glob", type=int, default=3)
ap.add_argument("--locals", dest="loc", type=int, default=7)
ap.add_argument("--modes", type=int, default=1)
ap.add_argument("--minutes-per-run", type=float, default=6.0)
ap.add_argument("--jobs", "--workers", dest="jobs", type=int, default=32,
                help="the --jobs value you will hand to "
                     "sim/benchmark_batch.py; the wall-clock line below is a "
                     "projection for exactly that setting (default 32)")
a = ap.parse_args()
if a.jobs < 1:
    sys.exit("--jobs must be >= 1")

per_map = a.tasks * a.seeds * a.glob * a.loc * a.modes
total = per_map * len(a.maps)
cpu_days = total * a.minutes_per_run / 60 / 24
wall_days = cpu_days / a.jobs

print(f"{'factor':28s}{'levels':>8s}")
for name, lv in (("maps", len(a.maps)), ("tasks / map", a.tasks),
                 ("crowd seeds / cell", a.seeds),
                 ("global planners", a.glob), ("local planners", a.loc),
                 ("pedestrian modes", a.modes)):
    print(f"{name:28s}{lv:>8d}")
print("-" * 36)
print(f"{'runs per map':28s}{per_map:>8d}")
print(f"{'TOTAL runs':28s}{total:>8d}")
print(f"est. {a.minutes_per_run} min/run -> {cpu_days:.1f} CPU-days")
if a.jobs == 1:
    print(f"projected {wall_days:.1f} wall-days sequentially "
          f"(benchmark_batch.py default --jobs 1)")
else:
    print(f"projected ~{wall_days:.1f} wall-days IF the sweep is launched as "
          f"`python sim/benchmark_batch.py --jobs {a.jobs} ...` on a machine "
          f"with >= {a.jobs} free cores -- ideal scaling, no I/O contention")
print("trim order if short (supervisor): pedestrian modes / density "
      "levels first; NEVER tasks or seeds.")
