#!/usr/bin/env python3
"""Factor table + run-count calculator (Wednesday deliverable).

python analysis/factor_table.py --maps map1_straight map2_crossing \
    map3_grid map4_london map5_ucl --tasks 10 --seeds 10 \
    --globals 3 --locals 7 --modes 1 --minutes-per-run 6 --workers 32
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--maps", nargs="+", required=True)
ap.add_argument("--tasks", type=int, default=10)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--globals", dest="glob", type=int, default=3)
ap.add_argument("--locals", dest="loc", type=int, default=7)
ap.add_argument("--modes", type=int, default=1)
ap.add_argument("--minutes-per-run", type=float, default=6.0)
ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()

per_map = a.tasks * a.seeds * a.glob * a.loc * a.modes
total = per_map * len(a.maps)
cpu_days = total * a.minutes_per_run / 60 / 24
wall_days = cpu_days / a.workers

print(f"{'factor':28s}{'levels':>8s}")
for name, lv in (("maps", len(a.maps)), ("tasks / map", a.tasks),
                 ("crowd seeds / cell", a.seeds),
                 ("global planners", a.glob), ("local planners", a.loc),
                 ("pedestrian modes", a.modes)):
    print(f"{name:28s}{lv:>8d}")
print("-" * 36)
print(f"{'runs per map':28s}{per_map:>8d}")
print(f"{'TOTAL runs':28s}{total:>8d}")
print(f"est. {a.minutes_per_run} min/run -> {cpu_days:.1f} CPU-days; "
      f"~{wall_days:.1f} wall-days at {a.workers} parallel workers")
print("trim order if short (supervisor): pedestrian modes / density "
      "levels first; NEVER tasks or seeds.")
