#!/usr/bin/env python3
"""THE canonical results-tree layout. Import this; never re-derive it.

Every run is written to

    <out_root>/<map_label>/<mode>/<algorithm>/seed_<seed>/

and `map_label` encodes the map plus the non-default levels of the route, task
and global-planner factors. That rule used to be copy-pasted into
benchmark_runner (twice) and benchmark_batch (twice). The copies drifted: for a
while the runner's success path excluded BOTH 'fixed' and 'dijkstra' from the
`__g-` suffix while every other site excluded only 'fixed', so on a sweep
crossing both global levels the dijkstra run silently overwrote the fixed run,
the batch reported "runner exited 0 but wrote no metrics" for every dijkstra
cell, and --skip-existing never matched one. On the full 10,500-run design that
is 3,500 lost runs and 3,500 false errors.

One definition, imported everywhere, so the copies cannot drift again.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Factor levels that are the protocol default and therefore NOT encoded in the
# directory name. Everything else gets an explicit tag, so a directory name is
# a faithful record of the cell that produced it.
DEFAULT_ROUTE = "default"
DEFAULT_GLOBAL_PLANNER = "fixed"


def map_label(map_name: str, route: str = DEFAULT_ROUTE,
              task: Optional[str] = None,
              global_planner: str = DEFAULT_GLOBAL_PLANNER) -> str:
    """Directory-name component encoding (map, route, task, global planner)."""
    label = map_name if route == DEFAULT_ROUTE else f"{map_name}__{route}"
    if task:
        label += f"__{task}"
    if global_planner != DEFAULT_GLOBAL_PLANNER:
        label += f"__g-{global_planner}"
    return label


def run_dir(out_root, map_name: str, mode: str, algorithm: str, seed: int,
            route: str = DEFAULT_ROUTE, task: Optional[str] = None,
            global_planner: str = DEFAULT_GLOBAL_PLANNER) -> Path:
    """Absolute run directory for one experimental cell."""
    return (Path(out_root)
            / map_label(map_name, route, task, global_planner)
            / mode / algorithm / f"seed_{seed}")


def route_label(route: str, waypoints: Optional[str]) -> str:
    """The route name the RUNNER will use.

    --waypoints overrides --route entirely (benchmark_runner sets route_name to
    "custom"), so a batch crossing several routes alongside --waypoints would
    enqueue byte-identical runs that all target one directory.
    """
    return "custom" if waypoints else route
