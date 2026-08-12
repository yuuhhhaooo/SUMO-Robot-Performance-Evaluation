#!/usr/bin/env python3
"""Dijkstra-controlled delivery robot in the SUMO sidewalk scene.

This baseline uses a grid over the north sidewalk and runs Dijkstra search on the
current local obstacle map. It is similar to the A* baseline, but the priority is
only accumulated path cost g(n), without the A* heuristic h(n).

Features match the other baseline scripts:
- random pedestrian scenario generation with --seed
- robot controlled as a SUMO person
- robot constrained to the north sidewalk
- run stops immediately when robot collides with a pedestrian
- trace CSV, metrics JSON and route plot are written per seed
"""
from __future__ import annotations

import argparse
import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    in_sidewalk,
    run_traci_with_planner,
)


class DijkstraPlanner:
    def __init__(self, cfg: PlannerConfig, seed: int):
        self.cfg = cfg
        # Same grid scale as the A* baseline, so the comparison is mainly the search strategy.
        self.x_res = 1.0
        self.y_res = 0.25
        self.replan_interval = 1.0
        self.last_plan_time = -1e9
        self.path: List[Tuple[float, float]] = []
        self.clearance = cfg.robot_radius + cfg.pedestrian_radius + 0.15
        # Cached grid geometry (cell centre coordinates + the in-sidewalk band), rebuilt
        # only if the config/resolution changes. Pure caching: no behaviour change.
        self._geom_key: Optional[Tuple[float, ...]] = None
        self._geom: Optional[Tuple] = None
        # Signature of the last search that returned no path, so an identical repeated
        # query is not re-searched. plan() is a pure function of this signature.
        self._failed_signature: Optional[Tuple] = None

    def xy_to_idx(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(round((x - self.cfg.sidewalk_x_min) / self.x_res))
        iy = int(round((y - self.cfg.sidewalk_y_min) / self.y_res))
        ix = max(0, min(ix, int(round((self.cfg.sidewalk_x_max - self.cfg.sidewalk_x_min) / self.x_res))))
        iy = max(0, min(iy, int(round((self.cfg.sidewalk_y_max - self.cfg.sidewalk_y_min) / self.y_res))))
        return ix, iy

    def idx_to_xy(self, idx: Tuple[int, int]) -> Tuple[float, float]:
        return (
            self.cfg.sidewalk_x_min + idx[0] * self.x_res,
            self.cfg.sidewalk_y_min + idx[1] * self.y_res,
        )

    def is_blocked(self, idx: Tuple[int, int], obstacles: Sequence[Obstacle], start_idx: Tuple[int, int], goal_idx: Tuple[int, int]) -> bool:
        """Reference occupancy predicate (unchanged).

        plan() no longer calls this per edge; it stamps the same predicate onto the grid
        once per call (see _geometry/_blocked_cells) and does a set lookup instead. This
        method is kept as the authoritative definition the cached grid is tested against.
        """
        if idx == start_idx or idx == goal_idx:
            return False
        x, y = self.idx_to_xy(idx)
        if not in_sidewalk(x, y, self.cfg, margin=0.03):
            return True
        for obs in obstacles:
            if math.hypot(x - obs.x, y - obs.y) < self.clearance:
                return True
        return False

    def _geometry(self) -> Tuple:
        """Cell coordinates and the in-sidewalk band, cached per grid geometry.

        Returns (xs, ys, max_ix, max_iy, bx_lo, bx_hi, by_lo, by_hi, outside).
        xs[ix]/ys[iy] are computed with the exact expression used by idx_to_xy(), so
        every downstream distance is bit-identical to the original. The band bounds
        reproduce in_sidewalk() exactly: the predicate is evaluated on every cell once,
        and if the accepted cells form a rectangle the bounds alone are sufficient;
        otherwise the rejected cells are returned in `outside` and folded into the
        blocked set instead.
        """
        cfg = self.cfg
        key = (
            cfg.sidewalk_x_min,
            cfg.sidewalk_x_max,
            cfg.sidewalk_y_min,
            cfg.sidewalk_y_max,
            self.x_res,
            self.y_res,
        )
        if self._geom_key == key and self._geom is not None:
            return self._geom

        max_ix = int(round((cfg.sidewalk_x_max - cfg.sidewalk_x_min) / self.x_res))
        max_iy = int(round((cfg.sidewalk_y_max - cfg.sidewalk_y_min) / self.y_res))
        xs = [cfg.sidewalk_x_min + ix * self.x_res for ix in range(max_ix + 1)]
        ys = [cfg.sidewalk_y_min + iy * self.y_res for iy in range(max_iy + 1)]

        inside = [
            (ix, iy)
            for ix in range(max_ix + 1)
            for iy in range(max_iy + 1)
            if in_sidewalk(xs[ix], ys[iy], cfg, margin=0.03)
        ]
        if inside:
            bx_lo = min(c[0] for c in inside)
            bx_hi = max(c[0] for c in inside)
            by_lo = min(c[1] for c in inside)
            by_hi = max(c[1] for c in inside)
            rectangular = len(inside) == (bx_hi - bx_lo + 1) * (by_hi - by_lo + 1)
        else:
            # No cell passes in_sidewalk(): an empty range rejects everything.
            bx_lo, bx_hi, by_lo, by_hi = 1, 0, 1, 0
            rectangular = True

        if rectangular:
            outside: frozenset = frozenset()
        else:
            bx_lo, bx_hi, by_lo, by_hi = 0, max_ix, 0, max_iy
            inside_set = set(inside)
            outside = frozenset(
                (ix, iy)
                for ix in range(max_ix + 1)
                for iy in range(max_iy + 1)
                if (ix, iy) not in inside_set
            )

        self._geom = (xs, ys, max_ix, max_iy, bx_lo, bx_hi, by_lo, by_hi, outside)
        self._geom_key = key
        return self._geom

    def _blocked_cells(
        self,
        obstacles: Sequence[Obstacle],
        xs: List[float],
        ys: List[float],
        max_ix: int,
        max_iy: int,
        outside: frozenset,
    ) -> set:
        """Stamp every obstacle clearance disc onto the grid once.

        Only the cells inside the disc's bounding box are visited (generously padded by
        one cell), and each candidate is accepted with the *same* comparison the original
        is_blocked() used - math.hypot(x - obs.x, y - obs.y) < self.clearance - so the
        resulting set matches the old predicate cell for cell.
        """
        blocked = set(outside)
        clearance = self.clearance
        if clearance <= 0.0:
            return blocked
        x_min = self.cfg.sidewalk_x_min
        y_min = self.cfg.sidewalk_y_min
        x_res = self.x_res
        y_res = self.y_res
        hypot = math.hypot
        floor = math.floor
        ceil = math.ceil
        for obs in obstacles:
            ox = obs.x
            oy = obs.y
            if not (math.isfinite(ox) and math.isfinite(oy)):
                # hypot() would be nan/inf, and "nan < c" / "inf < c" are both False:
                # the original predicate never blocked on such an obstacle either.
                continue
            ix_lo = int(floor((ox - clearance - x_min) / x_res)) - 1
            ix_hi = int(ceil((ox + clearance - x_min) / x_res)) + 1
            iy_lo = int(floor((oy - clearance - y_min) / y_res)) - 1
            iy_hi = int(ceil((oy + clearance - y_min) / y_res)) + 1
            if ix_lo < 0:
                ix_lo = 0
            if iy_lo < 0:
                iy_lo = 0
            if ix_hi > max_ix:
                ix_hi = max_ix
            if iy_hi > max_iy:
                iy_hi = max_iy
            for ix in range(ix_lo, ix_hi + 1):
                dx = xs[ix] - ox
                for iy in range(iy_lo, iy_hi + 1):
                    if hypot(dx, ys[iy] - oy) < clearance:
                        blocked.add((ix, iy))
        return blocked

    def reconstruct(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[float, float]]:
        path_idx = [current]
        while current in came_from:
            current = came_from[current]
            path_idx.append(current)
        path_idx.reverse()
        return [self.idx_to_xy(idx) for idx in path_idx]

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], obstacles: Sequence[Obstacle]) -> List[Tuple[float, float]]:
        start_idx = self.xy_to_idx(*start)
        goal_idx = self.xy_to_idx(*goal)

        xs, ys, max_ix, max_iy, bx_lo, bx_hi, by_lo, by_hi, outside = self._geometry()

        # plan() is a pure function of (start_idx, goal_idx, clearance, geometry,
        # obstacle positions). If the previous search with exactly these inputs found no
        # path, the answer is still no path - skip the (expensive, whole-grid) re-search.
        signature = (
            start_idx,
            goal_idx,
            self.clearance,
            self._geom_key,
            tuple((obs.x, obs.y) for obs in obstacles),
        )
        if self._failed_signature is not None and signature == self._failed_signature:
            return []

        blocked = self._blocked_cells(obstacles, xs, ys, max_ix, max_iy, outside)

        hypot = math.hypot

        # Dijkstra: priority is accumulated path length only. No heuristic term is added.
        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start_idx)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_idx: 0.0}
        visited = set()
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        inf = float("inf")
        heappush = heapq.heappush
        heappop = heapq.heappop

        while open_heap:
            current_cost, current = heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal_idx:
                return self.reconstruct(came_from, current)
            cix, ciy = current
            cx = xs[cix]
            cy = ys[ciy]
            for dx, dy in neighbors:
                nix = cix + dx
                niy = ciy + dy
                if nix < 0 or nix > max_ix or niy < 0 or niy > max_iy:
                    continue
                nb = (nix, niy)
                if nb != start_idx and nb != goal_idx:
                    # in_sidewalk() band test, then the stamped clearance discs.
                    if nix < bx_lo or nix > bx_hi or niy < by_lo or niy > by_hi:
                        continue
                    if nb in blocked:
                        continue
                nx = xs[nix]
                ny = ys[niy]
                step_cost = hypot(nx - cx, ny - cy)
                tentative = current_cost + step_cost
                if tentative < g_score.get(nb, inf):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    heappush(open_heap, (tentative, nb))
        self._failed_signature = signature
        return []

    def choose_waypoint(self, state: RobotState) -> Tuple[float, float]:
        if not self.path:
            return (min(self.cfg.sidewalk_x_max, state.x + 1.0), state.y)
        lookahead = 2.0
        for x, y in self.path[1:]:
            if math.hypot(x - state.x, y - state.y) >= lookahead:
                return x, y
        return self.path[-1]

    def compute_command(self, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle], sim_time: float):
        if (sim_time - self.last_plan_time >= self.replan_interval) or not self.path:
            self.path = self.plan((state.x, state.y), goal, obstacles)
            self.last_plan_time = sim_time
        if not self.path:
            # Fallback: move slowly toward the goal centreline if no grid path is found.
            dx, dy = goal[0] - state.x, goal[1] - state.y
            dist = max(math.hypot(dx, dy), 1e-9)
            return 0.3 * dx / dist, 0.3 * dy / dist, {"status": "no_path", "cost": float("inf")}
        wx, wy = self.choose_waypoint(state)
        dx, dy = wx - state.x, wy - state.y
        dist = max(math.hypot(dx, dy), 1e-9)
        speed = self.cfg.max_speed
        return speed * dx / dist, speed * dy / dist, {"status": "path_found", "cost": len(self.path)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dijkstra robot-as-pedestrian controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(args, lambda cfg, seed: DijkstraPlanner(cfg, seed), "dijkstra")


if __name__ == "__main__":
    main()
