#!/usr/bin/env python3
"""A*-controlled delivery robot in the SUMO sidewalk scene.

Features match the edited DWA baseline:
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

import numpy as np

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    in_sidewalk,
    run_traci_with_planner,
)


class AStarPlanner:
    def __init__(self, cfg: PlannerConfig, seed: int):
        self.cfg = cfg
        self.x_res = 1.0
        self.y_res = 0.25
        self.replan_interval = 1.0
        self.last_plan_time = -1e9
        self.path: List[Tuple[float, float]] = []
        self.clearance = cfg.robot_radius + cfg.pedestrian_radius + 0.15

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
        if idx == start_idx or idx == goal_idx:
            return False
        x, y = self.idx_to_xy(idx)
        if not in_sidewalk(x, y, self.cfg, margin=0.03):
            return True
        for obs in obstacles:
            if math.hypot(x - obs.x, y - obs.y) < self.clearance:
                return True
        return False

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
        max_ix = int(round((self.cfg.sidewalk_x_max - self.cfg.sidewalk_x_min) / self.x_res))
        max_iy = int(round((self.cfg.sidewalk_y_max - self.cfg.sidewalk_y_min) / self.y_res))

        def h(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            ax, ay = self.idx_to_xy(a)
            bx, by = self.idx_to_xy(b)
            return math.hypot(ax - bx, ay - by)

        open_heap: List[Tuple[float, Tuple[int, int]]] = [(h(start_idx, goal_idx), start_idx)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_idx: 0.0}
        visited = set()
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal_idx:
                return self.reconstruct(came_from, current)
            for dx, dy in neighbors:
                nb = (current[0] + dx, current[1] + dy)
                if nb[0] < 0 or nb[0] > max_ix or nb[1] < 0 or nb[1] > max_iy:
                    continue
                if self.is_blocked(nb, obstacles, start_idx, goal_idx):
                    continue
                cx, cy = self.idx_to_xy(current)
                nx, ny = self.idx_to_xy(nb)
                step_cost = math.hypot(nx - cx, ny - cy)
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    heapq.heappush(open_heap, (tentative + h(nb, goal_idx), nb))
        return []

    def choose_waypoint(self, state: RobotState) -> Tuple[float, float]:
        if not self.path:
            return (state.x + 1.0, state.y)
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
    p = argparse.ArgumentParser(description="A* robot-as-pedestrian controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(args, lambda cfg, seed: AStarPlanner(cfg, seed), "astar")


if __name__ == "__main__":
    main()
