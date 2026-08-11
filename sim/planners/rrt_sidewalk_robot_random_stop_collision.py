#!/usr/bin/env python3
"""RRT-controlled delivery robot in the SUMO sidewalk scene.

Features match the edited DWA baseline:
- random pedestrian scenario generation with --seed
- robot controlled as a SUMO person
- robot constrained to the north sidewalk
- run stops immediately when robot collides with a pedestrian
- trace CSV, metrics JSON and route plot are written per seed
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    in_sidewalk,
    run_traci_with_planner,
)


@dataclass
class RRTNode:
    x: float
    y: float
    parent: int


class RRTPlanner:
    def __init__(self, cfg: PlannerConfig, seed: int):
        self.cfg = cfg
        self.rng = random.Random(seed + 991)
        self.step_size = 3.0
        self.max_iter = 700
        self.goal_sample_rate = 0.18
        self.goal_connect_dist = 4.0
        self.replan_interval = 1.0
        self.last_plan_time = -1e9
        self.path: List[Tuple[float, float]] = []
        self.clearance = cfg.robot_radius + cfg.pedestrian_radius + 0.12

    def point_safe(self, x: float, y: float, obstacles: Sequence[Obstacle]) -> bool:
        if not in_sidewalk(x, y, self.cfg, margin=0.04):
            return False
        for obs in obstacles:
            if math.hypot(x - obs.x, y - obs.y) < self.clearance:
                return False
        return True

    def segment_safe(self, a: Tuple[float, float], b: Tuple[float, float], obstacles: Sequence[Obstacle]) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(dist / 0.25))
        for i in range(steps + 1):
            r = i / steps
            x = a[0] + r * (b[0] - a[0])
            y = a[1] + r * (b[1] - a[1])
            if not self.point_safe(x, y, obstacles):
                return False
        return True

    def sample(self, state: RobotState, goal: Tuple[float, float]) -> Tuple[float, float]:
        if self.rng.random() < self.goal_sample_rate:
            return goal
        # Bias samples toward the region in front of the robot because the task is start->goal along the sidewalk.
        x_min = max(self.cfg.sidewalk_x_min, state.x - 8.0)
        x = self.rng.uniform(x_min, self.cfg.sidewalk_x_max)
        y = self.rng.uniform(self.cfg.sidewalk_y_min + 0.05, self.cfg.sidewalk_y_max - 0.05)
        return x, y

    def nearest_index(self, nodes: List[RRTNode], sample: Tuple[float, float]) -> int:
        best_i = 0
        best_d = float("inf")
        for i, n in enumerate(nodes):
            d = (n.x - sample[0]) ** 2 + (n.y - sample[1]) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def steer(self, from_node: RRTNode, to_xy: Tuple[float, float]) -> Tuple[float, float]:
        dx, dy = to_xy[0] - from_node.x, to_xy[1] - from_node.y
        dist = math.hypot(dx, dy)
        if dist <= self.step_size:
            return to_xy
        return from_node.x + self.step_size * dx / dist, from_node.y + self.step_size * dy / dist

    def build_path(self, nodes: List[RRTNode], idx: int, goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        path = [goal]
        while idx >= 0:
            n = nodes[idx]
            path.append((n.x, n.y))
            idx = n.parent
        path.reverse()
        return self.shortcut(path)

    def shortcut(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        # Keep this conservative because dynamic obstacle checks happen in the next replanning step.
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = min(len(path) - 1, i + 5)
            out.append(path[j])
            i = j
        return out

    def plan(self, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle]) -> List[Tuple[float, float]]:
        start = (state.x, state.y)
        if self.segment_safe(start, goal, obstacles):
            return [start, goal]
        nodes = [RRTNode(state.x, state.y, -1)]
        for _ in range(self.max_iter):
            sample = self.sample(state, goal)
            nearest_i = self.nearest_index(nodes, sample)
            new_xy = self.steer(nodes[nearest_i], sample)
            if not self.segment_safe((nodes[nearest_i].x, nodes[nearest_i].y), new_xy, obstacles):
                continue
            nodes.append(RRTNode(new_xy[0], new_xy[1], nearest_i))
            new_i = len(nodes) - 1
            if math.hypot(new_xy[0] - goal[0], new_xy[1] - goal[1]) <= self.goal_connect_dist:
                if self.segment_safe(new_xy, goal, obstacles):
                    return self.build_path(nodes, new_i, goal)
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
            self.path = self.plan(state, goal, obstacles)
            self.last_plan_time = sim_time
        if not self.path:
            # Fallback: creep forward along the centreline.
            target = (min(goal[0], state.x + 2.0), self.cfg.sidewalk_center_y)
            status = "no_path"
        else:
            target = self.choose_waypoint(state)
            status = "path_found"
        dx, dy = target[0] - state.x, target[1] - state.y
        dist = max(math.hypot(dx, dy), 1e-9)
        speed = self.cfg.max_speed
        return speed * dx / dist, speed * dy / dist, {"status": status, "cost": len(self.path)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RRT robot-as-pedestrian controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(args, lambda cfg, seed: RRTPlanner(cfg, seed), "rrt")


if __name__ == "__main__":
    main()
