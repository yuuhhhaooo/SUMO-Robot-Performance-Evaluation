#!/usr/bin/env python3
"""ORCA-style velocity-obstacle delivery robot in the SUMO sidewalk scene.

This is a lightweight TraCI baseline inspired by ORCA / velocity obstacles. It does
not require the external RVO2 library. At every step it samples candidate robot
velocities and chooses the velocity closest to the preferred goal velocity while
avoiding predicted pedestrian collisions.

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
from typing import Sequence, Tuple

import numpy as np

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    in_sidewalk,
    run_traci_with_planner,
)


class ORCAStylePlanner:
    def __init__(self, cfg: PlannerConfig, seed: int):
        self.cfg = cfg
        self.time_horizon = 3.0
        self.collision_clearance = cfg.robot_radius + cfg.pedestrian_radius + 0.05
        self.social_clearance = cfg.social_distance
        self.centerline_gain = 0.06

    def time_to_closest_distance(self, state: RobotState, vx: float, vy: float, obs: Obstacle) -> Tuple[float, float]:
        # Relative position from robot to obstacle, relative velocity obstacle - robot.
        rx = obs.x - state.x
        ry = obs.y - state.y
        rvx = obs.vx - vx
        rvy = obs.vy - vy
        denom = rvx * rvx + rvy * rvy
        if denom < 1e-9:
            t = 0.0
        else:
            t = -((rx * rvx + ry * rvy) / denom)
            t = max(0.0, min(self.time_horizon, t))
        dx = rx + rvx * t
        dy = ry + rvy * t
        return t, math.hypot(dx, dy)

    def candidate_cost(self, state: RobotState, goal: Tuple[float, float], pref_vx: float, pref_vy: float, vx: float, vy: float, obstacles: Sequence[Obstacle]) -> float:
        next_x = state.x + vx * self.cfg.dt
        next_y = state.y + vy * self.cfg.dt
        if not in_sidewalk(next_x, next_y, self.cfg, margin=0.04):
            return float("inf")
        # Prefer velocities close to goal-directed velocity.
        cost = math.hypot(vx - pref_vx, vy - pref_vy)
        cost += self.centerline_gain * abs(next_y - self.cfg.sidewalk_center_y)
        for obs in obstacles:
            t, d = self.time_to_closest_distance(state, vx, vy, obs)
            current_d = math.hypot(obs.x - state.x, obs.y - state.y)
            if current_d < self.cfg.robot_radius + self.cfg.pedestrian_radius:
                return float("inf")
            if d < self.collision_clearance:
                return float("inf")
            # Soft social penalty. Earlier possible conflicts get higher penalty.
            if d < self.social_clearance:
                cost += (self.social_clearance - d) * (self.time_horizon - t + 0.5) * 2.0
            cost += 0.03 / max(d - self.collision_clearance, 0.03)
        return cost

    def compute_command(self, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle], sim_time: float):
        dx, dy = goal[0] - state.x, goal[1] - state.y
        dist_goal = max(math.hypot(dx, dy), 1e-9)
        pref_speed = self.cfg.max_speed
        pref_vx = pref_speed * dx / dist_goal
        pref_vy = pref_speed * dy / dist_goal

        pref_angle = math.atan2(pref_vy, pref_vx)
        speeds = np.linspace(0.0, self.cfg.max_speed, 7)
        angle_offsets = np.radians([-110, -85, -60, -35, -18, 0, 18, 35, 60, 85, 110])

        best = (0.0, 0.0)
        best_cost = float("inf")
        for s in speeds:
            if s < 1e-6:
                candidates = [(0.0, 0.0)]
            else:
                candidates = [(float(s * math.cos(pref_angle + off)), float(s * math.sin(pref_angle + off))) for off in angle_offsets]
            for vx, vy in candidates:
                cost = self.candidate_cost(state, goal, pref_vx, pref_vy, vx, vy, obstacles)
                if cost < best_cost:
                    best_cost = cost
                    best = (vx, vy)
        if math.isinf(best_cost):
            # Emergency fallback: move away from the closest obstacle if possible.
            if obstacles:
                closest = min(obstacles, key=lambda o: math.hypot(o.x - state.x, o.y - state.y))
                ax, ay = state.x - closest.x, state.y - closest.y
                norm = max(math.hypot(ax, ay), 1e-9)
                best = (0.2 * ax / norm, 0.2 * ay / norm)
            else:
                best = (0.0, 0.0)
            return best[0], best[1], {"status": "emergency", "cost": float("inf")}
        return best[0], best[1], {"status": "velocity_selected", "cost": best_cost}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ORCA-style robot-as-pedestrian controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(args, lambda cfg, seed: ORCAStylePlanner(cfg, seed), "orca")


if __name__ == "__main__":
    main()
