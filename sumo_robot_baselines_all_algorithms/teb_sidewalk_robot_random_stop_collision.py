#!/usr/bin/env python3
"""Simplified TEB-controlled delivery robot in the SUMO sidewalk scene.

This file implements a lightweight Python version inspired by Timed Elastic Band
(TEB) local planning. The full ROS teb_local_planner optimizes a timed trajectory
with kinodynamic constraints. For this SUMO sidewalk baseline, the x coordinates
of a short forward band are fixed and the optimizer deforms the y coordinates of
that timed band to avoid pedestrians while keeping the trajectory smooth.

Features match the other baselines:
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
from scipy.optimize import minimize

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    run_traci_with_planner,
)


class SimplifiedTEBPlanner:
    """Timed elastic band baseline specialized for a straight sidewalk.

    The band has N timed poses over a short horizon. Since the task is along a
    straight sidewalk from west to east, x positions are advanced forward and fixed;
    y positions are optimized. This keeps the implementation stable and fast while
    still capturing the TEB idea of deforming a timed local trajectory around obstacles.
    """

    def __init__(self, cfg: PlannerConfig, seed: int):
        self.cfg = cfg
        self.horizon = 10
        self.band_dt = cfg.dt
        self.lookahead_distance = 10.0
        self.margin = 0.06
        self.clearance = max(cfg.safe_distance, cfg.robot_radius + cfg.pedestrian_radius + 0.05)
        self.social_clearance = max(cfg.social_distance, self.clearance + 0.15)
        self.last_y: np.ndarray | None = None
        self.last_target: Tuple[float, float] | None = None

        # TEB-like objective weights.
        self.w_goal_y = 0.80
        self.w_centerline = 0.35
        self.w_obstacle = 2.0
        self.w_social = 0.45
        self.w_smooth = 8.0
        self.w_accel = 4.0
        self.w_short_time = 0.15
        self.w_lateral_speed = 2.50

    def _make_x_band(self, state: RobotState, goal: Tuple[float, float]) -> np.ndarray:
        remaining = max(goal[0] - state.x, 0.0)
        # Keep the band local but always move forward. The robot applies only the first step.
        forward_span = min(self.lookahead_distance, max(remaining, self.cfg.max_speed * self.cfg.dt))
        return np.linspace(state.x, min(goal[0], state.x + forward_span), self.horizon + 1)[1:]

    def _initial_y(self, state: RobotState) -> np.ndarray:
        if self.last_y is not None and len(self.last_y) == self.horizon:
            # Shift the previous band forward, append centerline as a mild prior.
            return np.r_[self.last_y[1:], self.cfg.sidewalk_center_y]
        return np.full(self.horizon, state.y, dtype=float)

    def _objective(self, y_values: np.ndarray, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle], x_band: np.ndarray) -> float:
        cfg = self.cfg
        y = np.asarray(y_values, dtype=float)
        total = 0.0
        prev_x, prev_y = state.x, state.y
        prev_dy = 0.0

        for k in range(self.horizon):
            xk = float(x_band[k])
            yk = float(y[k])
            t = (k + 1) * self.band_dt

            # Goal direction and centerline preference. Goal y is used weakly so the band
            # can move away from the center to pass pedestrians.
            total += self.w_goal_y * (yk - goal[1]) ** 2
            total += self.w_centerline * (yk - cfg.sidewalk_center_y) ** 2

            # Smoothness / elastic band deformation costs.
            dy = yk - prev_y
            dx = max(xk - prev_x, 1e-6)
            lateral_slope = dy / dx
            total += self.w_lateral_speed * lateral_slope * lateral_slope
            if k > 0:
                ddy = dy - prev_dy
                total += self.w_smooth * ddy * ddy
                total += self.w_accel * (ddy / max(self.band_dt, 1e-9)) ** 2
            prev_x, prev_y = xk, yk
            prev_dy = dy

            # Dynamic pedestrians with constant velocity prediction.
            for obs in obstacles:
                ox = obs.x + obs.vx * t
                oy = obs.y + obs.vy * t
                d = math.hypot(xk - ox, yk - oy)
                if d < self.clearance:
                    total += self.w_obstacle * (self.clearance - d + 1e-3) ** 2 * 100.0
                elif d < self.social_clearance:
                    total += self.w_social * (self.social_clearance - d) ** 2
                total += 0.012 / max(d - self.clearance + 0.05, 0.05)

        # Small terminal cost to pull the local band forward toward the global goal.
        total += 0.05 * ((goal[0] - x_band[-1]) ** 2 + (goal[1] - y[-1]) ** 2)
        total += self.w_short_time * self.horizon * self.band_dt
        return float(total)

    def compute_command(self, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle], sim_time: float):
        cfg = self.cfg
        x_band = self._make_x_band(state, goal)
        y0 = self._initial_y(state)
        bounds = [(cfg.sidewalk_y_min + self.margin, cfg.sidewalk_y_max - self.margin) for _ in range(self.horizon)]

        result = minimize(
            self._objective,
            y0,
            args=(state, goal, obstacles, x_band),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 35, "ftol": 1e-3, "maxls": 10},
        )

        if result.x is not None:
            y_opt = np.asarray(result.x, dtype=float)
            self.last_y = y_opt
            # Use the first or second pose as the local target for smoother behavior.
            target_i = 1 if self.horizon > 1 else 0
            tx = float(x_band[target_i])
            ty = float(y_opt[target_i])
            self.last_target = (tx, ty)
            dx = tx - state.x
            dy = ty - state.y
            dist = max(math.hypot(dx, dy), 1e-9)
            speed = min(cfg.max_speed, max(0.25, dist / max(cfg.dt, 1e-9)))
            vx = speed * dx / dist
            vy = speed * dy / dist
            # no backward motion in the sidewalk task
            vx = max(0.0, vx)
            status = "teb_optimized" if result.success else "teb_partial"
            return vx, vy, {"status": status, "cost": float(result.fun) if np.isfinite(result.fun) else float("inf")}

        # Fallback: creep forward along centerline.
        dx = min(goal[0], state.x + cfg.max_speed * cfg.dt) - state.x
        dy = cfg.sidewalk_center_y - state.y
        dist = max(math.hypot(dx, dy), 1e-9)
        return 0.35 * dx / dist, 0.35 * dy / dist, {"status": "teb_fallback", "cost": float("inf")}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simplified TEB robot-as-pedestrian controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(args, lambda cfg, seed: SimplifiedTEBPlanner(cfg, seed), "teb")


if __name__ == "__main__":
    main()
