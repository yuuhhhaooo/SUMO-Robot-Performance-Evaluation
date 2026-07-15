#!/usr/bin/env python3
"""MPC-controlled delivery robot in the SUMO sidewalk scene.

This is a lightweight receding-horizon MPC baseline designed to fit the existing
SUMO/TraCI framework. It does not require ROS. At every SUMO step, the planner
optimizes a short sequence of planar velocity commands and applies only the first
command, then replans at the next step.

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


class MPCPlanner:
    """Finite-horizon velocity MPC for sidewalk navigation.

    Decision variables are a sequence of velocity commands:
        u = [vx_0, vy_0, vx_1, vy_1, ...]
    The objective balances goal tracking, forward progress, obstacle separation,
    sidewalk boundary penalties, and smoothness. Dynamic pedestrians are predicted
    with a constant-velocity model over the short horizon.
    """

    def __init__(self, cfg: PlannerConfig, seed: int):
        self.cfg = cfg
        self.horizon = 8
        self.margin = 0.06
        self.clearance = max(cfg.safe_distance, cfg.robot_radius + cfg.pedestrian_radius + 0.05)
        self.social_clearance = max(cfg.social_distance, self.clearance + 0.15)
        self.last_solution: np.ndarray | None = None
        self.last_u = np.array([0.0, 0.0], dtype=float)

        # Cost weights. These are intentionally small enough for stable online use.
        self.w_terminal_goal = 9.0
        self.w_goal_path = 0.08
        self.w_obstacle = 8.0
        self.w_social = 0.8
        self.w_speed_ref = 0.45
        self.w_control_smooth = 0.55
        self.w_boundary = 60.0
        self.w_centerline = 0.03
        self.w_backward = 8.0
        self.w_lateral = 0.08

    def _initial_guess(self, state: RobotState, goal: Tuple[float, float]) -> np.ndarray:
        dx, dy = goal[0] - state.x, goal[1] - state.y
        dist = max(math.hypot(dx, dy), 1e-9)
        pref = np.array([self.cfg.max_speed * dx / dist, self.cfg.max_speed * dy / dist], dtype=float)
        pref[0] = max(0.0, pref[0])  # no backward motion for this sidewalk task
        if self.last_solution is not None and len(self.last_solution) == 2 * self.horizon:
            # Shift the previous optimal sequence forward by one step.
            shifted = np.r_[self.last_solution[2:], self.last_solution[-2:]]
            return shifted
        return np.tile(pref, self.horizon)

    def _objective(self, u_flat: np.ndarray, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle]) -> float:
        cfg = self.cfg
        x, y = state.x, state.y
        prev_u = self.last_u.copy()
        total = 0.0

        for k in range(self.horizon):
            vx = float(u_flat[2 * k])
            vy = float(u_flat[2 * k + 1])
            speed = math.hypot(vx, vy)
            t = (k + 1) * cfg.dt

            # Simulate robot forward.
            x += vx * cfg.dt
            y += vy * cfg.dt

            # Goal tracking along the horizon and stronger terminal goal cost.
            goal_dist_sq = (goal[0] - x) ** 2 + (goal[1] - y) ** 2
            total += self.w_goal_path * goal_dist_sq
            if k == self.horizon - 1:
                total += self.w_terminal_goal * goal_dist_sq

            # Encourage a reasonable reference speed without forcing unsafe motion.
            total += self.w_speed_ref * (speed - 0.85 * cfg.max_speed) ** 2
            if vx < 0.0:
                total += self.w_backward * (vx ** 2)
            total += self.w_lateral * (vy ** 2)

            # Penalize excessive speed beyond cfg.max_speed. Bounds already help, but this
            # keeps diagonal velocity magnitude under control.
            if speed > cfg.max_speed:
                total += 80.0 * (speed - cfg.max_speed) ** 2

            # Smooth control changes.
            dux = vx - prev_u[0]
            duy = vy - prev_u[1]
            total += self.w_control_smooth * (dux * dux + duy * duy)
            prev_u[:] = (vx, vy)

            # Sidewalk hard-boundary penalties.
            ymin = cfg.sidewalk_y_min + self.margin
            ymax = cfg.sidewalk_y_max - self.margin
            xmin = cfg.sidewalk_x_min + self.margin
            xmax = cfg.sidewalk_x_max - self.margin
            for violation in (xmin - x, x - xmax, ymin - y, y - ymax):
                if violation > 0.0:
                    total += self.w_boundary * violation * violation
            total += self.w_centerline * (y - cfg.sidewalk_center_y) ** 2

            # Dynamic-obstacle separation with constant-velocity pedestrian prediction.
            for obs in obstacles:
                ox = obs.x + obs.vx * t
                oy = obs.y + obs.vy * t
                d = math.hypot(x - ox, y - oy)
                if d < self.clearance:
                    total += self.w_obstacle * (self.clearance - d + 1e-3) ** 2 * 100.0
                elif d < self.social_clearance:
                    total += self.w_social * (self.social_clearance - d) ** 2
                # Soft reciprocal term to prefer larger clearance when alternatives exist.
                total += 0.015 / max(d - self.clearance + 0.05, 0.05)

        return float(total)

    def compute_command(self, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle], sim_time: float):
        cfg = self.cfg
        x0 = self._initial_guess(state, goal)
        bounds = []
        # vx forward, vy lateral. Lateral bound is limited because the sidewalk is narrow.
        for _ in range(self.horizon):
            bounds.append((0.0, cfg.max_speed))
            bounds.append((-0.75 * cfg.max_speed, 0.75 * cfg.max_speed))

        result = minimize(
            self._objective,
            x0,
            args=(state, goal, obstacles),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 35, "ftol": 1e-3, "maxls": 10},
        )

        if result.success or result.x is not None:
            u = np.asarray(result.x, dtype=float)
            self.last_solution = u
            vx = float(u[0])
            vy = float(u[1])
            speed = math.hypot(vx, vy)
            if speed > cfg.max_speed:
                scale = cfg.max_speed / max(speed, 1e-9)
                vx *= scale
                vy *= scale
            self.last_u[:] = (vx, vy)
            status = "mpc_optimized" if result.success else "mpc_partial"
            return vx, vy, {"status": status, "cost": float(result.fun) if np.isfinite(result.fun) else float("inf")}

        # Fallback: move slowly toward the goal centerline.
        dx = goal[0] - state.x
        dy = cfg.sidewalk_center_y - state.y
        norm = max(math.hypot(dx, dy), 1e-9)
        vx = 0.35 * dx / norm
        vy = 0.35 * dy / norm
        self.last_u[:] = (vx, vy)
        return vx, vy, {"status": "mpc_fallback", "cost": float("inf")}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MPC robot-as-pedestrian controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(args, lambda cfg, seed: MPCPlanner(cfg, seed), "mpc")


if __name__ == "__main__":
    main()
