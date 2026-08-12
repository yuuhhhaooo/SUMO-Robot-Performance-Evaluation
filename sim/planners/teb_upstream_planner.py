#!/usr/bin/env python3
"""Timed-Elastic-Band local planner driven by the PUBLISHED teb_local_planner.

Provenance
----------
This planner does NOT implement TEB.  Every optimisation step is executed by

    teb_local_planner 0.9.1  (rst-tu-dortmund/teb_local_planner, Roesmann et al.,
                              ROBOTIK 2012 / ECMR 2013), BSD.
    libg2o 2020.5.3          sparse Levenberg-Marquardt back end, BSD.

obtained as the prebuilt RoboStack conda package `ros-noetic-teb-local-planner`
(win-64).  `sim/third_party/pyteb/` holds a thin pybind11 bridge (`TebBridge`)
that fills the upstream `TebConfig`, hands it an upstream `ObstContainer` and
calls the upstream `TebOptimalPlanner::plan()` / `getVelocityCommand()`.
See `sim/third_party/pyteb/PATCHES.md` -- upstream code is unmodified.

Consequences that matter for the benchmark
------------------------------------------
The real TEB decision vector is the sequence of SE2 poses **and the time
differences dT_i between them**.  That is what lets it slow down, stop, and be
time-optimal.  The in-repo `SimplifiedTEBPlanner` has no dT and no velocity
variable at all (it optimises 10 lateral offsets and then reads the speed off
the band geometry), which is why it commands |v| = max_speed at essentially
every step.  `info["band_dt"]` below reports the first dT actually solved for,
so the difference is visible in the trace.

How the ROS pieces are replaced
-------------------------------
* global plan  -> a straight reference path from the robot to the leg goal,
  sampled at `plan_step` and truncated at `max_global_plan_lookahead_dist`,
  which is exactly what `TebLocalPlannerROS::computeVelocityCommands` hands to
  `plan()` after `transformGlobalPlan()`.
* local costmap -> two upstream `LineObstacle`s at the sidewalk kerbs plus one
  upstream `PointObstacle` per pedestrian in sensor range (with its measured
  velocity, so upstream's constant-velocity `EdgeDynamicObstacle` is used).
* No ROS node, master, tf or costmap_2d is involved.

Interface
---------
    UpstreamTEBPlanner(cfg: PlannerConfig, seed: int)
    compute_command(state, goal, obstacles, sim_time) -> (vx, vy, info)
in the LEG-LOCAL frame (sidewalk along +x, y in [0, band_width]).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    run_traci_with_planner,
)

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party" / "pyteb"

# Directories holding teb_local_planner.dll + its ROS/g2o/boost dependencies.
_DEFAULT_DLL_DIRS = (
    r"C:/Users/Mark/tebenv/Library/bin",
    r"C:/Users/Mark/tebenv",
)

_pyteb = None
_import_error: str | None = None


def _load_pyteb():
    """Import the vendored bridge, wiring up the native DLL search path first."""
    global _pyteb, _import_error
    if _pyteb is not None or _import_error is not None:
        return _pyteb
    dirs = os.environ.get("TEB_LOCAL_PLANNER_DLL_DIR")
    dirs = dirs.split(";") if dirs else list(_DEFAULT_DLL_DIRS)
    missing = [d for d in dirs if not Path(d).is_dir()]
    for d in dirs:
        if Path(d).is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(Path(d)))
    prebuilt = _THIRD_PARTY / "prebuilt"
    if str(prebuilt) not in sys.path:
        sys.path.insert(0, str(prebuilt))
    try:
        import pyteb  # type: ignore
        _pyteb = pyteb
    except Exception as exc:
        _import_error = (
            f"cannot import the upstream teb_local_planner bridge: {exc}. "
            f"Searched DLL dirs {dirs} (missing: {missing}); extension dir "
            f"{prebuilt}. Set TEB_LOCAL_PLANNER_DLL_DIR, or recreate the conda "
            f"env as documented in {_THIRD_PARTY / 'PATCHES.md'}."
        )
        raise ImportError(_import_error) from exc
    return _pyteb


class UpstreamTEBPlanner:
    """Adapter around upstream `teb_local_planner::TebOptimalPlanner`."""

    def __init__(self, cfg: PlannerConfig, seed: int,
                 lookahead: float = 6.0,
                 plan_step: float = 0.25,
                 use_homotopy: bool = False):
        pyteb = _load_pyteb()
        self.cfg = cfg
        self.seed = int(seed)
        self.lookahead = float(lookahead)
        self.plan_step = float(plan_step)
        self.use_homotopy = bool(use_homotopy)

        self.clearance = max(cfg.safe_distance,
                             cfg.robot_radius + cfg.pedestrian_radius + 0.05)
        # LineObstacle kerbs sit ON the band edge, so the robot centre must keep
        # min_obstacle_dist from them; use the robot radius there and let the
        # (larger) pedestrian clearance come from min_obstacle_dist below.
        self.wall_pad = 0.0

        # ---- upstream TebConfig ------------------------------------------
        # Everything not listed keeps the upstream default (teb_config.h).
        self.params = {
            # robot envelope == PlannerConfig, holonomic (max_vel_y > 0)
            "robot.max_vel_x": cfg.max_speed,
            "robot.max_vel_x_backwards": 0.02,      # sidewalk task: no reversing
            "robot.max_vel_y": 0.75 * cfg.max_speed,
            "robot.max_vel_theta": cfg.max_yaw_rate,
            "robot.acc_lim_x": cfg.max_accel,
            "robot.acc_lim_y": cfg.max_accel,
            "robot.acc_lim_theta": 2.0,
            "robot.min_turning_radius": 0.0,
            # trajectory resolution == the control step
            "trajectory.dt_ref": cfg.dt,
            "trajectory.dt_hysteresis": 0.1 * cfg.dt,
            "trajectory.min_samples": 5,
            "trajectory.max_samples": 100,
            "trajectory.max_global_plan_lookahead_dist": self.lookahead,
            "trajectory.global_plan_overwrite_orientation": 1,
            "trajectory.allow_init_with_backwards_motion": 0,
            "trajectory.exact_arc_length": 0,
            "trajectory.control_look_ahead_poses": 1,
            "trajectory.feasibility_check_no_poses": 3,
            # obstacles: pedestrians are dynamic, kerbs are static lines
            "obstacles.min_obstacle_dist": self.clearance,
            "obstacles.inflation_dist": cfg.social_distance,
            "obstacles.dynamic_obstacle_inflation_dist": cfg.social_distance,
            "obstacles.include_dynamic_obstacles": 1,
            "obstacles.include_costmap_obstacles": 0,
            "obstacles.costmap_obstacles_behind_robot_dist": 2.0,
            "optim.weight_inflation": 0.4,
            # holonomic robot: relax the non-holonomic kinematics edge
            "optim.weight_kinematics_nh": 1.0,
            "optim.weight_kinematics_forward_drive": 1.0,
            "optim.no_inner_iterations": 5,
            "optim.no_outer_iterations": 4,
            "goal_tolerance.xy_goal_tolerance": cfg.goal_tolerance,
            "goal_tolerance.free_goal_vel": 0,
        }
        if self.use_homotopy:
            self.params.update({
                "hcp.enable_homotopy_class_planning": 1,
                "hcp.enable_multithreading": 1,
                "hcp.max_number_classes": 4,
            })

        self.bridge = pyteb.TebBridge(self.params, cfg.robot_radius,
                                      self.use_homotopy)
        self._last_v = (0.0, 0.0)
        self._plans = 0
        self._fail = 0
        self._t_total = 0.0

    # ------------------------------------------------------------------ util
    def _reference_plan(self, state: RobotState,
                        goal: Tuple[float, float]) -> List[List[float]]:
        """Straight reference path robot -> leg goal, truncated at lookahead."""
        gx, gy = goal
        dx, dy = gx - state.x, gy - state.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return [[state.x, state.y, state.yaw],
                    [state.x + 1e-3, state.y, state.yaw]]
        heading = math.atan2(dy, dx)
        span = min(self.lookahead, dist)
        n = max(2, int(span / self.plan_step) + 1)
        ux, uy = dx / dist, dy / dist
        plan = [[state.x, state.y, state.yaw]]
        for i in range(1, n + 1):
            s = span * i / n
            plan.append([state.x + ux * s, state.y + uy * s, heading])
        return plan

    def _walls(self, state: RobotState) -> List[List[float]]:
        cfg = self.cfg
        x0 = max(cfg.sidewalk_x_min - 2.0, state.x - 3.0)
        x1 = min(cfg.sidewalk_x_max + 2.0, state.x + self.lookahead + 4.0)
        return [[x0, cfg.sidewalk_y_min, x1, cfg.sidewalk_y_min],
                [x0, cfg.sidewalk_y_max, x1, cfg.sidewalk_y_max]]

    # --------------------------------------------------------------- planner
    def compute_command(self, state: RobotState, goal: Tuple[float, float],
                        obstacles: Sequence[Obstacle], sim_time: float):
        cfg = self.cfg
        peds = [[float(o.x), float(o.y), float(o.vx), float(o.vy)]
                for o in obstacles
                if math.hypot(o.x - state.x, o.y - state.y) <= cfg.sensor_range]
        self.bridge.set_walls(self._walls(state))
        self.bridge.set_point_obstacles(peds)

        plan = self._reference_plan(state, goal)
        # current velocity expressed in the ROBOT frame, as ROS would supply it
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        wvx, wvy = self._last_v
        rvx = c * wvx + s * wvy
        rvy = -s * wvx + c * wvy

        t0 = time.perf_counter()
        try:
            ok = bool(self.bridge.plan(plan, rvx, rvy, 0.0, False))
            cmd = self.bridge.velocity_command(1)
        except Exception:
            ok, cmd = False, [0.0, 0.0, 0.0, 0.0]
        solve_ms = (time.perf_counter() - t0) * 1e3
        self._plans += 1
        self._t_total += solve_ms

        band = self.bridge.band() if ok else []
        band_dt = float(band[0][3]) if band else float("nan")
        theta0 = float(band[0][2]) if band else state.yaw

        if ok and cmd[0] > 0.5 and all(np.isfinite(cmd)):
            # upstream returns (vx, vy) in the frame of band pose 0
            bvx, bvy = float(cmd[1]), float(cmd[2])
            ct, st = math.cos(theta0), math.sin(theta0)
            vx = ct * bvx - st * bvy
            vy = st * bvx + ct * bvy
            sp = math.hypot(vx, vy)
            if sp > cfg.max_speed:                 # upstream limits are soft
                vx *= cfg.max_speed / sp
                vy *= cfg.max_speed / sp
            status = "teb_upstream"
        else:
            self._fail += 1
            sp_prev = math.hypot(*self._last_v)
            sp = max(0.0, sp_prev - cfg.max_accel * cfg.dt)
            if sp_prev > 1e-9:
                vx = self._last_v[0] / sp_prev * sp
                vy = self._last_v[1] / sp_prev * sp
            else:
                vx = vy = 0.0
            status = "teb_upstream_failed"
            self.bridge.clear_planner()            # drop the stale band

        self._last_v = (vx, vy)
        return vx, vy, {
            "status": status,
            "cost": float("nan"),
            "solve_ms": round(solve_ms, 3),
            "band_poses": len(band),
            "band_dt": band_dt,          # the TEB time-difference variable
            "n_peds": len(peds),
        }

    def timing_summary(self) -> dict:
        n = max(self._plans, 1)
        return {"plans": self._plans, "failures": self._fail,
                "mean_solve_ms": self._t_total / n,
                "total_solve_s": self._t_total / 1e3}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upstream teb_local_planner (0.9.1) robot-as-pedestrian "
                    "controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    p.add_argument("--teb-lookahead", type=float, default=6.0)
    p.add_argument("--teb-homotopy", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(
        args,
        lambda cfg, seed: UpstreamTEBPlanner(cfg, seed,
                                             lookahead=args.teb_lookahead,
                                             use_homotopy=args.teb_homotopy),
        "teb_upstream")


if __name__ == "__main__":
    main()
