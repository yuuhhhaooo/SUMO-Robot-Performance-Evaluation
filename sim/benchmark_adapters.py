#!/usr/bin/env python3
"""Adapters that plug the UNCHANGED uploaded planner files into the v7 maps.

Every adapter exposes the same call used by the classical baselines:
    compute_command(state, goal, obstacles, sim_time) -> (vx, vy, info)
in a LEG-LOCAL frame where the current sidewalk leg runs along +x inside the
band y in [0, band_width].  The benchmark runner owns the world<->leg
transform, the robot POI, the signal gate and all metrics.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

PLANNER_DIR = Path(__file__).resolve().parent / "planners"
if str(PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_DIR))

from sidewalk_robot_common import PlannerConfig, RobotState, Obstacle  # noqa: E402

ALGORITHMS = ["dwa", "astar", "dijkstra", "rrt", "orca", "mpc", "teb",
              "sarl", "cadrl", "lstm_rl"]
LEARNING = {"sarl", "cadrl", "lstm_rl"}

_CLASSICAL = {
    "astar": ("astar_sidewalk_robot_random_stop_collision", "AStarPlanner"),
    "dijkstra": ("dijkstra_sidewalk_robot_random_stop_collision", "DijkstraPlanner"),
    "rrt": ("rrt_sidewalk_robot_random_stop_collision", "RRTPlanner"),
    "orca": ("orca_sidewalk_robot_random_stop_collision", "ORCAStylePlanner"),
    "mpc": ("mpc_sidewalk_robot_random_stop_collision", "MPCPlanner"),
    "teb": ("teb_sidewalk_robot_random_stop_collision", "SimplifiedTEBPlanner"),
}

_SARL_CACHE: dict = {}


def leg_config(leg_len: float, band_w: float, dt: float, max_time: float) -> PlannerConfig:
    return PlannerConfig(
        dt=dt, max_time=max_time,
        sidewalk_x_min=0.0, sidewalk_x_max=max(leg_len, 1.0),
        sidewalk_y_min=0.0, sidewalk_y_max=band_w,
        sidewalk_center_y=band_w / 2.0,
    )


class DWAAdapter:
    """Unicycle DWA (module-level dwa_control) -> holonomic (vx, vy)."""

    def __init__(self, cfg: PlannerConfig, seed: int):
        import importlib
        self.mod = importlib.import_module("dwa_sidewalk_robot_random_stop_collision")
        self.cfg = self.mod.DWAConfig(
            dt=cfg.dt, max_time=cfg.max_time,
            sidewalk_x_min=cfg.sidewalk_x_min, sidewalk_x_max=cfg.sidewalk_x_max,
            sidewalk_y_min=cfg.sidewalk_y_min, sidewalk_y_max=cfg.sidewalk_y_max,
            sidewalk_center_y=cfg.sidewalk_center_y,
        )
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0

    def compute_command(self, state, goal, obstacles, sim_time):
        st = self.mod.RobotState(x=state.x, y=state.y, yaw=self.yaw,
                                 v=self.v, w=self.w)
        obs = [self.mod.Obstacle(o.pid, o.x, o.y, o.vx, o.vy) for o in obstacles]
        (v, w), _traj, info = self.mod.dwa_control(st, goal, obs, self.cfg)
        self.yaw = (self.yaw + w * self.cfg.dt + math.pi) % (2 * math.pi) - math.pi
        self.v, self.w = float(v), float(w)
        vx = self.v * math.cos(self.yaw)
        vy = self.v * math.sin(self.yaw)
        info = dict(info or {})
        info["status"] = info.get("status", "dwa")
        return vx, vy, info


class SARLAdapter:
    """CrowdNav SARL value network through SarlPolicy.predict."""

    def __init__(self, cfg: PlannerConfig, seed: int, model_path: Path, device: str):
        import torch
        from sarl_sumo_robot_unified import (SumoSarlConfig, SarlPolicy,
                                             FullState, HumanObservation,
                                             ObservableState)
        self.FullState = FullState
        self.HumanObservation = HumanObservation
        self.ObservableState = ObservableState
        self.cfg = cfg
        scfg = SumoSarlConfig(
            dt=cfg.dt, v_pref=min(1.0, cfg.max_speed),
            sidewalk_x_min=cfg.sidewalk_x_min, sidewalk_x_max=cfg.sidewalk_x_max,
            sidewalk_y_min=cfg.sidewalk_y_min, sidewalk_y_max=cfg.sidewalk_y_max,
            max_time=cfg.max_time,
        )
        key = (str(model_path), device)
        if key not in _SARL_CACHE:
            _SARL_CACHE[key] = SarlPolicy(Path(model_path), scfg,
                                          torch.device(device))
        self.policy = _SARL_CACHE[key]
        self.policy.cfg = scfg          # rebind geometry to the current leg
        self.last_v = (0.0, 0.0)

    def compute_command(self, state, goal, obstacles, sim_time):
        robot = self.FullState(px=state.x, py=state.y,
                               vx=self.last_v[0], vy=self.last_v[1],
                               radius=0.25, gx=goal[0], gy=goal[1],
                               v_pref=self.policy.cfg.v_pref, theta=0.0)
        humans = [self.HumanObservation(o.pid, self.ObservableState(
            o.x, o.y, o.vx, o.vy, 0.15)) for o in obstacles]
        action, value, att_pid, att_w = self.policy.predict(robot, humans)
        self.last_v = (float(action.vx), float(action.vy))
        return action.vx, action.vy, {"status": "sarl", "value": float(value),
                                      "attended": att_pid,
                                      "attention": float(att_w)}


def apply_params(planner, params):
    """Generic tuned-parameter override: set matching attributes on the
    planner or (one level deep) on its config-like members. Returns the
    list of keys that did NOT match anything (caller may warn)."""
    import inspect as _ins
    unmatched = []
    for k, v in (params or {}).items():
        hit = False
        if hasattr(planner, k):
            setattr(planner, k, v)
            hit = True
        for sub in vars(planner).values():
            if _ins.ismodule(sub):
                # the real consumers are usually Config CLASSES inside the
                # planner module (re-instantiated per leg): set the class
                # attribute so every future instance sees the tuned value
                for cls in vars(sub).values():
                    if _ins.isclass(cls) and hasattr(cls, k):
                        setattr(cls, k, v)
                        hit = True
            elif hasattr(sub, "__dict__") and hasattr(sub, k):
                setattr(sub, k, v)
                hit = True
        if not hit:
            unmatched.append(k)
    return unmatched


def build_planner(algorithm: str, cfg: PlannerConfig, seed: int,
                  model_dir: Path, device: str = "cpu", params=None):
    """Instantiate one planner for the current leg (planner files unchanged)."""
    import importlib
    if algorithm == "dwa":
        pl = DWAAdapter(cfg, seed)
        unm = apply_params(pl, params)
        if unm:
            print(f"params: unmatched keys {unm}")
        return pl
    if algorithm in _CLASSICAL:
        mod_name, cls_name = _CLASSICAL[algorithm]
        mod = importlib.import_module(mod_name)
        pl = getattr(mod, cls_name)(cfg, seed)
        unm = apply_params(pl, params)
        if unm:
            print(f"params: unmatched keys {unm}")
        return pl
    if algorithm == "sarl":
        return SARLAdapter(cfg, seed, model_dir / "sarl_rl_model.pth", device)
    if algorithm == "cadrl":
        mod = importlib.import_module("cadrl_sidewalk_robot_random_stop_collision")
        ns = SimpleNamespace(
            model_path=str(model_dir / "cadrl_rl_model.pth"), gpu=(device != "cpu"),
            cadrl_gamma=0.9, cadrl_speed_samples=5, cadrl_rotation_samples=16,
            cadrl_max_humans=5, cadrl_v_pref=None, cadrl_sidewalk_penalty=1.0,
            cadrl_centerline_penalty=0.02, cadrl_goal_lookahead=6.0,
            cadrl_progress_bonus=0.20)
        return mod.CADRLPlanner(cfg, seed, ns)
    if algorithm == "lstm_rl":
        import torch
        mod = importlib.import_module("lstm_rl_sidewalk_robot_random_stop_collision")
        return mod.LstmRLPlanner(cfg, seed,
                                 model_path=model_dir / "lstm_rl_model.pth",
                                 device=torch.device(device))
    raise SystemExit(f"unknown algorithm '{algorithm}'")
