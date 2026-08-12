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
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace

PLANNER_DIR = Path(__file__).resolve().parent / "planners"
if str(PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_DIR))

from sidewalk_robot_common import PlannerConfig, RobotState, Obstacle  # noqa: E402

ALGORITHMS = [
    # in-repo planners (the author's own implementations)
    "dwa", "astar", "dijkstra", "rrt", "orca_heuristic", "mpc", "teb",
    "sarl", "cadrl", "lstm_rl",
    # published implementations (see PUBLISHED_IMPL)
    "orca", "mpc_dompc", "teb_upstream",
    "sarl_upstream", "cadrl_upstream", "lstm_rl_upstream",
    "crowdnav_dsrnn", "crowdnav_attngraph",
]
LEARNING = {"sarl", "cadrl", "lstm_rl",
            "sarl_upstream", "cadrl_upstream", "lstm_rl_upstream",
            "crowdnav_dsrnn", "crowdnav_attngraph"}

# Algorithms backed by the ORIGINAL PUBLISHED implementation rather than an
# in-repo reimplementation. Recorded per run so the results table can say which
# is which, and cited in the paper.
PUBLISHED_IMPL = {
    "orca": "RVO2 (UNC GAMMA; van den Berg et al., ISRR 2009) "
            "via Python-RVO2 bindings",
    "mpc_dompc": "do-mpc 5.1.1 on CasADi 3.7.2 with Ipopt "
                 "(Lucia/Fiedler; Andersson et al.)",
    "teb_upstream": "teb_local_planner 0.9.1 (Roesmann et al.), the "
                    "published C++ library via pybind11 glue",
    "sarl_upstream": "CrowdNav (Chen et al., ICRA 2019), upstream network "
                     "+ the repo's shipped checkpoint",
    "cadrl_upstream": "CrowdNav (Chen et al., ICRA 2019), upstream network "
                      "+ the repo's shipped checkpoint",
    "lstm_rl_upstream": "CrowdNav (Chen et al., ICRA 2019), upstream network "
                        "+ the repo's shipped checkpoint",
    "crowdnav_dsrnn": "CrowdNav_DSRNN (Liu et al., ICRA 2021), upstream "
                      "network + published checkpoint",
    "crowdnav_attngraph": "CrowdNav_Prediction_AttnGraph (Liu et al., ICRA "
                          "2023) + GST predictor (Huang et al., RA-L 2022), "
                          "upstream networks + published checkpoints",
}

_CLASSICAL = {
    "astar": ("astar_sidewalk_robot_random_stop_collision", "AStarPlanner"),
    "dijkstra": ("dijkstra_sidewalk_robot_random_stop_collision", "DijkstraPlanner"),
    "rrt": ("rrt_sidewalk_robot_random_stop_collision", "RRTPlanner"),
    # 'orca' is the published RVO2 solver. The previous in-repo planner is kept
    # as 'orca_heuristic': an audit found it has no velocity-obstacle
    # half-planes, no linear program and no reciprocity factor, so it is a
    # reciprocal-force heuristic rather than ORCA. Both are runnable so the
    # difference can be reported rather than hidden.
    "orca": ("orca_rvo2_planner", "ORCARVO2Planner"),
    "orca_heuristic": ("orca_sidewalk_robot_random_stop_collision",
                       "ORCAStylePlanner"),
    "mpc": ("mpc_sidewalk_robot_random_stop_collision", "MPCPlanner"),
    "teb": ("teb_sidewalk_robot_random_stop_collision", "SimplifiedTEBPlanner"),
    # published solvers, same (cfg, seed) constructor
    "mpc_dompc": ("mpc_dompc_planner", "DoMPCPlanner"),
    "teb_upstream": ("teb_upstream_planner", "UpstreamTEBPlanner"),
}

# Learning planners backed by upstream networks + published checkpoints.
# All take (cfg, seed, model_path=None, device="cpu").
_LEARNED_PUBLISHED = {
    "sarl_upstream": ("crowdnav_upstream", "UpstreamSARLPlanner",
                      "sarl_rl_model.pth"),
    "cadrl_upstream": ("crowdnav_upstream", "UpstreamCADRLPlanner",
                       "cadrl_rl_model.pth"),
    "lstm_rl_upstream": ("crowdnav_upstream", "UpstreamLstmRLPlanner",
                         "lstm_rl_model.pth"),
    # these two vendor their OWN published checkpoints, so model_path stays
    # None and the adapter resolves it inside sim/third_party/
    "crowdnav_dsrnn": ("crowdnav_dsrnn_planner", "CrowdNavDSRNNPlanner", None),
    "crowdnav_attngraph": ("crowdnav_attngraph_planner",
                           "CrowdNavAttnGraphPlanner", None),
}

_SARL_CACHE: dict = {}

# cadrl / lstm_rl are rebuilt by build_planner on EVERY leg switch, and each
# rebuild re-reads a .pth checkpoint and re-creates the torch network.  Both
# planners are pure functions of (state, goal, obstacles) -- they write nothing
# to self outside __init__, and the LSTM-RL value net re-zeros (h0, c0) inside
# every forward() -- so the network can be reused across legs exactly like
# SARL's, as long as everything the constructor derived from cfg is recomputed
# for the new leg.  Keyed like _SARL_CACHE, plus the algorithm name.
# Value: (planner, frozenset of the attribute names it had when it was built).
_LEARNED_CACHE: dict = {}


# The leg goal sits at local x == leg_len. Every classical planner rejects
# candidate points within a small margin of sidewalk_x_max (in_sidewalk uses
# 0.02-0.04, MPC's terminal constraint uses 0.06), so a box ending exactly at
# the goal makes the goal itself an INFEASIBLE point: RRT can never connect
# its tree, A*/Dijkstra can never expand the goal cell, and MPC can never
# satisfy its terminal constraint -- all three then fall back to centreline
# following regardless of their parameters. Extend the box past the goal so
# the goal is strictly interior for every planner's margin.
LEG_GOAL_MARGIN = 0.5


def leg_config(leg_len: float, band_w: float, dt: float, max_time: float) -> PlannerConfig:
    return PlannerConfig(
        dt=dt, max_time=max_time,
        sidewalk_x_min=0.0,
        sidewalk_x_max=max(leg_len, 1.0) + LEG_GOAL_MARGIN,
        sidewalk_y_min=0.0, sidewalk_y_max=band_w,
        sidewalk_center_y=band_w / 2.0,
    )


# --------------------------------------------------------------------------
# PlannerConfig -> DWAConfig propagation.
#
# DWAConfig is a standalone dataclass inside the DWA planner file and carries
# its OWN defaults (max_speed 0.95 m/s, max_accel 0.80 m/s^2, max_yaw_rate
# 80 deg/s, safe_distance 0.20 m, social_distance 0.80 m, sensor_range 12 m,
# goal_tolerance 0.25 m).  Only dt / max_time / the five sidewalk-geometry
# fields used to be copied here, so DWA -- the statistical reference algorithm
# -- was benchmarked under a different robot envelope than the six planners
# that read PlannerConfig directly.  Every quantity that exists in BOTH
# dataclasses is now propagated.
#
# Fields below carry the same quantity, unit and meaning under the same name
# in both dataclasses; the list is explicit so a reader sees exactly what the
# adapter sets, and _dwa_config_kwargs() asserts every one of them still
# resolves.
_DWA_SHARED_FIELDS = (
    # robot / controller limits
    "max_speed", "min_speed", "max_accel", "max_yaw_rate", "dt",
    # geometry and safety distances
    "robot_radius", "pedestrian_radius", "safe_distance", "social_distance",
    "sensor_range", "goal_tolerance",
    # sidewalk / leg band
    "sidewalk_x_min", "sidewalk_x_max", "sidewalk_y_min", "sidewalk_y_max",
    "sidewalk_center_y",
    # episode budget
    "max_time",
)

# PlannerConfig field -> DWAConfig field for quantities that mean the same but
# are named differently.  Empty today: DWAConfig happens to reuse every name.
# Add an entry (with a comment naming the quantity) instead of letting a field
# fall through to _PLANNER_ONLY_FIELDS.
_DWA_RENAMED_FIELDS: dict[str, str] = {}

# PlannerConfig fields that deliberately have NO DWAConfig counterpart.  Empty
# today.  Anything not covered by the three collections raises, so a field
# added to PlannerConfig later cannot be silently dropped again.
_PLANNER_ONLY_FIELDS: frozenset[str] = frozenset()

# DWAConfig fields intentionally left at their DWAConfig default, because they
# are DWA *search* parameters rather than robot limits and have no
# PlannerConfig counterpart: max_delta_yaw_rate (yaw acceleration),
# v_resolution, yaw_rate_resolution, predict_time, goal_cost_gain,
# speed_cost_gain, obstacle_cost_gain, centerline_cost_gain, yaw_rate_cost_gain.


def _dwa_config_kwargs(cfg: PlannerConfig, dwa_config_cls) -> dict:
    """Resolve the PlannerConfig -> DWAConfig field mapping generically.

    Walks the PlannerConfig dataclass so a future field cannot be dropped
    unnoticed: it is copied when DWAConfig has a matching field, and raises
    otherwise unless it is declared in _PLANNER_ONLY_FIELDS.
    """
    dwa_names = {f.name for f in dataclass_fields(dwa_config_cls)}
    kwargs: dict = {}
    undeclared: list[str] = []
    for f in dataclass_fields(cfg):
        target = _DWA_RENAMED_FIELDS.get(f.name, f.name)
        if target in dwa_names:
            kwargs[target] = getattr(cfg, f.name)
        elif f.name not in _PLANNER_ONLY_FIELDS:
            undeclared.append(f.name)
    if undeclared:
        raise RuntimeError(
            "benchmark_adapters: PlannerConfig field(s) "
            f"{undeclared} have no DWAConfig counterpart. Add them to "
            "_DWA_RENAMED_FIELDS (same quantity, other name) or to "
            "_PLANNER_ONLY_FIELDS (no DWA equivalent) so the choice is explicit."
        )
    expected = {_DWA_RENAMED_FIELDS.get(n, n) for n in _DWA_SHARED_FIELDS}
    lost = sorted(expected - set(kwargs))
    if lost:
        raise RuntimeError(
            f"benchmark_adapters: expected DWAConfig field(s) {lost} to be set "
            "from PlannerConfig but they no longer resolve."
        )
    new = sorted(set(kwargs) - expected)
    if new:
        print(f"DWAAdapter: also propagating new shared config field(s) {new}")
    return kwargs


class DWAAdapter:
    """Unicycle DWA (module-level dwa_control) -> holonomic (vx, vy)."""

    def __init__(self, cfg: PlannerConfig, seed: int):
        import importlib
        self.mod = importlib.import_module("dwa_sidewalk_robot_random_stop_collision")
        # Run DWA under the SAME robot envelope as every other planner.
        self.cfg = self.mod.DWAConfig(**_dwa_config_kwargs(cfg, self.mod.DWAConfig))
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


def _cached_learned(key, build, retarget):
    """Return a cached learning planner, re-targeted to the current leg.

    `build()` constructs a fresh one; `retarget(planner)` rebinds everything
    the constructor derives from cfg/seed and resets any per-episode state.
    If the planner grew attributes since it was built -- i.e. it started
    keeping rollout state that a fresh instance would not have -- the cache
    entry is thrown away and a new planner is built, so reuse can never
    silently carry state from the previous leg.
    """
    hit = _LEARNED_CACHE.get(key)
    if hit is not None:
        planner, born_with = hit
        if frozenset(vars(planner)) == born_with:
            retarget(planner)
            return planner
        print(f"planner cache: {key[0]} grew per-episode state "
              f"{sorted(set(vars(planner)) - set(born_with))}; rebuilding "
              f"instead of reusing")
        _LEARNED_CACHE.pop(key, None)
    planner = build()
    _LEARNED_CACHE[key] = (planner, frozenset(vars(planner)))
    return planner


def _reset_episode_state(planner):
    """Drop anything that must not survive a leg switch.

    Today both cached planners are stateless between compute_command calls
    (the LSTM-RL net builds fresh zero (h0, c0) tensors on every forward), so
    this only clears the conventional hidden-state attribute names should a
    future revision of those files introduce one; _cached_learned's attribute
    snapshot catches the case where a brand-new attribute appears.
    """
    if hasattr(planner, "reset") and callable(planner.reset):
        planner.reset()
    for name in ("hidden", "hidden_state", "_hidden", "lstm_hidden",
                 "lstm_state", "rnn_state"):
        if hasattr(planner, name):
            setattr(planner, name, None)


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
    if algorithm in _LEARNED_PUBLISHED:
        # Upstream networks driven by published checkpoints. Cached like the
        # in-repo learning planners so a leg switch does not re-read the .pth
        # and rebuild the network; each class exposes reset() for per-episode
        # state, and re-targeting is just rebinding cfg.
        mod_name, cls_name, ckpt = _LEARNED_PUBLISHED[algorithm]
        model_path = (model_dir / ckpt) if ckpt else None

        def _build_pub():
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)(cfg, seed, model_path=model_path,
                                          device=device)

        def _retarget_pub(pl):
            # Rebinding cfg is NOT enough. These planners derive state from cfg
            # in __init__ -- the upstream policy's time_step, v_pref, the
            # DISCRETE ACTION SPACE built from v_pref, and both radii -- and a
            # cached instance would otherwise keep the FIRST leg's values for
            # the whole run. map5_ucl routes have 18+ legs, so a silent leak
            # here would affect every OSM result. Same reasoning, and the same
            # shape, as _retarget_cadrl below.
            #
            # With today's leg_config only the sidewalk geometry varies between
            # legs, so none of these actually change and the cache happens to
            # be safe -- but it is safe by accident, and one per-leg dt or
            # speed limit would break it silently. Recompute them explicitly.
            pl.cfg = cfg                # geometry of the NEW leg
            mod = importlib.import_module(mod_name)
            v_pref_const = getattr(mod, "UPSTREAM_V_PREF", None)
            if v_pref_const is not None and hasattr(pl, "v_pref") and \
                    hasattr(getattr(pl, "policy", None), "build_action_space"):
                # the three upstream-CrowdNav planners
                pl.policy.time_step = float(cfg.dt)
                v_pref = float(min(v_pref_const, cfg.max_speed))
                if v_pref != pl.v_pref:
                    pl.v_pref = v_pref
                    pl.policy.build_action_space(v_pref)
                pl.robot_radius = float(cfg.robot_radius)
                pl.human_radius = float(cfg.pedestrian_radius)
            # DS-RNN / AttnGraph deliberately feed upstream's CONSTANTS
            # (radius 0.3, v_pref 1.0) rather than cfg's values -- the networks
            # only ever saw those two numbers in those slots -- and read
            # cfg.dt / cfg.max_speed at call time, so rebinding cfg is all they
            # need. Do not "helpfully" overwrite those constants here.
            if hasattr(pl, "reset"):
                pl.reset()
            _reset_episode_state(pl)

        return _cached_learned((algorithm, str(model_path), device),
                               _build_pub, _retarget_pub)
    if algorithm == "sarl":
        return SARLAdapter(cfg, seed, model_dir / "sarl_rl_model.pth", device)
    if algorithm == "cadrl":
        model_path = model_dir / "cadrl_rl_model.pth"
        ns = SimpleNamespace(
            model_path=str(model_path), gpu=(device != "cpu"),
            cadrl_gamma=0.9, cadrl_speed_samples=5, cadrl_rotation_samples=16,
            cadrl_max_humans=5, cadrl_v_pref=None, cadrl_sidewalk_penalty=1.0,
            cadrl_centerline_penalty=0.02, cadrl_goal_lookahead=6.0,
            cadrl_progress_bonus=0.20)

        def _build_cadrl():
            mod = importlib.import_module(
                "cadrl_sidewalk_robot_random_stop_collision")
            return mod.CADRLPlanner(cfg, seed, ns)

        def _retarget_cadrl(pl):
            pl.cfg = cfg                # geometry of the NEW leg
            pl.seed = seed
            # CADRLPlanner.__init__: v_pref = cadrl_v_pref or cfg.max_speed,
            # and the discrete action space is derived from v_pref.
            v_pref = (float(ns.cadrl_v_pref) if ns.cadrl_v_pref is not None
                      else cfg.max_speed)
            if v_pref != pl.v_pref:
                pl.v_pref = v_pref
                pl.action_space = pl.build_action_space(v_pref)
            _reset_episode_state(pl)

        return _cached_learned(("cadrl", str(model_path), device),
                               _build_cadrl, _retarget_cadrl)
    if algorithm == "lstm_rl":
        import torch
        model_path = model_dir / "lstm_rl_model.pth"

        def _build_lstm():
            mod = importlib.import_module(
                "lstm_rl_sidewalk_robot_random_stop_collision")
            return mod.LstmRLPlanner(cfg, seed, model_path=model_path,
                                     device=torch.device(device))

        def _retarget_lstm(pl):
            pl.cfg = cfg                # geometry of the NEW leg
            pl.seed = seed
            # LstmRLPlanner.__init__: v_pref = min(<default 1.00>,
            # cfg.max_speed), and the action space is derived from it.
            v_pref = min(1.00, cfg.max_speed)
            if v_pref != pl.v_pref:
                pl.v_pref = v_pref
                pl.action_space = pl._build_action_space()
            _reset_episode_state(pl)

        return _cached_learned(("lstm_rl", str(model_path), device),
                               _build_lstm, _retarget_lstm)
    raise SystemExit(f"unknown algorithm '{algorithm}'")
