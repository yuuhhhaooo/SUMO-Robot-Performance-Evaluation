#!/usr/bin/env python3
"""ORCA via the ORIGINAL published RVO2 library (not a reimplementation).

Upstream: RVO2 by the UNC GAMMA group (Jur van den Berg, Stephen J. Guy, Jamie
Snape, Ming C. Lin, Dinesh Manocha) -- the reference implementation of
"Reciprocal n-Body Collision Avoidance" (ISRR 2009), used through the Cython
bindings `Python-RVO2` (sybrenstuvel/Python-RVO2).

Why this file exists
--------------------
The benchmark previously shipped an "ORCA" baseline that contained no
velocity-obstacle half-planes, no linear program and no reciprocity factor --
it was a reciprocal-force heuristic wearing ORCA's name. Publishing a results
row labelled ORCA that is not ORCA is not defensible in a benchmark paper, so
the real library now does the work. The old planner is kept, renamed to
`orca_heuristic`, so the two can be reported side by side.

The library is a compiled C++ extension and is NOT on PyPI; build it once with

    python sim/third_party/build_rvo2.py

Solver configuration follows the canonical CrowdNav ORCA baseline (Chen et al.,
"Crowd-Robot Interaction", ICRA 2019), which is how ORCA is normally reported in
the social-navigation literature: one RVO2 simulator per control step holding
the robot plus every observed pedestrian, the robot's preferred velocity aimed
at the goal, and the pedestrians' preferred velocity left at zero because their
goals are unobservable. That last choice matters and is exposed as
`ped_pref_mode` so its effect can be measured rather than assumed:

    "zero"    CrowdNav's convention. Pedestrians look like they want to stop, so
              the robot shoulders more of the avoidance. Conservative.
    "current" Pedestrians are assumed to continue at their observed velocity.
              Closer to reality here, since SUMO/JuPedSim pedestrians really do
              keep walking, but it makes the robot rely on reciprocity that the
              pedestrians do not actually provide.

Reciprocity caveat, stated rather than hidden: ORCA assumes every agent runs
ORCA and each pair splits the avoidance effort. In this benchmark the
pedestrians are driven by SUMO striping, the SFM layer or JuPedSim -- none of
them reciprocate. The robot therefore does less than half the work it needs
unless the pedestrians are made to look passive, which is exactly what
`ped_pref_mode="zero"` does.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

try:
    from sidewalk_robot_common import Obstacle, PlannerConfig, RobotState
except ImportError:                                  # direct import fallback
    from .sidewalk_robot_common import Obstacle, PlannerConfig, RobotState


_IMPORT_HINT = (
    "ORCA now uses the published RVO2 library, which is a compiled C++ "
    "extension and is not on PyPI. Build it once with:\n"
    "    python sim/third_party/build_rvo2.py\n"
    "or set RVO2_PATH to a directory containing the built rvo2 module."
)


def _import_rvo2():
    """Import the compiled RVO2 extension, looking where build_rvo2.py puts it.

    Honours RVO2_PATH first, then falls back to the build script's defaults so
    a fresh checkout works after `python sim/third_party/build_rvo2.py` without
    anyone having to remember to export anything.
    """
    import os
    import sys
    from pathlib import Path

    candidates = []
    extra = os.environ.get("RVO2_PATH")
    if extra:
        candidates.append(Path(extra))
    if os.name == "nt":
        candidates.append(Path("C:/Users") /
                          os.environ.get("USERNAME", "user") / "rvo2")
    candidates.append(Path.home() / ".cache" / "rvo2")
    candidates.append(Path(__file__).resolve().parents[1] / "third_party" / "rvo2")

    for c in candidates:
        if c.is_dir() and str(c) not in sys.path and \
                (list(c.glob("rvo2*.pyd")) or list(c.glob("rvo2*.so"))):
            sys.path.insert(0, str(c))
    try:
        import rvo2                                   # noqa: F401
        return rvo2
    except ImportError as exc:                        # pragma: no cover
        tried = "\n  ".join(str(c) for c in candidates)
        raise ImportError(f"{exc}\n\n{_IMPORT_HINT}\n\nSearched:\n  {tried}") from exc


class ORCARVO2Planner:
    """ORCA local planner backed by the published RVO2 solver."""

    def __init__(self, cfg: PlannerConfig, seed: int = 0):
        self.cfg = cfg
        self.rvo2 = _import_rvo2()

        # --- genuine ORCA parameters. These are the algorithm's real knobs,
        # so `configs/tuning_spaces.json` can finally tune ORCA as ORCA.
        self.neighbor_dist = float(cfg.sensor_range)
        self.max_neighbors = 10
        self.time_horizon = 5.0          # agent-agent VO horizon [s]
        self.time_horizon_obst = 2.0     # agent-obstacle VO horizon [s]
        self.safety_space = 0.01         # upstream CrowdNav's epsilon
        self.ped_pref_mode = "zero"      # see the module docstring
        self.use_band_obstacles = True

        self.robot_radius = float(cfg.robot_radius)
        self.ped_radius = float(cfg.pedestrian_radius)
        self.max_speed = float(cfg.max_speed)
        self._last_v: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ band
    def _add_band(self, sim) -> None:
        """Constrain the robot to the sidewalk band with RVO2 obstacles.

        RVO2 treats an obstacle vertex loop as solid on its left-hand side, so
        a region agents must stay INSIDE is given clockwise. Two long edges are
        enough here: the band is open at both ends because the leg goal sits at
        x = sidewalk_x_max and the robot must be able to reach it.
        """
        c = self.cfg
        x0, x1 = c.sidewalk_x_min - 50.0, c.sidewalk_x_max + 50.0
        y0, y1 = c.sidewalk_y_min, c.sidewalk_y_max
        # lower edge: solid below, walkable above
        sim.addObstacle([(x0, y0), (x1, y0)])
        # upper edge: solid above, walkable below
        sim.addObstacle([(x1, y1), (x0, y1)])
        sim.processObstacles()

    # --------------------------------------------------------------- control
    def compute_command(self, state: RobotState, goal: Tuple[float, float],
                        obstacles: Sequence[Obstacle],
                        sim_time: float) -> Tuple[float, float, Dict[str, Any]]:
        rvo2 = self.rvo2
        cfg = self.cfg
        dt = float(cfg.dt)

        # A fresh simulator per step: RVO2 has no removeAgent, and the observed
        # pedestrian set changes every step. The solver is C++ and this costs
        # microseconds for the crowd sizes here.
        sim = rvo2.PyRVOSimulator(
            dt, self.neighbor_dist, int(self.max_neighbors),
            self.time_horizon, self.time_horizon_obst,
            self.robot_radius + self.safety_space, self.max_speed)

        if self.use_band_obstacles:
            self._add_band(sim)

        robot = sim.addAgent(
            (float(state.x), float(state.y)),
            self.neighbor_dist, int(self.max_neighbors),
            self.time_horizon, self.time_horizon_obst,
            self.robot_radius + self.safety_space, self.max_speed,
            (float(self._last_v[0]), float(self._last_v[1])))

        ped_ids: List[int] = []
        for o in obstacles:
            ped_ids.append(sim.addAgent(
                (float(o.x), float(o.y)),
                self.neighbor_dist, int(self.max_neighbors),
                self.time_horizon, self.time_horizon_obst,
                self.ped_radius + self.safety_space, self.max_speed,
                (float(o.vx), float(o.vy))))

        # preferred velocity: straight at the goal, clipped to max_speed
        gx, gy = float(goal[0]), float(goal[1])
        dx, dy = gx - state.x, gy - state.y
        d = math.hypot(dx, dy)
        if d > 1e-9:
            scale = min(self.max_speed, d / max(dt, 1e-9)) / d
            pref = (dx * scale, dy * scale)
        else:
            pref = (0.0, 0.0)
        sim.setAgentPrefVelocity(robot, pref)

        for aid, o in zip(ped_ids, obstacles):
            if self.ped_pref_mode == "current":
                sim.setAgentPrefVelocity(aid, (float(o.vx), float(o.vy)))
            else:
                sim.setAgentPrefVelocity(aid, (0.0, 0.0))

        sim.doStep()
        vx, vy = sim.getAgentVelocity(robot)
        vx, vy = float(vx), float(vy)

        sp = math.hypot(vx, vy)
        if sp > self.max_speed > 0.0:
            vx, vy = vx / sp * self.max_speed, vy / sp * self.max_speed
        self._last_v = (vx, vy)

        return vx, vy, {
            "status": "orca_rvo2",
            "n_orca_lines": int(sim.getAgentNumORCALines(robot)),
            "n_neighbors": int(sim.getAgentNumAgentNeighbors(robot)),
            "pref_vx": round(pref[0], 4),
            "pref_vy": round(pref[1], 4),
        }
