#!/usr/bin/env python3
"""Reactive pedestrians driven by a PUBLISHED Social Force Model package.

This is the published-implementation counterpart of
`sim/social_pedestrians.py`. That file re-derives Helbing & Molnar by hand;
this one hands the state to an unmodified upstream library and lets the
library compute the forces and integrate them.

    --reactive-peds off        striping only (legacy)
    --reactive-peds sfm        the IN-REPO hand-rolled SFM (kept, unchanged)
    --reactive-peds jupedsim   JuPedSim 1.4.2
    --reactive-peds pysf       THIS layer -- PySocialForce / socialforce

Capture/release geometry (CAPTURE_R 12.0, RELEASE_R 18.0), robot handling and
metric keys deliberately mirror `sim/jupedsim_pedestrians.py`, so all four
settings are levels of ONE experimental factor and differ only in the
operational pedestrian model.


BACKENDS (both are pip dependencies, neither is vendored or patched)
--------------------------------------------------------------------
backend="pysocialforce"  (DEFAULT, primary)
    PySocialForce 1.1.2 -- github.com/yuxiang-gao/PySocialForce
    Extended/elliptical SFM: Helbing & Molnar 1995 goal force plus the
    Moussaid et al. 2010 directional interaction force, plus an obstacle
    force. numpy only. Chosen as primary because (a) it is the package the
    in-repo docstring already claims to follow, (b) it is the pedsim_ros
    lineage that social-robot-navigation papers benchmark against, (c) it
    ships an ObstacleForce, so the sidewalk boundary is handled INSIDE the
    model instead of by post-hoc snapping, and (d) it is ~4x faster than the
    torch backend (measured 20 ms vs 76 ms per 0.5 s SUMO step at 12 agents,
    10 sub-steps).

backend="socialforce"    (secondary, selectable)
    socialforce 0.2.3 -- github.com/svenkreiss/socialforce
    The circular-specification Helbing & Molnar 1995 potential
    V(b) = v0 * exp(-b/sigma) with the elliptical b, differentiated by torch
    autograd, leapfrog integration with built-in oversampling, and a binary
    200 deg field of view. Kept available so the author can report the
    circular and the elliptical published specifications side by side.

Neither package is modified. Their versions are recorded in the metrics as
`psf_package_version`.


WHAT THE PUBLISHED PARAMETERS ARE, VS THE IN-REPO HAND-ROLLED CONSTANTS
----------------------------------------------------------------------
The in-repo layer uses A_PED 4.5, B_PED 0.35, LAMBDA 0.30, TAU 0.5 with
    F_ij = A * exp((r_ij - d_ij)/B) * n_ij * (lambda + (1-lambda)(1+cos)/2)
This is the *circular* Helbing form with a continuous anisotropy weight.
NEITHER published package uses those four numbers, and only one of the four
even has a counterpart in the primary backend:

  in-repo             PySocialForce 1.1.2            socialforce 0.2.3
  ------------------  -----------------------------  ---------------------
  TAU    = 0.5 s      relaxation_time = 0.5 s        tau = 0.5 s
                      (same value, same role)        (same)
  A_PED  = 4.5        no counterpart. The repulsion  v0 = 2.1 m^2/s^2
   [m/s^2]            is not A*exp(): it is the      (a POTENTIAL depth, not
                      Moussaid directional force     an acceleration; the
                      with factor = 5.1 (dimension-  force is -grad V)
                      less gain) split into
                      force_velocity + force_angle
  B_PED  = 0.35 m     no counterpart. Range is set   sigma = 0.3 m
                      by B = gamma * ||D|| with      (fixed range of the
                      gamma = 0.35 -- numerically    exponential in b)
                      close to 0.35 but it is an
                      INTERACTION-VECTOR-scaled
                      range, not a fixed one
  LAMBDA = 0.30       lambda_importance = 2.0        FieldOfView(twophi=200)
   (anisotropy        (weight of the velocity        binary: 1.0 in view,
    weight in         difference inside the          0.5 out of view. Not a
    [lambda,1])       interaction direction, NOT     continuous cos weight
                      an FOV weight) + n = 2,        either.
                      n_prime = 3 angular shape
  -                   obstacle_force factor 10.0,    PedSpacePotential
                      sigma 0.2, threshold 3.0       u0 = 10, r = 0.2
  A_ROB = 6.0,        NO per-agent parameters. See   NO per-agent
  B_ROB = 0.45,       "ROBOT RADIUS" below.          parameters.
  R_ROB = 0.35
  R_PED = 0.30        agent_radius = 0.35, used      not used in ped-ped
                      ONLY by ObstacleForce
  dt = 0.5 s          step_width, upstream default   delta_t/oversampling,
  (== TAU: the        1.0 s -- ALSO >= tau. Set to   upstream default
   audit finding)     0.05 s here (10 sub-steps      0.4/10 = 0.04 s. Fine
                      per SUMO step) so dt << tau.   out of the box.

So the honest summary is: the published packages do NOT implement the
formula the in-repo docstring writes down, and their defaults are not the
in-repo constants. Only tau = 0.5 s is genuinely shared. Expect the two
levels to differ, and report the difference rather than treating this file
as a drop-in numerical replacement.

HOW BIG IS THE DIFFERENCE. Measured head-on repulsion magnitude [m/s^2] on
a pedestrian walking +x at 1.2 m/s with the robot closing head-on at
1.0 m/s, identical geometry through all three implementations:

    gap [m]   in-repo A_ROB term   pysocialforce 1.1.2   socialforce 0.2.3
      0.30            13.06                 4.35                0.00
      0.50             8.37                 3.91                4.46
      0.75             4.80                 3.43                1.36
      1.00             2.76                 3.00                0.55
      1.50             0.91                 2.31                0.10
      2.00             0.30                 1.77                0.02
      3.00             0.03                 1.04                0.00
      4.00             0.00                 0.61                0.00

Two structural differences, not just a gain mismatch:
  * NEAR FIELD. Inside ~0.8 m the in-repo term is 1.4-3x stronger than
    PySocialForce, because A_ROB * exp((R_ped+R_rob-d)/B_ROB) blows up as d
    approaches contact while the Moussaid force saturates at `factor`.
  * FAR FIELD. Beyond ~1.5 m PySocialForce is several times STRONGER,
    because its range B = gamma * ||D|| grows with the interaction vector,
    so the force decays slowly, whereas the in-repo term is essentially
    dead past 2 m.
  * socialforce 0.2.3 is the softest of the three, and its elliptical b
    degenerates at head-on contact range (the `in_sqrt` clamp flattens the
    potential, so the gradient -- and hence the force -- collapses to ~0 at
    a 0.3 m head-on gap). Treat its close-range behaviour with care; this
    is one more reason it is the secondary and not the primary backend.

CONSEQUENCE, observed end-to-end (map2_crossing, mixed flow 260, seed 1,
robot driving the first leg at 1.0 m/s, 180 s, identical demand):

    layer            median min gap [m]   ped_delay_s_mean   mean |lateral|
    off (striping)         1.258                 -                1.325
    sfm (in-repo)          1.056               1.24               1.510
    pysf (published)       0.735               0.51               1.073
    pysf, robot hidden     0.451               0.04               1.234

The published model, unmodified, yields SMALLER robot clearances and less
pedestrian delay than the author's hand-tuned reimplementation. That is a
result, not a defect: it is exactly the divergence the audit was looking
for, and it says the in-repo `sfm` numbers were partly produced by
hand-chosen gains (A_ROB = 6.0, B_ROB = 0.45) that no published package
uses. Report both levels.


ROBOT HANDLING (mirrors --robot-in-jps)
---------------------------------------
The robot is NOT visible to pedestrians by default, preserving the
benchmark's fairness rule that the robot does all the avoiding. With
`robot_in_layer=True` it is injected as a repulsive element:

  robot_as="agent" (default)
      an extra row in the model's state array at the robot's true pose, with
      its true velocity and a goal projected along its heading. This is what
      "the robot is an SFM agent" means in the published model.
      HONEST LIMITATION: neither published package supports per-agent radii
      in its pedestrian-pedestrian term (PySocialForce's Moussaid force is
      distance-based with no radius at all; socialforce's potential is a
      function of b only). So in this mode `robot_radius` does NOT scale the
      repulsion -- it is recorded, and used for the boundary clamp and the
      contact/personal-space bookkeeping only. The in-repo layer's separate
      A_ROB/B_ROB/R_ROB robot term is a non-published extension.

  robot_as="obstacle"  (pysocialforce backend only)
      the robot's footprint circle, sampled at `robot_radius`, is handed to
      the library's ObstacleForce alongside the sidewalk boundary. Here the
      radius IS a real physical quantity (a bigger circle produces a bigger
      wall force), at the cost of the robot being static-per-step rather
      than an anticipated moving agent.

As in the JuPedSim layer, the benchmark's own integration stays
AUTHORITATIVE for the recorded robot trajectory: the robot row is re-seeded
from the true pose every SUMO step, and the distance the model would have
moved it is reported as `psf_robot_track_err_*`.


METRIC COMPATIBILITY, stated explicitly because the two sibling layers
disagree with each other
------------------------------------------------------------------------
The ped_* definitions here follow `social_pedestrians.SocialForceLayer`
(the layer this one is meant to be compared against), NOT the JuPedSim one:
  * a pedestrian counts as "affected" once it comes within CAPTURE_R of the
    robot (JuPedSimLayer effectively requires PERSONAL_SPACE_R = 1.2 m, so
    its `ped_affected_n` is not comparable with either SFM layer);
  * ped_delay_s_mean is the integrated SPEED deficit inside 6 m of the
    robot, direction-agnostic (JuPedSimLayer uses distance-not-travelled
    over the whole capture);
  * ped_deflection_* and ped_personal_space_s_total match both siblings.
The HuNavSim social-work keys are emitted from the LIBRARY'S OWN force
terms (pysocialforce backend only; None for the torch backend).
"""
from __future__ import annotations

import logging
import math
import os

# Bubble geometry: byte-identical to social_pedestrians / jupedsim_pedestrians
# so that `sfm`, `jupedsim` and `pysf` differ only in the operational model.
CAPTURE_R = 12.0         # take over pedestrians within this range of the robot
RELEASE_R = 18.0         # hand back beyond this range
TARGET_LOOKAHEAD = 6.0   # how far ahead a captured pedestrian is steered [m]
PERSONAL_SPACE_R = 1.2   # pedestrian-side "robot is uncomfortably close" [m]
DELAY_ZONE_R = 6.0       # delay is only charged to the robot inside this [m]
SUBSTEP_DT = 0.05        # model integration timestep [s]; << tau = 0.5 s
OBSTACLE_R = 20.0        # sidewalk boundary handed to the model, around robot
CLAMP_SNAP_M = 1.0       # max snap-back distance onto the walkable surface

_BACKENDS = ("pysocialforce", "socialforce")
_ROBOT_MODES = ("agent", "obstacle")

_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pysocialforce_benchmark.toml")


def _import_pysocialforce():
    """Import PySocialForce without letting it hijack this process's logging.

    `pysocialforce.utils.logging` runs at import time and does

        logger = logging.getLogger("root")      # this IS the root logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(StreamHandler())      # DEBUG
        logger.addHandler(FileHandler("file.log"))

    Since Python 3.9 `getLogger("root")` returns the actual root logger, so a
    bare `import pysocialforce` switches the whole process to DEBUG and
    floods stdout with numba/matplotlib traces (hundreds of KB per run,
    which would bury the benchmark's own output) and opens `file.log` in the
    current working directory.

    This is a logging side effect, not part of the model, so it is undone
    here: records are globally muted below WARNING for the duration of the
    import (the flood is emitted by numba/matplotlib *while* pysocialforce is
    importing, so restoring afterwards alone is too late), and the handlers
    it installed on the root logger are removed afterwards. The library
    source itself is untouched.
    """
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    previously_disabled = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        import pysocialforce  # noqa: F401
    finally:
        logging.disable(previously_disabled)
        new = [h for h in root.handlers if h not in handlers]
        root.setLevel(level)
        root.handlers[:] = handlers
        for h in new:                       # close the stray file.log handle
            try:
                h.close()
            except Exception:
                pass
    return pysocialforce


def _package_version(name):
    try:
        import importlib.metadata as _md
        return _md.version(name)
    except Exception:
        return "unknown"


def _boundary_segments(geom, spacing):
    """Flatten a shapely (Multi)Polygon boundary into straight segments of at
    least `spacing` metres, ready for PySocialForce's EnvState.

    The length floor is not cosmetic. EnvState.obstacles resamples each
    segment with

        samples = int(norm(start - end) * resolution)
        line    = np.array(list(zip(np.linspace(...), np.linspace(...))))

    so a segment shorter than 1/resolution yields samples == 0, and
    `np.array(list(zip()))` is then a (0,) array rather than (0, 2).
    ObstacleForce later does `np.vstack(obstacles)`, which raises
    ValueError on the shape mismatch. A real OSM sidewalk boundary contains
    plenty of sub-decimetre segments, so this bites immediately. Resampling
    each ring at a uniform spacing avoids the degenerate case entirely
    instead of dropping boundary pieces (which would punch holes in the
    wall). Upstream is not patched.

    Pieces are made at least 2/resolution long rather than exactly
    1/resolution: at the exact floor, `int(0.25 * 4)` can evaluate to 0
    because interpolation leaves the length a few ulps short of 0.25. The
    factor of two puts the sample count at 2-3 per piece, safely clear of
    the cliff, and does not change the sampling density (linspace still
    spreads `resolution` points per metre).

    Returns a list of (x1, y1, x2, y2).
    """
    from shapely.geometry import LineString

    segs = []
    floor = 2.0 * spacing

    def _add(coords):
        if len(coords) < 2:
            return
        ls = LineString(coords)
        length = ls.length
        if length < floor:
            return                        # too short to sample; skip the ring
        k = max(1, int(length // floor))            # piece length >= 2*spacing
        pts = [ls.interpolate(i * length / k) for i in range(k + 1)]
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append((a.x, a.y, b.x, b.y))

    for g in list(getattr(geom, "geoms", [geom])):
        gt = g.geom_type
        if gt == "Polygon":
            _add(list(g.exterior.coords))
            for ring in g.interiors:
                _add(list(ring.coords))
        elif gt in ("LineString", "LinearRing"):
            _add(list(g.coords))
        elif gt in ("MultiPolygon", "GeometryCollection", "MultiLineString"):
            segs.extend(_boundary_segments(g, spacing))
    return segs


class PySocialForceLayer:
    """Drop-in sibling of SocialForceLayer / JuPedSimLayer backed by an
    unmodified published Social Force Model package."""

    def __init__(self, traci_mod, walk_union=None, walk_prep=None,
                 net_file=None, route_pts=None, backend="pysocialforce",
                 robot_radius=0.25, robot_in_layer=False, robot_as="agent",
                 robot_speed=1.0, seed=0, config_file=None,
                 substep_dt=SUBSTEP_DT, speed_calibration=True):
        self.traci = traci_mod
        self.union = walk_union
        self.uprep = walk_prep
        # pedestrian-steps, matching SocialForceLayer's semantics
        self.controlled_steps = 0
        self.active_steps = 0         # SUMO steps with >=1 controlled ped
        self.capture_events = 0
        self.robot_in_layer = bool(robot_in_layer)
        self.robot_radius = float(robot_radius)
        self.robot_speed = float(robot_speed)
        self.substep_dt = float(substep_dt)
        self.substeps = None
        self.speed_calibration = bool(speed_calibration)
        self.boundary_clamps = 0
        self._track_err = []
        self._done = []
        self.ctl = {}            # sumo pid -> bookkeeping dict
        self._order = []         # stable row order in the model state array
        # HuNavSim social-work accumulators, taken from the library's own
        # force terms (pysocialforce backend only)
        self.sf_on_robot = 0.0
        self.of_on_robot = 0.0
        self.sf_on_agents = 0.0
        self.robot_swork = 0.0   # back-compat alias of sf_on_robot

        if backend not in _BACKENDS:
            raise ValueError(f"unknown backend '{backend}'; "
                             f"choose from {list(_BACKENDS)}")
        if robot_as not in _ROBOT_MODES:
            raise ValueError(f"unknown robot_as '{robot_as}'; "
                             f"choose from {list(_ROBOT_MODES)}")
        self.backend = backend
        self.robot_as = robot_as
        if backend == "socialforce" and robot_as == "obstacle":
            raise ValueError("robot_as='obstacle' needs the pysocialforce "
                             "backend (socialforce has no obstacle force "
                             "over sampled circles in this layer)")

        self.config_file = config_file or _DEFAULT_CONFIG
        self._sim = None
        # cell the torch backend's wall point cloud was built for
        self._sim_space_at = None
        self._prev_acc = {}      # pid -> (ax, ay), socialforce leapfrog state

        if backend == "pysocialforce":
            self._psf = _import_pysocialforce()
            self.package = "pysocialforce"
            self.package_version = _package_version("pysocialforce")
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(
                    f"pysocialforce config not found: {self.config_file}")
        else:
            import socialforce as _sf
            import torch as _torch
            self._sf = _sf
            self._torch = _torch
            self.package = "socialforce"
            self.package_version = _package_version("socialforce")

        # Obstacle sampling density. Read from the same TOML the library will
        # read, so the segment length floor here and EnvState's resampler can
        # never disagree (see _boundary_segments).
        self.obstacle_resolution = 4.0
        if backend == "pysocialforce":
            try:
                import toml
                self.obstacle_resolution = float(
                    toml.load(self.config_file).get("resolution", 10.0))
            except Exception:
                pass
        self._spacing = 1.0 / max(self.obstacle_resolution, 1e-6)

        # --- sidewalk boundary, precomputed once
        self._segs = []
        self._seg_mid = []
        if walk_union is not None:
            try:
                self._segs = _boundary_segments(walk_union.boundary,
                                                self._spacing)
                self._seg_mid = [((a + c) * 0.5, (b + d) * 0.5)
                                 for (a, b, c, d) in self._segs]
            except Exception:
                self._segs, self._seg_mid = [], []

        print(f"reactive-peds pysf: {self.package} {self.package_version}, "
              f"{len(self._segs)} sidewalk boundary segments, "
              f"integration dt={self.substep_dt:g} s"
              + (f", robot injected as a repulsive {self.robot_as} "
                 f"(r={self.robot_radius:.2f} m)"
                 if self.robot_in_layer else
                 ", robot NOT visible to pedestrians"))

    # ------------------------------------------------------------------ util
    def _inside(self, pt):
        if self.uprep is None:
            return True
        from shapely.geometry import Point
        return self.uprep.covers(Point(pt[0], pt[1]))

    def _clamp(self, pt, fallback):
        """Keep a pedestrian on the walkable surface.

        The primary boundary handling is the LIBRARY's own wall force (the
        sidewalk boundary is fed to it every step); this is only a
        last-resort guard, and how often it fires is reported as
        `psf_boundary_clamps` so the reader can see whether the wall force
        was doing its job.
        """
        if self.uprep is None or self._inside(pt):
            return pt
        self.boundary_clamps += 1
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        P = Point(pt[0], pt[1])
        try:
            q = nearest_points(self.union, P)[0]
            if q.distance(P) <= CLAMP_SNAP_M:
                return (q.x, q.y)
        except Exception:
            pass
        return fallback

    def _on_internal(self, pid):
        """Junction cores are excluded, exactly as in the two sibling layers:
        remapping a person onto a junction-internal lane can corrupt SUMO's
        person state machine."""
        try:
            return self.traci.person.getRoadID(pid).startswith(":")
        except Exception:
            return True

    def _is_walking(self, pid):
        try:
            return self.traci.person.getStage(pid).type == 2   # 2 = walking
        except Exception:
            return False

    def _local_obstacles(self, rx, ry):
        """Sidewalk boundary near the robot, in PySocialForce's EnvState
        format (startx, endx, starty, endy)."""
        out = []
        r2 = OBSTACLE_R * OBSTACLE_R
        res = self.obstacle_resolution
        for (x1, y1, x2, y2), (mx, my) in zip(self._segs, self._seg_mid):
            if (mx - rx) ** 2 + (my - ry) ** 2 > r2:
                continue
            # hard guard: EnvState would emit a malformed (0,)-shaped array
            # for any segment the resampler rounds down to zero points, and
            # ObstacleForce's np.vstack would then raise. Never hand one over.
            if int(math.hypot(x2 - x1, y2 - y1) * res) < 1:
                continue
            out.append((x1, x2, y1, y2))
        return out

    def _robot_perturbation(self, mx, my, rx, ry, robot_v, dt):
        """How far the published model displaced the robot in one step.

        Measured against the robot's own DEAD-RECKONED pose (rx + vx*dt,
        ry + vy*dt), not against its pose at the start of the step: the
        latter would report ~|v|*dt (about 0.5 m here) even under a perfectly
        undisturbed robot, which is the same trap documented in
        JuPedSimLayer. This isolates the genuine model-induced residual.

        Structural caveat, worth knowing before reading the number: when the
        robot is stationary its goal is set to its own position, and
        PySocialForce's PedState.step zeroes the velocity of any agent within
        0.5 m of its goal. So a stopped robot cannot be displaced by the
        model and contributes exactly 0.0 to this metric.
        """
        rvx, rvy = robot_v
        return math.hypot(mx - (rx + rvx * dt), my - (ry + rvy * dt))

    def _robot_circle(self, rx, ry):
        """Robot footprint as obstacle chords, so `robot_radius` is a real
        physical quantity under robot_as='obstacle'.

        The vertex count is capped so that every chord is at least
        1/resolution long, for the same EnvState reason as the sidewalk
        boundary: chord = 2*R*sin(pi/n) >= spacing.
        """
        R = max(self.robot_radius, 1e-3)
        ratio = min(1.0, (2.0 * self._spacing) / (2.0 * R))
        n = int(math.pi / math.asin(ratio)) if ratio < 1.0 else 3
        n = max(3, min(16, n))
        pts = [(rx + R * math.cos(2 * math.pi * i / n),
                ry + R * math.sin(2 * math.pi * i / n))
               for i in range(n + 1)]
        return [(a[0], b[0], a[1], b[1]) for a, b in zip(pts[:-1], pts[1:])]

    # ------------------------------------------------------------------ main
    def step(self, robot_xy, robot_v, dt):
        t = self.traci
        rx, ry = robot_xy
        if self.substeps is None:
            self.substeps = max(1, int(round(dt / self.substep_dt)))

        try:
            pids = t.person.getIDList()
        except Exception:
            return
        alive = set(pids)

        # --- release: gone, or out of the bubble
        for pid in list(self.ctl):
            st = self.ctl[pid]
            gone = pid not in alive
            far = math.hypot(st["pos"][0] - rx, st["pos"][1] - ry) > RELEASE_R
            if gone or far or not self._is_walking(pid):
                self._release(pid)

        # --- capture: walking pedestrians inside the bubble
        for pid in pids:
            if pid == "robot0" or pid in self.ctl or pid.startswith("stand_"):
                continue                     # statics are never captured: it
            try:                             # would give them a walking speed
                px, py = t.person.getPosition(pid)
            except Exception:
                continue
            if math.hypot(px - rx, py - ry) >= CAPTURE_R:
                continue
            if self._on_internal(pid) or not self._inside((px, py)):
                continue
            if not self._is_walking(pid):
                continue
            self._capture(pid, (px, py))

        if not self.ctl:
            return
        # `controlled_steps` counts PEDESTRIAN-steps, matching
        # social_pedestrians.SocialForceLayer, because `sfm` is the level this
        # one is primarily compared against and benchmark_runner logs it as
        # `sfm_controlled_steps`. NOTE for whoever reads those columns:
        # JuPedSimLayer counts SUMO steps instead (one increment per step, not
        # per pedestrian), so its `sfm_controlled_steps` is on a different
        # scale from both SFM layers. The step count is also exposed
        # separately as `psf_layer_active_steps`.
        self.controlled_steps += len(self.ctl)
        self.active_steps += 1
        self._order = list(self.ctl)

        # --- advance the PUBLISHED model over one SUMO step
        if self.backend == "pysocialforce":
            new = self._step_pysocialforce((rx, ry), robot_v, dt)
        else:
            new = self._step_socialforce((rx, ry), robot_v, dt)
        if new is None:
            return

        # --- write positions back into SUMO, accumulate pedestrian-side cost
        for i, pid in enumerate(self._order):
            st = self.ctl.get(pid)
            if st is None:
                continue
            nx, ny, vx, vy = new[i]
            nx, ny = self._clamp((nx, ny), st["pos"])
            ex, ey = st["edir"]
            sx0, sy0 = st["start"]
            dr = math.hypot(nx - rx, ny - ry)
            spd = math.hypot(vx, vy)
            if dr < CAPTURE_R:
                st["nr"] = True
            if dr < DELAY_ZONE_R:
                st["delay"] += max(0.0, st["vdes"] - spd) * dt
            st["defl"] = max(st["defl"],
                             abs((nx - sx0) * (-ey) + (ny - sy0) * ex))
            if dr < PERSONAL_SPACE_R:
                st["ps"] += dt
            st["pos"] = (nx, ny)
            st["vel"] = (vx, vy)
            try:
                t.person.moveToXY(pid, "", nx, ny, keepRoute=2)
            except Exception:
                self._release(pid)

    # ------------------------------------------------------- backend: numpy
    def _step_pysocialforce(self, robot_xy, robot_v, dt):
        """One SUMO step of PySocialForce 1.1.2.

        The library owns the physics: we only assemble its state array
        (x, y, vx, vy, goal_x, goal_y), hand it the local sidewalk boundary,
        and call `Simulator.step(n_substeps)`.
        """
        import numpy as np
        rx, ry = robot_xy
        n = len(self._order)
        rows, v0 = [], []
        for pid in self._order:
            st = self.ctl[pid]
            px, py = st["pos"]
            vx, vy = st["vel"]
            ex, ey = st["edir"]
            rows.append([px, py, vx, vy,
                         px + ex * TARGET_LOOKAHEAD,
                         py + ey * TARGET_LOOKAHEAD])
            v0.append(st["vdes"])

        robot_row = None
        if self.robot_in_layer and self.robot_as == "agent":
            rvx, rvy = robot_v
            rsp = math.hypot(rvx, rvy)
            if rsp > 1e-3:
                gx = rx + rvx / rsp * TARGET_LOOKAHEAD
                gy = ry + rvy / rsp * TARGET_LOOKAHEAD
            else:
                gx, gy = rx, ry
            rows.append([rx, ry, rvx, rvy, gx, gy])
            v0.append(max(self.robot_speed, rsp, 0.1))
            robot_row = n

        state = np.asarray(rows, dtype=float)
        # PySocialForce's DesiredForce accelerates toward
        # `max_speeds = max_speed_multiplier * initial_speeds` (1.3 * v0) and
        # caps there too, so the free-walking speed is 1.3x the seeded speed.
        # `speed_calibration` divides the seed by the same 1.3 so that the
        # realised free-walking speed equals SUMO's desired speed and the
        # `pysf` level stays comparable with `sfm` / `jupedsim`. This is a
        # choice of INPUT, not a change to the library's maths; set
        # speed_calibration=False to run the raw published behaviour.
        mult = 1.3 if self.speed_calibration else 1.0
        init = np.asarray(v0, dtype=float) / mult

        obstacles = self._local_obstacles(rx, ry)
        if self.robot_in_layer and self.robot_as == "obstacle":
            obstacles = obstacles + self._robot_circle(rx, ry)

        if self._sim is None:
            self._sim = self._psf.Simulator(
                state.copy(), obstacles=obstacles,
                config_file=self.config_file)
            self._sim.peds.initial_speeds = init
            self._sim.peds.state = state.copy()
        else:
            # EnvState/PedState expose these as plain attributes; reassigning
            # them is how a fixed-size Simulator is reused for a changing
            # population. `initial_speeds` is latched on first assignment by
            # PedState's state setter, so it is set explicitly first --
            # otherwise max_speeds keeps the previous population's length.
            self._sim.env.obstacles = obstacles
            self._sim.peds.initial_speeds = init
            self._sim.peds.state = state.copy()

        # PedState.capped_velocity divides by the desired speed, which is
        # exactly 0 for an agent the model has brought to a stop; the library
        # then masks the resulting inf away itself (`factor[speeds == 0] = 0`).
        # Silencing the warning only stops the console noise.
        with np.errstate(divide="ignore", invalid="ignore"):
            self._sim.step(self.substeps)
        out = self._sim.peds.state.copy()

        # --- HuNavSim social work, read off the LIBRARY's own force terms
        try:
            forces = {type(f).__name__: f.get_force()
                      for f in self._sim.forces}
            fs = forces.get("SocialForce")
            fo = forces.get("ObstacleForce")
            if fs is not None:
                ped_rows = slice(0, n)
                self.sf_on_agents += float(
                    np.linalg.norm(fs[ped_rows], axis=1).sum()) * dt
                if robot_row is not None:
                    self.sf_on_robot += float(
                        np.linalg.norm(fs[robot_row])) * dt
            if fo is not None and robot_row is not None:
                self.of_on_robot += float(np.linalg.norm(fo[robot_row])) * dt
        except Exception:
            pass
        self.robot_swork = self.sf_on_robot

        # --- robot proxy: the benchmark's own integration stays
        # authoritative, so record how far the model would have moved it.
        if robot_row is not None:
            self._track_err.append(self._robot_perturbation(
                out[robot_row, 0], out[robot_row, 1], rx, ry, robot_v, dt))

        # PedState.state/groups setters append to unbounded history lists;
        # trimming keeps a long run from growing without bound. Purely
        # memory hygiene: nothing reads the history here.
        self._sim.peds.ped_states = self._sim.peds.ped_states[-1:]
        self._sim.peds.group_states = self._sim.peds.group_states[-1:]

        return [(out[i, 0], out[i, 1], out[i, 2], out[i, 3])
                for i in range(n)]

    # ------------------------------------------------------- backend: torch
    def _step_socialforce(self, robot_xy, robot_v, dt):
        """One SUMO step of socialforce 0.2.3 (svenkreiss).

        `Simulator.forward` is a pure function of the state, so a changing
        population needs no special handling: the full 10-column state
        (x, y, vx, vy, ax, ay, gx, gy, tau, v0) is rebuilt every step, which
        also preserves each pedestrian's preferred speed instead of letting
        `normalize_state` re-derive it from the (robot-slowed) current speed.
        """
        import numpy as np
        torch = self._torch
        rx, ry = robot_xy
        n = len(self._order)
        rows = []
        for pid in self._order:
            st = self.ctl[pid]
            px, py = st["pos"]
            vx, vy = st["vel"]
            ax, ay = self._prev_acc.get(pid, (0.0, 0.0))
            ex, ey = st["edir"]
            rows.append([px, py, vx, vy, ax, ay,
                         px + ex * TARGET_LOOKAHEAD,
                         py + ey * TARGET_LOOKAHEAD, 0.5, st["vdes"]])
        robot_row = None
        if self.robot_in_layer:
            rvx, rvy = robot_v
            rsp = math.hypot(rvx, rvy)
            if rsp > 1e-3:
                gx, gy = (rx + rvx / rsp * TARGET_LOOKAHEAD,
                          ry + rvy / rsp * TARGET_LOOKAHEAD)
            else:
                gx, gy = rx, ry
            rows.append([rx, ry, rvx, rvy, 0.0, 0.0, gx, gy, 0.5,
                         max(self.robot_speed, rsp, 0.1)])
            robot_row = n

        if self._sim is None or self._sim_space_at != self._space_key(rx, ry):
            space = self._space_tensors(rx, ry)
            self._sim = self._sf.Simulator(
                ped_space=(self._sf.potentials.PedSpacePotential(space)
                           if space else None),
                delta_t=dt, tau=0.5,
                oversampling=max(1, int(round(dt / self.substep_dt))))
            self._sim_space_at = self._space_key(rx, ry)

        with torch.no_grad():
            out = self._sim(torch.tensor(np.asarray(rows, dtype=np.float32)))
        out = out.detach().numpy()

        for i, pid in enumerate(self._order):
            self._prev_acc[pid] = (float(out[i, 4]), float(out[i, 5]))
        if robot_row is not None:
            self._track_err.append(self._robot_perturbation(
                out[robot_row, 0], out[robot_row, 1], rx, ry, robot_v, dt))
        return [(out[i, 0], out[i, 1], out[i, 2], out[i, 3])
                for i in range(n)]

    @staticmethod
    def _space_key(rx, ry):
        """Rebuild the boundary point cloud only when the robot has moved a
        whole cell, so the torch Simulator is not reconstructed every step."""
        return (int(rx // 10.0), int(ry // 10.0))

    def _space_tensors(self, rx, ry):
        import numpy as np
        torch = self._torch
        out = []
        res = 2.0
        for (x1, x2, y1, y2) in self._local_obstacles(rx, ry):
            L = math.hypot(x2 - x1, y2 - y1)
            k = max(2, int(L * res))
            out.append(torch.tensor(
                np.stack([np.linspace(x1, x2, k),
                          np.linspace(y1, y2, k)], axis=1),
                dtype=torch.float32))
        return out

    # ------------------------------------------------------------- internals
    def _capture(self, pid, pos):
        t = self.traci
        try:
            ang = math.radians(t.person.getAngle(pid))
            ex, ey = math.sin(ang), math.cos(ang)       # SUMO angle convention
            sp = float(t.person.getSpeed(pid))
            vdes = max(0.6, float(t.person.getMaxSpeed(pid)))
        except Exception:
            return
        self.ctl[pid] = {"pos": pos, "start": pos, "edir": (ex, ey),
                         "vel": (sp * ex, sp * ey), "vdes": vdes,
                         "delay": 0.0, "defl": 0.0, "ps": 0.0, "nr": False}
        self.capture_events += 1

    def _release(self, pid):
        st = self.ctl.pop(pid, None)
        self._prev_acc.pop(pid, None)
        if st is not None and st.get("nr"):
            self._done.append({"delay": st["delay"], "defl": st["defl"],
                               "ps": st["ps"]})

    # --------------------------------------------------------------- metrics
    def ped_metrics(self):
        allst = self._done + [{"delay": st["delay"], "defl": st["defl"],
                               "ps": st["ps"]}
                              for st in self.ctl.values() if st.get("nr")]
        n = len(allst)
        out = {
            "psf_backend": self.package,
            "psf_package_version": self.package_version,
            "psf_robot_in_sim": bool(self.robot_in_layer),
            "psf_robot_mode": self.robot_as if self.robot_in_layer else None,
            "psf_robot_radius_m": round(self.robot_radius, 3),
            "psf_substeps_per_step": self.substeps,
            "psf_substep_dt_s": self.substep_dt,
            "psf_speed_calibration": bool(self.speed_calibration),
            "psf_boundary_clamps": self.boundary_clamps,
            "psf_capture_events": self.capture_events,
            "psf_layer_active_steps": self.active_steps,
            "psf_obstacle_resolution_pts_per_m": self.obstacle_resolution,
        }
        if self._track_err:
            errs = sorted(self._track_err)
            out["psf_robot_track_err_mean_m"] = round(
                sum(errs) / len(errs), 3)
            out["psf_robot_track_err_p95_m"] = round(
                errs[min(len(errs) - 1, int(0.95 * len(errs)))], 3)
            out["psf_robot_track_err_max_m"] = round(errs[-1], 3)
        else:
            out["psf_robot_track_err_mean_m"] = None
            out["psf_robot_track_err_p95_m"] = None
            out["psf_robot_track_err_max_m"] = None

        if self.backend == "pysocialforce":
            out.update({
                "social_force_on_agents": round(self.sf_on_agents, 2),
                "social_force_on_robot": round(self.sf_on_robot, 2),
                "obstacle_force_on_robot": round(self.of_on_robot, 2),
                "social_work": round(self.sf_on_agents + self.sf_on_robot
                                     + self.of_on_robot, 2)})
        else:
            out.update({"social_force_on_agents": None,
                        "social_force_on_robot": None,
                        "obstacle_force_on_robot": None,
                        "social_work": None})

        if n == 0:
            out.update({"ped_affected_n": 0, "ped_delay_s_mean": 0.0,
                        "ped_deflection_m_mean": 0.0,
                        "ped_deflection_m_max": 0.0,
                        "ped_personal_space_s_total": 0.0})
            return out
        out.update({
            "ped_affected_n": n,
            "ped_delay_s_mean": round(sum(a["delay"] for a in allst) / n, 2),
            "ped_deflection_m_mean": round(
                sum(a["defl"] for a in allst) / n, 3),
            "ped_deflection_m_max": round(max(a["defl"] for a in allst), 3),
            "ped_personal_space_s_total": round(
                sum(a["ps"] for a in allst), 1),
        })
        return out
