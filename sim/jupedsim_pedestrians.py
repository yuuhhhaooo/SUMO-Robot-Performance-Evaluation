#!/usr/bin/env python3
"""Reactive pedestrians driven by JuPedSim inside an interaction bubble.

Same architecture as `social_pedestrians.SocialForceLayer`, and deliberately
the same capture/release geometry, so the two reactive layers are directly
comparable as levels of one experimental factor:

    --reactive-peds off   striping only (legacy, bit-compatible)
    --reactive-peds sfm   the in-repo Helbing-Molnar Social Force Model
    --reactive-peds jupedsim   JuPedSim (Juelich Pedestrian Simulator)

SUMO keeps macroscopic demand and long-range routing; inside a bubble around
the robot, pedestrian motion is taken over by JuPedSim's operational model and
written back with moveToXY. Pedestrians are released to SUMO's striping model
once they leave the bubble.

Robot visibility is OPT-IN (`--robot-in-jps`). By default the robot is NOT a
JuPedSim agent, which preserves the benchmark's fairness rule: pedestrians do
not see the robot and the robot does all of the avoiding. With the flag, the
robot is injected as a JuPedSim agent on a *direct steering stage*, so its
radius becomes a real physical quantity that pedestrians deflect around.

    IMPORTANT CONFOUND, recorded per run rather than hidden. A direct steering
    stage bypasses JuPedSim's strategic and tactical levels but NOT its
    operational level, so JuPedSim also applies collision avoidance to the
    robot: the realised robot position is the planner's command *plus* a
    JuPedSim correction. The layer therefore steers the robot toward the
    command and reports the tracking error (`jps_robot_track_err_*`). The
    benchmark's own integration remains authoritative for the recorded robot
    trajectory; the JuPedSim agent exists so that pedestrians have something
    to react to.

Two hard constraints of the JuPedSim API shape this file:
  * the accessible area must be a SINGLE CONNECTED polygon -- the walkable
    union of an OSM import generally is not, so it is unioned with the route
    corridor (the same trick benchmark_runner's strict-sidewalk layer uses)
    and the connected component containing the route is kept;
  * every steering target must lie INSIDE the accessible area, or
    Simulation.iterate() raises -- all targets are projected onto an eroded
    copy of the polygon before use.
"""
from __future__ import annotations

import math

# Bubble geometry: identical to social_pedestrians so that `sfm` and
# `jupedsim` differ only in the operational model, not in who is controlled.
CAPTURE_R = 12.0        # take over pedestrians within this range of the robot
RELEASE_R = 18.0        # hand back beyond this range
JPS_DT = 0.01           # JuPedSim internal timestep [s] (its documented value)
TARGET_LOOKAHEAD = 6.0  # how far ahead a captured pedestrian is steered [m]
INSET = 0.25            # erosion applied before projecting targets inside
PERSONAL_SPACE_R = 1.2  # pedestrian-side "robot is uncomfortably close" [m]
ROBOT_TRACK_SPEED = 3.0  # desired speed of the robot's JuPedSim proxy [m/s].
                         # Well above HARD_SPEED_CAP (1.6) so the proxy can
                         # always reach the robot's true position within one
                         # SUMO step; it is a tracking gain, not a robot limit.

_MODELS = {
    "collision_free_speed": "CollisionFreeSpeedModel",
    "social_force": "SocialForceModel",
    "anticipation_velocity": "AnticipationVelocityModel",
}


class JuPedSimLayer:
    """Drop-in sibling of SocialForceLayer backed by JuPedSim."""

    def __init__(self, traci_mod, walk_union=None, walk_prep=None,
                 net_file=None, route_pts=None, model="collision_free_speed",
                 robot_radius=0.25, robot_in_jps=False, robot_speed=1.0,
                 seed=0):
        import jupedsim as jps
        from shapely.geometry import LineString, MultiPolygon
        from shapely.ops import unary_union

        self.traci = traci_mod
        self.jps = jps
        self.controlled_steps = 0
        self.capture_events = 0
        self.robot_in_jps = bool(robot_in_jps)
        self.robot_radius = float(robot_radius)
        self.geometry_area_kept = 1.0
        self.iterations_per_step = None
        self._track_err = []          # robot commanded-vs-realised divergence
        self._done = []               # released pedestrians' accumulated cost
        self.ctl = {}                 # sumo pid -> bookkeeping dict
        self._jps_of = {}             # sumo pid -> jupedsim agent id
        self._robot_agent = None
        # junction no-control zone, REUSED from the SFM layer (see
        # social_pedestrians.build_junction_zone): never steer or write a
        # controlled pedestrian into junction internals -- doing so can
        # crash SUMO's person state machine (the sfm layer documents and
        # guards this; this layer previously did not).
        self.zone = self.zprep = None
        if net_file is not None:
            try:
                from social_pedestrians import build_junction_zone
                self.zone, self.zprep = build_junction_zone(net_file)
            except Exception as _ze:
                self.zone = self.zprep = None
                import sys as _sys
                print("jupedsim layer: junction-zone build FAILED "
                      f"({type(_ze).__name__}: {_ze}) -- running WITHOUT "
                      "the junction guard; long episodes may crash SUMO",
                      file=_sys.stderr)

        if model not in _MODELS:
            raise ValueError(f"unknown jupedsim model '{model}'; "
                             f"choose from {sorted(_MODELS)}")
        self.model_name = model

        if walk_union is None:
            raise ValueError("JuPedSimLayer needs the walkable surface")

        # --- accessible area: connected, and containing the robot's route
        geo = walk_union
        if route_pts and len(route_pts) >= 2:
            geo = unary_union(
                [geo, LineString([tuple(p) for p in route_pts]).buffer(1.2)])
        if isinstance(geo, MultiPolygon):
            parts = sorted(geo.geoms, key=lambda g: -g.area)
            total = sum(p.area for p in parts)
            pick = parts[0]
            if route_pts:
                start = LineString([tuple(p) for p in route_pts[:2]])
                for p in parts:                       # prefer the route's own
                    if p.intersects(start):           # component
                        pick = p
                        break
            self.geometry_area_kept = pick.area / max(total, 1e-9)
            geo = pick
        self.area = geo
        self.inner = geo.buffer(-INSET)
        if self.inner.is_empty:
            self.inner = geo

        self.sim = jps.Simulation(
            model=getattr(jps, _MODELS[model])(), geometry=geo, dt=JPS_DT)
        self._steer = self.sim.add_direct_steering_stage()
        self._journey = self.sim.add_journey(
            jps.JourneyDescription([self._steer]))
        print(f"reactive-peds jupedsim: {model}, accessible area "
              f"{geo.area:.0f} m^2 "
              f"({100.0 * self.geometry_area_kept:.1f}% of the walkable union)"
              + (f", robot injected as an agent (r={self.robot_radius:.2f} m)"
                 if self.robot_in_jps else ", robot NOT visible to pedestrians"))

    # ------------------------------------------------------------------ util
    def _clamp(self, pt):
        """Project a point onto the accessible area (targets must be inside)."""
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        P = Point(pt)
        if self.inner.contains(P):
            return (float(pt[0]), float(pt[1]))
        q = nearest_points(self.inner, P)[0]
        return (q.x, q.y)

    def _inside(self, pt):
        from shapely.geometry import Point
        return self.inner.contains(Point(pt))

    def _agent_params(self, pos, radius, speed):
        jps = self.jps
        cls = getattr(jps, _MODELS[self.model_name] + "AgentParameters")
        kw = dict(position=pos, journey_id=self._journey,
                  stage_id=self._steer, radius=float(radius))
        try:
            return cls(desired_speed=float(speed), **kw)
        except TypeError:                    # older builds name it v0
            return cls(v0=float(speed), **kw)

    def _in_zone(self, x, y):
        if self.zprep is None:
            return False
        P = getattr(self, "_Point", None)
        if P is None:
            from shapely.geometry import Point as P
            self._Point = P
        return self.zprep.covers(P(x, y))

    def _is_walking(self, pid):
        try:
            stg = self.traci.person.getStage(pid, 0)
            return getattr(stg, "type", 2) == 2      # 2 = walking
        except self.traci.exceptions.TraCIException:
            return False

    def _on_internal(self, pid):
        """Junction cores are excluded, exactly as in the SFM layer: remapping
        a person onto a junction-internal lane can corrupt SUMO's person state
        machine."""
        try:
            return self.traci.person.getRoadID(pid).startswith(":")
        except Exception:
            return True

    def _is_walking(self, pid):
        try:
            return self.traci.person.getSpeed(pid) is not None and \
                self.traci.person.getStage(pid).type != 0
        except Exception:
            return False

    # ------------------------------------------------------------------ main
    def step(self, robot_xy, robot_v, dt):
        t = self.traci
        rx, ry = robot_xy
        if self.iterations_per_step is None:
            self.iterations_per_step = max(1, int(round(dt / JPS_DT)))

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
            if gone or not self._is_walking(pid) \
                    or (far and not self._on_internal(pid)):
                self._release(pid)

        # --- capture: walking pedestrians inside the bubble
        for pid in pids:
            if pid == "robot0" or pid in self.ctl or pid.startswith("stand_"):
                continue                      # statics are never captured:
            try:                              # it would give them a walk speed
                px, py = t.person.getPosition(pid)
            except Exception:
                continue
            if math.hypot(px - rx, py - ry) >= CAPTURE_R:
                continue
            if self._on_internal(pid) or not self._inside((px, py)) \
                    or self._in_zone(px, py) \
                    or not self._is_walking(pid):
                continue
            self._capture(pid, (px, py))

        if self.ctl or self._robot_agent is not None:
            self.controlled_steps += 1

        # --- robot agent (opt-in)
        if self.robot_in_jps:
            self._sync_robot((rx, ry), robot_v, dt)

        # --- steer every controlled pedestrian along its frozen heading
        for pid, st in self.ctl.items():
            aid = self._jps_of.get(pid)
            if aid is None:
                continue
            try:
                a = self.sim.agent(aid)
            except Exception:
                continue
            ex, ey = st["edir"]
            px, py = a.position
            tgt = self._clamp((px + ex * TARGET_LOOKAHEAD,
                               py + ey * TARGET_LOOKAHEAD))
            if self._in_zone(*tgt) or self._in_zone(px, py):
                tgt = (px, py)   # hold at the junction boundary (SFM parity)
            a.target = tgt

        # --- advance JuPedSim over one SUMO step
        try:
            self.sim.iterate(self.iterations_per_step)
        except Exception as exc:
            # never let the pedestrian layer take down an episode
            print(f"jupedsim: iterate failed ({type(exc).__name__}: {exc}); "
                  f"releasing all controlled pedestrians")
            for pid in list(self.ctl):
                self._release(pid)
            return

        # --- robot proxy tracking error, measured AFTER the JuPedSim step.
        # Measuring before it would just report how far the robot moved during
        # the last SUMO step (~0.5 m) even under perfect tracking; measuring
        # after isolates the genuine residual, i.e. JuPedSim's own collision
        # avoidance pushing the proxy off the commanded pose.
        if self.robot_in_jps and self._robot_agent is not None:
            try:
                ax, ay = self.sim.agent(self._robot_agent).position
                self._track_err.append(math.hypot(ax - rx, ay - ry))
            except Exception:
                self._robot_agent = None

        # --- write positions back into SUMO and accumulate ped-side cost
        for pid, st in list(self.ctl.items()):
            aid = self._jps_of.get(pid)
            try:
                a = self.sim.agent(aid)
                nx, ny = a.position
            except Exception:
                self._release(pid)
                continue
            ox, oy = st["pos"]
            if self._in_zone(nx, ny):
                # junction no-control zone (SFM parity): hold at the
                # boundary -- never map a remote person onto junction
                # internals; the agent's target is pinned above so it
                # stops here too
                nx, ny = ox, oy
            moved = math.hypot(nx - ox, ny - oy)
            st["pos"] = (nx, ny)
            # delay = time not spent making progress at the desired speed
            st["delay"] += max(0.0, st["vdes"] * dt - moved)
            # deflection = lateral offset from the undisturbed walking line
            ex, ey = st["edir"]
            sx0, sy0 = st["start"]
            st["defl"] = max(st["defl"],
                             abs((nx - sx0) * (-ey) + (ny - sy0) * ex))
            if math.hypot(nx - rx, ny - ry) < PERSONAL_SPACE_R:
                st["ps"] += dt
                st["nr"] = True
            if math.hypot(nx - rx, ny - ry) < CAPTURE_R:
                st["nr"] = st.get("nr", False)
            try:
                t.person.moveToXY(pid, "", nx, ny, keepRoute=2)
            except Exception:
                self._release(pid)

    # ------------------------------------------------------------- internals
    def _capture(self, pid, pos):
        t = self.traci
        try:
            ang = math.radians(t.person.getAngle(pid))
            ex, ey = math.sin(ang), math.cos(ang)      # SUMO angle convention
            vdes = max(0.3, float(t.person.getSpeed(pid)) or 0.9)
        except Exception:
            return
        try:
            aid = self.sim.add_agent(self._agent_params(pos, 0.2, vdes))
        except Exception:
            return                                     # e.g. overlapping spawn
        self._jps_of[pid] = aid
        self.ctl[pid] = {"pos": pos, "start": pos, "edir": (ex, ey),
                         "vdes": vdes, "delay": 0.0, "defl": 0.0,
                         "ps": 0.0, "nr": False}
        self.capture_events += 1

    def _release(self, pid):
        st = self.ctl.pop(pid, None)
        aid = self._jps_of.pop(pid, None)
        if aid is not None:
            try:
                self.sim.mark_agent_for_removal(aid)
            except Exception:
                pass
        if st is not None and st.get("nr"):
            self._done.append({"delay": st["delay"], "defl": st["defl"],
                               "ps": st["ps"]})

    def _sync_robot(self, robot_xy, robot_v, dt):
        """Keep a JuPedSim agent on the robot so pedestrians can see it.

        The benchmark's own integration stays authoritative for the recorded
        trajectory; this agent is steered toward the commanded next pose and
        the residual is reported as the tracking error.
        """
        rx, ry = robot_xy
        vx, vy = robot_v
        # Target the robot's ACTUAL position, not a point ahead of it. A
        # direct-steered agent walks to its target and stops there, so any
        # lead distance becomes the steady-state tracking error one-for-one
        # (measured: a 1 m lead gives exactly 1.000 m of lag). Aiming at the
        # true position with a desired speed well above the robot's makes the
        # agent sit on the robot -- measured mean error 0.000 m on a straight
        # corridor -- so the only residual left is JuPedSim's own collision
        # avoidance, which is exactly the confound we want to quantify.
        target = self._clamp((rx, ry))
        if self._robot_agent is None:
            if not self._inside((rx, ry)):
                return
            try:
                self._robot_agent = self.sim.add_agent(
                    self._agent_params((rx, ry), self.robot_radius,
                                       ROBOT_TRACK_SPEED))
            except Exception:
                self._robot_agent = None
                return
        try:
            a = self.sim.agent(self._robot_agent)
        except Exception:
            self._robot_agent = None
            return
        a.target = target

    # -------------------------------------------------------------- metrics
    def ped_metrics(self):
        allst = self._done + [{"delay": st["delay"], "defl": st["defl"],
                               "ps": st["ps"]}
                              for st in self.ctl.values() if st.get("nr")]
        n = len(allst)
        out = {
            "jps_model": self.model_name,
            "jps_robot_in_sim": bool(self.robot_in_jps),
            "jps_robot_radius_m": round(self.robot_radius, 3),
            "jps_area_kept_frac": round(self.geometry_area_kept, 4),
            "jps_iterations_per_step": self.iterations_per_step,
        }
        if self._track_err:
            errs = sorted(self._track_err)
            out["jps_robot_track_err_mean_m"] = round(
                sum(errs) / len(errs), 3)
            out["jps_robot_track_err_p95_m"] = round(
                errs[min(len(errs) - 1, int(0.95 * len(errs)))], 3)
            out["jps_robot_track_err_max_m"] = round(errs[-1], 3)
        else:
            out["jps_robot_track_err_mean_m"] = None
            out["jps_robot_track_err_p95_m"] = None
            out["jps_robot_track_err_max_m"] = None
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
