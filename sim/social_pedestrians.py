#!/usr/bin/env python3
"""Reactive pedestrians via a Social Force Model interaction bubble.

Supervisor-mandated design (feedback 2026-08): SUMO keeps macroscopic demand
and long-range walking; inside an interaction bubble around the robot,
pedestrian motion is taken over by a Social Force Model (Helbing & Molnar
1995) that includes the ROBOT as a repulsive agent -- so pedestrians react
to the robot (give way, deflect, re-merge) instead of walking through it.

Force terms and default parameters follow the circular-specification SFM as
implemented in PySocialForce (github.com/yuxiang-gao/PySocialForce) /
socialforce (github.com/svenkreiss/socialforce):

    F_goal = (v_des * e_des - v) / tau                      tau = 0.5 s
    F_ij   = A * exp((r_ij - d_ij) / B) * n_ij * w(phi)     A=4.5, B=0.35
    w(phi) = lambda + (1 - lambda) * (1 + cos phi) / 2      lambda = 0.30
    robot uses the same form with its own radius and a stronger gain.

Boundary handling: after integration a pedestrian is snapped back onto the
walkable surface (sidewalks + crossings + walking areas), the same geometry
layer the benchmark already uses for the robot.

Handoff: pedestrians are captured when within CAPTURE_R of the robot and
released (remapped onto their SUMO route, which then resumes normal
striping control) once beyond RELEASE_R.
"""
from __future__ import annotations

import math
import zlib


def _stable_hash(pid: str) -> int:
    """Process-stable substitute for hash() on a pedestrian id.

    Python salts str.__hash__ per process (PYTHONHASHSEED), so seeding any
    behaviour from hash(pid) makes the run irreproducible: the same code with
    the same --seed produced different pedestrian trajectories and different
    ped_* metrics on consecutive runs, which silently invalidated the
    seed-matched paired comparisons the protocol depends on. crc32 is stable
    across processes, machines and Python versions.
    """
    return zlib.crc32(pid.encode("utf-8"))


# ---- SFM parameters (PySocialForce circular-specification defaults) ----
TAU = 0.5              # relaxation time [s]
A_PED = 4.5            # ped-ped repulsion gain [m/s^2]
B_PED = 0.35           # ped-ped repulsion range [m]
LAMBDA = 0.30          # anisotropy (field of view weighting)
R_PED = 0.30           # pedestrian body radius [m]
A_ROB = 6.0            # ped-robot repulsion gain [m/s^2]
B_ROB = 0.45
R_ROB = 0.35           # robot body radius [m]
NEIGH_R = 4.0          # neighbour cutoff for ped-ped forces [m]
CAPTURE_R = 12.0       # take over pedestrians within this range of robot
RELEASE_R = 18.0       # hand back beyond this range
STATIC_CAPTURE_R = 2.5 # ...or within this range of a standing pedestrian
STATIC_RELEASE_R = 4.5 # (striping walks through statics; SFM does not)
VMAX_FACTOR = 1.3      # speed clip relative to desired speed


class SocialForceLayer:
    def __init__(self, traci_mod, walk_union=None, walk_prep=None,
                 net_file=None):
        self.traci = traci_mod
        self.union = walk_union
        self.uprep = walk_prep
        self.ctl = {}          # pid -> dict(pos, vel, vdes, edir)
        self.controlled_steps = 0
        self.capture_events = 0
        # pedestrian-side metrics (supervisor feedback item 5):
        # cost imposed on pedestrians BY the robot, measured against each
        # captured pedestrian's own intent (desired speed + heading)
        self._done = []        # finished per-ped stats dicts
        # HuNavSim social-work components (arXiv:2305.01303 Table I):
        #   social_work = social_force_on_robot + obstacle_force_on_robot
        #                 + social_force_on_agents
        self.sf_on_robot = 0.0     # |social force agents->robot| integrated
        self.of_on_robot = 0.0     # |obstacle force walls->robot| integrated
        self.robot_swork = 0.0     # kept as alias = sf_on_robot (back-compat)
        self.PS_R = 1.2        # personal-space radius [m] (HuNavSim-style)
        # junction no-control zone: SFM never captures inside it and
        # never pushes a controlled pedestrian into it. Remote persons
        # mapped onto junction-internal lanes (walkingareas/crossings)
        # can crash SUMO's person state machine; excluding the junction
        # core makes that state unreachable. Pedestrians there stay
        # striping-controlled (documented limitation: reactivity applies
        # on sidewalk segments, not inside junction cores).
        self.zone = self.zprep = None
        if net_file is not None:
            try:
                self._build_zone(net_file)
            except Exception:
                self.zone = self.zprep = None

    def _build_zone(self, net_file):
        import xml.etree.ElementTree as ET
        from shapely.geometry import LineString, Point
        from shapely.ops import unary_union
        from shapely.prepared import prep
        polys = []
        for _ev, el in ET.iterparse(str(net_file)):
            if el.tag == "edge":
                func = el.get("function", "")
                if func == "walkingarea":   # crossings stay controllable
                    for lane in el.iter("lane"):
                        shp = lane.get("shape")
                        if not shp:
                            continue
                        P = [tuple(map(float, q.split(",")))
                             for q in shp.split()]
                        w = float(lane.get("width", "3.0"))
                        if len(P) >= 2:
                            polys.append(
                                LineString(P).buffer(w / 2 + 0.6))
                        elif P:
                            polys.append(Point(P[0]).buffer(w / 2 + 0.6))
                el.clear()
        if polys:
            self.zone = unary_union(polys)
            self.zprep = prep(self.zone)


    # ------------------------------------------------------------------
    def _desired_dir(self, pid):
        """Unit vector of the pedestrian's intended walking direction."""
        ang = math.radians(self.traci.person.getAngle(pid))
        # SUMO angle: 0 = north, clockwise
        return (math.sin(ang), math.cos(ang))

    def _capture(self, pid):
        t = self.traci
        x, y = t.person.getPosition(pid)
        sp = t.person.getSpeed(pid)
        ed = self._desired_dir(pid)
        vdes = max(0.6, t.person.getMaxSpeed(pid))
        self.ctl[pid] = {"pos": (x, y), "vel": (sp * ed[0], sp * ed[1]),
                         "vdes": vdes, "edir": ed,
                         "p0": (x, y), "delay": 0.0, "defl": 0.0,
                         "ps": 0.0, "swork": 0.0, "nr": False}
        self.capture_events += 1

    def _release(self, pid):
        """Remap onto the person's route so striping resumes control.
        Only safe on a normal edge: releasing while mapped to a junction-
        internal edge (':...') can crash SUMO, so callers defer in that
        case and retry on a later step."""
        st = self.ctl.pop(pid, None)
        if st is None:
            return
        if st.get("nr"):
            # robot-imposed-cost metrics: only pedestrians that came
            # near the robot are counted; static-bubble passers-by are
            # excluded so they do not dilute the means
            self._done.append({"delay": st["delay"], "defl": st["defl"],
                               "ps": st["ps"],
                               "swork": st.get("swork", 0.0)})
        if not self._is_walking(pid):
            return                      # drop control; no remap
        try:
            self.traci.person.moveToXY(pid, "", st["pos"][0], st["pos"][1],
                                       keepRoute=1)
        except self.traci.exceptions.TraCIException:
            pass

    def _on_internal(self, pid):
        try:
            rid = self.traci.person.getRoadID(pid)
        except self.traci.exceptions.TraCIException:
            return True
        return rid.startswith(":") or rid == ""

    def _in_zone(self, x, y):
        if self.zprep is None:
            return False
        from shapely.geometry import Point
        return self.zprep.covers(Point(x, y))

    def _is_walking(self, pid):
        try:
            stg = self.traci.person.getStage(pid, 0)
            return getattr(stg, "type", 2) == 2      # 2 = walking
        except self.traci.exceptions.TraCIException:
            return False

    # ------------------------------------------------------------------
    def step(self, robot_xy, robot_v, dt):
        t = self.traci
        rx, ry = robot_xy
        try:
            pids = t.person.getIDList()
        except t.exceptions.TraCIException:
            return
        alive = set(pids)
        # drop vanished, release far
        statics = []
        for pid in pids:
            if pid.startswith("stand_"):
                try:
                    statics.append(t.person.getPosition(pid))
                except t.exceptions.TraCIException:
                    pass

        def near_static(x_, y_):
            for (sx_, sy_) in statics:
                if math.hypot(x_ - sx_, y_ - sy_) < STATIC_CAPTURE_R:
                    return True
            return False

        def clear_of_statics(x_, y_):
            for (sx_, sy_) in statics:
                if math.hypot(x_ - sx_, y_ - sy_) < STATIC_RELEASE_R:
                    return False
            return True

        for pid in list(self.ctl):
            if pid not in alive:
                st = self.ctl.pop(pid, None)
                if st is not None and st.get("nr"):
                    self._done.append({"delay": st["delay"],
                                       "defl": st["defl"], "ps": st["ps"],
                                       "swork": st.get("swork", 0.0)})
                continue
            px, py = self.ctl[pid]["pos"]
            if math.hypot(px - rx, py - ry) > RELEASE_R and \
                    clear_of_statics(px, py) and \
                    not self._on_internal(pid):
                self._release(pid)
        # capture near
        near_states = {}
        for pid in pids:
            if pid == "robot0" or pid in self.ctl:
                continue
            px, py = t.person.getPosition(pid)
            d = math.hypot(px - rx, py - ry)
            near_states[pid] = (px, py)
            if (d < CAPTURE_R or near_static(px, py)) and \
                    not self._on_internal(pid) and \
                    self._is_walking(pid) and not self._in_zone(px, py) \
                    and not pid.startswith("stand_"):
                # statics (stand_*) are never captured: capturing would
                # assign them a walking desired speed and set them in
                # motion. They still repel controlled pedestrians as
                # neighbours via pos_all.
                self._capture(pid)
        if not self.ctl:
            return
        # neighbour set = controlled + uncontrolled positions
        pos_all = {pid: st["pos"] for pid, st in self.ctl.items()}
        for pid, p in near_states.items():
            if pid not in pos_all:
                pos_all[pid] = p
        # per-step stage sentinel: a controlled person whose stage is no
        # longer 'walking' (e.g. a static that reached its stop) must be
        # dropped BEFORE any further moveToXY -- moving a waiting person
        # can crash SUMO
        for pid in list(self.ctl):
            if not self._is_walking(pid):
                st = self.ctl.pop(pid)
                if st.get("nr"):
                    self._done.append({"delay": st["delay"],
                                       "defl": st["defl"],
                                       "ps": st["ps"],
                                       "swork": st.get("swork", 0.0)})
        # integrate every controlled pedestrian
        _rob_fx = _rob_fy = 0.0     # net social force on the robot this step
        for pid, st in self.ctl.items():
            px, py = st["pos"]
            vx, vy = st["vel"]
            ex, ey = st["edir"]
            vd = st["vdes"]
            # bypass steering: a static dead ahead on the frozen walking
            # line creates a head-on force deadlock (goal force vs
            # repulsion -> oscillation). Steer the DESIRED direction at a
            # waypoint 1.1 m beside the blocking static instead; once the
            # static is behind, the condition lapses and the original
            # heading resumes.
            blk = None
            bestd = 3.0
            for (sx0, sy0) in statics:
                ahead = (sx0 - px) * ex + (sy0 - py) * ey
                if 0.0 < ahead < bestd:
                    lat = (sx0 - px) * (-ey) + (sy0 - py) * ex
                    if abs(lat) < 0.9:
                        bestd = ahead
                        blk = (sx0, sy0, lat)
            gex, gey = ex, ey
            if blk is not None:
                sx0, sy0, lat = blk
                mem = st.get("byp")
                if mem and math.hypot(mem[0] - sx0, mem[1] - sy0) < 0.5:
                    side = mem[2]          # sticky: no per-step flapping
                else:
                    if abs(lat) < 0.05:
                        side = 1.0 if (_stable_hash(pid) & 1) else -1.0
                    else:
                        side = -1.0 if lat > 0 else 1.0
                    # walkable-aware: if the bypass point on the chosen
                    # side is off the sidewalk, take the other side
                    if self.uprep is not None:
                        from shapely.geometry import Point as _P
                        for cand in (side, -side):
                            bx_ = sx0 + (-ey) * cand * 1.1
                            by_ = sy0 + ex * cand * 1.1
                            if self.uprep.covers(_P(bx_, by_)):
                                side = cand
                                break
                    st["byp"] = (sx0, sy0, side)
                tx = sx0 + (-ey) * side * 1.1 + ex * 0.3
                ty = sy0 + ex * side * 1.1 + ey * 0.3
                dxb, dyb = tx - px, ty - py
                Lb = math.hypot(dxb, dyb) or 1e-9
                gex, gey = dxb / Lb, dyb / Lb
            else:
                st.pop("byp", None)
                # merge-back: after a bypass (or any displacement), steer
                # toward the pedestrian's ORIGINAL walking line (captured
                # at p0 along edir), with a small fixed per-ped random
                # offset so re-merged walkers spread naturally instead of
                # forming a single file
                jx = ((_stable_hash(pid) % 1000) / 1000.0 - 0.5) * 0.4
                lat0 = ((px - st["p0"][0]) * (-ey)
                        + (py - st["p0"][1]) * ex) - jx
                if abs(lat0) > 0.25:
                    ahead = ((px - st["p0"][0]) * ex
                             + (py - st["p0"][1]) * ey) + 3.0
                    tx = st["p0"][0] + ex * ahead + (-ey) * jx
                    ty = st["p0"][1] + ey * ahead + ex * jx
                    dxm, dym = tx - px, ty - py
                    Lm = math.hypot(dxm, dym) or 1e-9
                    gex, gey = dxm / Lm, dym / Lm
            fx = (vd * gex - vx) / TAU
            fy = (vd * gey - vy) / TAU
            # ped-ped repulsion (anisotropic)
            for qid, (qx, qy) in pos_all.items():
                if qid == pid:
                    continue
                dx, dy = px - qx, py - qy
                d = math.hypot(dx, dy)
                if d < 1e-6 or d > NEIGH_R:
                    continue
                nx_, ny_ = dx / d, dy / d
                mag = A_PED * math.exp((2 * R_PED - d) / B_PED)
                spd = math.hypot(vx, vy)
                if spd > 1e-6:
                    cosphi = (-(vx * nx_ + vy * ny_)) / spd
                    w = LAMBDA + (1 - LAMBDA) * (1 + cosphi) / 2.0
                else:
                    w = 1.0
                fx += mag * nx_ * w
                fy += mag * ny_ * w
            # robot repulsion (same form, robot as an agent)
            dx, dy = px - rx, py - ry
            d = math.hypot(dx, dy)
            if 1e-6 < d < NEIGH_R + 2.0:
                nx_, ny_ = dx / d, dy / d
                mag = A_ROB * math.exp((R_PED + R_ROB - d) / B_ROB)
                spd = math.hypot(vx, vy)
                if spd > 1e-6:
                    cosphi = (-(vx * nx_ + vy * ny_)) / spd
                    w = LAMBDA + (1 - LAMBDA) * (1 + cosphi) / 2.0
                else:
                    w = 1.0
                fx += mag * nx_ * w
                fy += mag * ny_ * w
                # --- HuNavSim social work accounting -------------------
                # pedestrian side: magnitude of the robot-induced social
                # force actually applied to this pedestrian, integrated
                # over time
                st["swork"] += mag * w * dt
                # robot side: reaction force (ped -> robot), anisotropy
                # weighted by the robot's own heading
                rvx, rvy = robot_v
                rsp = math.hypot(rvx, rvy)
                if rsp > 1e-6:
                    cosr = (rvx * nx_ + rvy * ny_) / rsp
                    wr = LAMBDA + (1 - LAMBDA) * (1 + cosr) / 2.0
                else:
                    wr = 1.0
                _rob_fx -= mag * wr * nx_
                _rob_fy -= mag * wr * ny_
            # integrate + clip
            vx += fx * dt
            vy += fy * dt
            sp = math.hypot(vx, vy)
            vmax = VMAX_FACTOR * vd
            if sp > vmax:
                vx, vy = vx / sp * vmax, vy / sp * vmax
            nx2, ny2 = px + vx * dt, py + vy * dt
            # junction no-control zone: a controlled pedestrian may not
            # enter it -- hold at the boundary (yields at the junction
            # mouth) so SUMO never maps a remote person onto internals
            if self._in_zone(nx2, ny2):
                nx2, ny2 = px, py
                vx *= 0.2
                vy *= 0.2
            # boundary: stay on the walkable surface
            if self.uprep is not None and not self.uprep.covers(
                    __import__("shapely.geometry", fromlist=["Point"])
                    .Point(nx2, ny2)):
                from shapely.geometry import Point
                from shapely.ops import nearest_points
                q = nearest_points(self.union, Point(nx2, ny2))[0]
                if q.distance(Point(nx2, ny2)) <= 1.0:
                    nx2, ny2 = q.x, q.y
                else:
                    nx2, ny2 = px, py
                vx *= 0.3
                vy *= 0.3
            # --- pedestrian-side accounting
            # delay = SPEED deficit only (direction-agnostic), so route
            # turns are not misread as robot-imposed delay; detours are
            # captured separately by the deflection metric
            spd_now = math.hypot(vx, vy)
            dr = math.hypot(nx2 - rx, ny2 - ry)   # fresh robot distance
            if dr < CAPTURE_R:
                st["nr"] = True
            if dr < 6.0:     # count cost only in the interaction zone
                st["delay"] += max(0.0, vd - spd_now) * dt
            lat = abs((nx2 - st["p0"][0]) * (-ey) +
                      (ny2 - st["p0"][1]) * ex)
            if lat > st["defl"]:
                st["defl"] = lat
            if dr < self.PS_R:
                st["ps"] += dt
            st["pos"] = (nx2, ny2)
            st["vel"] = (vx, vy)
            try:
                t.person.moveToXY(pid, "", nx2, ny2, keepRoute=2)
            except t.exceptions.TraCIException:
                pass
            self.controlled_steps += 1
        _sf = math.hypot(_rob_fx, _rob_fy) * dt
        self.sf_on_robot += _sf
        self.robot_swork += _sf
        # obstacle force on the robot: SFM wall repulsion from the walkable
        # boundary (robot treated as an SFM particle for metric purposes)
        if self.union is not None:
            try:
                from shapely.geometry import Point as _Pt
                d_wall = self.union.exterior.distance(_Pt(rx, ry)) \
                    if hasattr(self.union, "exterior") \
                    else self.union.boundary.distance(_Pt(rx, ry))
                self.of_on_robot += A_ROB * math.exp(
                    (R_ROB - max(d_wall, 1e-3)) / B_ROB) * dt
            except Exception:
                pass

    # ------------------------------------------------------------------
    def ped_metrics(self):
        """Aggregate pedestrian-side metrics over all captured pedestrians
        (finished episodes + still-controlled at run end)."""
        allst = self._done + [{"delay": st["delay"], "defl": st["defl"],
                               "ps": st["ps"],
                               "swork": st.get("swork", 0.0)}
                              for st in self.ctl.values() if st.get("nr")]
        n = len(allst)
        if n == 0:
            return {"ped_affected_n": 0, "ped_delay_s_mean": 0.0,
                    "ped_deflection_m_mean": 0.0,
                    "ped_deflection_m_max": 0.0,
                    "ped_personal_space_s_total": 0.0,
                    "social_force_on_agents": 0.0,
                    "social_force_on_robot": round(self.sf_on_robot, 2),
                    "obstacle_force_on_robot": round(self.of_on_robot, 2),
                    "social_work": round(self.sf_on_robot
                                         + self.of_on_robot, 2)}
        return {
            "ped_affected_n": n,
            "ped_delay_s_mean": round(sum(a["delay"] for a in allst) / n, 2),
            "ped_deflection_m_mean": round(
                sum(a["defl"] for a in allst) / n, 3),
            "ped_deflection_m_max": round(
                max(a["defl"] for a in allst), 3),
            "ped_personal_space_s_total": round(
                sum(a["ps"] for a in allst), 1),
            "social_force_on_agents": round(
                sum(a.get("swork", 0.0) for a in allst), 2),
            "social_force_on_robot": round(self.sf_on_robot, 2),
            "obstacle_force_on_robot": round(self.of_on_robot, 2),
            "social_work": round(self.sf_on_robot + self.of_on_robot
                                 + sum(a.get("swork", 0.0) for a in allst),
                                 2)}
