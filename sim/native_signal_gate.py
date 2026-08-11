#!/usr/bin/env python3
"""Signal gate for the fully native maps (Option B).

Everything in the simulation obeys the lights natively; ONLY the robot (a red
POI whose kinematics belong to the benchmark runner) needs to ask.  For every
signalised crossing, map_spec.json stores its TLS id and linkIndex; the gate
reads getRedYellowGreenState(tls)[linkIndex] each step:

    'G'/'g'  -> pedestrians may enter        (green)
    anything else (r, x during clearance)  -> wait in the 1.8 m strip

API (same shape as the option-A overlay, so planner gates are shared):
    gate.step(t)                       # refresh cached states (cheap)
    gate.zone_states()                 # [{"rect","green","strips","tls",...}]
    gate.ped_state_at(x, y, radius=6)  # nearest crossing or None
    gate.time_to_ped_green(zone, t)    # walk the native program forward
Robot rule of thumb: before entering zone["rect"], if not zone["green"], hold
inside the strip on your side and accumulate time_waiting_at_light_s.
"""
from __future__ import annotations

import math


class NativeSignalGate:
    def __init__(self, spec, traci_mod):
        self.traci = traci_mod
        self.crossings = [c for c in spec.get("crossings", []) if c.get("tls")]
        self.tls_ids = sorted({c["tls"] for c in self.crossings})
        self._logic = {t: self.traci.trafficlight.getAllProgramLogics(t)[0]
                       for t in self.tls_ids}
        # drop crossings that NEVER show green in their program (netconvert
        # artefacts at complex OSM junctions) -- holding for them would
        # deadlock; treat them as unsignalised instead
        served = []
        for c in self.crossings:
            lg = self._logic[c["tls"]]
            li = c["linkIndex"]
            if any(li < len(ph.state) and ph.state[li] in "Gg"
                   for ph in lg.phases):
                served.append(c)
        self.skipped_never_green = len(self.crossings) - len(served)
        self.crossings = served
        self._states = {}
        self._pub = []

    def step(self, t=None):
        self._states = {tid: self.traci.trafficlight
                        .getRedYellowGreenState(tid) for tid in self.tls_ids}
        self._pub = []
        for i, c in enumerate(self.crossings):
            st = self._states[c["tls"]][c["linkIndex"]]
            self._pub.append({"index": i, "id": c["id"], "rect": c["rect"],
                              "axis": c["axis"], "road": c.get("road"),
                              "tls": c["tls"], "linkIndex": c["linkIndex"],
                              "green": st in "Gg", "state_char": st,
                              "strips": c["strips"]})

    def zone_states(self):
        return self._pub

    def ped_state_at(self, x, y, radius=6.0):
        best, bd = None, radius
        for z in self._pub:
            x0, y0, x1, y1 = z["rect"]
            d = math.hypot(max(x0 - x, 0.0, x - x1),
                           max(y0 - y, 0.0, y - y1))
            if d <= bd:
                bd, best = d, z
        return best

    def time_to_ped_green(self, zone, t):
        if zone["green"]:
            return 0.0
        tls, li = zone["tls"], zone["linkIndex"]
        lg = self._logic[tls]
        pi = self.traci.trafficlight.getPhase(tls)
        acc = max(0.0, self.traci.trafficlight.getNextSwitch(tls) - t)
        for k in range(1, 2 * len(lg.phases) + 1):
            ph = lg.phases[(pi + k) % len(lg.phases)]
            if ph.state[li] in "Gg":
                return acc
            acc += ph.duration
        return acc
