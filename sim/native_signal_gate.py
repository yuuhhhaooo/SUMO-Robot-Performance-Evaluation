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
        self._logic = {t: self._active_logic(t) for t in self.tls_ids}
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

    @staticmethod
    def _program_id(logic):
        """Program name of a traci Logic object (API name differs by
        SUMO version: ``programID`` on modern traci, ``subID`` on old)."""
        pid = getattr(logic, "programID", None)
        if pid is None:
            pid = getattr(logic, "subID", None)
        return pid

    def _active_logic(self, tls_id):
        """The logic of the program that is CURRENTLY loaded on this TLS.

        getAllProgramLogics() returns every program defined for the junction;
        index 0 is only the active one by coincidence.  Any TLS that carries
        an extra program (e.g. an "off"/actuated variant, or a program set
        via setProgram) would otherwise be walked with the wrong phase table
        by time_to_ped_green(), and screened with the wrong phase states in
        the never-green filter below.
        """
        logics = list(self.traci.trafficlight.getAllProgramLogics(tls_id))
        if not logics:
            raise RuntimeError(
                f"native_signal_gate: TLS {tls_id} has no program logic")
        try:
            active = self.traci.trafficlight.getProgram(tls_id)
        except Exception as exc:
            print(f"native_signal_gate: cannot read active program of TLS "
                  f"{tls_id} ({exc}); using program "
                  f"'{self._program_id(logics[0])}' (index 0)")
            return logics[0]
        for lg in logics:
            if self._program_id(lg) == active:
                return lg
        print(f"native_signal_gate: TLS {tls_id} reports active program "
              f"'{active}' but getAllProgramLogics() only offers "
              f"{[self._program_id(lg) for lg in logics]}; falling back to "
              f"program '{self._program_id(logics[0])}' (index 0)")
        return logics[0]

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
        # Shallow copy: callers may not append to / reorder / clear the gate's
        # own list.  The per-zone dicts are deliberately NOT copied -- they are
        # rebuilt from scratch by every step() call, so they are per-step
        # scratch objects, and copying them each call would add work on the
        # hot path (this is called for every robot step).
        return list(self._pub)

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
            # same bounds guard as the never-green screen in __init__: a
            # phase whose state string is shorter than linkIndex simply does
            # not serve this crossing (netconvert emits ragged states at some
            # OSM junctions) -- it must not raise IndexError here.
            if li < len(ph.state) and ph.state[li] in "Gg":
                return acc
            acc += ph.duration
        return acc
