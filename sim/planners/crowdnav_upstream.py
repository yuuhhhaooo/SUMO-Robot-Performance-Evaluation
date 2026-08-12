#!/usr/bin/env python3
"""Benchmark planners driven by the ORIGINAL published CrowdNav code.

Upstream: https://github.com/vita-epfl/CrowdNav @ 20d678085c06831e658a65b9e20c8bb6f6ecdc10
Paper:    Chen, Liu, Kreiss, Alahi, "Crowd-Robot Interaction: Crowd-aware Robot
          Navigation with Attention-based Deep Reinforcement Learning", ICRA 2019.
License:  MIT (c) 2018 VITA lab at EPFL -- see sim/third_party/crowdnav/LICENSE

The vendored tree lives in sim/third_party/crowdnav/ with its LICENSE, the exact
commit in COMMIT, and the two import-guard patches described in PATCHES.md.  The
files that contain the algorithms --
    crowd_nav/policy/cadrl.py
    crowd_nav/policy/multi_human_rl.py
    crowd_nav/policy/sarl.py
    crowd_nav/policy/lstm_rl.py
    crowd_nav/configs/policy.config
    crowd_sim/envs/utils/{state,action}.py
    crowd_sim/envs/policy/policy.py
-- are byte-for-byte upstream.  Nothing in this file re-implements any part of
SARL / CADRL / LSTM-RL.  This file only

  (a) builds the upstream policy object from upstream's own policy.config,
  (b) loads the shipped checkpoint into upstream's own network class,
  (c) converts the benchmark's leg-local (state, goal, obstacles) into upstream's
      FullState / ObservableState / JointState,
  (d) supplies the one-step-lookahead the upstream policies ask their
      environment for (there is no CrowdSim env here -- SUMO is the env),
  (e) clips the leg goal to a LOCAL goal inside upstream's training
      distribution, and truncates the crowd to upstream's training crowd size.

Action selection, the value networks, the state rotation, the reward and the
discrete action space are all executed by the vendored upstream code.


================================================================================
CHECKPOINT / ARCHITECTURE COMPATIBILITY -- THE RESULT
================================================================================
All three shipped checkpoints under sim/planners/models/ load into the
UNMODIFIED upstream network classes with strict=True, with identical key sets
and identical tensor shapes, using upstream's own default policy.config.  They
are genuine CrowdNav weights.  No retraining is required.

  sarl_rl_model.pth  ->  crowd_nav.policy.sarl.ValueNetwork
        (input_dim=13, self_state_dim=6, mlp1=[150,100], mlp2=[100,50],
         attention=[100,100,1], mlp3=[150,100,100,1], with_global_state=True)
        mlp1.0 (150,13)  mlp1.2 (100,150)
        mlp2.0 (100,100) mlp2.2 (50,100)
        attention.0 (100,200)  attention.2 (100,100)  attention.4 (1,100)
        mlp3.0 (150,56)  mlp3.2 (100,150)  mlp3.4 (100,100)  mlp3.6 (1,100)
        -- attention.0 fan-in 200 == mlp1_dims[-1]*2 confirms with_global_state
           was True at training time; mlp3.0 fan-in 56 == mlp2_dims[-1] + 6
           confirms with_om was False (no 48-wide occupancy map appended).

  cadrl_rl_model.pth ->  crowd_nav.policy.cadrl.ValueNetwork
        (input_dim=13 == self_state_dim 6 + human_state_dim 7,
         mlp_dims=[150,100,100,1])
        value_network.0 (150,13) .2 (100,150) .4 (100,100) .6 (1,100)

  lstm_rl_model.pth  ->  crowd_nav.policy.lstm_rl.ValueNetwork1
        (input_dim=13, self_state_dim=6, mlp_dims=[150,100,100,1],
         lstm_hidden_dim=50)
        lstm.weight_ih_l0 (200,13)  lstm.weight_hh_l0 (200,50)
        lstm.bias_ih_l0 (200,)      lstm.bias_hh_l0 (200,)
        mlp.0 (150,56)  mlp.2 (100,150)  mlp.4 (100,100)  mlp.6 (1,100)
        -- the presence of `lstm.*` and the absence of any `mlp1.*` confirms
           ValueNetwork1, i.e. with_interaction_module=False; mlp.0 fan-in
           56 == 6 + 50 confirms with_om=False.

Run `python sim/planners/crowdnav_upstream.py` to re-verify this on demand.
================================================================================


LOCAL-GOAL CLIPPING
-------------------
Upstream is trained and tested on `circle_crossing` with `circle_radius = 4`
(crowd_nav/configs/env.config), so the robot starts 8 m from its goal and the
goal distance `dg` -- the FIRST feature of the rotated state vector that both
value networks consume -- is only ever in [0, 8] m during training.  The
benchmark's leg goal can be several hundred metres away.  Feeding dg = 300 into
a network whose training support tops out at 8 is far out of distribution, and
its output there carries no information about which action makes progress.

This module therefore hands the policy a LOCAL goal placed
`GOAL_HORIZON = 2.0` m along the straight line from the robot to the true leg
goal, and the true goal itself once it is nearer than that.  On a straight
sidewalk leg the clipped goal is collinear with the true goal, so clipping
changes only dg -- never the goal DIRECTION.

WHY 2.0 m.  It was chosen by measurement, not by taste.  Sweep over a synthetic
30 m leg, 2 m band, dt = 0.5 s, 60 s budget, 2 seeds; the cell is
"legs completed / 2" (mean net forward progress, m):

    horizon      SARL         CADRL        LSTM-RL      crowd
      1.0     2/2 (29.4)   2/2 (29.4)   2/2 (29.3)     14 peds
      2.0     2/2 (29.3)   2/2 (29.2)   2/2 (29.4)     14 peds
      3.0     2/2 (29.5)   2/2 (29.3)   2/2 (29.4)     14 peds
      4.0     1/2 (27.1)   0/2 (-12.0)  2/2 (29.4)     14 peds
      2.0     2/2 (29.5)   2/2 (29.3)   2/2 (29.4)      4 peds
      4.0     2/2 (29.4)   0/2 ( -8.6)  2/2 (29.4)      4 peds
      8.0     2/2 (29.4)   2/2 (25.9)   2/2 (29.4)      4 peds
     20.0     0/2 ( 3.6)   0/2 ( -4.1)  1/2 (18.9)      4 peds
    300.0*    1/2 (27.5)   0/2 ( -5.8)  2/2 (29.1)      4 peds
    (* 300 m = no clipping at all, i.e. what the leg goal is today.)

Negative progress means the robot drove backwards down the leg.  Everything
from 4 m up is erratic: which policy survives which horizon is essentially
arbitrary, which is exactly the signature of an out-of-distribution input.
H <= 3 m is uniformly stable for all three policies at both crowd densities.

Within the stable band, 2.0 m rather than 1.0 m because:
  * upstream's one-step lookahead awards the terminal reward +1 when the
    PROPAGATED next state is within robot_radius of the goal.  The robot moves
    at most v_pref*dt = 0.5 m per step, so a horizon below
    v_pref*dt + robot_radius = 0.75 m would let that +1 fire on every step and
    swamp the value net entirely.  2.0 m keeps a 1.25 m margin; 1.0 m keeps
    only 0.25 m.
  * it is 4 planning steps of lookahead, enough that the goal-pull direction is
    stable step to step.
1.0 m is marginally faster to the goal (30.8 s vs 34.8 s for SARL) -- that is
the speed/margin trade-off, and `goal_horizon` is a constructor argument and an
instance attribute precisely so it can be reported as a sensitivity.

CROWD SIZE
----------
Upstream trains with `human_num = 5` (env.config).  Humans are filtered to
`cfg.sensor_range` and then truncated to the `max_humans = 5` nearest, matching
the training crowd size.  Also exposed as a constructor argument.
"""
from __future__ import annotations

import math
import sys
from configparser import RawConfigParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party" / "crowdnav"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

_PLANNER_DIR = Path(__file__).resolve().parent
if str(_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(_PLANNER_DIR))

# --- upstream, unmodified --------------------------------------------------
from crowd_nav.policy.cadrl import CADRL as _UpstreamCADRL          # noqa: E402
from crowd_nav.policy.lstm_rl import LstmRL as _UpstreamLstmRL      # noqa: E402
from crowd_nav.policy.multi_human_rl import MultiHumanRL as _MultiHumanRL  # noqa: E402
from crowd_nav.policy.sarl import SARL as _UpstreamSARL            # noqa: E402
from crowd_sim.envs.utils.action import ActionXY                    # noqa: E402
from crowd_sim.envs.utils.state import (FullState, JointState,      # noqa: E402
                                        ObservableState)
# ---------------------------------------------------------------------------

from sidewalk_robot_common import PlannerConfig                     # noqa: E402

UPSTREAM_COMMIT = "20d678085c06831e658a65b9e20c8bb6f6ecdc10"
UPSTREAM_REPO = "https://github.com/vita-epfl/CrowdNav"
POLICY_CONFIG = _THIRD_PARTY / "crowd_nav" / "configs" / "policy.config"
ENV_CONFIG = _THIRD_PARTY / "crowd_nav" / "configs" / "env.config"

#: Local goal horizon, metres. Chosen by sweep; see "LOCAL-GOAL CLIPPING" above.
#: Must stay above v_pref*dt + robot_radius (0.75 m here) and below ~3 m.
GOAL_HORIZON = 2.0
#: Crowd truncation. == upstream env.config human_num.
MAX_HUMANS = 5
#: Upstream trains at v_pref = 1.0 m/s (env.config [robot] v_pref).
UPSTREAM_V_PREF = 1.0
#: torch intra-op threads. Upstream's per-action loop issues 81 batch-of-1
#: forward passes per control step; the default 16-thread pool makes that ~30x
#: slower with bit-identical results. See _CrowdNavUpstreamPlanner.__init__.
TORCH_THREADS: Optional[int] = 1


class _OneStepLookaheadEnv:
    """The `policy.env` the upstream policies query, backed by SUMO's crowd.

    Upstream's shipped `policy.config` sets `query_env = true`, so
    `MultiHumanRL.predict` (SARL, LSTM-RL) and `CADRL.predict` both call
    `self.env.onestep_lookahead(action)` to obtain the next human states and the
    immediate reward for each candidate action.  In upstream that is
    `CrowdSim.step(action, update=False)`, which rolls the humans forward with
    their own ORCA policies.

    There is no CrowdSim here -- SUMO is the environment, and it cannot be
    rolled forward speculatively.  This shim therefore reproduces exactly what
    upstream itself does when `query_env = false`, which is upstream's own
    supported no-environment branch (see `MultiHumanRL.predict`):

        next_human_states = [self.propagate(h, ActionXY(h.vx, h.vy)) for h in ...]
        reward            = self.compute_reward(next_self_state, next_human_states)

    Both calls below dispatch into the vendored upstream methods; the constant-
    velocity human prediction and the reward function are upstream's, not ours.
    `CADRL.predict` has no `query_env` branch of its own, so routing it through
    this shim is what puts it on the same upstream code path.
    """

    __slots__ = ("policy", "self_state", "human_states")

    def __init__(self, policy):
        self.policy = policy
        self.self_state: Optional[FullState] = None
        self.human_states: Sequence[ObservableState] = ()

    def set_state(self, self_state: FullState,
                  human_states: Sequence[ObservableState]) -> None:
        self.self_state = self_state
        self.human_states = human_states

    def onestep_lookahead(self, action):
        policy = self.policy
        next_self_state = policy.propagate(self.self_state, action)
        next_human_states = [
            policy.propagate(h, ActionXY(h.vx, h.vy)) for h in self.human_states
        ]
        # Upstream's own reward (crowd_nav/policy/multi_human_rl.py). CADRL does
        # not inherit it, so it is called unbound; it only needs policy.time_step.
        reward = _MultiHumanRL.compute_reward(policy, next_self_state,
                                              next_human_states)
        return next_human_states, reward, False, None


class _CrowdNavUpstreamPlanner:
    """Common benchmark wrapper around an upstream CrowdNav policy.

    Benchmark interface:
        __init__(cfg: PlannerConfig, seed: int, model_path=..., device=...)
        compute_command(state, goal, obstacles, sim_time) -> (vx, vy, info)
    All quantities are in the leg-local frame (x along travel, y across the
    band in [0, cfg.sidewalk_y_max]).
    """

    #: subclass hooks
    policy_cls: Any = None
    config_section: str = ""
    default_model: str = ""
    name: str = ""
    cpu_only: bool = False          # see PATCHES.md "Deliberately NOT patched"

    def __init__(self, cfg: PlannerConfig, seed: int,
                 model_path: Optional[Path] = None,
                 device: str = "cpu",
                 goal_horizon: float = GOAL_HORIZON,
                 max_humans: int = MAX_HUMANS,
                 torch_threads: Optional[int] = TORCH_THREADS):
        self.cfg = cfg
        self.seed = int(seed)
        self.goal_horizon = float(goal_horizon)
        self.max_humans = int(max_humans)

        # Upstream predict() scores the 81 discrete actions in a Python loop with
        # one forward pass per action on a batch of 1, so each control step is 81
        # tiny matmuls. torch's default intra-op thread pool (16 on this machine)
        # spends far more time synchronising than these matmuls take to compute.
        # Measured on this box, batch-of-5-humans scene, ms per control step:
        #
        #     threads      SARL      CADRL    LSTM-RL
        #        16      5024.4     3829.4     6528.8
        #         1       171.3      109.2      142.8
        #
        # i.e. ~30x, and 5 s per control step would make the benchmark
        # unrunnable. The outputs are bit-for-bit identical between the two
        # settings (verified over 18 scenes x 3 policies), so this is a pure
        # runtime knob, not a numerical one. Pass torch_threads=None to leave
        # the process-global setting alone.
        if torch_threads is not None:
            torch.set_num_threads(int(torch_threads))

        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if self.cpu_only and dev.type != "cpu":
            raise RuntimeError(
                f"{self.name}: upstream lstm_rl.ValueNetwork1.forward allocates its "
                "LSTM initial state with torch.zeros(...) and no device= argument, "
                "so it only runs on CPU. Patching that would mean editing upstream "
                "code; see sim/third_party/crowdnav/PATCHES.md. Pass device='cpu'."
            )
        self.device = dev

        if model_path is None:
            model_path = _PLANNER_DIR / "models" / self.default_model
        self.model_path = Path(model_path)

        # --- build the upstream policy from UPSTREAM's own config file ------
        config = RawConfigParser()
        if not config.read(POLICY_CONFIG):
            raise FileNotFoundError(f"missing vendored config {POLICY_CONFIG}")
        self.policy = self.policy_cls()
        self.policy.configure(config)          # upstream: builds the network
        self.policy.set_device(self.device)
        self.policy.set_phase("test")
        self.policy.set_epsilon(0.0)
        self.policy.time_step = float(cfg.dt)

        state_dict = torch.load(self.model_path, map_location=self.device,
                                weights_only=True)
        # strict=True is the whole point: it fails loudly if the checkpoint and
        # upstream's architecture ever stop agreeing.
        self.policy.get_model().load_state_dict(state_dict, strict=True)
        self.policy.get_model().eval()
        for p in self.policy.get_model().parameters():
            p.requires_grad_(False)

        self.env = _OneStepLookaheadEnv(self.policy)
        self.policy.set_env(self.env)

        # Upstream trains at v_pref = 1.0 m/s; do not exceed the benchmark robot.
        self.v_pref = float(min(UPSTREAM_V_PREF, cfg.max_speed))
        self.policy.build_action_space(self.v_pref)

        self.robot_radius = float(cfg.robot_radius)
        self.human_radius = float(cfg.pedestrian_radius)
        #: last command issued; kept for reporting and for reset(). It is NOT
        #: fed back into the policy -- the robot velocity handed to upstream
        #: comes from the measured state (see compute_command).
        self.last_v: Tuple[float, float] = (0.0, 0.0)
        # `seed` is accepted for interface compatibility and recorded, but at
        # phase 'test' upstream's predict() is fully deterministic: the only
        # random draw it makes is the epsilon-greedy one, which is discarded
        # outside phase 'train' (and whose RNG side effect compute_command
        # undoes). Two runs with different seeds give identical commands.

    # -- helpers ------------------------------------------------------------
    def _local_goal(self, px: float, py: float,
                    goal: Sequence[float]) -> Tuple[float, float, float]:
        """Clip the leg goal into upstream's training distribution.

        Returns (gx, gy, true_distance).  Direction is preserved exactly; only
        the goal-distance feature dg is brought back in range.
        """
        gx, gy = float(goal[0]), float(goal[1])
        dx, dy = gx - px, gy - py
        dist = math.hypot(dx, dy)
        if dist > self.goal_horizon and dist > 1e-9:
            s = self.goal_horizon / dist
            return px + dx * s, py + dy * s, dist
        return gx, gy, dist

    def _humans(self, px: float, py: float,
                obstacles: Sequence[Any]) -> List[ObservableState]:
        """Sensor-range filter + truncate to upstream's training crowd size."""
        scored = []
        rng = float(self.cfg.sensor_range)
        for o in obstacles or ():
            d = math.hypot(float(o.x) - px, float(o.y) - py)
            if d <= rng:
                scored.append((d, o))
        scored.sort(key=lambda t: t[0])
        return [ObservableState(float(o.x), float(o.y),
                                float(getattr(o, "vx", 0.0)),
                                float(getattr(o, "vy", 0.0)),
                                self.human_radius)
                for _, o in scored[:self.max_humans]]

    def _goal_direct(self, px, py, gx, gy) -> ActionXY:
        """Fallback for an empty crowd.

        Upstream's value networks take a (batch, #humans, 13) tensor and its
        environment guarantees human_num >= 1, so `predict` cannot be called
        with zero humans (torch.cat of an empty list raises).  With no
        pedestrian in sensor range there is also nothing for a crowd-navigation
        policy to decide, so the robot drives straight at the (clipped) goal at
        v_pref -- the same convention the in-repo adapter uses.  Reported as
        status='goal_direct' so these steps can be excluded from any analysis
        of the policy itself.
        """
        dx, dy = gx - px, gy - py
        d = math.hypot(dx, dy)
        if d <= 1e-9:
            return ActionXY(0.0, 0.0)
        s = min(self.v_pref, d) / d
        return ActionXY(dx * s, dy * s)

    def _order_humans(self, px: float, py: float,
                      humans: List[ObservableState]) -> List[ObservableState]:
        """Order in which humans are presented. Nearest-first by default.

        Only LSTM-RL is order-sensitive; see UpstreamLstmRLPlanner.
        """
        return humans

    def _extra_info(self, joint_state) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        """Clear everything that belongs to the finished leg/episode.

        Clearing last_v alone is not enough: the one-step-lookahead env still
        holds the PREVIOUS leg's self_state and human_states, and upstream's
        policy keeps its action_values from the last predict(). Neither is
        observable today, because compute_command always calls set_state()
        before predict(), but leaving stale pose data on a reused planner is
        exactly the kind of thing that becomes a silent cross-leg leak the
        moment the call order changes.
        """
        self.last_v = (0.0, 0.0)
        env = getattr(self, "env", None)
        if env is not None:
            env.self_state = None
            env.human_states = None
        pol = getattr(self, "policy", None)
        if pol is not None:
            pol.action_values = None

    # -- benchmark entry point ---------------------------------------------
    def compute_command(self, state, goal, obstacles, sim_time):
        px, py = float(state.x), float(state.y)
        gx, gy, true_dist = self._local_goal(px, py, goal)
        humans = self._order_humans(px, py, self._humans(px, py, obstacles))

        info: Dict[str, Any] = {
            "status": self.name,
            "goal_dist": true_dist,
            "local_goal_dist": math.hypot(gx - px, gy - py),
            "n_humans": len(humans),
        }

        if true_dist <= max(self.robot_radius, 1e-6):
            self.last_v = (0.0, 0.0)
            info["status"] = "at_goal"
            return 0.0, 0.0, info

        if not humans:
            action = self._goal_direct(px, py, gx, gy)
            self.last_v = (float(action.vx), float(action.vy))
            info["status"] = "goal_direct"
            return self.last_v[0], self.last_v[1], info

        # Upstream's FullState carries the robot's ACTUAL velocity (it feeds the
        # vx/vy columns of rotate()), so take it from the measured state rather
        # than from the last command, which the runner may have capped, gated at
        # a traffic light, or clamped at the band edge.
        rvx = float(state.v) * math.cos(float(state.yaw))
        rvy = float(state.v) * math.sin(float(state.yaw))
        self_state = FullState(px, py, rvx, rvy,
                               self.robot_radius, gx, gy, self.v_pref,
                               float(state.yaw))
        joint_state = JointState(self_state, humans)
        self.env.set_state(self_state, humans)

        # Upstream's predict() calls np.random.random() unconditionally and only
        # uses the draw when phase == 'train'. At phase 'test' the value is
        # discarded but the process-global numpy RNG has still been advanced,
        # which would silently perturb any RNG stream the benchmark runner
        # shares. Save and restore the global state around the call so the
        # planner is a no-op on it. (Patching upstream instead would mean
        # editing the algorithm file; this is equivalent and leaves it pristine.)
        rng_state = np.random.get_state()
        try:
            with torch.no_grad():
                # >>> upstream code decides the action <<<
                action = self.policy.predict(joint_state)
        finally:
            np.random.set_state(rng_state)

        vals = getattr(self.policy, "action_values", None)
        if vals:
            info["value"] = float(max(vals))
        info.update(self._extra_info(joint_state))

        self.last_v = (float(action.vx), float(action.vy))
        return self.last_v[0], self.last_v[1], info


class UpstreamSARLPlanner(_CrowdNavUpstreamPlanner):
    """CrowdNav SARL (ICRA 2019) -- upstream crowd_nav.policy.sarl.SARL."""
    policy_cls = _UpstreamSARL
    config_section = "sarl"
    default_model = "sarl_rl_model.pth"
    name = "sarl_upstream"

    def _extra_info(self, joint_state) -> Dict[str, Any]:
        """Attention weights for the SELECTED action.

        `policy.get_attention_weights()` holds the weights of whichever action
        was evaluated LAST, not the argmax, so the chosen action's scene is
        re-forwarded once.  This is a read-out only: it does not affect the
        action already returned by upstream's predict().
        """
        out: Dict[str, Any] = {}
        try:
            vals = self.policy.action_values
            best = self.policy.action_space[int(np.argmax(vals))]
            nxt_humans, _r, _d, _i = self.env.onestep_lookahead(best)
            nxt_self = self.policy.propagate(joint_state.self_state, best)
            batch = torch.cat([torch.Tensor([nxt_self + h]).to(self.device)
                               for h in nxt_humans], dim=0)
            with torch.no_grad():
                self.policy.get_model()(self.policy.rotate(batch).unsqueeze(0))
            w = self.policy.get_attention_weights()
            if w is not None and len(w):
                k = int(np.argmax(w))
                out["attention"] = float(w[k])
                out["attended_index"] = k
        except Exception:       # read-out must never break an episode
            pass
        return out


class UpstreamCADRLPlanner(_CrowdNavUpstreamPlanner):
    """CADRL -- upstream crowd_nav.policy.cadrl.CADRL (CrowdNav's reproduction).

    Note this is upstream's single-human value net: `CADRL.predict` scores each
    candidate action by the MINIMUM value over the humans present (its
    pessimistic pairwise aggregation), which is upstream behaviour.
    """
    policy_cls = _UpstreamCADRL
    config_section = "cadrl"
    default_model = "cadrl_rl_model.pth"
    name = "cadrl_upstream"


class UpstreamLstmRLPlanner(_CrowdNavUpstreamPlanner):
    """LSTM-RL -- upstream crowd_nav.policy.lstm_rl.LstmRL (ValueNetwork1).

    Upstream `LstmRL.predict` sorts humans by DECREASING distance before
    delegating to `MultiHumanRL.predict`, so the LSTM consumes the nearest
    human last and its final hidden state is dominated by the nearest human.

    That sort does not actually reach the network in upstream, though: with the
    shipped `query_env = true`, `MultiHumanRL.predict` builds its batch from the
    list returned by `env.onestep_lookahead(...)`, not from the sorted
    `state.human_states`, so the environment's own (arbitrary, episode-fixed)
    human ordering wins.  Upstream is internally inconsistent here -- the sorted
    order IS used for the training target it stores in `transform()`.

    Humans are therefore pre-sorted farthest-first before being handed to BOTH
    the JointState and the lookahead shim, so the two agree and the network sees
    the ordering `LstmRL.predict` asks for.  Upstream's own `sorted(...)` call
    then becomes a no-op on an already-sorted list.  This is an input-ordering
    choice, not a change to the algorithm; `_order_humans` can be overridden to
    reproduce the inconsistent upstream behaviour if the author wants to
    quantify it.

    CPU only -- see PATCHES.md.
    """
    policy_cls = _UpstreamLstmRL
    config_section = "lstm_rl"
    default_model = "lstm_rl_model.pth"
    name = "lstm_rl_upstream"
    cpu_only = True

    def _order_humans(self, px, py, humans):
        # upstream LstmRL.predict: "sort human order by decreasing distance"
        return sorted(humans,
                      key=lambda h: math.hypot(h.px - px, h.py - py),
                      reverse=True)


PLANNERS = {
    "sarl_upstream": UpstreamSARLPlanner,
    "cadrl_upstream": UpstreamCADRLPlanner,
    "lstm_rl_upstream": UpstreamLstmRLPlanner,
}


def _self_check() -> int:
    """Verify checkpoint <-> upstream-architecture compatibility, key by key."""
    import collections
    ok = True
    print(f"upstream {UPSTREAM_REPO} @ {UPSTREAM_COMMIT}")
    print(f"torch {torch.__version__}  numpy {np.__version__}  "
          f"python {sys.version.split()[0]}\n")
    for name, cls in PLANNERS.items():
        path = _PLANNER_DIR / "models" / cls.default_model
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = RawConfigParser()
        config.read(POLICY_CONFIG)
        pol = cls.policy_cls()
        pol.configure(config)
        pol.set_device(torch.device("cpu"))
        net = pol.get_model()
        up = collections.OrderedDict(
            (k, tuple(v.shape)) for k, v in net.state_dict().items())
        ck = collections.OrderedDict((k, tuple(v.shape)) for k, v in ckpt.items())
        print(f"--- {name}: {cls.default_model} vs "
              f"{type(net).__module__}.{type(net).__name__}")
        if up == ck:
            net.load_state_dict(ckpt, strict=True)
            print(f"    MATCH: {len(up)} tensors, all keys and shapes identical; "
                  f"strict load OK")
        else:
            ok = False
            print("    MISMATCH")
            for k in sorted(set(up) | set(ck)):
                if up.get(k) != ck.get(k):
                    print(f"      {k}: upstream={up.get(k)} checkpoint={ck.get(k)}")
    print("\nRESULT:", "checkpoints ARE upstream-compatible" if ok
          else "checkpoints are NOT upstream-compatible -- retraining required")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
