#!/usr/bin/env python3
"""DS-RNN via the ORIGINAL published CrowdNav_DSRNN code and weights.

Upstream: https://github.com/Shuijing725/CrowdNav_DSRNN @ 91fb53c0, vendored
verbatim under ``sim/third_party/crowdnav_dsrnn/`` (MIT, see its LICENSE,
COMMIT and PATCHES.md).

    Shuijing Liu, Peixin Chang, Weihang Liang, Neeloy Chakraborty,
    Katherine Driggs-Campbell, "Decentralized Structural-RNN for Robot Crowd
    Navigation with Deep Reinforcement Learning", ICRA 2021.
    https://arxiv.org/abs/2011.04820

Nothing in the network or the maths is reimplemented here: this file builds
upstream's ``pytorchBaselines.a2c_ppo_acktr.model.Policy`` with upstream's own
``Config``, loads upstream's own published checkpoint, and calls
``Policy.act(..., deterministic=True)`` exactly as upstream's ``test.py`` /
``pytorchBaselines/evaluation.py`` do.

WEIGHTS
-------
This is an RL policy: without trained weights it is meaningless. Upstream ships
two trained checkpoints IN THE REPOSITORY, and both are vendored:

    data/example_model/checkpoints/27776.pt           holonomic  <- used here
    data/example_model_unicycle/checkpoints/55554.pt  unicycle

The holonomic one is used because the benchmark's planner interface is a
holonomic (vx, vy). All 47 state_dict keys load with ``strict=True`` against
the upstream network definition; if a checkpoint is missing or does not match,
this adapter raises. It never falls back to random weights or a heuristic.

WHAT THE OBSERVATION IS, AND WHAT WE MAP ONTO IT
------------------------------------------------
Taken from upstream ``crowd_sim/envs/crowd_sim_dict.py::generate_ob`` (kept for
audit at ``third_party/crowdnav_dsrnn/_ref/crowd_sim/envs/crowd_sim_dict.py``),
the policy consumes a dict of three arrays:

    robot_node     (1, 7)  [px, py, radius, gx, gy, v_pref, theta]
                           = Agent.get_full_state_list_noV()
    temporal_edges (1, 2)  [robot vx, robot vy]
    spatial_edges  (5, 2)  human_i_position - robot_position, ABSOLUTE axes,
                           one row per human, upstream's human index order

The benchmark's ``Obstacle(pid, x, y, vx, vy)`` maps on as follows:

    Obstacle.x, .y  ->  spatial_edges row = (o.x - state.x, o.y - state.y)
    Obstacle.vx,.vy ->  NOT USED. DS-RNN's spatial edge is position only; the
                        temporal edge carries the ROBOT's velocity, not the
                        pedestrians'. Pedestrian motion reaches the policy only
                        through the edge GRU's memory across steps, which is
                        the whole point of the architecture. Feeding velocities
                        anywhere would be inventing an input the network never
                        saw.
    Obstacle.pid    ->  used only to order/deduplicate; DS-RNN has no identity
                        tracking (upstream's slot i is just "human i").

FRAME: a TRANSLATION of the benchmark's leg-local frame into upstream's +/-6 m
arena. The leg frame is NOT rotated: upstream samples robot start and goal
independently from U(-6, 6)^2 (``crowd_sim.py::generate_robot_humans``, FoV
branch), so the goal bearing is uniformly random in training and there is no
canonical heading to rotate to. The translation matters because robot_node
carries ABSOLUTE px, py, gx, gy -- it is a network input, not bookkeeping.
``origin_mode`` selects it:

    "centred_pair" (default) robot at -(goal_clip/2) * u, clipped goal at
                             +(goal_clip/2) * u, so the robot/goal pair
                             straddles the arena centre. This is the modal
                             training configuration for a given remaining
                             distance: with start and goal both ~U(-6, 6)^2 and
                             separated by >= 6, a robot with 6 m still to run
                             sits roughly 3 m short of centre with its goal
                             roughly 3 m past it.
    "robot_origin"           robot at (0, 0), goal at goal_clip * u.

Measured: the choice barely matters for this policy (open-field cross-track bias
over 30 s changed from a mean of -5.0 deg to -4.1 deg off the goal line), which
is itself worth knowing -- DS-RNN is close to translation invariant in practice
even though it is not by construction.

ASSUMPTIONS THIS WRAPPER HAS TO MAKE (all exposed as attributes)
----------------------------------------------------------------
* ``max_humans`` = 5. Upstream's ``sim.human_num`` is 5 and the observation has
  exactly 5 slots, always filled (DS-RNN's robot has no sensor range and a 2*pi
  FOV, so all 5 humans are always visible). No network weight depends on this
  number, but the policy only ever saw 5, so 5 it is. When the benchmark reports
  MORE than 5 pedestrians we keep the 5 nearest; when it reports FEWER we pad
  with upstream's own dummy human, which sits at absolute (15, 15) with the
  robot at the origin (``crowd_sim.py::update_last_human_states``, reset
  branch), i.e. a spatial edge of (15, 15). Order within the 5 is by ascending
  distance; upstream's order is the arbitrary human index, so this is a free
  choice that also makes the "keep nearest 5" truncation well defined.
* ``goal_clip_m`` = 6.0. The benchmark's leg goal can be hundreds of metres
  away; upstream's goal is drawn inside a +/-6 m box with a start-goal distance
  of at least 6 m, so a goal at (6, 0) relative to a robot at the origin is a
  typical mid-episode state and a goal at (300, 0) is not a state the policy has
  ever seen. The goal handed to the network is therefore the real goal moved
  along the robot->goal ray to at most 6.0 m. 6.0 is upstream's own
  ``sim.circle_radius`` / arena half-width, not a tuned number.
* ``radius`` = 0.3 and ``v_pref`` = 1.0 are upstream's CONSTANTS for every
  training episode (``config.robot.radius`` / ``config.robot.v_pref``), so they
  are fed as-is rather than as the benchmark's 0.25 m / cfg.max_speed: the
  network has only ever seen those two numbers in those slots. The benchmark's
  speed limit is applied to the OUTPUT instead (see below).
* ``theta`` = pi/2. For holonomic kinematics upstream never updates
  ``Agent.theta`` (``crowd_sim/envs/utils/agent.py::step`` only assigns theta in
  the unicycle branch), so robot_node[6] is the constant pi/2 in 100% of the
  training data. Feeding the robot's true heading would put an input outside its
  entire training support.
* Control period. Upstream trains at ``env.time_step`` = 0.25 s; the benchmark
  defaults to ``--step-length 0.5``. The action is a velocity so the units are
  unaffected, but the edge/node GRUs advance once per control step, so their
  effective time constant doubles at dt = 0.5. Run the benchmark with
  ``--step-length 0.25`` to match upstream exactly.
* Static geometry. DS-RNN has no notion of walls or kerbs -- upstream's arena is
  open space. The sidewalk band is enforced only by the benchmark's
  ``apply_velocity`` clamp, so band-violation metrics measure the policy being
  clamped, not the policy avoiding a kerb. Nothing is added to the observation
  to fake it.

Action post-processing is upstream's, in upstream's order: the deterministic
mode of the DiagGaussian is clipped to v_pref by the norm clip in
``crowd_nav/policy/srnn.py::SRNN.clip_action`` (holonomic branch), and only then
clipped again to the benchmark's ``cfg.max_speed``.

MEASURED (torch 2.12, numpy 2.4, Python 3.13, CPU, single thread)
-----------------------------------------------------------------
* 2.5 ms per control step (median over 0/1/3/5/6/12/25-pedestrian scenes,
  including pedestrians pinned to both band edges), p90 2.9 ms, max 3.2 ms.
  Construction ~2.6 s, then cached per (checkpoint, device).
* Sanity check on upstream's OWN scenario (5 humans, circle radius 6, robot
  (0,-6) -> (0,+6), dt 0.25 s, NON-reactive humans, which is harder than
  upstream's cooperating ORCA humans): goal reached in 12/12 seeds, mean 16.1 s
  against a 12 s straight-line time, mean closest-pedestrian distance 0.89 m,
  11/12 seeds never closer than 0.6 m. The policy behaves as trained.
* KNOWN TRANSPLANT EFFECT, report it rather than tune it away: in an open field
  with the goal held at the clip distance, the policy does not track the goal
  line exactly. Cross-track angle over 30 s, swept over eight goal bearings:
  mean -5 deg, sd 8 deg, worst 16 deg. It is direction-dependent in the world
  frame, so it is not a simple "keep right" convention, and it survives both
  origin_mode settings and the presence or absence of pedestrians. On a 4 m
  sidewalk band with the goal hundreds of metres along +x, a persistent few
  degrees of drift walks the robot into a band edge, where the benchmark's
  ``apply_velocity`` clamp holds it. That is a real property of a
  circle-crossing-trained policy placed in a corridor, not a wrapper artefact.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

try:
    from sidewalk_robot_common import Obstacle, PlannerConfig, RobotState
except ImportError:                                  # direct import fallback
    from .sidewalk_robot_common import Obstacle, PlannerConfig, RobotState


VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "crowdnav_dsrnn"
DEFAULT_MODEL = VENDOR_ROOT / "data" / "example_model" / "checkpoints" / "27776.pt"
DEFAULT_CONFIG = VENDOR_ROOT / "data" / "example_model" / "configs" / "config.py"

# Shapes of every parameter of upstream's Policy(base='srnn') under upstream's
# shipped Config. Checked against the checkpoint BEFORE load_state_dict so a
# mismatch reports which tensor is wrong instead of a wall of key names.
_EXPECTED_SHAPES: Dict[str, Tuple[int, ...]] = {
    "base.humanNodeRNN.gru.weight_ih_l0": (384, 128),
    "base.humanNodeRNN.gru.weight_hh_l0": (384, 128),
    "base.humanNodeRNN.encoder_linear.weight": (64, 3),
    "base.humanNodeRNN.edge_embed.weight": (64, 256),
    "base.humanNodeRNN.edge_attention_embed.weight": (64, 512),
    "base.humanNodeRNN.output_linear.weight": (256, 128),
    "base.humanhumanEdgeRNN_spatial.gru.weight_ih_l0": (768, 64),
    "base.humanhumanEdgeRNN_spatial.gru.weight_hh_l0": (768, 256),
    "base.humanhumanEdgeRNN_spatial.encoder_linear.weight": (64, 2),
    "base.humanhumanEdgeRNN_temporal.gru.weight_ih_l0": (768, 64),
    "base.humanhumanEdgeRNN_temporal.encoder_linear.weight": (64, 2),
    "base.attn.temporal_edge_layer.0.weight": (64, 256),
    "base.attn.spatial_edge_layer.0.weight": (64, 256),
    "base.robot_linear.weight": (3, 7),
    "base.critic_linear.weight": (1, 256),
    "dist.fc_mean.weight": (2, 256),
    "dist.logstd._bias": (2, 1),
}

_MISSING_WEIGHTS_HINT = (
    "DS-RNN is a reinforcement-learning policy: without its trained weights it "
    "is not DS-RNN, it is a random network, and a random network in a benchmark "
    "is worse than an absent one.\n"
    "Upstream ships the trained checkpoint inside the repository and it is "
    "vendored with this benchmark at\n"
    f"    {DEFAULT_MODEL}\n"
    "If that file is gone, restore it from\n"
    "    https://github.com/Shuijing725/CrowdNav_DSRNN @ 91fb53c0\n"
    "    data/example_model/checkpoints/27776.pt\n"
    "(sha256 09eb9965e60159a0f1ab00ecfd420b31fa512fa12953678b796e5448b6bb3499)"
)


# The benchmark rebuilds a planner per leg. Building this one costs ~3 s (torch
# import + checkpoint load), so the network is cached per (checkpoint, device).
# ONLY the nn.Module is shared: every recurrent state lives on the planner
# instance and is passed into Policy.act(), which stores nothing on the module,
# so two planners sharing a cache entry cannot leak state into each other.
_POLICY_CACHE: Dict[Tuple[str, str], Any] = {}


def _load_upstream_config(config_path: Path):
    """Exec upstream's own config.py and return its Config().

    Loaded by file path rather than by package import so that no top-level
    ``crowd_nav`` package appears on sys.path -- another vendored CrowdNav lives
    under third_party/crowdnav/ and must not be shadowed.
    """
    spec = importlib.util.spec_from_file_location("_crowdnav_dsrnn_config", config_path)
    if spec is None or spec.loader is None:            # pragma: no cover
        raise ImportError(f"cannot load upstream DS-RNN config from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Config()


class CrowdNavDSRNNPlanner:
    """DS-RNN local planner backed by upstream's network and published weights."""

    def __init__(self, cfg: PlannerConfig, seed: int = 0,
                 model_path: Any = None, device: str = "cpu"):
        import torch

        self.cfg = cfg
        self.seed = int(seed)
        self.device = torch.device(device)
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"DS-RNN checkpoint not found: {self.model_path}\n\n{_MISSING_WEIGHTS_HINT}")

        if str(VENDOR_ROOT) not in sys.path:
            sys.path.insert(0, str(VENDOR_ROOT))

        # Upstream's test.py does this; on this machine it is a 27x speedup
        # (96 ms -> 3.6 ms per step) because the tensors are tiny and the
        # default 16-thread OpenMP pool spends all its time synchronising.
        # It is a PROCESS-GLOBAL setting: set torch_threads=None to skip it.
        self.torch_threads = 1
        if self.torch_threads is not None:
            torch.set_num_threads(int(self.torch_threads))

        self.upstream_config = _load_upstream_config(DEFAULT_CONFIG)
        self.upstream_config.training.cuda = (self.device.type == "cuda")

        # --- upstream constants, exposed so they can be measured, not hidden
        self.max_humans = int(self.upstream_config.sim.human_num)      # 5
        self.dummy_human_xy = 15.0        # crowd_sim.py::update_last_human_states
        self.goal_clip_m = 6.0            # == upstream sim.circle_radius
        self.radius = float(self.upstream_config.robot.radius)         # 0.3
        self.v_pref = float(self.upstream_config.robot.v_pref)         # 1.0
        self.theta = math.pi / 2.0        # constant for holonomic upstream
        self.sensor_range = None          # DS-RNN has none; None = use all given
        self.deterministic = True         # upstream evaluation.py
        # Where inside upstream's +/-6 m arena we place the robot. robot_node
        # carries ABSOLUTE px, py, gx, gy, so this is a real input, not a
        # bookkeeping detail. See the module docstring.
        #   "centred_pair" -- robot at -(goal_clip/2)*u, goal at +(goal_clip/2)*u
        #   "robot_origin" -- robot at (0, 0), goal at goal_clip*u
        self.origin_mode = "centred_pair"

        key = (str(self.model_path), str(self.device))
        if key not in _POLICY_CACHE:
            _POLICY_CACHE[key] = self._build_policy()
        self.policy = _POLICY_CACHE[key]
        self.hxs = self._zero_hidden()
        self.masks = None                 # set by reset()
        self.reset()

    # ------------------------------------------------------------ construction
    def _build_policy(self):
        import torch
        from gymnasium import spaces
        from pytorchBaselines.a2c_ppo_acktr.model import Policy

        cfgu = self.upstream_config
        obs_space = {
            "robot_node": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7),
                                     dtype=np.float32),
            "temporal_edges": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2),
                                         dtype=np.float32),
            "spatial_edges": spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.max_humans, 2), dtype=np.float32),
        }
        high = np.inf * np.ones([2], dtype=np.float32)
        action_space = spaces.Box(-high, high, dtype=np.float32)

        policy = Policy(obs_space, action_space, base_kwargs=cfgu,
                        base=cfgu.robot.policy)

        state_dict = torch.load(self.model_path, map_location=self.device)
        self._verify_state_dict(state_dict, policy)
        policy.load_state_dict(state_dict, strict=True)   # raises on any mismatch
        policy.base.nenv = 1                              # upstream test.py
        policy.to(self.device)
        policy.eval()
        return policy

    def _verify_state_dict(self, state_dict, policy) -> None:
        """Check the published weights really are this network's weights."""
        own = policy.state_dict()
        missing = sorted(set(own) - set(state_dict))
        extra = sorted(set(state_dict) - set(own))
        if missing or extra:
            raise RuntimeError(
                f"DS-RNN checkpoint {self.model_path} does not match upstream's "
                f"network definition.\n  missing keys: {missing}\n"
                f"  unexpected keys: {extra}\n\n{_MISSING_WEIGHTS_HINT}")
        bad = []
        for k, want in _EXPECTED_SHAPES.items():
            got = tuple(state_dict[k].shape)
            if got != want:
                bad.append(f"{k}: checkpoint {got} != upstream {want}")
        for k, ref in own.items():
            if tuple(state_dict[k].shape) != tuple(ref.shape):
                bad.append(f"{k}: checkpoint {tuple(state_dict[k].shape)} != "
                           f"network {tuple(ref.shape)}")
        if bad:
            raise RuntimeError(
                f"DS-RNN checkpoint {self.model_path} has the right keys but the "
                "wrong shapes:\n  " + "\n  ".join(bad) + f"\n\n{_MISSING_WEIGHTS_HINT}")

    def _zero_hidden(self) -> dict:
        """Hidden state exactly as upstream's evaluation.py allocates it."""
        import torch
        srnn = self.upstream_config.SRNN
        return {
            "human_node_rnn": torch.zeros(1, 1, srnn.human_node_rnn_size,
                                          device=self.device),
            "human_human_edge_rnn": torch.zeros(1, self.max_humans + 1,
                                                srnn.human_human_edge_rnn_size,
                                                device=self.device),
        }

    def reset(self) -> None:
        """Start a new episode/leg: zero the GRUs and re-raise the done mask.

        Upstream's evaluation loop starts with masks = 0 (which multiplies the
        hidden state by zero on the first forward) and uses 1 thereafter.
        """
        import torch
        self.hxs = self._zero_hidden()
        self.masks = torch.zeros(1, 1, device=self.device)

    # ------------------------------------------------------------ observation
    def _build_obs(self, state: RobotState, goal: Tuple[float, float],
                   obstacles: Sequence[Obstacle]) -> Tuple[dict, int, Tuple[float, float]]:
        import torch

        # local goal, clipped into the training support along the robot->goal ray
        dx, dy = float(goal[0]) - state.x, float(goal[1]) - state.y
        d = math.hypot(dx, dy)
        if d > self.goal_clip_m:
            s = self.goal_clip_m / d
            gx, gy = dx * s, dy * s
        else:
            gx, gy = dx, dy

        # absolute robot position inside upstream's arena (see origin_mode)
        if self.origin_mode == "centred_pair":
            ox, oy = -gx / 2.0, -gy / 2.0
        elif self.origin_mode == "robot_origin":
            ox, oy = 0.0, 0.0
        else:
            raise ValueError(f"unknown origin_mode {self.origin_mode!r}")

        # robot velocity in the leg frame (the benchmark integrates and limits
        # the command, so this is what the robot actually did, which is what
        # upstream's Agent.vx/vy hold)
        vx = state.v * math.cos(state.yaw)
        vy = state.v * math.sin(state.yaw)

        rel = []
        for o in obstacles:
            hx, hy = float(o.x) - state.x, float(o.y) - state.y
            if self.sensor_range is not None and math.hypot(hx, hy) > self.sensor_range:
                continue
            rel.append((hx * hx + hy * hy, hx, hy))
        rel.sort(key=lambda r: r[0])
        n_used = min(len(rel), self.max_humans)

        # padding rows: upstream's unseen-human placeholder is at ABSOLUTE
        # (15, 15), so its spatial edge is (15 - px, 15 - py)
        spatial = np.empty((self.max_humans, 2), dtype=np.float32)
        spatial[:, 0] = self.dummy_human_xy - ox
        spatial[:, 1] = self.dummy_human_xy - oy
        for i in range(n_used):
            spatial[i, 0] = rel[i][1]
            spatial[i, 1] = rel[i][2]

        robot_node = np.array([[ox, oy, self.radius, ox + gx, oy + gy,
                                self.v_pref, self.theta]], dtype=np.float32)
        temporal = np.array([[vx, vy]], dtype=np.float32)

        obs = {
            "robot_node": torch.from_numpy(robot_node).unsqueeze(0).to(self.device),
            "temporal_edges": torch.from_numpy(temporal).unsqueeze(0).to(self.device),
            "spatial_edges": torch.from_numpy(spatial).unsqueeze(0).to(self.device),
        }
        return obs, n_used, (gx, gy)

    # ---------------------------------------------------------------- control
    def compute_command(self, state: RobotState, goal: Tuple[float, float],
                        obstacles: Sequence[Obstacle],
                        sim_time: float) -> Tuple[float, float, Dict[str, Any]]:
        import torch

        obs, n_used, (gx, gy) = self._build_obs(state, goal, obstacles)

        with torch.no_grad():
            value, action, _logp, hxs = self.policy.act(
                obs, self.hxs, self.masks, deterministic=self.deterministic)
        self.hxs = hxs
        self.masks = torch.ones(1, 1, device=self.device)

        a = action[0].detach().cpu().numpy().astype(float)
        vx, vy = float(a[0]), float(a[1])

        # upstream crowd_nav/policy/srnn.py::SRNN.clip_action, holonomic branch
        n = math.hypot(vx, vy)
        if n > self.v_pref:
            vx, vy = vx / n * self.v_pref, vy / n * self.v_pref
        # benchmark envelope on top (apply_velocity also enforces it)
        n = math.hypot(vx, vy)
        if n > self.cfg.max_speed > 0.0:
            vx, vy = vx / n * self.cfg.max_speed, vy / n * self.cfg.max_speed

        return vx, vy, {
            "status": "crowdnav_dsrnn",
            "value": round(float(value.item()), 4),
            "n_humans_used": int(n_used),
            "n_humans_seen": int(len(obstacles)),
            "local_goal_x": round(gx, 3),
            "local_goal_y": round(gy, 3),
        }
