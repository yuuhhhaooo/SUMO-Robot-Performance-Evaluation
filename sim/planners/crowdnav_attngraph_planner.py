#!/usr/bin/env python3
"""CrowdNav++ (attention-based interaction graph + GST intention prediction)
via the ORIGINAL published code and weights.

Upstream: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph @ 39077313,
vendored verbatim under ``sim/third_party/crowdnav_attngraph/`` (MIT, see its
LICENSE, COMMIT and PATCHES.md).

    Shuijing Liu, Peixin Chang, Zhe Huang, Neeloy Chakraborty, Kaiwen Hong,
    Weihang Liang, D. Livingston McPherson, Junyi Geng, Katherine
    Driggs-Campbell, "Intention Aware Robot Crowd Navigation with
    Attention-Based Interaction Graph", ICRA 2023.
    https://arxiv.org/abs/2203.01821

The policy is only half of the method. Its observation contains PREDICTED future
pedestrian positions produced by a second published network, the Gumbel Social
Transformer:

    Zhe Huang, Ruohua Li, Kazuki Shin, Katherine Driggs-Campbell, "Learning
    Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory
    Prediction", RA-L 2022.

Both networks are upstream's, both are used unmodified, and both trained
checkpoints ship in upstream's repository. Nothing is reimplemented here: this
file builds ``rl.networks.model.Policy(base='selfAttn_merge_srnn')`` with
upstream's own ``arguments.get_args()`` and ``Config``, builds upstream's
``CrowdNavPredInterfaceMultiEnv``, assembles the observation exactly as
``rl/vec_env/vec_pretext_normalize.py::VecPretextNormalize.process_obs_rew``
does, and calls ``Policy.act(..., deterministic=True)`` as
``rl/evaluation.py`` does.

WEIGHTS
-------
Two policy checkpoints and two matching predictor checkpoints ship upstream and
are vendored:

    trained_models/GST_predictor_non_rand/checkpoints/41200.pt   <- default
    trained_models/GST_predictor_rand/checkpoints/41665.pt
    gst_updated/results/100-...-seed_1000/sj/checkpoint/epoch_100.pt       (non_rand)
    gst_updated/results/100-...-seed_1000_rand/sj/checkpoint/epoch_100.pt  (rand)

The policy<->predictor pairing is upstream's, stated in
``gst_updated/results/README.md``; ``model_path`` and ``gst_model_dir`` are
paired automatically and both are overridable. All 47 policy state_dict keys
load with ``strict=True``. If a checkpoint is missing or does not match, this
adapter raises: it never falls back to random weights or a heuristic.

Two upstream files in ``trained_models/*/`` (``arguments.py``, ``configs/config.py``)
are STALE relative to their own checkpoints -- they carry the repository
defaults ``--env-name CrowdSimVarNum-v0`` and ``sim.predict_method='none'``,
which build a 2-dimensional spatial edge. The shipped weights have
``base.spatial_attn.embedding_layer.0.weight`` of shape (128, 12), which
``rl/networks/selfAttn_srnn_temp_node.py::SpatialEdgeSelfAttn`` only produces
for ``env_name in {CrowdSimPred-v0, CrowdSimPredRealGST-v0}``. The weights
therefore settle the question: this adapter uses ``CrowdSimPredRealGST-v0`` and
the root ``crowd_nav/configs/config.py`` (``predict_method='inferred'``), which
is also what the model directory's name says.

WHAT THE OBSERVATION IS, AND WHAT WE MAP ONTO IT
------------------------------------------------
From ``crowd_sim/envs/crowd_sim_var_num.py::generate_ob``,
``crowd_sim/envs/crowd_sim_pred_real_gst.py::generate_ob`` and the wrapper
``rl/vec_env/vec_pretext_normalize.py`` (all kept for audit under
``third_party/crowdnav_attngraph/_ref/``):

    robot_node         (1, 7)   [px, py, radius, gx, gy, v_pref, theta]
    temporal_edges     (1, 2)   [robot vx, robot vy]
                                -> concatenated to the 9-d robot state that
                                   base.robot_linear (256, 9) consumes
    spatial_edges      (20, 12) per human, 2*(1 + predict_steps) numbers:
                                cols 0:2   current position - robot position
                                cols 2:12  GST-predicted positions at t+1..t+5
                                           control steps, MINUS THE CURRENT
                                           robot position (not the future one)
                                rows sorted by ascending current distance;
                                rows for undetected humans are all 15.0
    detected_human_num (1,)     number of detected humans, floored at 1
    visible_masks      (20,)    bool, per slot; feeds the predictor's
                                partial-detection mask

The benchmark's ``Obstacle(pid, x, y, vx, vy)`` maps on as follows:

    Obstacle.x, .y  ->  spatial_edges[:, 0:2] = (o.x - state.x, o.y - state.y),
                        and the per-pedestrian position history fed to GST.
    Obstacle.pid    ->  the identity key for that history. This one matters:
                        GST predicts from a 5-frame track per pedestrian, so
                        the wrapper must know which detection at step k is the
                        same person as which detection at step k-1. Upstream
                        gets identity for free from its simulator's human index;
                        here the SUMO/JuPedSim person id does the same job.
    Obstacle.vx,.vy ->  NOT fed to either network. Neither the policy's spatial
                        edge nor GST's input has a velocity slot -- GST is given
                        positions and takes first differences itself
                        (``crowd_nav_interface_parallel.py``: obs_traj_rel).
                        The benchmark's vx/vy are finite differences of the same
                        positions, so feeding them anywhere would be inventing
                        an input the networks never saw.

FRAME: a TRANSLATION of the benchmark's leg-local frame into upstream's +/-6 m
arena, no rotation. Upstream draws robot start and goal independently from
U(-6, 6)^2 with a start-goal distance >= 8
(``crowd_sim_var_num.py::generate_robot_humans``, sim branch), so the goal
bearing is uniformly random in training and there is no canonical heading to
rotate to. The translation is a real network input (robot_node carries absolute
px, py, gx, gy) and ``origin_mode`` selects it:

    "centred_pair" (default) robot at -(goal_clip/2) * u, clipped goal at
                             +(goal_clip/2) * u, straddling the arena centre --
                             the modal training configuration for a given
                             remaining distance.
    "robot_origin"           robot at (0, 0), goal at goal_clip * u.

GST is translation invariant (it consumes first differences and pairwise
position differences and adds the last observed position back at the end), so
the pedestrian history is expressed in the same translated frame and upstream's
own ``out_traj - robot_node[:2]`` line then yields robot-relative predictions
exactly as it does upstream. Measured: the choice makes almost no difference
(open-field cross-track bias mean -4.4 deg vs -3.0 deg off the goal line).

ASSUMPTIONS THIS WRAPPER HAS TO MAKE (all exposed as attributes)
----------------------------------------------------------------
* ``goal_clip_m`` = 6.0. The benchmark's leg goal can be hundreds of metres
  away; upstream's goal always lies inside a +/-6 m box. The goal handed to the
  network is the real goal moved along the robot->goal ray to at most 6.0 m.
  6.0 is upstream's own ``sim.arena_size``, not a tuned number.
* ``sensor_range`` = 5.0 m, upstream's ``config.robot.sensor_range``, measured
  surface-to-surface as in ``crowd_sim.py::detect_visible``
  (dist - r_robot - r_human). The benchmark hands over everything inside
  ``cfg.sensor_range`` (11 m by default); pedestrians beyond upstream's 5 m were
  never visible to this policy in training, so they are dropped rather than
  shown to it. Set ``sensor_range=None`` to pass everything through and measure
  the difference.
* ``max_humans`` = 20 = upstream ``sim.human_num + sim.human_num_range``. No
  weight depends on it (the attention is masked and variable length), but it
  sets the padded observation width upstream trained with. If more than 20
  pedestrians are inside the sensor range the nearest 20 are kept and
  ``info['n_humans_dropped']`` is non-zero.
* ``radius`` = 0.3 and ``v_pref`` = 1.0 are upstream's constants for every
  training episode, so they are fed as-is rather than as the benchmark's 0.25 m
  / cfg.max_speed. The benchmark's speed limit is applied to the OUTPUT.
* ``theta`` = pi/2. For holonomic kinematics upstream never updates
  ``Agent.theta``, so robot_node[6] is the constant pi/2 in 100% of the training
  data.
* Prediction time base. GST was trained to predict on a ``data.pred_timestep``
  = 0.25 s grid and the policy consumes 5 such steps. The benchmark's control
  period is ``cfg.dt`` (0.5 s by default), so the per-pedestrian position
  history is LINEARLY INTERPOLATED onto the 0.25 s grid from the recorded
  (sim_time, x, y) samples. Grid slots earlier than a pedestrian's first
  detection are marked invalid through upstream's own partial-detection mask,
  which is the mechanism GST was built for; a pedestrian detected for the first
  time this step gets no prediction at all and its spatial edge stays at the
  tiled current position, exactly as upstream behaves at episode start. Note
  the CONSEQUENCE: the predicted horizon is 1.25 s of real time regardless of
  cfg.dt, whereas the policy advances its GRU once per control step -- run with
  ``--step-length 0.25`` to match upstream exactly.
* Static geometry. CrowdNav++ has no notion of walls or kerbs; upstream's arena
  is open space. The sidewalk band is enforced only by the benchmark's
  ``apply_velocity`` clamp. Nothing is added to the observation to fake it.

Action post-processing is upstream's: the deterministic mode of the DiagGaussian
is norm-clipped to v_pref (``crowd_nav/policy/srnn.py::clip_action``, holonomic
branch), then clipped again to the benchmark's ``cfg.max_speed``.

MEASURED (torch 2.12, numpy 2.4, Python 3.13, CPU, single thread)
-----------------------------------------------------------------
* 15.2 ms per control step (median over 0/1/3/5/6/12/25-pedestrian scenes,
  including pedestrians pinned to both band edges), p90 16.5 ms, max 24 ms --
  roughly 6x DS-RNN, and the GST forward pass is essentially all of it.
  Construction ~0.8 s once torch is warm, then cached per
  (checkpoint, GST dir, device).
* Sanity check on upstream's OWN scenario (20 humans, circle radius 6*sqrt(2),
  robot (0,-R) -> (0,+R), dt 0.25 s, NON-reactive humans, harder than upstream's
  cooperating ORCA humans): goal reached in 12/12 seeds, mean 23.5 s against a
  17 s straight-line time, mean closest-pedestrian distance 0.78 m, 11/12 seeds
  never closer than 0.6 m. The policy and predictor behave as trained.
* KNOWN TRANSPLANT EFFECT, report it rather than tune it away: with the goal held
  at the clip distance in an open field, cross-track angle over 30 s swept across
  eight goal bearings is mean -3 deg but sd 18 deg, worst 33 deg -- markedly more
  direction-dependent than DS-RNN. Part of it is upstream's own construction for
  an empty neighbourhood: when nothing is inside the 5 m sensor range every
  spatial-edge row is the 15.0 placeholder and ``detected_human_num`` is floored
  at 1, so the attention is pointed at a phantom pedestrian at (15, 15). Moving
  that placeholder to (-15, -15) visibly changes the bearing pattern, which is
  how this was identified. Upstream rarely meets an empty neighbourhood (20
  humans in a 6*sqrt(2) circle) whereas a sidewalk often will. The rest of the
  bias persists with pedestrians continuously in view (sd 17 deg). On a 4 m band
  with the goal hundreds of metres along +x this walks the robot into a band
  edge, where the benchmark's ``apply_velocity`` clamp holds it.
"""
from __future__ import annotations

import importlib.util
import math
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from sidewalk_robot_common import Obstacle, PlannerConfig, RobotState
except ImportError:                                  # direct import fallback
    from .sidewalk_robot_common import Obstacle, PlannerConfig, RobotState


VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "crowdnav_attngraph"
_GST_BASE = ("100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-"
             "edge_head_0-ebd_64-snl_1-snh_8-seed_1000")

DEFAULT_MODEL = (VENDOR_ROOT / "trained_models" / "GST_predictor_non_rand" /
                 "checkpoints" / "41200.pt")
# upstream gst_updated/results/README.md: non-randomised humans <-> plain dir,
# randomised humans <-> the _rand dir.
_GST_FOR_MODEL = {
    "41200.pt": VENDOR_ROOT / "gst_updated" / "results" / _GST_BASE / "sj",
    "41665.pt": VENDOR_ROOT / "gst_updated" / "results" / (_GST_BASE + "_rand") / "sj",
}
DEFAULT_GST = _GST_FOR_MODEL["41200.pt"]

# Parameter shapes of upstream's Policy(base='selfAttn_merge_srnn') built with
# env_name='CrowdSimPredRealGST-v0'. Checked before load_state_dict so a
# mismatch names the offending tensor.
_EXPECTED_SHAPES: Dict[str, Tuple[int, ...]] = {
    "base.humanNodeRNN.gru.weight_ih_l0": (384, 128),
    "base.humanNodeRNN.gru.weight_hh_l0": (384, 128),
    "base.humanNodeRNN.encoder_linear.weight": (64, 256),
    "base.humanNodeRNN.edge_attention_embed.weight": (64, 256),
    "base.humanNodeRNN.output_linear.weight": (256, 128),
    "base.attn.temporal_edge_layer.0.weight": (64, 256),
    "base.attn.spatial_edge_layer.0.weight": (64, 256),
    "base.robot_linear.0.weight": (256, 9),        # temporal_edges(2) + robot_node(7)
    "base.spatial_attn.embedding_layer.0.weight": (128, 12),   # 2*(1+predict_steps)
    "base.spatial_attn.embedding_layer.2.weight": (512, 128),
    "base.spatial_attn.multihead_attn.in_proj_weight": (1536, 512),
    "base.spatial_linear.0.weight": (256, 512),
    "base.critic_linear.weight": (1, 256),
    "dist.fc_mean.weight": (2, 256),
    "dist.logstd._bias": (2, 1),
}

_MISSING_WEIGHTS_HINT = (
    "CrowdNav++ is a reinforcement-learning policy driven by a learned "
    "trajectory predictor: without the trained weights it is not CrowdNav++, it "
    "is two random networks, and a random network in a benchmark is worse than "
    "an absent one.\n"
    "Upstream ships both checkpoints inside the repository and both are "
    "vendored with this benchmark at\n"
    f"    {DEFAULT_MODEL}\n"
    f"    {DEFAULT_GST / 'checkpoint' / 'epoch_100.pt'}\n"
    "If they are gone, restore them from\n"
    "    https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph @ 39077313\n"
    "(sha256 of 41200.pt: "
    "b829fbf511efcc95ff745e9ff9daa6c0ad5aacdf45f599127aec9990e3c52236)"
)

# torch.load defaults to weights_only=True since torch 2.6. The GST checkpoint
# is a full training checkpoint and carries numpy scalars (epoch losses)
# alongside the tensors, so it needs those two globals allowlisted. This keeps
# weights_only=True -- it does NOT fall back to arbitrary-code unpickling, and
# it needs no edit to upstream's crowd_nav_interface_parallel.py.
def _numpy_safe_globals() -> list:
    import numpy.core.multiarray as _nm
    allow = [(_nm.scalar, "numpy.core.multiarray.scalar"), (np.dtype, "numpy.dtype")]
    for name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType"):
        dt = getattr(np.dtypes, name, None)
        if dt is not None:
            allow.append((dt, f"numpy.dtypes.{name}"))
    return allow


# The benchmark rebuilds a planner per leg. Building this one loads two
# checkpoints, so the pair is cached per (policy checkpoint, GST dir, device).
# ONLY the nn.Modules are shared: every recurrent state and every pedestrian
# track lives on the planner instance, and neither Policy.act() nor the GST
# interface stores state on its module.
_BUILD_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _load_upstream_args_and_config(device_is_cuda: bool):
    """Return upstream's (args, Config()) built through upstream's own parser.

    ``arguments.get_args()`` calls ``parser.parse_args()``, and
    ``crowd_nav/configs/config.py`` imports and calls it at class-body time, so
    argv is replaced with upstream's own flags rather than any value being poked
    in afterwards. ``arguments`` is registered in sys.modules only for the
    duration of the config exec, so no generic top-level module name is left
    behind, and no ``crowd_nav`` package is put on sys.path (another vendored
    CrowdNav lives under third_party/crowdnav/ and must not be shadowed).
    """
    spec = importlib.util.spec_from_file_location(
        "_crowdnav_attngraph_arguments", VENDOR_ROOT / "arguments.py")
    if spec is None or spec.loader is None:            # pragma: no cover
        raise ImportError(f"cannot load {VENDOR_ROOT / 'arguments.py'}")
    argmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(argmod)

    argv = ["arguments.py", "--env-name", "CrowdSimPredRealGST-v0"]
    if not device_is_cuda:
        argv.append("--no-cuda")

    old_argv, old_mod = sys.argv, sys.modules.get("arguments")
    sys.argv = argv
    sys.modules["arguments"] = argmod
    try:
        args = argmod.get_args()
        cspec = importlib.util.spec_from_file_location(
            "_crowdnav_attngraph_config", VENDOR_ROOT / "configs" / "config.py")
        cmod = importlib.util.module_from_spec(cspec)
        cspec.loader.exec_module(cmod)
        config = cmod.Config()
    finally:
        sys.argv = old_argv
        if old_mod is None:
            sys.modules.pop("arguments", None)
        else:
            sys.modules["arguments"] = old_mod
    return args, config


class CrowdNavAttnGraphPlanner:
    """CrowdNav++ local planner backed by upstream's networks and weights."""

    def __init__(self, cfg: PlannerConfig, seed: int = 0,
                 model_path: Any = None, device: str = "cpu"):
        import torch

        self.cfg = cfg
        self.seed = int(seed)
        self.device = torch.device(device)
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL
        self.gst_model_dir = _GST_FOR_MODEL.get(self.model_path.name, DEFAULT_GST)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"CrowdNav++ policy checkpoint not found: {self.model_path}\n\n"
                f"{_MISSING_WEIGHTS_HINT}")
        gst_ckpt = self.gst_model_dir / "checkpoint" / "epoch_100.pt"
        if not gst_ckpt.is_file():
            raise FileNotFoundError(
                f"GST predictor checkpoint not found: {gst_ckpt}\n"
                "This policy's observation contains GST predictions; without the "
                "predictor the observation cannot be built at all.\n\n"
                f"{_MISSING_WEIGHTS_HINT}")

        if str(VENDOR_ROOT) not in sys.path:
            sys.path.insert(0, str(VENDOR_ROOT))

        # Upstream's test.py does this; the tensors are tiny and the default
        # OpenMP pool costs far more than the maths. PROCESS-GLOBAL: set
        # torch_threads=None to skip it.
        self.torch_threads = 1
        if self.torch_threads is not None:
            torch.set_num_threads(int(self.torch_threads))

        self.upstream_args, self.upstream_config = _load_upstream_args_and_config(
            self.device.type == "cuda")
        self.upstream_args.num_processes = 1          # nenv at inference

        ucfg = self.upstream_config
        # --- upstream constants, exposed so they can be measured, not hidden
        self.max_humans = int(ucfg.sim.human_num + ucfg.sim.human_num_range)   # 20
        self.predict_steps = int(ucfg.sim.predict_steps)                       # 5
        self.pred_timestep = float(ucfg.data.pred_timestep)                    # 0.25 s
        self.sensor_range = float(ucfg.robot.sensor_range)                     # 5.0 m
        self.human_radius = float(ucfg.humans.radius)                          # 0.3
        self.radius = float(ucfg.robot.radius)                                 # 0.3
        self.v_pref = float(ucfg.robot.v_pref)                                 # 1.0
        self.theta = math.pi / 2.0
        self.dummy_human_xy = 15.0     # crowd_sim_var_num.py::generate_ob
        self.goal_clip_m = 6.0         # == upstream sim.arena_size
        self.deterministic = True      # upstream rl/evaluation.py
        # Where inside upstream's +/-6 m arena we place the robot. robot_node
        # carries ABSOLUTE px, py, gx, gy, so this is a real input, not a
        # bookkeeping detail. See the module docstring.
        #   "centred_pair" -- robot at -(goal_clip/2)*u, goal at +(goal_clip/2)*u
        #   "robot_origin" -- robot at (0, 0), goal at goal_clip*u
        self.origin_mode = "centred_pair"

        key = (str(self.model_path), str(self.gst_model_dir), str(self.device))
        if key not in _BUILD_CACHE:
            _BUILD_CACHE[key] = self._build()
        self.policy, self.predictor, self.gst_args = _BUILD_CACHE[key]
        self.obs_seq_len = int(self.gst_args.obs_seq_len)                      # 5
        # seconds of per-pedestrian history GST needs, plus one control step of
        # slack so the oldest grid point is interpolatable rather than clamped
        self.history_span_s = (self.obs_seq_len - 1) * self.pred_timestep

        self.track: Dict[str, deque] = {}   # pid -> deque[(t, x, y)] leg-local
        self.hxs = self._zero_hidden()
        self.masks = None
        self.reset()

    # ------------------------------------------------------------ construction
    def _build(self):
        import torch
        from gymnasium import spaces
        from rl.networks.model import Policy
        from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import (
            CrowdNavPredInterfaceMultiEnv)

        dim = 2 * (self.predict_steps + 1)
        obs_space = {
            "robot_node": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7),
                                     dtype=np.float32),
            "temporal_edges": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2),
                                         dtype=np.float32),
            "spatial_edges": spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.max_humans, dim), dtype=np.float32),
            "detected_human_num": spaces.Box(low=-np.inf, high=np.inf, shape=(1,),
                                             dtype=np.float32),
            "visible_masks": spaces.Box(low=0, high=1, shape=(self.max_humans,),
                                        dtype=bool),
        }
        high = np.inf * np.ones([2], dtype=np.float32)
        action_space = spaces.Box(-high, high, dtype=np.float32)

        policy = Policy(obs_space, action_space, base_kwargs=self.upstream_args,
                        base=self.upstream_config.robot.policy)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self._verify_state_dict(state_dict, policy)
        policy.load_state_dict(state_dict, strict=True)
        policy.base.nenv = 1                          # upstream test.py
        policy.to(self.device)
        policy.eval()

        with open(self.gst_model_dir / "checkpoint" / "args.pickle", "rb") as f:
            gst_args = pickle.load(f)
        with torch.serialization.safe_globals(_numpy_safe_globals()):
            predictor = CrowdNavPredInterfaceMultiEnv(
                load_path=str(self.gst_model_dir), device=self.device,
                config=gst_args, num_env=1)
        return policy, predictor, gst_args

    def _verify_state_dict(self, state_dict, policy) -> None:
        own = policy.state_dict()
        missing = sorted(set(own) - set(state_dict))
        extra = sorted(set(state_dict) - set(own))
        if missing or extra:
            raise RuntimeError(
                f"CrowdNav++ checkpoint {self.model_path} does not match "
                f"upstream's network definition.\n  missing keys: {missing}\n"
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
                f"CrowdNav++ checkpoint {self.model_path} has the right keys but "
                "the wrong shapes:\n  " + "\n  ".join(bad) +
                f"\n\n{_MISSING_WEIGHTS_HINT}")

    def _zero_hidden(self) -> dict:
        """Hidden state exactly as upstream's rl/evaluation.py allocates it."""
        import torch
        base = self.policy.base
        return {
            "human_node_rnn": torch.zeros(1, 1, base.human_node_rnn_size,
                                          device=self.device),
            "human_human_edge_rnn": torch.zeros(1, base.human_num + 1,
                                                base.human_human_edge_rnn_size,
                                                device=self.device),
        }

    def reset(self) -> None:
        """Start a new episode/leg: zero the GRU, drop every pedestrian track."""
        import torch
        self.hxs = self._zero_hidden()
        self.masks = torch.zeros(1, 1, device=self.device)
        self.track.clear()

    # ------------------------------------------------------------- pedestrians
    def _visible(self, state: RobotState,
                 obstacles: Sequence[Obstacle]) -> Tuple[List[Tuple[str, float, float]], int]:
        """Apply upstream's sensor range, keep the nearest ``max_humans``.

        Upstream ``crowd_sim.py::detect_visible`` measures the range
        surface-to-surface: dist - r_robot - r_human <= sensor_range. The FOV
        test is skipped because upstream's robot.FOV is 2*pi, so it passes for
        every pedestrian.
        """
        out = []
        for o in obstacles:
            dx, dy = float(o.x) - state.x, float(o.y) - state.y
            d = math.hypot(dx, dy)
            if self.sensor_range is not None and \
                    d - self.radius - self.human_radius > self.sensor_range:
                continue
            out.append((d, str(o.pid), dx, dy))
        out.sort(key=lambda r: r[0])
        dropped = max(0, len(out) - self.max_humans)
        return [(pid, dx, dy) for _d, pid, dx, dy in out[:self.max_humans]], dropped

    def _record(self, sim_time: float, state: RobotState,
                visible: Sequence[Tuple[str, float, float]]) -> None:
        """Append leg-local absolute positions to each pedestrian's track."""
        keep = max(2, int(math.ceil(self.history_span_s / max(self.cfg.dt, 1e-6))) + 2)
        seen = set()
        for pid, dx, dy in visible:
            seen.add(pid)
            trk = self.track.get(pid)
            if trk is None or trk.maxlen != keep:
                trk = deque(trk or (), maxlen=keep)
                self.track[pid] = trk
            trk.append((float(sim_time), state.x + dx, state.y + dy))
        # forget people who left the sensor range: upstream's mask does the same
        for pid in [p for p in self.track if p not in seen]:
            del self.track[pid]

    def _history_grid(self, pid: str, sim_time: float,
                      state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """Sample one pedestrian's track onto upstream's obs_seq_len x 0.25 s grid.

        Returns (positions [obs_seq_len, 2] relative to the CURRENT robot
        position, valid mask [obs_seq_len]). Slots earlier than the pedestrian's
        first detection are invalid; GST is built for exactly that case.
        """
        pos = np.zeros((self.obs_seq_len, 2), dtype=np.float32)
        valid = np.zeros(self.obs_seq_len, dtype=bool)
        trk = self.track.get(pid)
        if not trk:
            return pos, valid
        ts = [p[0] for p in trk]
        for k in range(self.obs_seq_len):
            t = sim_time - (self.obs_seq_len - 1 - k) * self.pred_timestep
            if t < ts[0] - 1e-9 or t > ts[-1] + 1e-9:
                continue
            j = int(np.searchsorted(ts, t, side="left"))
            if j <= 0:
                x, y = trk[0][1], trk[0][2]
            elif j >= len(ts):
                x, y = trk[-1][1], trk[-1][2]
            else:
                t0, x0, y0 = trk[j - 1]
                t1, x1, y1 = trk[j]
                w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                x, y = x0 + w * (x1 - x0), y0 + w * (y1 - y0)
            pos[k, 0] = x - state.x
            pos[k, 1] = y - state.y
            valid[k] = True
        return pos, valid

    # ---------------------------------------------------------------- control
    def compute_command(self, state: RobotState, goal: Tuple[float, float],
                        obstacles: Sequence[Obstacle],
                        sim_time: float) -> Tuple[float, float, Dict[str, Any]]:
        import torch

        visible, dropped = self._visible(state, obstacles)
        self._record(sim_time, state, visible)
        n_det = len(visible)

        # ---- local goal, clipped into the training support
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

        vx_r = state.v * math.cos(state.yaw)
        vy_r = state.v * math.sin(state.yaw)

        # ---- crowd_sim_var_num.py::generate_ob, then crowd_sim_pred_real_gst.py
        dim = 2 * (self.predict_steps + 1)
        cur = np.full((self.max_humans, 2), np.inf, dtype=np.float64)
        vis_mask = np.zeros(self.max_humans, dtype=bool)
        for i, (_pid, hx, hy) in enumerate(visible):
            cur[i] = (hx, hy)
            vis_mask[i] = True
        cur[np.isinf(cur)] = self.dummy_human_xy
        spatial = np.tile(cur, self.predict_steps + 1).astype(np.float32)

        robot_node = np.array([[ox, oy, self.radius, ox + gx, oy + gy,
                                self.v_pref, self.theta]], dtype=np.float32)
        O = {
            "robot_node": torch.from_numpy(robot_node).unsqueeze(0).to(self.device),
            "temporal_edges": torch.from_numpy(
                np.array([[vx_r, vy_r]], dtype=np.float32)).unsqueeze(0).to(self.device),
            "spatial_edges": torch.from_numpy(spatial).unsqueeze(0).to(self.device),
            "visible_masks": torch.from_numpy(vis_mask).unsqueeze(0).to(self.device),
            "detected_human_num": torch.tensor([[float(max(n_det, 1))]],
                                               device=self.device),
        }

        # ---- GST inference, mirroring VecPretextNormalize.process_obs_rew
        in_traj = np.zeros((1, self.max_humans, self.obs_seq_len, 2), dtype=np.float32)
        in_mask = np.zeros((1, self.max_humans, self.obs_seq_len, 1), dtype=np.float32)
        for i, (pid, _hx, _hy) in enumerate(visible):
            p, v = self._history_grid(pid, sim_time, state)
            # same absolute frame as robot_node, so upstream's
            # "out_traj - robot_node[:2]" below yields robot-relative positions
            in_traj[0, i, :, 0] = p[:, 0] + ox
            in_traj[0, i, :, 1] = p[:, 1] + oy
            in_mask[0, i, :, 0] = v.astype(np.float32)
        out_traj, out_mask = self.predictor.forward(
            input_traj=torch.from_numpy(in_traj).to(self.device),
            input_binary_mask=torch.from_numpy(in_mask).to(self.device))
        out_mask = out_mask.bool()

        # predictions are absolute in the frame we fed GST, whose origin is the
        # current robot position -- so robot_pos here is (0, 0), which is
        # upstream's subtraction with our translation folded in
        robot_pos = O["robot_node"][:, :, :2].unsqueeze(1)
        out_traj[:, :, :, :2] = out_traj[:, :, :, :2] - robot_pos
        rep = out_mask.repeat(1, 1, self.predict_steps * 2)
        new_edges = out_traj[:, :, :, :2].reshape(1, self.max_humans, -1)
        O["spatial_edges"][:, :, 2:][rep] = new_edges[rep]

        # sort by current distance (padding rows sit at 15 -> they sort last)
        hr = torch.linalg.norm(O["spatial_edges"][:, :, :2], dim=-1)
        O["spatial_edges"][0] = O["spatial_edges"][0][torch.argsort(hr, dim=1)[0]]

        with torch.no_grad():
            value, action, _logp, hxs = self.policy.act(
                O, self.hxs, self.masks, deterministic=self.deterministic)
        self.hxs = hxs
        self.masks = torch.ones(1, 1, device=self.device)

        a = action[0].detach().cpu().numpy().astype(float)
        vx, vy = float(a[0]), float(a[1])
        n = math.hypot(vx, vy)
        if n > self.v_pref:                       # upstream SRNN.clip_action
            vx, vy = vx / n * self.v_pref, vy / n * self.v_pref
        n = math.hypot(vx, vy)
        if n > self.cfg.max_speed > 0.0:          # benchmark envelope
            vx, vy = vx / n * self.cfg.max_speed, vy / n * self.cfg.max_speed

        return vx, vy, {
            "status": "crowdnav_attngraph",
            "value": round(float(value.item()), 4),
            "n_humans_used": int(n_det),
            "n_humans_seen": int(len(obstacles)),
            "n_humans_dropped": int(dropped),
            "n_predicted": int(out_mask.sum().item()),
            "local_goal_x": round(gx, 3),
            "local_goal_y": round(gy, 3),
        }
