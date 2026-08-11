#!/usr/bin/env python3
"""Run a trained CrowdNav SARL model as a SUMO sidewalk robot controller.

This unified edition supports the same randomized scenario arguments and per-seed
output layout as the existing classical baseline scripts.

This script is a self-contained SUMO adapter for a standard CrowdNav SARL
``rl_model.pth`` state_dict. It does not require the CrowdNav package itself.

Expected model architecture (verified from the supplied rl_model.pth):
    rotated joint-state input: 13
    mlp1:      13 -> 150 -> 100
    mlp2:     100 -> 100 -> 50
    attention: 200 -> 100 -> 100 -> 1   (with global state)
    mlp3:      56 -> 150 -> 100 -> 100 -> 1

The robot is represented as a SUMO person and moved with TraCI moveToXY().
SARL is a value-based policy: at every SUMO step it samples candidate
holonomic velocity actions, predicts the next robot/human states, evaluates
those states with the learned value network, and selects the best action.

Typical Windows commands
------------------------
1. Put this file and rl_model.pth next to BasicConfig.sumocfg.
2. Make sure SUMO_HOME is configured and CrowdNav is NOT required.
3. Run with GUI:

    python sarl_sumo_robot.py --cfg BasicConfig.sumocfg \
        --model rl_model.pth --sumo-gui

Headless:

    python sarl_sumo_robot.py --cfg BasicConfig.sumocfg \
        --model rl_model.pth

Use --dt 0.25 to stay close to the original CrowdNav training time step.
For direct comparison with an existing 0.5 s baseline, pass --dt 0.5 to all
algorithms consistently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - environment-specific
    raise RuntimeError("PyTorch is required. Install it with: pip install torch") from exc


def resolve_project_path(value: str, base_dir: Path) -> Path:
    """Resolve a CLI path relative to the SUMO project directory."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


# ---------------------------------------------------------------------------
# State and action types compatible with the original CrowdNav SARL semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionXY:
    vx: float
    vy: float


@dataclass(frozen=True)
class FullState:
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    gx: float
    gy: float
    v_pref: float
    theta: float

    def as_tuple(self) -> Tuple[float, ...]:
        return (
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        )


@dataclass(frozen=True)
class ObservableState:
    px: float
    py: float
    vx: float
    vy: float
    radius: float

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.px, self.py, self.vx, self.vy, self.radius)


@dataclass(frozen=True)
class HumanObservation:
    person_id: str
    state: ObservableState


@dataclass
class RobotRuntimeState:
    x: float
    y: float
    vx: float
    vy: float
    yaw: float

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class SumoSarlConfig:
    # Original CrowdNav SARL defaults / model assumptions.
    gamma: float = 0.9
    speed_samples: int = 5
    rotation_samples: int = 16
    v_pref: float = 1.0
    dt: float = 0.25

    # Robot, pedestrian and scene geometry.
    robot_radius: float = 0.25
    pedestrian_radius: float = 0.15
    sensor_range: float = 12.0
    max_humans: int = 30
    goal_tolerance: float = 0.25
    discomfort_dist: float = 0.20

    # Current north-side sidewalk geometry from the earlier SUMO baseline.
    sidewalk_x_min: float = 0.0
    sidewalk_x_max: float = 300.0
    sidewalk_y_min: float = 3.0
    sidewalk_y_max: float = 5.0

    max_time: float = 420.0

    @property
    def collision_distance(self) -> float:
        return self.robot_radius + self.pedestrian_radius


# ---------------------------------------------------------------------------
# SARL network (same parameter names as the supplied state_dict)
# ---------------------------------------------------------------------------


def mlp(input_dim: int, dims: Sequence[int], last_relu: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    all_dims = [input_dim, *dims]
    for i in range(len(all_dims) - 1):
        layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
        if i != len(all_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SarlValueNetwork(nn.Module):
    """Network matching the uploaded CrowdNav SARL state_dict exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.self_state_dim = 6
        self.global_state_dim = 100
        self.mlp1 = mlp(13, [150, 100], last_relu=True)
        self.mlp2 = mlp(100, [100, 50])
        self.attention = mlp(200, [100, 100, 1])
        self.mlp3 = mlp(56, [150, 100, 100, 1])
        self.attention_weights: Optional[np.ndarray] = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (batch, number_of_humans, 13)
        size = state.shape
        if state.ndim != 3 or size[2] != 13:
            raise ValueError(f"SARL expects (batch, humans, 13), got {tuple(size)}")

        self_state = state[:, 0, : self.self_state_dim]
        mlp1_output = self.mlp1(state.reshape(-1, size[2]))
        mlp2_output = self.mlp2(mlp1_output)

        global_state = torch.mean(
            mlp1_output.reshape(size[0], size[1], -1), dim=1, keepdim=True
        )
        global_state = (
            global_state.expand(size[0], size[1], self.global_state_dim)
            .contiguous()
            .reshape(-1, self.global_state_dim)
        )
        attention_input = torch.cat([mlp1_output, global_state], dim=1)
        scores = self.attention(attention_input).reshape(size[0], size[1])

        # Numerically stable equivalent of the original masked softmax. The model
        # normally produces non-zero scores; clamp prevents a zero denominator.
        scores_exp = torch.exp(scores - torch.max(scores, dim=1, keepdim=True).values)
        weights = scores_exp / torch.clamp(
            torch.sum(scores_exp, dim=1, keepdim=True), min=1e-12
        )
        self.attention_weights = weights[0].detach().cpu().numpy()

        features = mlp2_output.reshape(size[0], size[1], -1)
        weighted_feature = torch.sum(weights.unsqueeze(2) * features, dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        return self.mlp3(joint_state)


def safe_load_state_dict(path: Path, device: torch.device) -> OrderedDict[str, torch.Tensor]:
    """Load a state_dict without permitting arbitrary pickle object execution."""
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Older PyTorch versions do not support weights_only.
        raise RuntimeError(
            "Your PyTorch version is too old for safe weights-only loading. "
            "Upgrade PyTorch, or load only a model file you fully trust."
        )

    if not isinstance(obj, (dict, OrderedDict)):
        raise TypeError(
            "Expected rl_model.pth to contain a state_dict, "
            f"but found {type(obj).__name__}."
        )

    # Some training wrappers save {'state_dict': ...}.
    if "state_dict" in obj and isinstance(obj["state_dict"], (dict, OrderedDict)):
        obj = obj["state_dict"]

    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor value in model state_dict at key: {key}")
        clean_key = str(key)
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value
    return cleaned


def validate_model_shapes(state_dict: Dict[str, torch.Tensor]) -> None:
    expected = {
        "mlp1.0.weight": (150, 13),
        "mlp1.2.weight": (100, 150),
        "mlp2.0.weight": (100, 100),
        "mlp2.2.weight": (50, 100),
        "attention.0.weight": (100, 200),
        "attention.2.weight": (100, 100),
        "attention.4.weight": (1, 100),
        "mlp3.0.weight": (150, 56),
        "mlp3.2.weight": (100, 150),
        "mlp3.4.weight": (100, 100),
        "mlp3.6.weight": (1, 100),
    }
    problems: List[str] = []
    for key, shape in expected.items():
        if key not in state_dict:
            problems.append(f"missing {key}")
        elif tuple(state_dict[key].shape) != shape:
            problems.append(
                f"{key}: expected {shape}, got {tuple(state_dict[key].shape)}"
            )
    if problems:
        raise ValueError(
            "The supplied model is not the expected standard SARL architecture:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# SARL action selection
# ---------------------------------------------------------------------------


class SarlPolicy:
    def __init__(
        self,
        model_path: Path,
        cfg: SumoSarlConfig,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model = SarlValueNetwork().to(device)
        state_dict = safe_load_state_dict(model_path, device)
        validate_model_shapes(state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Model key mismatch. Missing={list(missing)}, unexpected={list(unexpected)}"
            )
        self.model.eval()
        self.action_space = self._build_action_space()

    def _build_action_space(self) -> List[ActionXY]:
        # Same exponential speed sampling as CrowdNav.
        speeds = [
            (math.exp((i + 1) / self.cfg.speed_samples) - 1.0)
            / (math.e - 1.0)
            * self.cfg.v_pref
            for i in range(self.cfg.speed_samples)
        ]
        rotations = np.linspace(
            0.0, 2.0 * math.pi, self.cfg.rotation_samples, endpoint=False
        )
        actions = [ActionXY(0.0, 0.0)]
        for rotation in rotations:
            for speed in speeds:
                actions.append(
                    ActionXY(
                        float(speed * math.cos(float(rotation))),
                        float(speed * math.sin(float(rotation))),
                    )
                )
        return actions

    def _propagate_robot(self, state: FullState, action: ActionXY) -> FullState:
        return FullState(
            px=state.px + action.vx * self.cfg.dt,
            py=state.py + action.vy * self.cfg.dt,
            vx=action.vx,
            vy=action.vy,
            radius=state.radius,
            gx=state.gx,
            gy=state.gy,
            v_pref=state.v_pref,
            theta=state.theta,
        )

    def _propagate_human(self, state: ObservableState) -> ObservableState:
        return ObservableState(
            px=state.px + state.vx * self.cfg.dt,
            py=state.py + state.vy * self.cfg.dt,
            vx=state.vx,
            vy=state.vy,
            radius=state.radius,
        )

    def _rotate(self, state: torch.Tensor) -> torch.Tensor:
        """Original CrowdNav robot-centric 14D -> 13D transformation."""
        # Raw layout:
        # px, py, vx, vy, r, gx, gy, v_pref, theta,
        # human_px, human_py, human_vx, human_vy, human_r
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape(batch, 1)
        dy = (state[:, 6] - state[:, 1]).reshape(batch, 1)
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.linalg.vector_norm(torch.cat([dx, dy], dim=1), dim=1, keepdim=True)
        v_pref = state[:, 7].reshape(batch, 1)
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape(batch, 1)
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape(batch, 1)
        radius = state[:, 4].reshape(batch, 1)

        # This supplied model is holonomic, so theta is intentionally zero.
        theta = torch.zeros_like(v_pref)

        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape(batch, 1)
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape(batch, 1)
        px1 = (
            (state[:, 9] - state[:, 0]) * torch.cos(rot)
            + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        ).reshape(batch, 1)
        py1 = (
            (state[:, 10] - state[:, 1]) * torch.cos(rot)
            - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        ).reshape(batch, 1)
        radius1 = state[:, 13].reshape(batch, 1)
        radius_sum = radius + radius1
        da = torch.linalg.vector_norm(
            torch.cat(
                [
                    (state[:, 0] - state[:, 9]).reshape(batch, 1),
                    (state[:, 1] - state[:, 10]).reshape(batch, 1),
                ],
                dim=1,
            ),
            dim=1,
            keepdim=True,
        )

        return torch.cat(
            [
                dg,
                v_pref,
                theta,
                radius,
                vx,
                vy,
                px1,
                py1,
                vx1,
                vy1,
                radius1,
                da,
                radius_sum,
            ],
            dim=1,
        )

    def _reward(self, robot: FullState, humans: Sequence[ObservableState]) -> float:
        # SUMO adapter addition: prevent the model from leaving the sidewalk.
        if not (
            self.cfg.sidewalk_x_min <= robot.px <= self.cfg.sidewalk_x_max
            and self.cfg.sidewalk_y_min <= robot.py <= self.cfg.sidewalk_y_max
        ):
            return -0.25

        dmin = float("inf")
        collision = False
        for human in humans:
            clearance = (
                math.hypot(robot.px - human.px, robot.py - human.py)
                - robot.radius
                - human.radius
            )
            if clearance < 0.0:
                collision = True
                break
            dmin = min(dmin, clearance)

        reaching_goal = math.hypot(robot.px - robot.gx, robot.py - robot.gy) < robot.radius
        if collision:
            return -0.25
        if reaching_goal:
            return 1.0
        if dmin < self.cfg.discomfort_dist:
            return (dmin - self.cfg.discomfort_dist) * 0.5 * self.cfg.dt
        return 0.0

    @staticmethod
    def _goal_action(robot: FullState) -> ActionXY:
        dx = robot.gx - robot.px
        dy = robot.gy - robot.py
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            return ActionXY(0.0, 0.0)
        speed = min(robot.v_pref, distance)
        return ActionXY(speed * dx / distance, speed * dy / distance)

    def predict(
        self,
        robot: FullState,
        humans: Sequence[HumanObservation],
    ) -> Tuple[ActionXY, float, str, float]:
        """Return action, action value, most-attended person id, attention weight."""
        if math.hypot(robot.gx - robot.px, robot.gy - robot.py) <= robot.radius:
            return ActionXY(0.0, 0.0), 0.0, "", 0.0

        # The SARL network needs at least one human dimension. In an empty local
        # scene, direct goal motion is both simpler and consistent with the task.
        if not humans:
            return self._goal_action(robot), 0.0, "", 0.0

        human_states = [h.state for h in humans]
        best_action: Optional[ActionXY] = None
        best_value = float("-inf")
        best_attention: Optional[np.ndarray] = None

        with torch.no_grad():
            for action in self.action_space:
                next_robot = self._propagate_robot(robot, action)
                next_humans = [self._propagate_human(human) for human in human_states]
                reward = self._reward(next_robot, next_humans)

                raw_rows = [
                    next_robot.as_tuple() + human.as_tuple() for human in next_humans
                ]
                raw = torch.tensor(raw_rows, dtype=torch.float32, device=self.device)
                rotated = self._rotate(raw).unsqueeze(0)
                next_value = float(self.model(rotated).item())
                value = reward + math.pow(
                    self.cfg.gamma, self.cfg.dt * robot.v_pref
                ) * next_value

                if value > best_value:
                    best_value = value
                    best_action = action
                    if self.model.attention_weights is not None:
                        best_attention = self.model.attention_weights.copy()

        if best_action is None:
            raise RuntimeError("SARL failed to select an action.")

        attended_id = ""
        attended_weight = 0.0
        if best_attention is not None and len(best_attention) == len(humans):
            index = int(np.argmax(best_attention))
            attended_id = humans[index].person_id
            attended_weight = float(best_attention[index])

        return best_action, best_value, attended_id, attended_weight


# ---------------------------------------------------------------------------
# SUMO adapter and evaluation
# ---------------------------------------------------------------------------


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def choose_edge_hint(x: float, y: float) -> str:
    # Three horizontal strips (U/M/L) and three 100 m longitudinal sections.
    if y >= 4.335:
        row = "U"
    elif y >= 3.665:
        row = "M"
    else:
        row = "L"

    if x < 100.0:
        column = "A"
    elif x < 200.0:
        column = "B"
    else:
        column = "C"
    return f"walk_{column}{row}"


def edge_local_position(x: float) -> float:
    if x < 100.0:
        return x
    if x < 200.0:
        return x - 100.0
    return x - 200.0


def sumo_angle_to_math(angle_deg: float) -> float:
    # SUMO: 0=north, 90=east, clockwise. Math: 0=east, CCW.
    return math.radians(90.0 - angle_deg)


def math_angle_to_sumo(angle_rad: float) -> float:
    return (90.0 - math.degrees(angle_rad)) % 360.0


def clamp_to_sidewalk(x: float, y: float, cfg: SumoSarlConfig) -> Tuple[float, float]:
    epsilon = 0.03
    return (
        float(np.clip(x, cfg.sidewalk_x_min + epsilon, cfg.sidewalk_x_max - epsilon)),
        float(np.clip(y, cfg.sidewalk_y_min + epsilon, cfg.sidewalk_y_max - epsilon)),
    )


def get_human_observations(
    traci: Any,
    robot_id: str,
    robot: RobotRuntimeState,
    previous_positions: Dict[str, Tuple[float, float]],
    cfg: SumoSarlConfig,
) -> List[HumanObservation]:
    observations: List[HumanObservation] = []
    current_positions: Dict[str, Tuple[float, float]] = {}

    for person_id in traci.person.getIDList():
        if person_id == robot_id:
            continue
        try:
            x, y = traci.person.getPosition(person_id)
        except Exception:
            continue

        x = float(x)
        y = float(y)
        current_positions[person_id] = (x, y)
        distance = math.hypot(x - robot.x, y - robot.y)
        if distance > cfg.sensor_range:
            continue

        if person_id in previous_positions:
            old_x, old_y = previous_positions[person_id]
            vx = (x - old_x) / max(cfg.dt, 1e-9)
            vy = (y - old_y) / max(cfg.dt, 1e-9)
        else:
            # Better first-step estimate than assuming every moving person is static.
            try:
                speed = float(traci.person.getSpeed(person_id))
                angle = sumo_angle_to_math(float(traci.person.getAngle(person_id)))
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)
            except Exception:
                vx = 0.0
                vy = 0.0

        observations.append(
            HumanObservation(
                person_id=person_id,
                state=ObservableState(
                    px=x,
                    py=y,
                    vx=float(vx),
                    vy=float(vy),
                    radius=cfg.pedestrian_radius,
                ),
            )
        )

    previous_positions.clear()
    previous_positions.update(current_positions)

    observations.sort(
        key=lambda h: math.hypot(h.state.px - robot.x, h.state.py - robot.y)
    )
    return observations[: cfg.max_humans]


def nearest_person(
    x: float, y: float, humans: Sequence[HumanObservation]
) -> Tuple[float, str]:
    if not humans:
        return float("inf"), ""
    person = min(
        humans,
        key=lambda h: math.hypot(h.state.px - x, h.state.py - y),
    )
    return math.hypot(person.state.px - x, person.state.py - y), person.person_id


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(
    rows: Sequence[Dict[str, Any]],
    cfg: SumoSarlConfig,
    goal: Tuple[float, float],
    success: bool,
    termination_reason: str,
) -> Dict[str, Any]:
    if len(rows) < 2:
        return {
            "algorithm": "SARL",
            "success": bool(success),
            "steps": len(rows),
            "termination_reason": termination_reason,
        }

    xs = np.asarray([float(row["x"]) for row in rows])
    ys = np.asarray([float(row["y"]) for row in rows])
    speeds = np.asarray([float(row["v"]) for row in rows])
    yaw_rates = np.asarray([float(row["w"]) for row in rows])
    times = np.asarray([float(row["time"]) for row in rows])
    min_distances = np.asarray([float(row["min_person_dist"]) for row in rows])

    dt = float(np.median(np.diff(times))) if len(times) > 1 else cfg.dt
    path_length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    straight_distance = float(math.hypot(goal[0] - xs[0], goal[1] - ys[0]))
    finite_distances = min_distances[np.isfinite(min_distances)]
    min_person_distance = (
        float(np.min(finite_distances)) if finite_distances.size else float("inf")
    )
    collision_steps = (
        int(np.sum(finite_distances < cfg.collision_distance))
        if finite_distances.size
        else 0
    )
    hazard_steps = (
        int(np.sum(finite_distances < 0.8)) if finite_distances.size else 0
    )

    acceleration = np.diff(speeds) / max(dt, 1e-9)
    jerk = np.diff(acceleration) / max(dt, 1e-9)
    sidewalk_violations = int(
        np.sum(
            (xs < cfg.sidewalk_x_min)
            | (xs > cfg.sidewalk_x_max)
            | (ys < cfg.sidewalk_y_min)
            | (ys > cfg.sidewalk_y_max)
        )
    )

    return {
        "algorithm": "SARL",
        "success": bool(success and collision_steps == 0),
        "termination_reason": termination_reason,
        "steps": int(len(rows)),
        "total_time_s": round(float(times[-1] - times[0]), 3),
        "path_length_m": round(path_length, 3),
        "straight_line_distance_m": round(straight_distance, 3),
        "extra_distance_ratio": round(
            path_length / max(straight_distance, 1e-9), 4
        ),
        "final_goal_distance_m": round(
            float(math.hypot(goal[0] - xs[-1], goal[1] - ys[-1])), 3
        ),
        "min_person_distance_m": (
            round(min_person_distance, 3)
            if math.isfinite(min_person_distance)
            else None
        ),
        "collision_steps": collision_steps,
        "hazard_steps_lt_0_8m": hazard_steps,
        "hazard_time_ratio": round(hazard_steps / max(len(rows), 1), 4),
        "sidewalk_violation_steps": sidewalk_violations,
        "sidewalk_violation_ratio": round(
            sidewalk_violations / max(len(rows), 1), 4
        ),
        "mean_speed_mps": round(float(np.mean(speeds)), 3),
        "mean_abs_yaw_rate_radps": round(float(np.mean(np.abs(yaw_rates))), 3),
        "mean_abs_accel_mps2": (
            round(float(np.mean(np.abs(acceleration))), 3)
            if acceleration.size
            else 0.0
        ),
        "mean_abs_jerk_mps3": (
            round(float(np.mean(np.abs(jerk))), 3) if jerk.size else 0.0
        ),
    }


def plot_route(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    cfg: SumoSarlConfig,
    goal: Tuple[float, float],
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; route plot was skipped.")
        return

    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axhspan(cfg.sidewalk_y_min, cfg.sidewalk_y_max, alpha=0.15)
    ax.hlines(
        [cfg.sidewalk_y_min, cfg.sidewalk_y_max],
        cfg.sidewalk_x_min,
        cfg.sidewalk_x_max,
        linewidth=1.3,
    )
    ax.plot(xs, ys, linewidth=1.7, label="SARL robot route")
    ax.scatter([xs[0]], [ys[0]], marker="o", s=55, label="start")
    ax.scatter([goal[0]], [goal[1]], marker="*", s=120, label="goal")
    ax.set_xlim(cfg.sidewalk_x_min - 2.0, cfg.sidewalk_x_max + 2.0)
    ax.set_ylim(cfg.sidewalk_y_min - 1.0, cfg.sidewalk_y_max + 1.0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("SARL robot route in the SUMO sidewalk scene")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_sumo(args: argparse.Namespace, cfg: SumoSarlConfig) -> None:
    try:
        import traci  # type: ignore
        try:
            from sumolib import checkBinary  # type: ignore
        except Exception:
            checkBinary = None
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import SUMO TraCI. Set SUMO_HOME and add %SUMO_HOME%/tools "
            "to PYTHONPATH, or run inside the SUMO Python environment."
        ) from exc

    cfg_path = Path(args.cfg).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"SUMO config not found: {cfg_path}")

    work_dir = cfg_path.parent
    model_path = resolve_project_path(args.model, Path.cwd())
    if not model_path.is_file():
        raise FileNotFoundError(f"SARL model not found: {model_path}")

    seed = int(args.seed)
    output_root = Path(args.output_dir).resolve()
    output_dir = output_root / f"seed_{seed}" if args.random_scenario else output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    route_override: Optional[Path] = None
    if args.random_scenario:
        try:
            from sidewalk_robot_common import generate_random_demand  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "--random-scenario requires sidewalk_robot_common.py in the same folder."
            ) from exc
        base_route = resolve_project_path(args.rou, work_dir)
        if not base_route.is_file():
            raise FileNotFoundError(f"Base SUMO route file not found: {base_route}")
        route_override = output_dir / f"BasicDemand_random_seed_{seed}.rou.xml"
        generate_random_demand(base_route, route_override, args, seed)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading SARL model: {model_path}")
    policy = SarlPolicy(model_path=model_path, cfg=cfg, device=device)
    print("SARL model loaded successfully.")

    sumo_name = "sumo-gui" if args.sumo_gui else "sumo"
    sumo_binary = checkBinary(sumo_name) if checkBinary else sumo_name
    command = [
        sumo_binary,
        "-c",
        str(cfg_path),
        "--step-length",
        str(cfg.dt),
        "--seed",
        str(seed),
        "--quit-on-end",
    ]
    if route_override is not None:
        command += ["--route-files", str(route_override)]
    if args.sumo_gui:
        command += ["--start"]

    start = (float(args.start_x), float(args.start_y))
    goal = (float(args.goal_x), float(args.goal_y))
    robot_id = args.robot_id
    runtime = RobotRuntimeState(
        x=start[0], y=start[1], vx=0.0, vy=0.0, yaw=0.0
    )

    trace_csv = output_dir / "sarl_robot_trace.csv"
    metrics_json = output_dir / "sarl_robot_metrics.json"
    route_png = output_dir / "sarl_robot_route.png"

    rows: List[Dict[str, Any]] = []
    previous_human_positions: Dict[str, Tuple[float, float]] = {}
    success = False
    termination_reason = "max_time"

    old_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        traci.start(command)

        if robot_id not in traci.person.getIDList():
            start_edge = choose_edge_hint(*start)
            traci.person.add(
                robot_id,
                start_edge,
                pos=edge_local_position(start[0]),
                depart=0,
                typeID="DEFAULT_PEDTYPE",
            )
            traci.person.appendWaitingStage(
                robot_id,
                duration=cfg.max_time + 1000.0,
                description="sarl_controlled_robot",
                stopID="",
            )

        traci.simulationStep()
        try:
            traci.person.setColor(robot_id, (255, 0, 0, 255))
        except Exception:
            pass
        traci.person.moveToXY(
            robot_id,
            choose_edge_hint(*start),
            start[0],
            start[1],
            angle=90.0,
            keepRoute=2,
            matchThreshold=20.0,
        )

        maximum_steps = int(math.ceil(cfg.max_time / cfg.dt))
        for _ in range(maximum_steps + 1):
            simulation_time = float(traci.simulation.getTime())
            humans = get_human_observations(
                traci,
                robot_id,
                runtime,
                previous_human_positions,
                cfg,
            )

            full_state = FullState(
                px=runtime.x,
                py=runtime.y,
                vx=runtime.vx,
                vy=runtime.vy,
                radius=cfg.robot_radius,
                gx=goal[0],
                gy=goal[1],
                v_pref=cfg.v_pref,
                theta=runtime.yaw,
            )
            action, action_value, attended_id, attended_weight = policy.predict(
                full_state, humans
            )

            next_x = runtime.x + action.vx * cfg.dt
            next_y = runtime.y + action.vy * cfg.dt
            next_x, next_y = clamp_to_sidewalk(next_x, next_y, cfg)
            speed = math.hypot(action.vx, action.vy)
            next_yaw = (
                math.atan2(action.vy, action.vx)
                if speed > 1e-8
                else runtime.yaw
            )
            yaw_rate = (next_yaw - runtime.yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate /= max(cfg.dt, 1e-9)

            runtime = RobotRuntimeState(
                x=next_x,
                y=next_y,
                vx=action.vx,
                vy=action.vy,
                yaw=next_yaw,
            )

            traci.person.moveToXY(
                robot_id,
                choose_edge_hint(runtime.x, runtime.y),
                runtime.x,
                runtime.y,
                angle=math_angle_to_sumo(runtime.yaw),
                keepRoute=2,
                matchThreshold=20.0,
            )

            min_distance, closest_id = nearest_person(
                runtime.x, runtime.y, humans
            )
            goal_distance = math.hypot(goal[0] - runtime.x, goal[1] - runtime.y)
            rows.append(
                {
                    "time": round(simulation_time, 3),
                    "x": round(runtime.x, 4),
                    "y": round(runtime.y, 4),
                    "yaw": round(runtime.yaw, 6),
                    "v": round(runtime.speed, 4),
                    "w": round(yaw_rate, 6),
                    "vx": round(runtime.vx, 4),
                    "vy": round(runtime.vy, 4),
                    "goal_distance": round(goal_distance, 4),
                    "min_person_dist": (
                        round(min_distance, 4)
                        if math.isfinite(min_distance)
                        else float("inf")
                    ),
                    "closest_person": closest_id,
                    "visible_humans": len(humans),
                    "sarl_action_value": round(action_value, 6),
                    "most_attended_person": attended_id,
                    "attention_weight": round(attended_weight, 6),
                    "action_linear_x": round(runtime.vx, 4),
                    "action_linear_y": round(runtime.vy, 4),
                    "observation_state_road_type": "Sidewalk",
                }
            )

            if goal_distance <= cfg.goal_tolerance:
                success = True
                termination_reason = "goal_reached"
                break
            if math.isfinite(min_distance) and min_distance < cfg.collision_distance:
                success = False
                termination_reason = f"collision_with_{closest_id or 'person'}"
                break
            if simulation_time >= cfg.max_time:
                termination_reason = "max_time"
                break

            traci.simulationStep()
    finally:
        try:
            traci.close(False)
        except Exception:
            pass
        os.chdir(old_cwd)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trace_csv, rows)
    metrics = compute_metrics(rows, cfg, goal, success, termination_reason)
    metrics.update(
        {
            "model_file": str(model_path),
            "device": str(device),
            "sumo_step_length_s": cfg.dt,
            "v_pref_mps": cfg.v_pref,
            "speed_samples": cfg.speed_samples,
            "rotation_samples": cfg.rotation_samples,
            "seed": seed,
            "random_scenario": bool(args.random_scenario),
            "route_file": str(route_override) if route_override is not None else None,
        }
    )
    metrics_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_route(route_png, rows, cfg, goal)

    print(f"Trace:   {trace_csv}")
    print(f"Metrics: {metrics_json}")
    print(f"Plot:    {route_png}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained CrowdNav SARL model in the SUMO sidewalk scene"
    )
    parser.add_argument("--cfg", default="BasicConfig.sumocfg")
    parser.add_argument("--rou", default="BasicDemand.rou.xml", help="Base route file used by --random-scenario")
    parser.add_argument("--net", default="BasicNetwork.net.xml", help="Accepted for compatibility with the other runners")
    parser.add_argument("--model", default="rl_model.pth")
    parser.add_argument("--random-scenario", action="store_true", help="Generate the same seeded random pedestrian scenario format as the classical baselines")
    parser.add_argument("--flow-mode", choices=["probability", "personsPerHour"], default="personsPerHour")
    parser.add_argument("--speed-min", type=float, default=0.80)
    parser.add_argument("--speed-max", type=float, default=1.60)
    parser.add_argument("--flow-min", type=float, default=80.0)
    parser.add_argument("--flow-max", type=float, default=350.0)
    parser.add_argument("--static-min", type=int, default=4)
    parser.add_argument("--static-max", type=int, default=14)
    parser.add_argument("--static-min-gap", type=float, default=6.0)
    parser.add_argument("--scenario-begin", type=float, default=0.0)
    parser.add_argument("--scenario-end", type=float, default=36000.0)
    parser.add_argument("--sumo-gui", "--gui", dest="sumo_gui", action="store_true")
    parser.add_argument("--output-dir", default="sarl_outputs")
    parser.add_argument("--robot-id", default="robot0")
    parser.add_argument("--start-x", type=float, default=2.0)
    parser.add_argument("--start-y", type=float, default=4.0)
    parser.add_argument("--goal-x", type=float, default=298.0)
    parser.add_argument("--goal-y", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--max-time", type=float, default=420.0)
    parser.add_argument("--v-pref", type=float, default=1.0)
    parser.add_argument("--sensor-range", type=float, default=12.0)
    parser.add_argument("--max-humans", type=int, default=30)
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--pedestrian-radius", type=float, default=0.15)
    parser.add_argument("--speed-samples", type=int, default=5)
    parser.add_argument("--rotation-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument("--sidewalk-x-min", type=float, default=0.0)
    parser.add_argument("--sidewalk-x-max", type=float, default=300.0)
    parser.add_argument("--sidewalk-y-min", type=float, default=3.0)
    parser.add_argument("--sidewalk-y-max", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dt <= 0:
        raise ValueError("--dt must be positive")
    if args.v_pref <= 0:
        raise ValueError("--v-pref must be positive")
    if args.speed_samples <= 0 or args.rotation_samples <= 0:
        raise ValueError("Action-space sample counts must be positive")

    cfg = SumoSarlConfig(
        dt=args.dt,
        max_time=args.max_time,
        v_pref=args.v_pref,
        sensor_range=args.sensor_range,
        max_humans=args.max_humans,
        robot_radius=args.robot_radius,
        pedestrian_radius=args.pedestrian_radius,
        speed_samples=args.speed_samples,
        rotation_samples=args.rotation_samples,
        sidewalk_x_min=args.sidewalk_x_min,
        sidewalk_x_max=args.sidewalk_x_max,
        sidewalk_y_min=args.sidewalk_y_min,
        sidewalk_y_max=args.sidewalk_y_max,
    )
    run_sumo(args, cfg)


if __name__ == "__main__":
    main()
