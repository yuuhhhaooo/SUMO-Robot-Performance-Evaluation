#!/usr/bin/env python3
"""Run a trained CrowdNav LSTM-RL model as a SUMO sidewalk robot controller.

This file follows the same interface as the other SUMO baseline planners:
    - it supports --random-scenario and per-seed output folders
    - it writes robot_trace.csv, robot_metrics.json and robot_route.png
    - it is compatible with run_random_batch_overlay_all.py

The uploaded LSTM-RL model is a value network. At every SUMO step this adapter:
    1. samples candidate holonomic velocity actions,
    2. predicts one step of robot/human motion,
    3. transforms the joint state into the original CrowdNav 13D rotated format,
    4. evaluates the state with the trained LSTM value network,
    5. selects the best action and lets sidewalk_robot_common.py move the robot.

It does not require the full CrowdNav package, but it does require PyTorch.
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - environment-specific
    raise RuntimeError("PyTorch is required for LSTM-RL inference. Install torch first.") from exc

from sidewalk_robot_common import (
    PlannerConfig,
    RobotState,
    Obstacle,
    add_common_arguments,
    run_traci_with_planner,
)


def mlp(input_dim: int, dims: Sequence[int], last_relu: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    all_dims = [input_dim, *dims]
    for i in range(len(all_dims) - 1):
        layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
        if i != len(all_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class LstmRLValueNetwork(nn.Module):
    """ValueNetwork1 from CrowdNav LSTM-RL.

    Expected state_dict shapes for the uploaded model:
        lstm.weight_ih_l0: (200, 13)
        lstm.weight_hh_l0: (200, 50)
        mlp.0.weight:      (150, 56)
        mlp.2.weight:      (100, 150)
        mlp.4.weight:      (100, 100)
        mlp.6.weight:      (1, 100)
    """

    def __init__(self) -> None:
        super().__init__()
        self.self_state_dim = 6
        self.lstm_hidden_dim = 50
        self.lstm = nn.LSTM(13, self.lstm_hidden_dim, batch_first=True)
        self.mlp = mlp(self.self_state_dim + self.lstm_hidden_dim, [150, 100, 100, 1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state shape: (batch_size, num_humans, 13)
        if state.ndim != 3 or state.shape[2] != 13:
            raise ValueError(f"LSTM-RL expects state shape (batch, humans, 13), got {tuple(state.shape)}")

        batch_size = state.shape[0]
        self_state = state[:, 0, : self.self_state_dim]
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=state.device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=state.device)
        _, (hn, _) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        return self.mlp(joint_state)


def safe_load_state_dict(path: Path, device: torch.device) -> OrderedDict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"LSTM-RL model file not found: {path}")

    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not have weights_only.
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]

    if not isinstance(obj, (dict, OrderedDict)):
        raise TypeError(f"Expected a state_dict, got {type(obj).__name__}")

    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor entry in state_dict: {key}")
        clean_key = str(key)
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value
    return cleaned


def validate_lstm_rl_shapes(state_dict: Dict[str, torch.Tensor]) -> None:
    expected = {
        "mlp.0.weight": (150, 56),
        "mlp.2.weight": (100, 150),
        "mlp.4.weight": (100, 100),
        "mlp.6.weight": (1, 100),
        "lstm.weight_ih_l0": (200, 13),
        "lstm.weight_hh_l0": (200, 50),
        "lstm.bias_ih_l0": (200,),
        "lstm.bias_hh_l0": (200,),
    }
    problems: List[str] = []
    for key, shape in expected.items():
        if key not in state_dict:
            problems.append(f"missing {key}")
        elif tuple(state_dict[key].shape) != shape:
            problems.append(f"{key}: expected {shape}, got {tuple(state_dict[key].shape)}")
    if problems:
        raise ValueError(
            "The supplied model is not the expected LSTM-RL ValueNetwork1 architecture:\n  - "
            + "\n  - ".join(problems)
        )


class LstmRLPlanner:
    def __init__(
        self,
        cfg: PlannerConfig,
        seed: int,
        model_path: Path,
        device: torch.device,
        gamma: float = 0.90,
        speed_samples: int = 5,
        rotation_samples: int = 16,
        max_humans: int = 30,
        v_pref: float = 1.00,
        goal_lookahead: float = 8.0,
        discomfort_dist: float = 0.20,
        progress_bonus: float = 0.04,
        centerline_penalty: float = 0.01,
        sidewalk_penalty: float = 0.25,
    ) -> None:
        self.cfg = cfg
        self.seed = seed
        self.gamma = gamma
        self.speed_samples = speed_samples
        self.rotation_samples = rotation_samples
        self.max_humans = max_humans
        self.v_pref = min(v_pref, cfg.max_speed)
        self.goal_lookahead = goal_lookahead
        self.discomfort_dist = discomfort_dist
        self.progress_bonus = progress_bonus
        self.centerline_penalty = centerline_penalty
        self.sidewalk_penalty = sidewalk_penalty
        self.device = device
        # The candidate-action batch is a stack of tiny GEMMs on <=30-row
        # tensors; a wide intra-op thread pool is pure oversubscription here.
        try:
            torch.set_num_threads(1)
        except Exception:  # pragma: no cover - some builds forbid late changes
            pass

        self.model = LstmRLValueNetwork().to(device)
        state_dict = safe_load_state_dict(model_path, device)
        validate_lstm_rl_shapes(state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(f"LSTM-RL model key mismatch. Missing={missing}, unexpected={unexpected}")
        self.model.eval()
        self.action_space = self._build_action_space()

    def _build_action_space(self) -> List[Tuple[float, float]]:
        speeds = [
            (math.exp((i + 1) / self.speed_samples) - 1.0) / (math.e - 1.0) * self.v_pref
            for i in range(self.speed_samples)
        ]
        rotations = np.linspace(0.0, 2.0 * math.pi, self.rotation_samples, endpoint=False)
        actions: List[Tuple[float, float]] = [(0.0, 0.0)]
        for theta in rotations:
            for speed in speeds:
                actions.append((float(speed * math.cos(float(theta))), float(speed * math.sin(float(theta)))))
        return actions

    def _local_goal(self, state: RobotState, goal: Tuple[float, float]) -> Tuple[float, float]:
        dx = goal[0] - state.x
        dy = goal[1] - state.y
        dist = math.hypot(dx, dy)
        if dist <= self.goal_lookahead:
            return goal
        scale = self.goal_lookahead / max(dist, 1e-9)
        return (state.x + dx * scale, state.y + dy * scale)

    def _goal_action(self, x: float, y: float, goal: Tuple[float, float]) -> Tuple[float, float]:
        dx = goal[0] - x
        dy = goal[1] - y
        dist = math.hypot(dx, dy)
        if dist <= 1e-9:
            return (0.0, 0.0)
        speed = min(self.v_pref, dist / max(self.cfg.dt, 1e-9))
        return (speed * dx / dist, speed * dy / dist)

    def _rotate_raw_to_13d(self, raw: torch.Tensor) -> torch.Tensor:
        """CrowdNav robot-centric rotation transform.

        Raw layout per row:
            robot px, py, vx, vy, radius, gx, gy, v_pref, theta,
            human px, py, vx, vy, radius
        Output layout:
            dg, v_pref, theta, robot_radius, robot_vx, robot_vy,
            human_px_rel, human_py_rel, human_vx, human_vy,
            human_radius, distance_abs, radius_sum
        """
        batch = raw.shape[0]
        dx = (raw[:, 5] - raw[:, 0]).reshape(batch, 1)
        dy = (raw[:, 6] - raw[:, 1]).reshape(batch, 1)
        rot = torch.atan2(raw[:, 6] - raw[:, 1], raw[:, 5] - raw[:, 0])
        dg = torch.linalg.vector_norm(torch.cat([dx, dy], dim=1), dim=1, keepdim=True)

        v_pref = raw[:, 7].reshape(batch, 1)
        theta = torch.zeros_like(v_pref)  # holonomic adapter
        radius = raw[:, 4].reshape(batch, 1)

        vx = (raw[:, 2] * torch.cos(rot) + raw[:, 3] * torch.sin(rot)).reshape(batch, 1)
        vy = (raw[:, 3] * torch.cos(rot) - raw[:, 2] * torch.sin(rot)).reshape(batch, 1)

        px1 = ((raw[:, 9] - raw[:, 0]) * torch.cos(rot) + (raw[:, 10] - raw[:, 1]) * torch.sin(rot)).reshape(batch, 1)
        py1 = ((raw[:, 10] - raw[:, 1]) * torch.cos(rot) - (raw[:, 9] - raw[:, 0]) * torch.sin(rot)).reshape(batch, 1)
        vx1 = (raw[:, 11] * torch.cos(rot) + raw[:, 12] * torch.sin(rot)).reshape(batch, 1)
        vy1 = (raw[:, 12] * torch.cos(rot) - raw[:, 11] * torch.sin(rot)).reshape(batch, 1)
        radius1 = raw[:, 13].reshape(batch, 1)

        da = torch.linalg.vector_norm(
            torch.cat([(raw[:, 0] - raw[:, 9]).reshape(batch, 1), (raw[:, 1] - raw[:, 10]).reshape(batch, 1)], dim=1),
            dim=1,
            keepdim=True,
        )
        radius_sum = radius + radius1

        return torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

    def _reward(
        self,
        next_x: float,
        next_y: float,
        local_goal: Tuple[float, float],
        next_obstacles: Sequence[Obstacle],
        old_goal_dist: float,
    ) -> float:
        if not (
            self.cfg.sidewalk_x_min <= next_x <= self.cfg.sidewalk_x_max
            and self.cfg.sidewalk_y_min <= next_y <= self.cfg.sidewalk_y_max
        ):
            return -self.sidewalk_penalty

        collision = False
        dmin = float("inf")
        for obs in next_obstacles:
            clearance = (
                math.hypot(next_x - obs.x, next_y - obs.y)
                - self.cfg.robot_radius
                - self.cfg.pedestrian_radius
            )
            if clearance < 0.0:
                collision = True
                break
            dmin = min(dmin, clearance)

        if collision:
            return -0.25

        new_goal_dist = math.hypot(local_goal[0] - next_x, local_goal[1] - next_y)
        if new_goal_dist < self.cfg.goal_tolerance:
            return 1.0

        reward = 0.0
        if dmin < self.discomfort_dist:
            reward += (dmin - self.discomfort_dist) * 0.5 * self.cfg.dt

        progress = old_goal_dist - new_goal_dist
        reward += self.progress_bonus * progress
        reward -= self.centerline_penalty * abs(next_y - self.cfg.sidewalk_center_y)
        return reward

    def _evaluate_action(
        self,
        state: RobotState,
        local_goal: Tuple[float, float],
        obstacles: Sequence[Obstacle],
        action: Tuple[float, float],
    ) -> float:
        vx, vy = action
        next_x = state.x + vx * self.cfg.dt
        next_y = state.y + vy * self.cfg.dt
        next_obstacles = [
            Obstacle(
                pid=o.pid,
                x=o.x + o.vx * self.cfg.dt,
                y=o.y + o.vy * self.cfg.dt,
                vx=o.vx,
                vy=o.vy,
            )
            for o in obstacles
        ]
        old_goal_dist = math.hypot(local_goal[0] - state.x, local_goal[1] - state.y)
        reward = self._reward(next_x, next_y, local_goal, next_obstacles, old_goal_dist)

        if not next_obstacles:
            return reward

        # LSTM-RL sorts humans by decreasing distance before prediction.
        next_obstacles = sorted(
            next_obstacles,
            key=lambda o: math.hypot(o.x - next_x, o.y - next_y),
            reverse=True,
        )[: self.max_humans]

        raw_rows: List[Tuple[float, ...]] = []
        for obs in next_obstacles:
            raw_rows.append(
                (
                    next_x,
                    next_y,
                    vx,
                    vy,
                    self.cfg.robot_radius,
                    local_goal[0],
                    local_goal[1],
                    self.v_pref,
                    0.0,
                    obs.x,
                    obs.y,
                    obs.vx,
                    obs.vy,
                    self.cfg.pedestrian_radius,
                )
            )

        raw = torch.tensor(raw_rows, dtype=torch.float32, device=self.device)
        rotated = self._rotate_raw_to_13d(raw).unsqueeze(0)
        with torch.no_grad():
            value = float(self.model(rotated).item())

        return reward + math.pow(self.gamma, self.cfg.dt * self.v_pref) * value

    @torch.no_grad()
    def compute_command(
        self,
        state: RobotState,
        goal: Tuple[float, float],
        obstacles: Sequence[Obstacle],
        sim_time: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        local_goal = self._local_goal(state, goal)
        obstacles = sorted(
            list(obstacles),
            key=lambda o: math.hypot(o.x - state.x, o.y - state.y),
        )[: self.max_humans]

        if not obstacles:
            vx, vy = self._goal_action(state.x, state.y, local_goal)
            return vx, vy, {"status": "lstm_rl_no_human", "cost": 0.0}

        dt = self.cfg.dt
        actions = self.action_space
        num_actions = len(actions)

        # The one-step human prediction does not depend on the candidate action,
        # so it is hoisted out of the action loop.
        next_obstacles = [
            Obstacle(pid=o.pid, x=o.x + o.vx * dt, y=o.y + o.vy * dt, vx=o.vx, vy=o.vy)
            for o in obstacles
        ]
        old_goal_dist = math.hypot(local_goal[0] - state.x, local_goal[1] - state.y)

        next_x = np.empty(num_actions, dtype=np.float64)
        next_y = np.empty(num_actions, dtype=np.float64)
        rewards: List[float] = []
        for index, (vx, vy) in enumerate(actions):
            nx = state.x + vx * dt
            ny = state.y + vy * dt
            next_x[index] = nx
            next_y[index] = ny
            rewards.append(
                self._reward(nx, ny, local_goal, next_obstacles, old_goal_dist)
            )

        obs_x = np.fromiter((o.x for o in next_obstacles), dtype=np.float64, count=len(next_obstacles))
        obs_y = np.fromiter((o.y for o in next_obstacles), dtype=np.float64, count=len(next_obstacles))
        obs_vx = np.fromiter((o.vx for o in next_obstacles), dtype=np.float64, count=len(next_obstacles))
        obs_vy = np.fromiter((o.vy for o in next_obstacles), dtype=np.float64, count=len(next_obstacles))

        # LSTM-RL orders humans by DECREASING distance to the *predicted* robot
        # pose, which is action dependent -- so the sort cannot be hoisted, only
        # vectorised. A stable argsort on the negated distance reproduces
        # ``sorted(..., reverse=True)`` including its tie ordering.
        distances = np.hypot(
            obs_x[None, :] - next_x[:, None], obs_y[None, :] - next_y[:, None]
        )
        order = np.argsort(-distances, axis=1, kind="stable")[:, : self.max_humans]
        num_rows = order.shape[1]

        robot_block = np.empty((num_actions, 9), dtype=np.float64)
        robot_block[:, 0] = next_x
        robot_block[:, 1] = next_y
        robot_block[:, 2] = [vx for vx, _ in actions]
        robot_block[:, 3] = [vy for _, vy in actions]
        robot_block[:, 4] = self.cfg.robot_radius
        robot_block[:, 5] = local_goal[0]
        robot_block[:, 6] = local_goal[1]
        robot_block[:, 7] = self.v_pref
        robot_block[:, 8] = 0.0

        raw_np = np.empty((num_actions, num_rows, 14), dtype=np.float32)
        raw_np[:, :, :9] = robot_block[:, None, :]
        raw_np[:, :, 9] = obs_x[order]
        raw_np[:, :, 10] = obs_y[order]
        raw_np[:, :, 11] = obs_vx[order]
        raw_np[:, :, 12] = obs_vy[order]
        raw_np[:, :, 13] = self.cfg.pedestrian_radius

        raw = torch.as_tensor(
            raw_np.reshape(num_actions * num_rows, 14),
            dtype=torch.float32,
            device=self.device,
        )
        # The rotation is row-independent, so it runs on the flat stack; the
        # (num_actions, num_rows, 13) shape is restored so the LSTM still folds
        # over the humans of ONE candidate scene per batch element.
        rotated = self._rotate_raw_to_13d(raw).reshape(num_actions, num_rows, 13)
        net_values = self.model(rotated).reshape(num_actions).detach().cpu().numpy()

        discount = math.pow(self.gamma, self.cfg.dt * self.v_pref)
        best_action: Optional[Tuple[float, float]] = None
        best_value = float("-inf")
        for index in range(num_actions):
            value = rewards[index] + discount * float(net_values[index])
            if value > best_value:
                best_value = value
                best_action = actions[index]

        if best_action is None:
            best_action = self._goal_action(state.x, state.y, local_goal)
            best_value = 0.0

        return best_action[0], best_action[1], {"status": "lstm_rl", "cost": best_value}


def choose_device(args: argparse.Namespace) -> torch.device:
    # --gpu is kept for compatibility with CADRL-style commands.
    requested = str(getattr(args, "device", "auto")).lower()
    if getattr(args, "gpu", False):
        requested = "cuda"
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LSTM-RL SUMO sidewalk robot controller")
    add_common_arguments(p)
    p.add_argument("--model-path", "--model", dest="model_path", default="models/lstm_rl_model.pth", help="Path to trained LSTM-RL .pth model")
    p.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, cuda, cuda:0")
    p.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    p.add_argument("--dt", type=float, default=None, help="Override SUMO/controller step length")
    p.add_argument("--lstm-gamma", type=float, default=0.90)
    p.add_argument("--lstm-speed-samples", type=int, default=5)
    p.add_argument("--lstm-rotation-samples", type=int, default=16)
    p.add_argument("--lstm-max-humans", type=int, default=30)
    p.add_argument("--lstm-v-pref", type=float, default=1.00)
    p.add_argument("--lstm-goal-lookahead", type=float, default=8.0)
    p.add_argument("--lstm-progress-bonus", type=float, default=0.04)
    p.add_argument("--lstm-centerline-penalty", type=float, default=0.01)
    p.add_argument("--lstm-sidewalk-penalty", type=float, default=0.25)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = PlannerConfig()
    if args.dt is not None:
        cfg.dt = float(args.dt)
    if args.max_time is not None:
        cfg.max_time = float(args.max_time)

    device = choose_device(args)
    model_path = Path(args.model_path)

    def factory(local_cfg: PlannerConfig, seed: int) -> LstmRLPlanner:
        # Resolve relative model paths from the current project folder.
        resolved_model = model_path if model_path.is_absolute() else (Path.cwd() / model_path)
        return LstmRLPlanner(
            local_cfg,
            seed,
            resolved_model,
            device=device,
            gamma=float(args.lstm_gamma),
            speed_samples=int(args.lstm_speed_samples),
            rotation_samples=int(args.lstm_rotation_samples),
            max_humans=int(args.lstm_max_humans),
            v_pref=float(args.lstm_v_pref),
            goal_lookahead=float(args.lstm_goal_lookahead),
            progress_bonus=float(args.lstm_progress_bonus),
            centerline_penalty=float(args.lstm_centerline_penalty),
            sidewalk_penalty=float(args.lstm_sidewalk_penalty),
        )

    run_traci_with_planner(args, factory, "lstm_rl", cfg)


if __name__ == "__main__":
    main()
