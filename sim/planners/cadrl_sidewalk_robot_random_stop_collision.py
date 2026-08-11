#!/usr/bin/env python3
"""CADRL value-network controlled delivery robot in the SUMO sidewalk scene.

This script connects a CrowdNav CADRL .pth model to the same SUMO/TraCI
batch-evaluation framework used by the other baselines.

The implementation is intentionally self-contained: it reads the uploaded
CADRL value-network state_dict directly and performs CADRL-style one-step
action evaluation over a discrete holonomic action space.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    in_sidewalk,
    run_traci_with_planner,
)


class CADRLValueNetwork(nn.Module):
    """Reconstruct CrowdNav CADRL ValueNetwork from a saved state_dict."""

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        # The CrowdNav CADRL state_dict usually has keys like:
        # value_network.0.weight, value_network.0.bias, value_network.2.weight, ...
        linear_indices = sorted(
            int(key.split(".")[1])
            for key in state_dict.keys()
            if key.startswith("value_network.") and key.endswith(".weight")
        )
        if not linear_indices:
            raise ValueError("Could not find value_network.*.weight keys in the CADRL model file")

        layers: List[nn.Module] = []
        for layer_no, idx in enumerate(linear_indices):
            weight = state_dict[f"value_network.{idx}.weight"]
            out_dim, in_dim = int(weight.shape[0]), int(weight.shape[1])
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_no != len(linear_indices) - 1:
                layers.append(nn.ReLU())
        self.value_network = nn.Sequential(*layers)
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_network(x)


class CADRLPlanner:
    def __init__(self, cfg: PlannerConfig, seed: int, args: argparse.Namespace):
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(args, "gpu", False) else "cpu")

        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"CADRL model file not found: {model_path}\n"
                "Put cadrl_rl_model.pth under models/, or pass --model-path PATH."
            )

        raw_state = torch.load(model_path, map_location="cpu")
        if not isinstance(raw_state, dict):
            raise ValueError(f"Unsupported CADRL model format: {type(raw_state)}")
        self.model = CADRLValueNetwork(raw_state).to(self.device).eval()

        self.gamma = float(args.cadrl_gamma)
        self.speed_samples = int(args.cadrl_speed_samples)
        self.rotation_samples = int(args.cadrl_rotation_samples)
        self.max_humans = int(args.cadrl_max_humans)
        self.v_pref = float(args.cadrl_v_pref) if args.cadrl_v_pref is not None else cfg.max_speed
        self.sidewalk_penalty = float(args.cadrl_sidewalk_penalty)
        self.centerline_penalty = float(args.cadrl_centerline_penalty)
        self.goal_lookahead = float(args.cadrl_goal_lookahead)
        self.progress_bonus = float(args.cadrl_progress_bonus)
        self.action_space = self.build_action_space(self.v_pref)

    def build_action_space(self, v_pref: float) -> List[Tuple[float, float]]:
        """CrowdNav-style discrete holonomic action space."""
        speeds = [
            (math.exp((i + 1) / self.speed_samples) - 1.0) / (math.e - 1.0) * v_pref
            for i in range(self.speed_samples)
        ]
        rotations = np.linspace(0.0, 2.0 * math.pi, self.rotation_samples, endpoint=False)
        actions: List[Tuple[float, float]] = [(0.0, 0.0)]
        for rot in rotations:
            for speed in speeds:
                actions.append((float(speed * math.cos(rot)), float(speed * math.sin(rot))))
        return actions

    def nearest_humans(self, state: RobotState, obstacles: Sequence[Obstacle]) -> List[Obstacle]:
        humans = sorted(obstacles, key=lambda o: math.hypot(o.x - state.x, o.y - state.y))
        return humans[: self.max_humans]

    def propagate_robot(self, state: RobotState, vx: float, vy: float) -> Tuple[float, float, float, float, float]:
        next_x = state.x + vx * self.cfg.dt
        next_y = state.y + vy * self.cfg.dt
        theta = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-6 else state.yaw
        return next_x, next_y, vx, vy, theta

    def propagate_human(self, human: Obstacle) -> Tuple[float, float, float, float, float]:
        next_x = human.x + human.vx * self.cfg.dt
        next_y = human.y + human.vy * self.cfg.dt
        return next_x, next_y, human.vx, human.vy, self.cfg.pedestrian_radius

    def local_goal(self, state: RobotState, final_goal: Tuple[float, float]) -> Tuple[float, float]:
        """Limit the CADRL goal distance to the scale seen during CrowdNav training.

        CrowdNav CADRL is trained in compact crowd-navigation scenes. The SUMO
        sidewalk goal is almost 300 m away, which is far outside that training
        scale. We therefore feed the value network a short lookahead goal along
        the sidewalk, while the evaluation still uses the real final goal.
        """
        dx = final_goal[0] - state.x
        if abs(dx) <= self.goal_lookahead:
            return final_goal
        step_x = state.x + math.copysign(self.goal_lookahead, dx)
        return (step_x, final_goal[1])

    def compute_reward(self, state: RobotState, next_self: Tuple[float, float, float, float, float], humans: Sequence[Obstacle], value_goal: Tuple[float, float], final_goal: Tuple[float, float]) -> float:
        px, py, _, _, _ = next_self
        dmin = float("inf")
        collision = False
        for human in humans:
            hx = human.x + human.vx * self.cfg.dt
            hy = human.y + human.vy * self.cfg.dt
            dist = math.hypot(px - hx, py - hy) - self.cfg.robot_radius - self.cfg.pedestrian_radius
            if dist < 0.0:
                collision = True
                break
            dmin = min(dmin, dist)

        if collision:
            reward = -0.25
        elif math.hypot(px - final_goal[0], py - final_goal[1]) < self.cfg.robot_radius:
            reward = 1.0
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.cfg.dt
        else:
            reward = 0.0

        old_dist = math.hypot(value_goal[0] - state.x, value_goal[1] - state.y)
        new_dist = math.hypot(value_goal[0] - px, value_goal[1] - py)
        reward += self.progress_bonus * (old_dist - new_dist)

        # The original CADRL does not know our SUMO sidewalk boundary. Add a small
        # adapter penalty so the policy remains usable in this constrained sidewalk.
        if not in_sidewalk(px, py, self.cfg, margin=0.03):
            reward -= self.sidewalk_penalty
        reward -= self.centerline_penalty * (py - self.cfg.sidewalk_center_y) ** 2
        return float(reward)

    def rotate_batch(self, batch_state: torch.Tensor) -> torch.Tensor:
        """Same 14D -> 13D agent-centric transform used by CrowdNav CADRL.

        Input columns are:
        self px, py, vx, vy, radius, gx, gy, v_pref, theta,
        human px, py, vx, vy, radius.
        """
        batch = batch_state.shape[0]
        dx = (batch_state[:, 5] - batch_state[:, 0]).reshape((batch, -1))
        dy = (batch_state[:, 6] - batch_state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(batch_state[:, 6] - batch_state[:, 1], batch_state[:, 5] - batch_state[:, 0])
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = batch_state[:, 7].reshape((batch, -1))
        vx = (batch_state[:, 2] * torch.cos(rot) + batch_state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (batch_state[:, 3] * torch.cos(rot) - batch_state[:, 2] * torch.sin(rot)).reshape((batch, -1))
        radius = batch_state[:, 4].reshape((batch, -1))
        theta = torch.zeros_like(v_pref)  # holonomic CADRL does not use theta
        vx1 = (batch_state[:, 11] * torch.cos(rot) + batch_state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (batch_state[:, 12] * torch.cos(rot) - batch_state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (batch_state[:, 9] - batch_state[:, 0]) * torch.cos(rot) + (batch_state[:, 10] - batch_state[:, 1]) * torch.sin(rot)
        py1 = (batch_state[:, 10] - batch_state[:, 1]) * torch.cos(rot) - (batch_state[:, 9] - batch_state[:, 0]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = py1.reshape((batch, -1))
        radius1 = batch_state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(
            torch.cat([
                (batch_state[:, 0] - batch_state[:, 9]).reshape((batch, -1)),
                (batch_state[:, 1] - batch_state[:, 10]).reshape((batch, -1)),
            ], dim=1),
            2,
            dim=1,
            keepdim=True,
        )
        return torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

    def batch_next_states(self, next_self: Tuple[float, float, float, float, float], humans: Sequence[Obstacle], value_goal: Tuple[float, float]) -> torch.Tensor:
        sx, sy, svx, svy, theta = next_self
        rows = []
        if not humans:
            # Dummy far-away human keeps the CADRL value network input shape valid.
            dummy = Obstacle("dummy", sx + 100.0, sy + 100.0, 0.0, 0.0)
            humans = [dummy]
        for human in humans:
            hx, hy, hvx, hvy, hr = self.propagate_human(human)
            rows.append([
                sx, sy, svx, svy, self.cfg.robot_radius, value_goal[0], value_goal[1], self.v_pref, theta,
                hx, hy, hvx, hvy, hr,
            ])
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def compute_command(self, state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle], sim_time: float):
        humans = self.nearest_humans(state, obstacles)
        value_goal = self.local_goal(state, goal)
        best_value = -float("inf")
        best_action = (0.0, 0.0)
        best_info: Dict[str, float | str | int] = {"status": "cadrl", "num_humans": len(humans)}

        for vx, vy in self.action_space:
            next_self = self.propagate_robot(state, vx, vy)
            next_states = self.batch_next_states(next_self, humans, value_goal)
            rotated = self.rotate_batch(next_states)
            values = self.model(rotated).reshape(-1)
            min_value = float(torch.min(values).item())
            reward = self.compute_reward(state, next_self, humans, value_goal, goal)
            total_value = reward + (self.gamma ** (self.cfg.dt * max(self.v_pref, 1e-6))) * min_value

            if total_value > best_value:
                best_value = total_value
                best_action = (vx, vy)
                best_info = {
                    "status": "cadrl",
                    "value": float(total_value),
                    "reward": float(reward),
                    "network_value": float(min_value),
                    "num_humans": len(humans),
                    "local_goal_x": float(value_goal[0]),
                    "local_goal_y": float(value_goal[1]),
                }

        # Safety fallback: if all action values are very poor near the sidewalk boundary,
        # move slowly toward the goal along the sidewalk centerline.
        if not math.isfinite(best_value):
            dx = goal[0] - state.x
            dy = self.cfg.sidewalk_center_y - state.y
            norm = max(math.hypot(dx, dy), 1e-9)
            best_action = (0.3 * dx / norm, 0.3 * dy / norm)
            best_info = {"status": "cadrl_fallback", "value": best_value, "num_humans": len(humans)}

        return best_action[0], best_action[1], best_info


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CADRL trained-model controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    p.add_argument("--model-path", default="models/cadrl_rl_model.pth", help="Path to CADRL .pth model")
    p.add_argument("--gpu", action="store_true", help="Use CUDA for CADRL value inference if available")
    p.add_argument("--cadrl-gamma", type=float, default=0.9)
    p.add_argument("--cadrl-speed-samples", type=int, default=5)
    p.add_argument("--cadrl-rotation-samples", type=int, default=16)
    p.add_argument("--cadrl-max-humans", type=int, default=5)
    p.add_argument("--cadrl-v-pref", type=float, default=None, help="Preferred speed used by CADRL action sampling; default uses max_speed")
    p.add_argument("--cadrl-sidewalk-penalty", type=float, default=1.0)
    p.add_argument("--cadrl-centerline-penalty", type=float, default=0.02)
    p.add_argument("--cadrl-goal-lookahead", type=float, default=6.0, help="Local lookahead goal distance fed to CADRL")
    p.add_argument("--cadrl-progress-bonus", type=float, default=0.20, help="Small reward for progress toward the local CADRL goal")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(args, lambda cfg, seed: CADRLPlanner(cfg, seed, args), "cadrl")


if __name__ == "__main__":
    main()
