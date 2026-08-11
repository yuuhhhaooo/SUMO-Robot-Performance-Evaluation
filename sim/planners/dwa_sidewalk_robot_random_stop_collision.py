#!/usr/bin/env python3
"""
DWA-controlled delivery robot in a SUMO sidewalk scene.

Robot representation:
- The robot is inserted as a SUMO person, not as a vehicle.
- Its position is controlled at every simulation step with traci.person.moveToXY().
- The DWA planner is constrained to the north sidewalk polygon only.

Run with SUMO GUI:
    python dwa_sidewalk_robot.py --cfg BasicConfig.sumocfg --sumo-gui

Run headless:
    python dwa_sidewalk_robot.py --cfg BasicConfig.sumocfg

This edited version stops a run immediately when the robot collides with a pedestrian.

Create an offline preview plot without SUMO/TraCI:
    python dwa_sidewalk_robot.py --offline-preview --net BasicNetwork.net.xml --rou BasicDemand.rou.xml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DWAConfig:
    # Robot / controller limits. Keep this sidewalk-delivery-robot-like, not car-like.
    max_speed: float = 0.95              # m/s
    min_speed: float = 0.00              # m/s
    max_yaw_rate: float = math.radians(80.0)       # rad/s
    max_accel: float = 0.80              # m/s^2
    max_delta_yaw_rate: float = math.radians(120.0)  # rad/s^2

    # Search resolution.
    v_resolution: float = 0.02           # m/s
    yaw_rate_resolution: float = math.radians(8.0) # rad/s
    dt: float = 0.50                     # s, same as the supplied SUMO config
    predict_time: float = 2.50           # s

    # Cost weights.
    goal_cost_gain: float = 2.0
    speed_cost_gain: float = 0.6
    obstacle_cost_gain: float = 0.85
    centerline_cost_gain: float = 0.12
    yaw_rate_cost_gain: float = 0.18

    # Geometry / safety.
    robot_radius: float = 0.25           # m
    pedestrian_radius: float = 0.15      # m
    safe_distance: float = 0.2          # m, collision/hard safety threshold
    social_distance: float = 0.80        # m, comfort / hazard threshold
    sensor_range: float = 12.0           # m
    goal_tolerance: float = 0.25         # m

    # North sidewalk polygon in the supplied files.
    sidewalk_x_min: float = 0.00
    sidewalk_x_max: float = 300.00
    sidewalk_y_min: float = 3.00
    sidewalk_y_max: float = 5.00
    sidewalk_center_y: float = 4.00

    max_time: float = 900.0              # s


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float
    w: float


@dataclass
class Obstacle:
    pid: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def motion(state: RobotState, v: float, w: float, dt: float) -> RobotState:
    """Unicycle model for local planning and direct position control."""
    yaw = angle_wrap(state.yaw + w * dt)
    x = state.x + v * math.cos(yaw) * dt
    y = state.y + v * math.sin(yaw) * dt
    return RobotState(x=x, y=y, yaw=yaw, v=v, w=w)


def in_sidewalk(x: float, y: float, cfg: DWAConfig, margin: float = 0.02) -> bool:
    return (
        cfg.sidewalk_x_min + margin <= x <= cfg.sidewalk_x_max - margin
        and cfg.sidewalk_y_min + margin <= y <= cfg.sidewalk_y_max - margin
    )


def clamp_to_sidewalk(state: RobotState, cfg: DWAConfig) -> RobotState:
    # The DWA already rejects outside trajectories. This is a last safety guard.
    eps = 0.03
    return RobotState(
        x=float(np.clip(state.x, cfg.sidewalk_x_min + eps, cfg.sidewalk_x_max - eps)),
        y=float(np.clip(state.y, cfg.sidewalk_y_min + eps, cfg.sidewalk_y_max - eps)),
        yaw=state.yaw,
        v=state.v,
        w=state.w,
    )


def calc_dynamic_window(state: RobotState, cfg: DWAConfig) -> Tuple[float, float, float, float]:
    vs = (cfg.min_speed, cfg.max_speed, -cfg.max_yaw_rate, cfg.max_yaw_rate)
    vd = (
        state.v - cfg.max_accel * cfg.dt,
        state.v + cfg.max_accel * cfg.dt,
        state.w - cfg.max_delta_yaw_rate * cfg.dt,
        state.w + cfg.max_delta_yaw_rate * cfg.dt,
    )
    return (
        max(vs[0], vd[0]),
        min(vs[1], vd[1]),
        max(vs[2], vd[2]),
        min(vs[3], vd[3]),
    )


def predict_trajectory(state: RobotState, v: float, w: float, cfg: DWAConfig) -> np.ndarray:
    traj = []
    s = RobotState(state.x, state.y, state.yaw, state.v, state.w)
    t = 0.0
    while t <= cfg.predict_time + 1e-9:
        s = motion(s, v, w, cfg.dt)
        traj.append((s.x, s.y, s.yaw, s.v, s.w))
        t += cfg.dt
    return np.array(traj, dtype=float)


def calc_obstacle_cost(traj: np.ndarray, obstacles: Sequence[Obstacle], cfg: DWAConfig) -> Tuple[float, float]:
    """Return obstacle cost and minimum clearance over the predicted trajectory.

    Obstacles are treated as locally static over the short DWA horizon. Since the planner
    replans every SUMO step, this is usually enough for a first baseline.
    """
    if not obstacles:
        return 0.0, float("inf")

    min_clearance = float("inf")
    collision_distance = cfg.robot_radius + cfg.pedestrian_radius
    for row in traj:
        px, py = row[0], row[1]
        for obs in obstacles:
            d = math.hypot(px - obs.x, py - obs.y) - collision_distance
            min_clearance = min(min_clearance, d)
            if d < 0.0:
                return float("inf"), min_clearance

    # A small epsilon avoids exploding cost for almost touching states.
    return 1.0 / max(min_clearance, 1e-3), min_clearance


def dwa_control(state: RobotState, goal: Tuple[float, float], obstacles: Sequence[Obstacle], cfg: DWAConfig) -> Tuple[Tuple[float, float], np.ndarray, Dict[str, float]]:
    dw = calc_dynamic_window(state, cfg)
    best_cost = float("inf")
    best_u = (0.0, 0.0)
    best_traj = predict_trajectory(state, 0.0, 0.0, cfg)
    best_info = {
        "goal_cost": float("inf"),
        "speed_cost": float("inf"),
        "obstacle_cost": float("inf"),
        "centerline_cost": float("inf"),
        "yaw_rate_cost": float("inf"),
        "min_pred_clearance": float("inf"),
        "total_cost": float("inf"),
    }

    # Use linspace-like inclusive loops to avoid missing the upper dynamic-window boundary.
    v_values = np.arange(dw[0], dw[1] + cfg.v_resolution * 0.5, cfg.v_resolution)
    w_values = np.arange(dw[2], dw[3] + cfg.yaw_rate_resolution * 0.5, cfg.yaw_rate_resolution)

    for v in v_values:
        for w in w_values:
            traj = predict_trajectory(state, float(v), float(w), cfg)

            # Hard sidewalk constraint: all predicted states must stay inside the sidewalk.
            if any(not in_sidewalk(px, py, cfg) for px, py in traj[:, :2]):
                continue

            dx = goal[0] - traj[-1, 0]
            dy = goal[1] - traj[-1, 1]
            goal_cost = cfg.goal_cost_gain * math.hypot(dx, dy)
            speed_cost = cfg.speed_cost_gain * (cfg.max_speed - traj[-1, 3])
            obs_cost_raw, min_pred_clearance = calc_obstacle_cost(traj, obstacles, cfg)
            if math.isinf(obs_cost_raw):
                continue
            obstacle_cost = cfg.obstacle_cost_gain * obs_cost_raw
            centerline_cost = cfg.centerline_cost_gain * float(np.mean((traj[:, 1] - cfg.sidewalk_center_y) ** 2))
            yaw_rate_cost = cfg.yaw_rate_cost_gain * abs(float(w))

            total_cost = goal_cost + speed_cost + obstacle_cost + centerline_cost + yaw_rate_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_u = (float(v), float(w))
                best_traj = traj
                best_info = {
                    "goal_cost": goal_cost,
                    "speed_cost": speed_cost,
                    "obstacle_cost": obstacle_cost,
                    "centerline_cost": centerline_cost,
                    "yaw_rate_cost": yaw_rate_cost,
                    "min_pred_clearance": min_pred_clearance,
                    "total_cost": total_cost,
                }

    # No feasible trajectory: brake and keep heading.
    if math.isinf(best_cost):
        brake_v = max(cfg.min_speed, state.v - cfg.max_accel * cfg.dt)
        best_u = (brake_v, 0.0)
        best_traj = predict_trajectory(state, best_u[0], best_u[1], cfg)
        best_info["total_cost"] = float("inf")

    return best_u, best_traj, best_info


def parse_lane_shapes(net_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    root = ET.parse(net_path).getroot()
    shapes: Dict[str, List[Tuple[float, float]]] = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        edge_id = edge.get("id") or ""
        lane = edge.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        pts = []
        for item in shape_str.split():
            x_str, y_str = item.split(",")[:2]
            pts.append((float(x_str), float(y_str)))
        shapes[edge_id] = pts
    return shapes


def edge_pos_to_xy(edge_shapes: Dict[str, List[Tuple[float, float]]], edge_id: str, pos: float) -> Tuple[float, float]:
    pts = edge_shapes[edge_id]
    if len(pts) < 2:
        return pts[0]
    # Supplied sidewalk edges are straight. This also handles a simple polyline.
    remaining = pos
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if remaining <= seg_len or seg_len <= 1e-9:
            r = 0.0 if seg_len <= 1e-9 else remaining / seg_len
            return (x0 + r * (x1 - x0), y0 + r * (y1 - y0))
        remaining -= seg_len
    return pts[-1]


def parse_static_pedestrians(rou_path: Path, net_path: Path) -> List[Obstacle]:
    """Read the nearly-stationary pedestrians from the uploaded route file.

    This is mainly used by --offline-preview. SUMO/TraCI mode reads live persons instead.
    """
    edge_shapes = parse_lane_shapes(net_path)
    root = ET.parse(rou_path).getroot()
    obstacles: List[Obstacle] = []
    for person in root.findall("person"):
        pid = person.get("id", "person")
        walk = person.find("walk")
        if walk is None:
            continue
        speed = float(walk.get("speed", "1.3"))
        if speed > 0.05:
            continue
        edges = (walk.get("edges") or "").split()
        if not edges:
            continue
        depart_pos = float(person.get("departPos", walk.get("departPos", "0")))
        edge_id = edges[0]
        if edge_id not in edge_shapes:
            continue
        x, y = edge_pos_to_xy(edge_shapes, edge_id, depart_pos)
        obstacles.append(Obstacle(pid=pid, x=x, y=y))
    return obstacles


def choose_edge_hint(x: float, y: float) -> str:
    # Map continuous sidewalk coordinates to the closest of the three north sidewalk strips.
    if y >= 4.335:
        row = "U"
    elif y >= 3.665:
        row = "M"
    else:
        row = "L"
    if x < 100.0:
        col = "A"
    elif x < 200.0:
        col = "B"
    else:
        col = "C"
    return f"walk_{col}{row}"


def min_distance_to_obstacles(x: float, y: float, obstacles: Sequence[Obstacle]) -> Tuple[float, str]:
    if not obstacles:
        return float("inf"), ""
    best_d = float("inf")
    best_id = ""
    for obs in obstacles:
        d = math.hypot(x - obs.x, y - obs.y)
        if d < best_d:
            best_d = d
            best_id = obs.pid
    return best_d, best_id


def compute_metrics(rows: List[Dict[str, float]], cfg: DWAConfig, goal: Tuple[float, float], success: bool, termination_reason: str = "unknown") -> Dict[str, float | int | bool | str]:
    if len(rows) < 2:
        return {"success": bool(success), "steps": len(rows), "termination_reason": termination_reason}

    xs = np.array([r["x"] for r in rows], dtype=float)
    ys = np.array([r["y"] for r in rows], dtype=float)
    vs = np.array([r["v"] for r in rows], dtype=float)
    ws = np.array([r["w"] for r in rows], dtype=float)
    ts = np.array([r["time"] for r in rows], dtype=float)
    min_ds = np.array([r["min_person_dist"] for r in rows], dtype=float)

    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else cfg.dt
    path_length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    straight_distance = float(math.hypot(goal[0] - xs[0], goal[1] - ys[0]))
    total_time = float(ts[-1] - ts[0])
    reference_time = straight_distance / max(cfg.max_speed, 1e-9)

    valid_min_ds = min_ds[np.isfinite(min_ds)]
    min_person_dist = float(np.min(valid_min_ds)) if len(valid_min_ds) else float("inf")
    collision_steps = int(np.sum(valid_min_ds < (cfg.robot_radius + cfg.pedestrian_radius))) if len(valid_min_ds) else 0
    hazard_steps = int(np.sum(valid_min_ds < cfg.social_distance)) if len(valid_min_ds) else 0
    social_steps = int(np.sum(valid_min_ds < max(cfg.sensor_range, cfg.social_distance))) if len(valid_min_ds) else 0

    accel = np.diff(vs) / dt if len(vs) > 1 else np.array([])
    jerk = np.diff(accel) / dt if len(accel) > 1 else np.array([])

    sidewalk_violations = int(
        np.sum(
            (xs < cfg.sidewalk_x_min)
            | (xs > cfg.sidewalk_x_max)
            | (ys < cfg.sidewalk_y_min)
            | (ys > cfg.sidewalk_y_max)
        )
    )

    return {
        "success": bool(success and collision_steps == 0),
        "termination_reason": termination_reason,
        "steps": int(len(rows)),
        "total_time_s": round(total_time, 3),
        "path_length_m": round(path_length, 3),
        "straight_line_distance_m": round(straight_distance, 3),
        "extra_distance_ratio": round(path_length / max(straight_distance, 1e-9), 4),
        "normalized_time_ratio": round(total_time / max(reference_time, 1e-9), 4),
        "final_goal_distance_m": round(float(math.hypot(goal[0] - xs[-1], goal[1] - ys[-1])), 3),
        "min_person_distance_m": round(min_person_dist, 3) if math.isfinite(min_person_dist) else None,
        "collision_steps": collision_steps,
        "hazard_steps_lt_social_distance": hazard_steps,
        "hazard_time_ratio": round(hazard_steps / max(len(rows), 1), 4),
        "hazard_ratio_given_social_presence": round(hazard_steps / max(social_steps, 1), 4),
        "sidewalk_violation_steps": sidewalk_violations,
        "sidewalk_violation_ratio": round(sidewalk_violations / max(len(rows), 1), 4),
        "mean_speed_mps": round(float(np.mean(vs)), 3),
        "mean_abs_yaw_rate_radps": round(float(np.mean(np.abs(ws))), 3),
        "mean_abs_accel_mps2": round(float(np.mean(np.abs(accel))), 3) if len(accel) else 0.0,
        "mean_abs_jerk_mps3": round(float(np.mean(np.abs(jerk))), 3) if len(jerk) else 0.0,
    }


def write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def get_effective_seed(args: argparse.Namespace) -> int:
    """Return a reproducible seed if supplied, otherwise create one for this run."""
    if args.seed is not None:
        return int(args.seed)
    # A random seed is still printed and saved, so the scenario can be reproduced later.
    return int(np.random.SeedSequence().entropy) % (2**31 - 1)


def random_scenario_output_dir(args: argparse.Namespace, seed: int) -> Path:
    out_dir = Path(args.output_dir)
    if args.random_scenario:
        out_dir = out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def generate_random_demand(base_rou: Path, out_rou: Path, args: argparse.Namespace, seed: int) -> Dict[str, object]:
    """Generate a random pedestrian scenario for the fixed SUMO network.

    The road/sidewalk geometry is not changed. This function randomizes the *scenario*
    on the north sidewalk: pedestrian flow density, walking speed, and the number/positions
    of nearly stationary pedestrians. If --flow-mode probability is used, SUMO's --seed
    will also change the exact pedestrian depart times.
    """
    rng = random.Random(seed)
    tree = ET.parse(base_rou)
    root = tree.getroot()

    # Remove the original north-side moving flows and original static pedestrians.
    for child in list(root):
        child_id = child.get("id", "")
        if child.tag == "personFlow" and (child_id.startswith("walk_east_") or child_id.startswith("walk_west_")):
            root.remove(child)
        elif child.tag == "person" and child_id.startswith("stand_"):
            root.remove(child)

    moving_specs = [
        ("walk_east_up", "walk_AU", "5.00", "95.00"),
        ("walk_east_mid", "walk_AM", "5.00", "95.00"),
        ("walk_east_dn", "walk_AL", "5.00", "95.00"),
        ("walk_west_up", "walk_CU", "95.00", "5.00"),
        ("walk_west_mid", "walk_CM", "95.00", "5.00"),
        ("walk_west_dn", "walk_CL", "95.00", "5.00"),
    ]

    meta: Dict[str, object] = {
        "seed": seed,
        "base_route_file": str(base_rou),
        "generated_route_file": str(out_rou),
        "flow_mode": args.flow_mode,
        "moving_flows": [],
        "static_pedestrians": [],
    }

    root.append(ET.Comment(f" random scenario generated by dwa_sidewalk_robot.py, seed={seed} "))

    # Generate randomized moving flows. Density is saved as persons/hour for readability.
    for flow_id, edge_id, depart_pos, arrival_pos in moving_specs:
        density = rng.uniform(args.flow_min, args.flow_max)
        speed = rng.uniform(args.speed_min, args.speed_max)
        attrs = {
            "id": flow_id,
            "begin": f"{args.scenario_begin:.2f}",
            "end": f"{args.scenario_end:.2f}",
            "departPos": depart_pos,
        }
        if args.flow_mode == "probability":
            # probability is evaluated every simulation second. This keeps the same expected
            # density but makes pedestrian depart times change with --seed.
            attrs["probability"] = f"{density / 3600.0:.6f}"
        else:
            attrs["personsPerHour"] = f"{density:.2f}"
        pf = ET.SubElement(root, "personFlow", attrs)
        ET.SubElement(pf, "walk", {
            "edges": edge_id,
            "arrivalPos": arrival_pos,
            "speed": f"{speed:.3f}",
        })
        meta["moving_flows"].append({
            "id": flow_id,
            "edge": edge_id,
            "density_persons_per_hour": round(density, 3),
            "speed_mps": round(speed, 3),
        })

    # Generate randomized nearly stationary pedestrians in the centre zone.
    row_to_edge = {"up": "walk_BU", "mid": "walk_BM", "dn": "walk_BL"}
    n_static = rng.randint(args.static_min, args.static_max)
    placed_by_edge: Dict[str, List[float]] = {edge: [] for edge in row_to_edge.values()}

    for i in range(n_static):
        edge_id = rng.choice(list(row_to_edge.values()))
        # Try to avoid static pedestrians being unrealistically stacked on top of each other.
        pos = None
        for _ in range(80):
            candidate = rng.uniform(8.0, 92.0)
            if all(abs(candidate - old) >= args.static_min_gap for old in placed_by_edge[edge_id]):
                pos = candidate
                break
        if pos is None:
            pos = rng.uniform(8.0, 92.0)
        placed_by_edge[edge_id].append(pos)
        arrival_pos = pos + rng.choice([-1.0, 1.0])
        arrival_pos = max(0.5, min(99.5, arrival_pos))

        person = ET.SubElement(root, "person", {
            "id": f"stand_{i}",
            "depart": "0.00",
            "departPos": f"{pos:.2f}",
        })
        ET.SubElement(person, "walk", {
            "edges": edge_id,
            "arrivalPos": f"{arrival_pos:.2f}",
            "speed": "0.00002",
        })
        meta["static_pedestrians"].append({
            "id": f"stand_{i}",
            "edge": edge_id,
            "departPos": round(pos, 2),
            "arrivalPos": round(arrival_pos, 2),
        })

    if hasattr(ET, "indent"):
        ET.indent(tree, space="    ")
    out_rou.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_rou, encoding="UTF-8", xml_declaration=True)
    meta_path = out_rou.with_suffix(".scenario.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Random scenario route: {out_rou}")
    print(f"Random scenario metadata: {meta_path}")
    return meta

def plot_route(path: Path, rows: List[Dict[str, float]], obstacles: Sequence[Obstacle], cfg: DWAConfig, goal: Tuple[float, float], title: str) -> None:
    import matplotlib.pyplot as plt

    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axhspan(cfg.sidewalk_y_min, cfg.sidewalk_y_max, color="#f5f5f5", zorder=0)
    ax.hlines(
        [cfg.sidewalk_y_min, cfg.sidewalk_y_max],
        cfg.sidewalk_x_min,
        cfg.sidewalk_x_max,
        colors="black",
        linewidth=1.5,
        label="sidewalk",
        zorder=1,
    )

    if obstacles:
        ax.scatter(
            [o.x for o in obstacles],
            [o.y for o in obstacles],
            marker="x",
            s=55,
            linewidths=1.6,
            label="obstacles",
            zorder=3,
        )

    ax.plot(xs, ys, linewidth=2.4, color="#d62728", label="robot path", zorder=4)
    ax.scatter([xs[0]], [ys[0]], marker="o", s=70, color="#ff7f0e", label="start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=130, color="#2ca02c", label="goal", zorder=5)
    ax.set_xlim(-5, 305)
    ax.set_ylim(2.7, 5.3)
    ax.set_aspect("auto")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(title)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        fontsize=8,
        frameon=True,
        edgecolor="#cccccc",
    )
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.35)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def offline_preview(args: argparse.Namespace, cfg: DWAConfig) -> None:
    base_dir = Path.cwd()
    net_path = resolve_path(args.net, base_dir)
    seed = get_effective_seed(args)
    out_dir = random_scenario_output_dir(args, seed)
    rou_path = resolve_path(args.rou, base_dir)
    if args.random_scenario:
        random_rou = out_dir / f"BasicDemand_random_seed_{seed}.rou.xml"
        generate_random_demand(rou_path, random_rou, args, seed)
        rou_path = random_rou
    obstacles = parse_static_pedestrians(rou_path, net_path)

    start = (args.start_x, args.start_y)
    goal = (args.goal_x, args.goal_y)
    state = RobotState(x=start[0], y=start[1], yaw=0.0, v=0.0, w=0.0)

    rows: List[Dict[str, float | str]] = []
    t = 0.0
    success = False
    termination_reason = "max_time"
    collision_distance = cfg.robot_radius + cfg.pedestrian_radius
    for step in range(int(cfg.max_time / cfg.dt) + 1):
        nearby = [o for o in obstacles if math.hypot(state.x - o.x, state.y - o.y) <= cfg.sensor_range]
        (v, w), _, info = dwa_control(state, goal, nearby, cfg)
        state = motion(state, v, w, cfg.dt)
        state = clamp_to_sidewalk(state, cfg)
        dmin, closest = min_distance_to_obstacles(state.x, state.y, obstacles)
        rows.append(
            {
                "time": round(t, 3),
                "x": round(state.x, 4),
                "y": round(state.y, 4),
                "yaw": round(state.yaw, 5),
                "v": round(state.v, 4),
                "w": round(state.w, 5),
                "goal_distance": round(math.hypot(goal[0] - state.x, goal[1] - state.y), 4),
                "min_person_dist": round(dmin, 4) if math.isfinite(dmin) else float("inf"),
                "closest_person": closest,
                "total_cost": round(info["total_cost"], 4) if math.isfinite(info["total_cost"]) else float("inf"),
            }
        )
        if math.isfinite(dmin) and dmin < collision_distance:
            success = False
            termination_reason = "collision"
            print(f"Collision detected in offline preview at t={t:.2f}s with {closest}; stopping this run.")
            break
        if math.hypot(goal[0] - state.x, goal[1] - state.y) <= cfg.goal_tolerance:
            success = True
            termination_reason = "goal_reached"
            break
        t += cfg.dt

    trace_csv = out_dir / "offline_dwa_robot_trace.csv"
    metrics_json = out_dir / "offline_dwa_robot_metrics.json"
    plot_png = out_dir / "offline_dwa_robot_route.png"

    write_csv(trace_csv, rows)
    metrics = compute_metrics(rows, cfg, goal, success, termination_reason)
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_route(plot_png, rows, obstacles, cfg, goal, "Offline DWA route preview on the north sidewalk")

    print(f"Offline preview finished. Trace: {trace_csv}")
    print(f"Metrics: {metrics_json}")
    print(f"Plot: {plot_png}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def get_live_obstacles(traci, robot_id: str, state: RobotState, previous_positions: Dict[str, Tuple[float, float]], cfg: DWAConfig) -> List[Obstacle]:
    obstacles: List[Obstacle] = []
    current_positions: Dict[str, Tuple[float, float]] = {}
    for pid in traci.person.getIDList():
        if pid == robot_id:
            continue
        try:
            x, y = traci.person.getPosition(pid)
        except Exception:
            continue
        current_positions[pid] = (x, y)
        if math.hypot(x - state.x, y - state.y) > cfg.sensor_range:
            continue
        if pid in previous_positions:
            px, py = previous_positions[pid]
            vx = (x - px) / max(cfg.dt, 1e-9)
            vy = (y - py) / max(cfg.dt, 1e-9)
        else:
            vx = vy = 0.0
        obstacles.append(Obstacle(pid=pid, x=x, y=y, vx=vx, vy=vy))
    previous_positions.clear()
    previous_positions.update(current_positions)
    return obstacles


def run_traci(args: argparse.Namespace, cfg: DWAConfig) -> None:
    try:
        import traci  # type: ignore
        try:
            from sumolib import checkBinary  # type: ignore
        except Exception:
            checkBinary = None
    except Exception as exc:
        raise RuntimeError(
            "Cannot import traci. Run this inside a SUMO Python environment, or set SUMO_HOME/tools on PYTHONPATH. "
            "For a plot without SUMO, use --offline-preview."
        ) from exc

    cfg_path = Path(args.cfg).resolve()
    work_dir = cfg_path.parent
    sumo_name = "sumo-gui" if args.sumo_gui else "sumo"
    sumo_binary = checkBinary(sumo_name) if checkBinary else sumo_name

    seed = get_effective_seed(args)
    out_dir = random_scenario_output_dir(args, seed)
    route_override: Optional[Path] = None
    if args.random_scenario:
        base_rou = resolve_path(args.rou, work_dir)
        route_override = out_dir / f"BasicDemand_random_seed_{seed}.rou.xml"
        generate_random_demand(base_rou, route_override, args, seed)

    cmd = [
        sumo_binary,
        "-c",
        str(cfg_path),
        "--step-length",
        str(cfg.dt),
        "--quit-on-end",
    ]
    if route_override is not None:
        cmd += ["--route-files", str(route_override)]
    if args.random_scenario or args.seed is not None:
        cmd += ["--seed", str(seed)]
    if args.sumo_gui:
        cmd += ["--start"]

    start = (args.start_x, args.start_y)
    goal = (args.goal_x, args.goal_y)
    state = RobotState(x=start[0], y=start[1], yaw=0.0, v=0.0, w=0.0)
    robot_id = args.robot_id

    trace_csv = out_dir / "dwa_robot_trace.csv"
    metrics_json = out_dir / "dwa_robot_metrics.json"
    plot_png = out_dir / "dwa_robot_route.png"

    rows: List[Dict[str, float | str]] = []
    all_obstacles_for_plot: Dict[str, Obstacle] = {}
    previous_positions: Dict[str, Tuple[float, float]] = {}
    success = False
    termination_reason = "max_time"
    collision_distance = cfg.robot_radius + cfg.pedestrian_radius

    old_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        traci.start(cmd)
        # Insert the robot as a pedestrian/person. A long waiting stage keeps it alive while
        # external control moves it with moveToXY every step.
        if robot_id not in traci.person.getIDList():
            traci.person.add(robot_id, choose_edge_hint(start[0], start[1]), pos=start[0], depart=0, typeID="DEFAULT_PEDTYPE")
            traci.person.appendWaitingStage(robot_id, duration=cfg.max_time + 1000.0, description="dwa_controlled_robot", stopID="")
        traci.simulationStep()
        try:
            traci.person.setColor(robot_id, (255, 0, 0, 255))
        except Exception:
            pass
        traci.person.moveToXY(robot_id, choose_edge_hint(start[0], start[1]), start[0], start[1], angle=90.0, keepRoute=2, matchThreshold=20.0)

        n_steps = int(cfg.max_time / cfg.dt)
        for _ in range(n_steps + 1):
            sim_time = float(traci.simulation.getTime())

            live_obstacles = get_live_obstacles(traci, robot_id, state, previous_positions, cfg)
            for obs in live_obstacles:
                all_obstacles_for_plot[obs.pid] = obs

            (v, w), _, info = dwa_control(state, goal, live_obstacles, cfg)
            state = motion(state, v, w, cfg.dt)
            state = clamp_to_sidewalk(state, cfg)

            edge_hint = choose_edge_hint(state.x, state.y)
            traci.person.moveToXY(
                robot_id,
                edge_hint,
                state.x,
                state.y,
                angle=math.degrees(state.yaw),
                keepRoute=2,
                matchThreshold=20.0,
            )

            dmin, closest = min_distance_to_obstacles(state.x, state.y, live_obstacles)
            rows.append(
                {
                    "time": round(sim_time, 3),
                    "x": round(state.x, 4),
                    "y": round(state.y, 4),
                    "yaw": round(state.yaw, 5),
                    "v": round(state.v, 4),
                    "w": round(state.w, 5),
                    "goal_distance": round(math.hypot(goal[0] - state.x, goal[1] - state.y), 4),
                    "min_person_dist": round(dmin, 4) if math.isfinite(dmin) else float("inf"),
                    "closest_person": closest,
                    "total_cost": round(info["total_cost"], 4) if math.isfinite(info["total_cost"]) else float("inf"),
                    # R-KNav-like action/state fields: command output and executed state.
                    "action_linear_x": round(v, 4),
                    "action_angular_z": round(w, 5),
                    "observation_state_linear_x": round(state.v, 4),
                    "observation_state_angular_z": round(state.w, 5),
                    "observation_state_road_type": "Sidewalk",
                }
            )

            if math.isfinite(dmin) and dmin < collision_distance:
                success = False
                termination_reason = "collision"
                print(f"Collision detected at t={sim_time:.2f}s with {closest}; stopping this seed.")
                break

            if math.hypot(goal[0] - state.x, goal[1] - state.y) <= cfg.goal_tolerance:
                success = True
                termination_reason = "goal_reached"
                break

            traci.simulationStep()
    finally:
        try:
            traci.close(False)
        except Exception:
            pass
        os.chdir(old_cwd)

    write_csv(trace_csv, rows)
    metrics = compute_metrics(rows, cfg, goal, success, termination_reason)
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_route(plot_png, rows, list(all_obstacles_for_plot.values()), cfg, goal, "DWA robot route in SUMO sidewalk scene")

    print(f"Trace: {trace_csv}")
    print(f"Metrics: {metrics_json}")
    print(f"Plot: {plot_png}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DWA robot-as-pedestrian controller for a SUMO sidewalk scene")
    p.add_argument("--cfg", default="BasicConfig.sumocfg", help="SUMO configuration file")
    p.add_argument("--net", default="BasicNetwork.net.xml", help="SUMO network file, used by --offline-preview")
    p.add_argument("--rou", default="BasicDemand.rou.xml", help="Base SUMO route file. With --random-scenario, this file is used as a template")
    p.add_argument("--random-scenario", action="store_true", help="Generate a random pedestrian scenario before running SUMO")
    p.add_argument("--seed", type=int, default=None, help="Random seed for scenario generation and SUMO randomness")
    p.add_argument("--flow-mode", choices=["probability", "personsPerHour"], default="probability", help="probability makes depart times random under different SUMO seeds")
    p.add_argument("--speed-min", type=float, default=0.80, help="Minimum random pedestrian speed in m/s")
    p.add_argument("--speed-max", type=float, default=1.60, help="Maximum random pedestrian speed in m/s")
    p.add_argument("--flow-min", type=float, default=80.0, help="Minimum random density per moving flow in persons/hour")
    p.add_argument("--flow-max", type=float, default=350.0, help="Maximum random density per moving flow in persons/hour")
    p.add_argument("--static-min", type=int, default=4, help="Minimum number of random static pedestrians")
    p.add_argument("--static-max", type=int, default=14, help="Maximum number of random static pedestrians")
    p.add_argument("--static-min-gap", type=float, default=6.0, help="Minimum position gap between static pedestrians on the same centre edge")
    p.add_argument("--scenario-begin", type=float, default=0.0, help="Begin time for generated personFlow elements")
    p.add_argument("--scenario-end", type=float, default=36000.0, help="End time for generated personFlow elements")
    p.add_argument("--sumo-gui", "--gui", dest="sumo_gui", action="store_true", help="Run with sumo-gui instead of sumo")
    p.add_argument("--offline-preview", action="store_true", help="Generate a route preview without starting SUMO/TraCI")
    p.add_argument("--output-dir", default="dwa_outputs", help="Directory for trace CSV, metrics JSON and route PNG")
    p.add_argument("--robot-id", default="robot0", help="SUMO person id for the robot")
    p.add_argument("--start-x", type=float, default=2.0)
    p.add_argument("--start-y", type=float, default=4.0)
    p.add_argument("--goal-x", type=float, default=298.0)
    p.add_argument("--goal-y", type=float, default=4.0)
    p.add_argument("--max-time", type=float, default=900.0, help="Maximum simulated time per seed in seconds")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = DWAConfig()
    cfg.max_time = float(args.max_time)
    if args.offline_preview:
        offline_preview(args, cfg)
    else:
        run_traci(args, cfg)


if __name__ == "__main__":
    main()
