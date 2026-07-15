#!/usr/bin/env python3
"""Shared utilities for SUMO sidewalk robot baseline planners.

The robot is controlled as a SUMO person with TraCI moveToXY().
This file is used by A*, RRT and ORCA-style baseline scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PlannerConfig:
    # Common robot limits
    max_speed: float = 1.00
    min_speed: float = 0.00
    max_accel: float = 0.50
    max_yaw_rate: float = math.radians(120.0)
    dt: float = 0.50

    # Geometry / safety
    robot_radius: float = 0.25
    pedestrian_radius: float = 0.15
    safe_distance: float = 0.42
    social_distance: float = 0.85
    sensor_range: float = 11.0
    goal_tolerance: float = 0.35

    # North sidewalk polygon in the supplied SUMO network
    sidewalk_x_min: float = 0.00
    sidewalk_x_max: float = 300.00
    sidewalk_y_min: float = 3.00
    sidewalk_y_max: float = 5.00
    sidewalk_center_y: float = 4.00

    max_time: float = 900.0


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


def in_sidewalk(x: float, y: float, cfg: PlannerConfig, margin: float = 0.03) -> bool:
    return (
        cfg.sidewalk_x_min + margin <= x <= cfg.sidewalk_x_max - margin
        and cfg.sidewalk_y_min + margin <= y <= cfg.sidewalk_y_max - margin
    )


def clamp_xy(x: float, y: float, cfg: PlannerConfig) -> Tuple[float, float]:
    eps = 0.03
    return (
        float(np.clip(x, cfg.sidewalk_x_min + eps, cfg.sidewalk_x_max - eps)),
        float(np.clip(y, cfg.sidewalk_y_min + eps, cfg.sidewalk_y_max - eps)),
    )


def apply_velocity(state: RobotState, vx: float, vy: float, cfg: PlannerConfig) -> RobotState:
    speed = math.hypot(vx, vy)
    if speed > cfg.max_speed:
        scale = cfg.max_speed / max(speed, 1e-9)
        vx *= scale
        vy *= scale
        speed = cfg.max_speed

    # Acceleration limit for fairer comparison with DWA.
    if speed > state.v + cfg.max_accel * cfg.dt:
        limited = state.v + cfg.max_accel * cfg.dt
        scale = limited / max(speed, 1e-9)
        vx *= scale
        vy *= scale
        speed = limited
    elif speed < max(cfg.min_speed, state.v - cfg.max_accel * cfg.dt):
        limited = max(cfg.min_speed, state.v - cfg.max_accel * cfg.dt)
        scale = limited / max(speed, 1e-9) if speed > 1e-9 else 0.0
        vx *= scale
        vy *= scale
        speed = math.hypot(vx, vy)

    new_x = state.x + vx * cfg.dt
    new_y = state.y + vy * cfg.dt
    new_x, new_y = clamp_xy(new_x, new_y, cfg)

    if speed > 1e-6:
        new_yaw = math.atan2(vy, vx)
    else:
        new_yaw = state.yaw
    yaw_delta = angle_wrap(new_yaw - state.yaw)
    max_delta = cfg.max_yaw_rate * cfg.dt
    if abs(yaw_delta) > max_delta:
        new_yaw = angle_wrap(state.yaw + math.copysign(max_delta, yaw_delta))
    w = angle_wrap(new_yaw - state.yaw) / max(cfg.dt, 1e-9)
    return RobotState(x=new_x, y=new_y, yaw=new_yaw, v=speed, w=w)


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


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def get_effective_seed(args: argparse.Namespace) -> int:
    if getattr(args, "seed", None) is not None:
        return int(args.seed)
    return int(np.random.SeedSequence().entropy) % (2**31 - 1)


def random_scenario_output_dir(args: argparse.Namespace, seed: int) -> Path:
    out_dir = Path(args.output_dir)
    if args.random_scenario:
        out_dir = out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def generate_random_demand(base_rou: Path, out_rou: Path, args: argparse.Namespace, seed: int) -> Dict[str, object]:
    """Generate a random pedestrian scenario for the fixed SUMO network.

    Geometry is unchanged. Only north-side moving flow density/speed and static
    pedestrian locations are randomized. This keeps every seed comparable.
    """
    rng = random.Random(seed)
    tree = ET.parse(base_rou)
    root = tree.getroot()

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

    root.append(ET.Comment(f" random scenario generated for baseline planner, seed={seed} "))

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
            attrs["probability"] = f"{density / 3600.0:.6f}"
        else:
            attrs["personsPerHour"] = f"{density:.2f}"
        pf = ET.SubElement(root, "personFlow", attrs)
        ET.SubElement(pf, "walk", {"edges": edge_id, "arrivalPos": arrival_pos, "speed": f"{speed:.3f}"})
        meta["moving_flows"].append({
            "id": flow_id,
            "edge": edge_id,
            "density_persons_per_hour": round(density, 3),
            "speed_mps": round(speed, 3),
        })

    row_edges = ["walk_BU", "walk_BM", "walk_BL"]
    n_static = rng.randint(args.static_min, args.static_max)
    placed_by_edge: Dict[str, List[float]] = {edge: [] for edge in row_edges}
    for i in range(n_static):
        edge_id = rng.choice(row_edges)
        pos = None
        for _ in range(80):
            candidate = rng.uniform(8.0, 92.0)
            if all(abs(candidate - old) >= args.static_min_gap for old in placed_by_edge[edge_id]):
                pos = candidate
                break
        if pos is None:
            pos = rng.uniform(8.0, 92.0)
        placed_by_edge[edge_id].append(pos)
        arrival_pos = max(0.5, min(99.5, pos + rng.choice([-1.0, 1.0])))
        person = ET.SubElement(root, "person", {"id": f"stand_{i}", "depart": "0.00", "departPos": f"{pos:.2f}"})
        ET.SubElement(person, "walk", {"edges": edge_id, "arrivalPos": f"{arrival_pos:.2f}", "speed": "0.00002"})
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


def get_live_obstacles(traci: Any, robot_id: str, state: RobotState, previous_positions: Dict[str, Tuple[float, float]], cfg: PlannerConfig) -> List[Obstacle]:
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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(rows: List[Dict[str, Any]], cfg: PlannerConfig, goal: Tuple[float, float], success: bool, termination_reason: str) -> Dict[str, Any]:
    if len(rows) < 2:
        return {"success": bool(success), "steps": len(rows), "termination_reason": termination_reason}

    xs = np.array([float(r["x"]) for r in rows], dtype=float)
    ys = np.array([float(r["y"]) for r in rows], dtype=float)
    vs = np.array([float(r["v"]) for r in rows], dtype=float)
    ws = np.array([float(r["w"]) for r in rows], dtype=float)
    ts = np.array([float(r["time"]) for r in rows], dtype=float)
    min_ds = np.array([float(r["min_person_dist"]) for r in rows], dtype=float)

    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else cfg.dt
    path_length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    straight_distance = float(math.hypot(goal[0] - xs[0], goal[1] - ys[0]))
    total_time = float(ts[-1] - ts[0])
    reference_time = straight_distance / max(cfg.max_speed, 1e-9)

    valid_min_ds = min_ds[np.isfinite(min_ds)]
    min_person_dist = float(np.min(valid_min_ds)) if len(valid_min_ds) else float("inf")
    collision_distance = max(cfg.safe_distance, cfg.robot_radius + cfg.pedestrian_radius)
    collision_steps = int(np.sum(valid_min_ds < collision_distance)) if len(valid_min_ds) else 0
    hazard_steps = int(np.sum(valid_min_ds < cfg.social_distance)) if len(valid_min_ds) else 0
    social_steps = int(np.sum(valid_min_ds < max(cfg.sensor_range, cfg.social_distance))) if len(valid_min_ds) else 0

    accel = np.diff(vs) / dt if len(vs) > 1 else np.array([])
    jerk = np.diff(accel) / dt if len(accel) > 1 else np.array([])
    sidewalk_violations = int(np.sum((xs < cfg.sidewalk_x_min) | (xs > cfg.sidewalk_x_max) | (ys < cfg.sidewalk_y_min) | (ys > cfg.sidewalk_y_max)))

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


def plot_route(path: Path, rows: List[Dict[str, Any]], obstacles: Sequence[Obstacle], cfg: PlannerConfig, goal: Tuple[float, float], title: str) -> None:
    import matplotlib.pyplot as plt

    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axhspan(cfg.sidewalk_y_min, cfg.sidewalk_y_max, color="#f5f5f5", zorder=0)
    ax.hlines([cfg.sidewalk_y_min, cfg.sidewalk_y_max], cfg.sidewalk_x_min, cfg.sidewalk_x_max, colors="black", linewidth=1.5, label="sidewalk", zorder=1)
    ax.hlines([cfg.sidewalk_center_y], cfg.sidewalk_x_min, cfg.sidewalk_x_max, colors="#bbbbbb", linewidth=0.8, linestyles="dashed", zorder=1)
    if obstacles:
        ax.scatter([o.x for o in obstacles], [o.y for o in obstacles], marker="x", s=55, linewidths=1.6, label="nearby pedestrians", zorder=3)
    if xs:
        ax.plot(xs, ys, linewidth=2.4, label="robot path", zorder=4)
        ax.scatter([xs[0]], [ys[0]], marker="o", s=70, label="start", zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=130, label="goal", zorder=5)
    ax.set_xlim(-5, 305)
    ax.set_ylim(2.7, 5.3)
    ax.set_aspect("auto")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=8, frameon=True, edgecolor="#cccccc")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.35)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def add_common_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cfg", default="BasicConfig.sumocfg", help="SUMO configuration file")
    p.add_argument("--net", default="BasicNetwork.net.xml", help="SUMO network file")
    p.add_argument("--rou", default="BasicDemand.rou.xml", help="Base SUMO route file; used as template with --random-scenario")
    p.add_argument("--random-scenario", action="store_true", help="Generate a random pedestrian scenario before running SUMO")
    p.add_argument("--seed", type=int, default=None, help="Random seed for scenario generation and SUMO randomness")
    p.add_argument("--flow-mode", choices=["probability", "personsPerHour"], default="personsPerHour")
    p.add_argument("--speed-min", type=float, default=0.80)
    p.add_argument("--speed-max", type=float, default=1.60)
    p.add_argument("--flow-min", type=float, default=80.0)
    p.add_argument("--flow-max", type=float, default=350.0)
    p.add_argument("--static-min", type=int, default=4)
    p.add_argument("--static-max", type=int, default=14)
    p.add_argument("--static-min-gap", type=float, default=6.0)
    p.add_argument("--scenario-begin", type=float, default=0.0)
    p.add_argument("--scenario-end", type=float, default=36000.0)
    p.add_argument("--sumo-gui", "--gui", dest="sumo_gui", action="store_true", help="Run with sumo-gui instead of sumo")
    p.add_argument("--output-dir", default="baseline_outputs", help="Directory for trace CSV, metrics JSON and route PNG")
    p.add_argument("--robot-id", default="robot0", help="SUMO person id for the robot")
    p.add_argument("--start-x", type=float, default=2.0)
    p.add_argument("--start-y", type=float, default=4.0)
    p.add_argument("--goal-x", type=float, default=298.0)
    p.add_argument("--goal-y", type=float, default=4.0)
    p.add_argument("--max-time", type=float, default=900.0, help="Maximum simulated time per seed in seconds")


def run_traci_with_planner(args: argparse.Namespace, planner_factory: Any, algorithm_name: str, cfg: Optional[PlannerConfig] = None) -> None:
    if cfg is None:
        cfg = PlannerConfig()
    if hasattr(args, "max_time") and args.max_time is not None:
        cfg.max_time = float(args.max_time)
    try:
        import traci  # type: ignore
        try:
            from sumolib import checkBinary  # type: ignore
        except Exception:
            checkBinary = None
    except Exception as exc:
        raise RuntimeError(
            "Cannot import traci. Set SUMO_HOME/tools on PYTHONPATH or use the SUMO Python environment."
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

    cmd = [sumo_binary, "-c", str(cfg_path), "--step-length", str(cfg.dt), "--quit-on-end"]
    if route_override is not None:
        cmd += ["--route-files", str(route_override)]
    if args.random_scenario or args.seed is not None:
        cmd += ["--seed", str(seed)]
    if args.sumo_gui:
        cmd += ["--start"]

    start = (args.start_x, args.start_y)
    goal = (args.goal_x, args.goal_y)
    state = RobotState(x=start[0], y=start[1], yaw=0.0, v=0.0, w=0.0)
    planner = planner_factory(cfg, seed)
    robot_id = args.robot_id

    trace_csv = out_dir / "robot_trace.csv"
    metrics_json = out_dir / "robot_metrics.json"
    plot_png = out_dir / "robot_route.png"

    rows: List[Dict[str, Any]] = []
    all_obstacles_for_plot: Dict[str, Obstacle] = {}
    previous_positions: Dict[str, Tuple[float, float]] = {}
    success = False
    termination_reason = "max_time"
    collision_distance = max(cfg.safe_distance, cfg.robot_radius + cfg.pedestrian_radius)

    old_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        traci.start(cmd)
        if robot_id not in traci.person.getIDList():
            traci.person.add(robot_id, choose_edge_hint(start[0], start[1]), pos=start[0], depart=0, typeID="DEFAULT_PEDTYPE")
            traci.person.appendWaitingStage(robot_id, duration=cfg.max_time + 1000.0, description=f"{algorithm_name}_controlled_robot", stopID="")
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

            vx, vy, info = planner.compute_command(state, goal, live_obstacles, sim_time)
            state = apply_velocity(state, vx, vy, cfg)

            edge_hint = choose_edge_hint(state.x, state.y)
            traci.person.moveToXY(robot_id, edge_hint, state.x, state.y, angle=math.degrees(state.yaw), keepRoute=2, matchThreshold=20.0)

            dmin, closest = min_distance_to_obstacles(state.x, state.y, live_obstacles)
            rows.append({
                "time": round(sim_time, 3),
                "x": round(state.x, 4),
                "y": round(state.y, 4),
                "yaw": round(state.yaw, 5),
                "v": round(state.v, 4),
                "w": round(state.w, 5),
                "goal_distance": round(math.hypot(goal[0] - state.x, goal[1] - state.y), 4),
                "min_person_dist": round(dmin, 4) if math.isfinite(dmin) else float("inf"),
                "closest_person": closest,
                "planner_status": str(info.get("status", "")),
                "planner_cost": round(float(info.get("cost", 0.0)), 4) if isinstance(info.get("cost", 0.0), (int, float)) and math.isfinite(float(info.get("cost", 0.0))) else "",
                "action_linear_x": round(state.v, 4),
                "action_angular_z": round(state.w, 5),
                "observation_state_linear_x": round(state.v, 4),
                "observation_state_angular_z": round(state.w, 5),
                "observation_state_road_type": "Sidewalk",
            })

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
    metrics["algorithm"] = algorithm_name
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_route(plot_png, rows, list(all_obstacles_for_plot.values()), cfg, goal, f"{algorithm_name.upper()} robot route in SUMO sidewalk scene")

    print(f"Trace: {trace_csv}")
    print(f"Metrics: {metrics_json}")
    print(f"Plot: {plot_png}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
