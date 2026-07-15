#!/usr/bin/env python3
"""Run many randomized SUMO pedestrian scenarios for DWA, A*, Dijkstra, RRT or ORCA-style baselines.

This unified batch runner supports all five baseline scripts and overlays all robot
paths from different seeds into one figure.

Examples:
    python run_random_batch_overlay_all.py --algorithm dwa   --num-seeds 100 --flow-mode personsPerHour
    python run_random_batch_overlay_all.py --algorithm astar --num-seeds 100 --flow-mode personsPerHour
    python run_random_batch_overlay_all.py --algorithm orca  --num-seeds 100 --flow-mode personsPerHour --sumo-gui
    python run_random_batch_overlay_all.py --algorithm rrt   --num-seeds 100 --flow-mode personsPerHour
    python run_random_batch_overlay_all.py --algorithm dijkstra --num-seeds 100 --flow-mode personsPerHour
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

SCRIPT_BY_ALGORITHM = {
    "dwa": "dwa_sidewalk_robot_random_stop_collision.py",
    "astar": "astar_sidewalk_robot_random_stop_collision.py",
    "dijkstra": "dijkstra_sidewalk_robot_random_stop_collision.py",
    "rrt": "rrt_sidewalk_robot_random_stop_collision.py",
    "orca": "orca_sidewalk_robot_random_stop_collision.py",
    "mpc": "mpc_sidewalk_robot_random_stop_collision.py",
    "teb": "teb_sidewalk_robot_random_stop_collision.py",
}

METRICS_BY_ALGORITHM = {
    "dwa": "dwa_robot_metrics.json",
    "astar": "robot_metrics.json",
    "dijkstra": "robot_metrics.json",
    "rrt": "robot_metrics.json",
    "orca": "robot_metrics.json",
    "mpc": "robot_metrics.json",
    "teb": "robot_metrics.json",
}

TRACE_BY_ALGORITHM = {
    "dwa": "dwa_robot_trace.csv",
    "astar": "robot_trace.csv",
    "dijkstra": "robot_trace.csv",
    "rrt": "robot_trace.csv",
    "orca": "robot_trace.csv",
    "mpc": "robot_trace.csv",
    "teb": "robot_trace.csv",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified batch runner for randomized sidewalk robot baseline planners")
    p.add_argument("--algorithm", choices=sorted(SCRIPT_BY_ALGORITHM), required=True)
    p.add_argument("--script", default=None, help="Override script path")
    p.add_argument("--cfg", default="BasicConfig.sumocfg")
    p.add_argument("--rou", default="BasicDemand.rou.xml")
    p.add_argument("--net", default="BasicNetwork.net.xml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--start-seed", type=int, default=1)
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--sumo-gui", action="store_true", help="Use GUI; not recommended for very large batches")
    p.add_argument("--speed-min", type=float, default=0.80)
    p.add_argument("--speed-max", type=float, default=1.60)
    p.add_argument("--flow-min", type=float, default=80.0)
    p.add_argument("--flow-max", type=float, default=350.0)
    p.add_argument("--static-min", type=int, default=4)
    p.add_argument("--static-max", type=int, default=14)
    p.add_argument("--flow-mode", choices=["probability", "personsPerHour"], default="personsPerHour")
    p.add_argument("--no-combined-plot", action="store_true")
    p.add_argument("--start-x", type=float, default=2.0)
    p.add_argument("--start-y", type=float, default=4.0)
    p.add_argument("--goal-x", type=float, default=298.0)
    p.add_argument("--goal-y", type=float, default=4.0)
    p.add_argument("--max-time", type=float, default=900.0, help="Maximum simulated time per seed in seconds")
    return p


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_runs": len(rows)}
    if not rows:
        return summary
    keys = set().union(*(row.keys() for row in rows))
    for key in sorted(keys):
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            summary[f"{key}_mean"] = round(mean(values), 6)
            summary[f"{key}_std"] = round(stdev(values), 6) if len(values) > 1 else 0.0
    return summary


def read_trace_xy(path: Path) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    if not path.exists():
        return xs, ys
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
            except (KeyError, TypeError, ValueError):
                continue
    return xs, ys


def plot_all_seed_paths(output_path: Path, output_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axhspan(3.0, 5.0, color="#f5f5f5", zorder=0)
    ax.hlines([3.0, 5.0], 0.0, 300.0, colors="black", linewidth=1.4, label="sidewalk", zorder=1)
    ax.hlines([4.0], 0.0, 300.0, colors="#bbbbbb", linewidth=0.8, linestyles="dashed", zorder=1)

    trace_name = TRACE_BY_ALGORITHM[args.algorithm]
    plotted = 0
    for row in sorted(rows, key=lambda item: int(item.get("seed", 0))):
        seed = int(row["seed"])
        trace_path = output_dir / f"seed_{seed}" / trace_name
        xs, ys = read_trace_xy(trace_path)
        if not xs:
            continue
        reason = str(row.get("termination_reason", ""))
        success = bool(row.get("success", False))
        ax.plot(xs, ys, linewidth=1.2, alpha=0.78)
        if reason == "collision" or int(row.get("collision_steps", 0) or 0) > 0:
            ax.scatter([xs[-1]], [ys[-1]], marker="x", s=45, linewidths=1.7, zorder=5)
        elif success:
            ax.scatter([xs[-1]], [ys[-1]], marker="o", s=24, zorder=5)
        else:
            ax.scatter([xs[-1]], [ys[-1]], marker="s", s=24, zorder=5)
        plotted += 1

    ax.scatter([args.start_x], [args.start_y], marker="o", s=90, edgecolors="black", label="start", zorder=6)
    ax.scatter([args.goal_x], [args.goal_y], marker="*", s=170, edgecolors="black", label="goal", zorder=6)
    ax.set_xlim(-5, 305)
    ax.set_ylim(2.7, 5.3)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"{args.algorithm.upper()} robot paths under different random seeds, n={plotted}")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    script = args.script or SCRIPT_BY_ALGORITHM[args.algorithm]
    output_dir = Path(args.output_dir or f"batch_{args.algorithm}_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    seeds = range(args.start_seed, args.start_seed + args.num_seeds)
    for seed in seeds:
        cmd = [
            sys.executable, script,
            "--cfg", args.cfg,
            "--rou", args.rou,
            "--net", args.net,
            "--output-dir", str(output_dir),
            "--random-scenario",
            "--seed", str(seed),
            "--flow-mode", args.flow_mode,
            "--speed-min", str(args.speed_min),
            "--speed-max", str(args.speed_max),
            "--flow-min", str(args.flow_min),
            "--flow-max", str(args.flow_max),
            "--static-min", str(args.static_min),
            "--static-max", str(args.static_max),
            "--start-x", str(args.start_x),
            "--start-y", str(args.start_y),
            "--goal-x", str(args.goal_x),
            "--goal-y", str(args.goal_y),
            "--max-time", str(args.max_time),
        ]
        if args.sumo_gui:
            cmd.append("--sumo-gui")
        print(f"\n=== Running {args.algorithm} seed {seed} ===")
        subprocess.run(cmd, check=True)

        metrics_path = output_dir / f"seed_{seed}" / METRICS_BY_ALGORITHM[args.algorithm]
        metrics = load_metrics(metrics_path)
        metrics["seed"] = seed
        metrics["algorithm"] = args.algorithm
        rows.append(metrics)

    per_run_csv = output_dir / "batch_per_seed_metrics.csv"
    summary_json = output_dir / "batch_summary.json"
    write_rows_csv(per_run_csv, rows)
    summary = numeric_summary(rows)
    summary["algorithm"] = args.algorithm
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_combined_plot:
        combined_plot = output_dir / "batch_all_seed_paths.png"
        plot_all_seed_paths(combined_plot, output_dir, rows, args)
        print(f"Combined seed paths plot: {combined_plot}")

    print(f"\nPer-seed metrics: {per_run_csv}")
    print(f"Summary metrics: {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
