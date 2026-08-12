#!/usr/bin/env python3
"""Equal-budget automated parameter tuning (supervisor protocol).

Every classical planner gets the SAME Optuna budget (--n-trials), the same
objective, and the same held-out tuning episodes (tuning maps x tuning
seeds 1000-1002 from configs/seeds.json; evaluation maps/seeds are never
touched). Best parameters are frozen to configs/<algo>.json and the full
trial history is archived -- both are part of the published protocol.

Objective per episode (hierarchical, weights predeclared per the
12-combination protocol doc): failures dominate --
    score = 0                                   on any failure
    score = 1 - 0.15*T_norm - 0.05*Social_norm  on success
where T_norm = sim_time/max_time and Social_norm =
min(ped_personal_space_s_total / 60 s, 1). Jerk is not logged by the
runner and is omitted (disclosed). Trial score = mean over episodes;
MedianPruner kills bad trials early.

    python sim/tune.py --algorithm dwa --n-trials 50
    python sim/tune.py --algorithm dwa --check          # space<->attr audit
Learning-based planners (sarl/cadrl/lstm_rl) are excluded by design:
their parameters live in trained weights (retrain-or-drop decision).

Note on local rrt: its tree search is fallback-dominated in crowded
corridors (parameters largely unexercised), so its tuning run doubles as
a SENTINEL -- near-identical objective values across trials are the
expected outcome and are archived as evidence of that diagnosis.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

# Tunable = whatever configs/tuning_spaces.json actually defines a space for.
# Keeping a second hand-maintained list here let the two drift: the file gained
# an 'orca_heuristic' space (added when the published RVO2 solver took over the
# 'orca' id) that --algorithm then refused to accept.
def _tunable_algorithms():
    try:
        import json as _json
        spaces = _json.loads(
            (Path(__file__).resolve().parent.parent / "configs"
             / "tuning_spaces.json").read_text())
        return sorted(spaces)
    except Exception:                       # fall back to the historical list
        return ["astar", "dijkstra", "dwa", "mpc", "orca", "rrt", "teb"]


TUNABLE = _tunable_algorithms()


RRT_JOINT_SPACE = [
    {"name": "step_m", "low": 4.0, "high": 14.0},
    {"name": "goal_bias", "low": 0.05, "high": 0.30},
    {"name": "corridor_sample", "low": 0.4, "high": 0.9},
    {"name": "max_iters", "low": 15000, "high": 60000, "int": True},
]


def run_episode(algo, params_file, mp, seed, max_time,
                reactive="sfm", gp="fixed", task=None, task_file=None,
                grrt_file=None, mode="mixed"):
    out = Path(tempfile.mkdtemp())
    cmd = [sys.executable, str(ROOT / "benchmark_runner.py"),
           "--map", mp, "--mode", mode, "--algorithm", algo,
           "--seed", str(seed), "--max-time", str(max_time),
           "--out-root", str(out), "--params-file", str(params_file),
           "--reactive-peds", reactive]
    if gp != "fixed":
        cmd += ["--global-planner", gp]
    if task:
        cmd += ["--task", task, "--task-file", str(task_file)]
    if grrt_file:
        cmd += ["--global-rrt-params", str(grrt_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return 0.0
    try:
        line = res.stdout.strip().splitlines()[-1]
        m = json.loads(line)
    except Exception:
        return 0.0
    if not m.get("success"):
        return 0.0
    t_norm = min(float(m.get("sim_time_s", max_time)) / max_time, 1.0)
    soc = min(float(m.get("ped_personal_space_s_total") or 0.0)
              / 60.0, 1.0)
    return 1.0 - 0.15 * t_norm - 0.05 * soc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", required=True, choices=TUNABLE)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--maps", nargs="+",
                    default=["map1_straight", "map3_grid"])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=json.loads(
                        (REPO / "configs" / "seeds.json").read_text()
                    )["tuning_seeds"])
    ap.add_argument("--max-time", type=float, default=600.0)
    ap.add_argument("--tune-mode", default="mixed",
                    choices=["same", "opposite", "mixed", "static", "all"],
                    help="pedestrian mode for the tuning episodes. RULE: "
                         "match the evaluation mode (tune under the same "
                         "environment the evaluation will use).")
    ap.add_argument("--episodes-mode", choices=["maps", "tasks"],
                    default="maps",
                    help="'maps' = default routes on the tuning maps "
                         "(legacy); 'tasks' = held-out TUNING tasks "
                         "(configs/tuning_tasks_<map>.json) -- required "
                         "for route-style contrast, per the "
                         "12-combination protocol")
    ap.add_argument("--tuning-maps", nargs="+",
                    default=["map3_grid", "map4_london"],
                    help="maps for --episodes-mode tasks")
    ap.add_argument("--joint-global-rrt", action="store_true",
                    help="RRT-stack joint tuning: add the global RRT "
                         "sampling parameters to the search space "
                         "(only meaningful with --tune-globals rrt); "
                         "outputs a sibling <algo>__g-rrt.globalrrt.json")
    ap.add_argument("--tune-globals", nargs="+", default=["fixed"],
                    choices=["fixed", "dijkstra", "astar", "rrt"],
                    help="global-planner condition(s) for the tuning "
                         "episodes. One value = tune under that condition "
                         "(output configs/<algo>__g-<gp>.json when not "
                         "'fixed'); several values = episodes cycle "
                         "through them (route-style-robust single set).")
    ap.add_argument("--reactive-peds", choices=["off", "sfm"],
                    default="sfm",
                    help="tune under the SAME pedestrian model the "
                         "evaluation will use (default: sfm)")
    ap.add_argument("--check", action="store_true",
                    help="audit which space keys match planner attributes, "
                         "then exit")
    args = ap.parse_args()

    spaces = json.loads(
        (REPO / "configs" / "tuning_spaces.json").read_text())
    space = spaces.get(args.algorithm)
    if not space:
        sys.exit(f"no tuning space for {args.algorithm}")

    if args.check:
        from benchmark_adapters import build_planner, apply_params
        from benchmark_adapters import PlannerConfig
        cfg = PlannerConfig()
        pl = build_planner(args.algorithm, cfg, 0,
                           ROOT / "planners" / "models")
        unm = apply_params(pl, {d["name"]: d["low"] for d in space})
        print(f"{args.algorithm}: {len(space) - len(unm)}/{len(space)} "
              f"space keys match planner attributes; unmatched: {unm}")
        return

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.episodes_mode == "tasks":
        base = []
        for mp in args.tuning_maps:
            tf = REPO / "configs" / f"tuning_tasks_{mp}.json"
            ids = [t["id"] for t in
                   json.loads(tf.read_text())["tasks"]]
            for j, tid in enumerate(ids):
                base.append((mp, args.seeds[j % len(args.seeds)],
                             tid, tf))
    else:
        base = [(mp, sd, None, None)
                for mp in args.maps for sd in args.seeds]
    episodes = [(mp, sd, tid, tf,
                 args.tune_globals[i % len(args.tune_globals)])
                for i, (mp, sd, tid, tf) in enumerate(base)]

    joint = bool(args.joint_global_rrt and "rrt" in args.tune_globals)

    def objective(trial):
        params = {}
        grrt = {}
        for d in (RRT_JOINT_SPACE if joint else []):
            key = f"grrt_{d['name']}"
            if d.get("int"):
                grrt[d["name"]] = trial.suggest_int(
                    key, int(d["low"]), int(d["high"]))
            else:
                grrt[d["name"]] = trial.suggest_float(
                    key, d["low"], d["high"])
        for d in space:
            if d.get("int"):
                params[d["name"]] = trial.suggest_int(
                    d["name"], int(d["low"]), int(d["high"]))
            else:
                params[d["name"]] = trial.suggest_float(
                    d["name"], d["low"], d["high"],
                    log=bool(d.get("log")))
        pf = Path(tempfile.mkdtemp()) / "params.json"
        pf.write_text(json.dumps(params))
        gf = None
        if grrt:
            gf = pf.parent / "grrt.json"
            gf.write_text(json.dumps(grrt))
        scores = []
        for i, (mp, sd, tid, tf, gp) in enumerate(episodes):
            scores.append(run_episode(args.algorithm, pf, mp, sd,
                                      args.max_time, args.reactive_peds,
                                      gp, tid, tf, gf, args.tune_mode))
            trial.report(sum(scores) / len(scores), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return sum(scores) / len(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=args.n_trials,
                   show_progress_bar=False)

    best_all = study.best_params
    best = {k: v for k, v in best_all.items()
            if not k.startswith("grrt_")}
    best_grrt = {k[5:]: v for k, v in best_all.items()
                 if k.startswith("grrt_")}
    tag = ""
    if len(args.tune_globals) == 1 and args.tune_globals[0] != "fixed":
        tag = f"__g-{args.tune_globals[0]}"
    out = REPO / "configs" / f"{args.algorithm}{tag}.json"
    out.write_text(json.dumps(best, indent=2))
    if best_grrt:
        gout = out.with_name(out.stem + ".globalrrt.json")
        gout.write_text(json.dumps(best_grrt, indent=2))
        print(f"joint global-RRT params -> {gout}")
    hist_dir = REPO / "configs" / "tuning_history"
    hist_dir.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(
        hist_dir / f"{args.algorithm}{tag}_trials.csv", index=False)
    print(f"tuning condition: globals={args.tune_globals}, "
          f"mode={args.tune_mode}, reactive={args.reactive_peds}")
    print(f"{args.algorithm}: best score "
          f"{study.best_value:.3f} over {args.n_trials} trials "
          f"-> {out}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
