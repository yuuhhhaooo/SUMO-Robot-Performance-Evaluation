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
Their LOCAL parameters therefore stay frozen; --global-only nevertheless
tunes the global-RRT sampling parameters of the stack ABOVE any
registered planner (published / learning-based included), e.g.:

    python sim/tune.py --algorithm sarl_upstream --tune-globals rrt \
        --global-only --episodes-mode tasks

Per-global-condition outputs carry a tag: configs/<algo>__g-<gp>.json.

Note on local rrt: its tree search is fallback-dominated in crowded
corridors (parameters largely unexercised), so its tuning run doubles as
a SENTINEL -- near-identical objective values across trials are the
expected outcome and are archived as evidence of that diagnosis.
(That diagnosis predates the always-empty plan() defect recorded in
docs/code_audit.md; the current tree carries the fix -- re-verify with
one manual rrt run before leaning on the sentinel interpretation.)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

# Tunable = whatever configs/tuning_spaces.json actually defines a space for.
# Keeping a second hand-maintained list here let the two drift: the file gained
# an 'orca_heuristic' space (added when the published RVO2 solver took over the
# 'orca' id) that --algorithm then refused to accept.
def _tunable_algorithms():
    spaces_path = (Path(__file__).resolve().parent.parent / "configs"
                   / "tuning_spaces.json")
    try:
        return sorted(json.loads(spaces_path.read_text()))
    except Exception as e:                  # no silent fallback: it drifts
        sys.exit(f"cannot read {spaces_path}: {e} -- refusing to fall "
                 f"back to a hand-maintained algorithm list (the comment "
                 f"above records how such a list drifted before)")


TUNABLE = _tunable_algorithms()


RRT_JOINT_SPACE = [
    {"name": "step_m", "low": 4.0, "high": 14.0},
    {"name": "goal_bias", "low": 0.05, "high": 0.30},
    {"name": "corridor_sample", "low": 0.4, "high": 0.9},
    {"name": "max_iters", "low": 15000, "high": 60000, "int": True},
]


EP_STATS = {"success": 0, "planner_fail": 0, "infra_fail": 0,
            "timeout": 0}
_EP_LOCK = threading.Lock()


def _bump(*keys):
    with _EP_LOCK:
        for k in keys:
            EP_STATS[k] += 1


def run_episode(algo, params_file, mp, seed, max_time,
                reactive="sfm", gp="fixed", task=None, task_file=None,
                grrt_file=None, mode="mixed",
                jps_model="collision_free_speed", timeout_s=None):
    out = Path(tempfile.mkdtemp())
    cmd = [sys.executable, str(ROOT / "benchmark_runner.py"),
           "--map", mp, "--mode", mode, "--algorithm", algo,
           "--seed", str(seed), "--max-time", str(max_time),
           "--out-root", str(out), "--params-file", str(params_file),
           "--reactive-peds", reactive]
    if reactive == "jupedsim":
        cmd += ["--jps-model", jps_model]
    if gp != "fixed":
        cmd += ["--global-planner", gp]
    if task:
        cmd += ["--task", task, "--task-file", str(task_file)]
    if grrt_file:
        cmd += ["--global-rrt-params", str(grrt_file)]
    _to = timeout_s if timeout_s else 4 * max_time + 600
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        _stdout, _stderr = proc.communicate(timeout=_to)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":       # kill the whole tree (SUMO too)
            subprocess.run(["taskkill", "/F", "/T", "/PID",
                            str(proc.pid)], capture_output=True)
        else:
            proc.kill()
        proc.communicate()
        _bump("timeout", "infra_fail")
        print(f"[episode timeout] {algo} {mp} seed={seed} gp={gp}: "
              f"killed after {_to:.0f}s wall (raise --episode-timeout "
              f"if legitimate runs need longer, e.g. when running "
              f"several studies in parallel)", file=sys.stderr)
        shutil.rmtree(out, ignore_errors=True)
        return 0.0
    import types
    res = types.SimpleNamespace(returncode=proc.returncode,
                                stdout=_stdout, stderr=_stderr)
    if res.returncode != 0:
        _tail = (res.stderr or "").strip().splitlines()
        print(f"[episode failed] {algo} {mp} seed={seed} gp={gp}: "
              + (_tail[-1] if _tail else f"exit code {res.returncode}"),
              file=sys.stderr)
        _bump("infra_fail")
        shutil.rmtree(out, ignore_errors=True)
        return 0.0
    try:
        line = next(l for l in reversed(res.stdout.strip().splitlines())
                    if l.lstrip().startswith("{"))
        m = json.loads(line)
    except Exception:
        _bump("infra_fail")
        shutil.rmtree(out, ignore_errors=True)
        return 0.0
    shutil.rmtree(out, ignore_errors=True)
    if not m.get("success"):
        _bump("planner_fail")
        return 0.0
    _bump("success")
    t_norm = min(float(m.get("sim_time_s", max_time)) / max_time, 1.0)
    soc = min(float(m.get("ped_personal_space_s_total") or 0.0)
              / 60.0, 1.0)
    return 1.0 - 0.15 * t_norm - 0.05 * soc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", required=True,
                    help=f"local planner id. Without --global-only: one "
                         f"of {TUNABLE}. With --global-only: any "
                         f"registered algorithm; its local parameters "
                         f"stay at their defaults/checkpoint.")
    ap.add_argument("--global-only", action="store_true",
                    help="tune ONLY the global-RRT sampling parameters "
                         "(RRT_JOINT_SPACE) for this local planner; the "
                         "local planner runs frozen at defaults. For "
                         "planners with no local tuning space (published "
                         "/ learning-based). Requires --tune-globals "
                         "rrt. Outputs configs/<algo>__g-rrt.json as an "
                         "empty {} placeholder (so the batch's sibling "
                         "lookup finds it) plus the tuned "
                         "<algo>__g-rrt.globalrrt.json.")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--maps", nargs="+",
                    default=["map1_straight", "map3_grid"])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=json.loads(
                        (REPO / "configs" / "seeds.json").read_text()
                    )["tuning_seeds"])
    ap.add_argument("--max-time", type=float, default=600.0)
    ap.add_argument("--episode-timeout", type=float, default=None,
                    help="wall-clock kill limit per episode in seconds "
                         "(default: 4*max_time+600). Raise it when "
                         "running several studies in parallel -- a too-"
                         "tight limit silently scores slow-but-valid "
                         "parameter candidates 0 and BIASES the search")
    ap.add_argument("--interleave-maps", action="store_true",
                    help="round-robin the tuning episodes across maps "
                         "instead of map-by-map, so the MedianPruner's "
                         "early kills are not dominated by the first "
                         "map's performance (docs/code_audit.md 2.27). "
                         "OFF by default: enabling changes the episode "
                         "order and therefore the search trajectory -- "
                         "a protocol decision.")
    ap.add_argument("--episode-jobs", type=int, default=1,
                    help="run each trial's episodes in parallel with "
                         "this many workers. 1 (default) = the original "
                         "serial loop with per-episode pruning. >1 "
                         "disables intermediate pruning (single final "
                         "report) to keep results deterministic -- a "
                         "protocol decision; also raise "
                         "--episode-timeout accordingly.")
    ap.add_argument("--max-infra-frac", type=float, default=0.5,
                    help="refuse to freeze parameters when more than "
                         "this fraction of episodes were INFRASTRUCTURE "
                         "failures (timeouts/crashes/missing deps), "
                         "since those score 0 and bias the search. "
                         "1.0 disables the guard.")
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
    ap.add_argument("--reactive-peds",
                    choices=["off", "sfm", "jupedsim", "pysf"],
                    default="sfm",
                    help="tune under the SAME pedestrian model the "
                         "evaluation will use (default: sfm). jupedsim "
                         "and pysf need their extra packages; a missing "
                         "dependency makes every episode score 0 -- the "
                         "all-zero guard below then refuses to freeze "
                         "parameters")
    ap.add_argument("--jps-model", default="collision_free_speed",
                    choices=["collision_free_speed", "social_force",
                             "anticipation_velocity"],
                    help="JuPedSim operational model, forwarded to the "
                         "runner (only used with --reactive-peds "
                         "jupedsim; match the evaluation's choice)")
    ap.add_argument("--check", action="store_true",
                    help="audit which space keys match planner attributes, "
                         "then exit")
    args = ap.parse_args()

    if args.global_only:
        try:
            from benchmark_adapters import ALGORITHMS
        except ImportError as e:
            sys.exit(f"--global-only needs the algorithm registry from "
                     f"benchmark_adapters, which failed to import: {e}")
        if args.algorithm not in ALGORITHMS:
            sys.exit(f"--global-only: unknown algorithm "
                     f"'{args.algorithm}'; registered: {ALGORITHMS}")
        if args.tune_globals != ["rrt"]:
            sys.exit("--global-only tunes the global-RRT sampling "
                     "parameters, so it requires --tune-globals rrt "
                     "(the other globals are parameter-free).")
    elif args.algorithm not in TUNABLE:
        sys.exit(f"argument --algorithm: invalid choice: "
                 f"'{args.algorithm}' (choose from {TUNABLE}, or use "
                 f"--global-only for planners without a local space)")

    if args.episodes_mode == "maps" and args.tune_globals != ["fixed"]:
        print("WARNING: --episodes-mode maps is the legacy default; "
              "the 12-combination protocol requires --episodes-mode "
              "tasks (route-style contrast between global planners "
              "only exists on task pairs).", file=sys.stderr)

    spaces = json.loads(
        (REPO / "configs" / "tuning_spaces.json").read_text())
    space = spaces.get(args.algorithm)
    if args.global_only:
        space = []            # local planner frozen at defaults
    elif not space:
        sys.exit(f"no tuning space for {args.algorithm}")

    if args.check:
        if args.global_only:
            print(f"{args.algorithm}: --global-only has no local space "
                  f"to audit; the global-RRT dims (RRT_JOINT_SPACE) are "
                  f"fixed-range and consumed by the runner directly")
            return
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
    if args.interleave_maps:
        from itertools import zip_longest
        _by_map = {}
        for _e in base:
            _by_map.setdefault(_e[0], []).append(_e)
        base = [_e for _grp in zip_longest(*_by_map.values())
                for _e in _grp if _e is not None]
    episodes = [(mp, sd, tid, tf,
                 args.tune_globals[i % len(args.tune_globals)])
                for i, (mp, sd, tid, tf) in enumerate(base)]

    joint = bool(args.joint_global_rrt and "rrt" in args.tune_globals)
    if args.global_only:
        joint = True          # grrt dims are the whole search space

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
        if args.episode_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(args.episode_jobs) as ex:
                scores = list(ex.map(
                    lambda e: run_episode(
                        args.algorithm, pf, e[0], e[1],
                        args.max_time, args.reactive_peds, e[4],
                        e[2], e[3], gf if e[4] == "rrt" else None,
                        args.tune_mode, jps_model=args.jps_model,
                        timeout_s=args.episode_timeout),
                    episodes))
            trial.report(sum(scores) / len(scores), len(episodes) - 1)
            return sum(scores) / len(scores)
        for i, (mp, sd, tid, tf, gp) in enumerate(episodes):
            scores.append(run_episode(args.algorithm, pf, mp, sd,
                                      args.max_time, args.reactive_peds,
                                      gp, tid, tf,
                                      gf if gp == "rrt" else None,
                                      args.tune_mode,
                                      jps_model=args.jps_model,
                                      timeout_s=args.episode_timeout))
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

    print(f"episode summary: {EP_STATS['success']} scored, "
          f"{EP_STATS['planner_fail']} planner failures, "
          f"{EP_STATS['infra_fail']} INFRASTRUCTURE failures "
          f"(of which {EP_STATS['timeout']} timeouts)")
    if EP_STATS["infra_fail"]:
        print("WARNING: infrastructure failures score 0 exactly like "
              "planner failures and BIAS the search away from affected "
              "candidates -- investigate the [episode failed]/[episode "
              "timeout] lines before trusting this study.",
              file=sys.stderr)

    _total = (EP_STATS["success"] + EP_STATS["planner_fail"]
              + EP_STATS["infra_fail"])
    if (_total and args.max_infra_frac < 1.0
            and EP_STATS["infra_fail"] / _total > args.max_infra_frac):
        sys.exit(f"{args.algorithm}: {EP_STATS['infra_fail']}/{_total} "
                 f"episodes were infrastructure failures "
                 f"(> --max-infra-frac {args.max_infra_frac}). The "
                 f"search was shaped by a broken environment, not the "
                 f"objective. REFUSING to freeze parameters.")

    if study.best_value <= 0.0:
        sys.exit(f"{args.algorithm}: best score is 0.0 over "
                 f"{args.n_trials} trials -- every episode failed "
                 f"(broken environment, missing dependency, or an "
                 f"always-failing planner). REFUSING to freeze "
                 f"parameters; nothing written to configs/. See the "
                 f"[episode failed]/[episode timeout] lines above.")

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
    meta = {
        "algorithm": args.algorithm, "global_only": args.global_only,
        "tune_globals": args.tune_globals, "tune_mode": args.tune_mode,
        "reactive_peds": args.reactive_peds,
        "jps_model": (args.jps_model
                      if args.reactive_peds == "jupedsim" else None),
        "episodes_mode": args.episodes_mode,
        "maps": (args.tuning_maps if args.episodes_mode == "tasks"
                 else args.maps),
        "seeds": args.seeds, "n_trials": args.n_trials,
        "max_time": args.max_time,
        "interleave_maps": args.interleave_maps,
        "episode_jobs": args.episode_jobs,
        "best_score": study.best_value,
        "episode_stats": dict(EP_STATS),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out.with_name(out.stem + ".meta.json")).write_text(
        json.dumps(meta, indent=2))
    print(f"condition fingerprint -> {out.stem}.meta.json")
    study.trials_dataframe().to_csv(
        hist_dir / f"{args.algorithm}{tag}_trials.csv", index=False)
    print(f"tuning condition: globals={args.tune_globals}, "
          f"mode={args.tune_mode}, reactive={args.reactive_peds}"
          + (f", jps={args.jps_model}"
             if args.reactive_peds == "jupedsim" else ""))
    print(f"{args.algorithm}: best score "
          f"{study.best_value:.3f} over {args.n_trials} trials "
          f"-> {out}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
