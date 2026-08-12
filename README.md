# Sidewalk-Robot Social-Navigation Benchmark (SUMO)

A benchmark for comparing robot navigation algorithms on **sidewalks with
signalised crossings**, on four controlled synthetic maps and one real urban
map (UCL / Bloomsbury, imported from OpenStreetMap). Ten algorithms are
evaluated as the **local-planning component** of a fixed global–local stack:
DWA, A\*, Dijkstra, RRT (the last three as receding-horizon local variants),
ORCA, MPC, TEB, SARL, CADRL, LSTM-RL.

Repository layout:

```
sim/        simulation code: runner, batch driver, map builders, demand,
            signal gate, planner adapters, planners/ (+ pretrained models)
analysis/   plotting / analysis (kept separate from simulation code)
maps/       map1_straight … map4_london (synthetic), map5_ucl (OSM import)
configs/    fixed seed lists (evaluation + reserved tuning seeds)
examples/   minimal hand-written OSM extract for testing the import pipeline
docs/       usage reference and map/route preview figures
results/    generated outputs (git-ignored; layout documented inside)
```

## Install

```bash
pip install -r requirements.txt          # pinned; includes SUMO via pip
# or: docker build -t swbench . 
```

A standalone SUMO **1.27.1** installation (with `SUMO_HOME` set) can replace
the pip SUMO. All commands below are run from the repository root.

## Smoke test (bit-reproducible reference run)

```bash
python sim/benchmark_runner.py --map map2_crossing --mode mixed \
    --algorithm dwa --seed 1 --max-time 200
```

Expected final JSON: `"termination_reason": "collision"`,
`"path_length_m": 138.5`. This exact number is the cross-machine
regression check.

> **Re-baselined 2026-08-12** (was `102.59`). DWA was previously benchmarked
> under the `DWAConfig` module defaults rather than the shared `PlannerConfig`
> envelope, so it ran at `max_speed` 0.95 m/s while every other planner ran at
> 1.00 m/s — a systematic handicap on the algorithm that is also the reference
> level in the statistical models. The full envelope is now propagated; six
> fields both changed value and are read by `dwa_control`:
>
> | field | was (DWAConfig) | now (PlannerConfig) |
> |---|---|---|
> | `max_speed` | 0.95 m/s | 1.00 m/s |
> | `max_accel` | 0.80 m/s² | 0.50 m/s² |
> | `max_yaw_rate` | 80 °/s | 120 °/s |
> | `social_distance` | 0.80 m | 0.85 m |
> | `sensor_range` | 12.0 m | 11.0 m |
> | `goal_tolerance` | 0.25 m | 0.35 m |
>
> (`safe_distance` also moved 0.20→0.42 but is never read by DWA.)
> Net effect on the reference run: `avg_speed_mps` 0.933→0.986. See
> `docs/code_audit.md` §0. Results produced before this date are not
> comparable for DWA.

## Reproduce one figure (end to end)

```bash
python sim/benchmark_batch.py --maps map2_crossing --modes mixed \
    --algorithms dwa astar orca sarl --seeds 1 2 3 --max-time 420 \
    --out-root results
python analysis/benchmark_plots.py --results results
```

With three or more seeds per cell the plot suite REPLACES the per-
algorithm multi-seed overlay with
`envelope_<algo>_<map>_<mode>.png`: the median path with a 10-90 %
quantile envelope in route-aligned coordinates (supervisor item 7 --
shows WHERE trajectories diverge instead of overplotting runs; metric
computation never uses simplified geometry, only drawn lines are
Douglas-Peucker-reduced). Produces
`results/plots/metrics_map2_crossing_mixed.png` (per-algorithm
metric bars) and `results/plots/overlay_seed1_map2_crossing_mixed.png`
(strip-view trajectories of all algorithms against the same seed-1 crowd).

## Full evaluation protocol

Fixed seed list: `configs/seeds.json` (evaluation seeds 1–50; seeds
1000–1002 are reserved for parameter tuning and never used in evaluation).

```bash
# PowerShell:  --seeds $(Get-Content configs/seeds_eval.txt)
# bash:        --seeds $(cat configs/seeds_eval.txt)
python sim/benchmark_batch.py \
    --maps map1_straight map2_crossing map3_grid map4_london map5_ucl \
    --routes default path2 \
    --global-planners dijkstra astar rrt \
    --modes same opposite mixed static all \
    --algorithms dwa orca mpc teb sarl cadrl lstm_rl \
    --seeds $(cat configs/seeds_eval.txt) \
    --max-time 3000 --veh-period 5 --reactive-peds sfm
python analysis/benchmark_plots.py --results results
```

`--skip-existing` resumes an interrupted sweep without recomputation.
Per-run raw logs (`robot_trace.csv`, `robot_metrics.json`,
`scenario.json`, including every sampled demand parameter) are written for
every cell; see `results/README.md` for the layout.

## Task sampling (Option B protocol, supervisor decision 2026-08-09)

```bash
python sim/sample_tasks.py --map map5_ucl --n-tasks 10 --bins 3 \
    --length-min 300 --length-max 1100
```

Ten start-goal tasks per map, stratified into path-length bins with equal
counts; unreachable pairs and same-edge pairs are screened out. The
sampling script, its seed (20260810) and the bin edges are committed next
to each task list (`configs/tasks_<map>.json`) -- the lists are
reproducible random draws, not hand-picked sets. Per-task geometry
features (path length, number of turns, minimum sidewalk width,
signalised-junction count) are logged and entered as standardized fixed
effects in the mixed models, so the analysis reports WHICH topology
properties drive ranking changes; pedestrian flow along the route is a
per-run covariate (density is sampled per crowd seed and recorded in every
run's metrics). Evaluation crosses tasks with crowd seeds (10 x 10 per
cell; tasks and seeds are never traded away for other factors -- trim
pedestrian modes or density levels instead). Every run records its task
ID. Protocol definition of a shared crowd seed: it fixes demand, spawn
times, walking speeds and appearance -- NOT identical realised
trajectories once reactive pedestrians interact with the robot. Tasks
enter the models as a crossed random effect alongside crowd seed, which
also blocks the algorithm comparison (same task, same seeds for every
algorithm).

Full design size: `python analysis/factor_table.py --maps map1_straight
map2_crossing map3_grid map4_london map5_ucl --tasks 10 --seeds 10
--globals 3 --locals 7 --modes 1` -> 10,500 runs.

## Equal-budget parameter tuning (supervisor protocol)

```bash
python sim/tune.py --algorithm dwa --check      # space <-> attribute audit
python sim/tune.py --algorithm dwa --n-trials 50
```

Two tuning designs are implemented (supervisor ruling pending; both feed
the SAME Option-B evaluation, only `--params-file` differs):

*Version F (default, global fixed during tuning).* One tuned set per
local planner (7 studies); evaluation reuses it across all global levels:
`--params-file configs/{algo}.json`.

*Version C (per-combination).* Each local planner is tuned separately
under each global condition via `--tune-globals <gp>` (outputs
`configs/<algo>__g-<gp>.json`, 7x3 = 21 studies; the g-astar and
g-dijkstra conditions are expected to coincide since their routes are
identical -- a built-in validity check). Evaluation resolves the file per
combination: `--params-file "configs/{algo}__g-{gp}.json"`.

*Doc protocol (12-combination, per the working methodology document).*
Local axis = {dwa, orca, mpc, teb}; A*/Dijkstra stacks tune the local
parameters only (their globals are parameter-free); RRT stacks JOINTLY
tune the local parameters plus four global-RRT sampling parameters
(step_m, goal_bias, corridor_sample, max_iters), producing a sibling
`configs/<algo>__g-rrt.globalrrt.json` that the batch attaches
automatically on rrt cells. Tuning episodes use HELD-OUT TUNING TASKS
(`configs/tuning_tasks_map{3_grid,4_london}.json`, sampling seed
20260811, screened to stay clear of the evaluation tasks) because route-
style contrast between the global planners exists on task pairs but not
on the default corridor routes; map5 remains fully unseen. Objective is
hierarchical (failures dominate; then time 0.15, personal-space social
cost 0.05; jerk not logged, disclosed). Commands:

```bash
# A*/Dijkstra columns (local-only tuning, 8 studies)
for a in dwa orca mpc teb; do for g in dijkstra astar; do \
  python sim/tune.py --algorithm $a --tune-globals $g \
    --episodes-mode tasks --n-trials 50 --max-time 600; done; done
# RRT column (joint stack tuning, 4 studies)
for a in dwa orca mpc teb; do \
  python sim/tune.py --algorithm $a --tune-globals rrt \
    --joint-global-rrt --episodes-mode tasks --n-trials 50 \
    --max-time 600; done
# evaluation: --params-file "configs/{algo}__g-{gp}.json" (rrt cells
# pick up the .globalrrt.json sibling automatically)
```

`--tune-globals dijkstra rrt` (several values) gives the middle option:
episodes cycle through route styles, producing ONE route-robust set per
planner. In every version the global planners' own parameters remain
fixed protocol constants (`GLOBAL_RRT_PARAMS`).

Every classical planner (dwa, astar, dijkstra, rrt, orca, mpc, teb) gets an
identical Optuna (TPE + median pruner) budget on held-out tuning episodes
(maps 1+3 x tuning seeds 1000-1002 from `configs/seeds.json`; evaluation
maps/seeds are never used). Objective: 0.8*success + 0.2*(1 - normalised
time). Search spaces (`configs/tuning_spaces.json`) use the planners' real
attribute names (audited: 7/7 fully matched) with ranges centred on the
shipped defaults. Best parameters are frozen to `configs/<algo>.json`
(consumed via `benchmark_runner --params-file`, recorded in metrics) and
the full trial history is archived under `configs/tuning_history/`.
Three tuning-condition rulings are supported by the same machinery
(awaiting supervisor decision; default = first): (1) one set per local
planner tuned under the fixed route (`--tune-globals fixed`, output
`configs/<algo>.json`); (2) one route-style-robust set from mixed
episodes (`--tune-globals dijkstra rrt`, same output name); (3) per-
combination sets (`--tune-globals rrt` -> `configs/<algo>__g-rrt.json`,
consumed in batch via `--params-file "configs/{algo}__g-{gp}.json"`).
Injection is verified behaviourally (extreme parameters change outcomes
for dwa/orca/mpc/astar/dijkstra; teb shifts are small on the reference
scenario). Note: the receding-horizon *local* RRT variant is fallback-dominated in
crowded strips (its tree search rarely succeeds, so it tracks the
centreline regardless of parameters). It is still tuned with the same
budget as everyone else -- its near-flat trial history serves as the
archived sentinel evidence for that diagnosis -- and RRT's primary role
in the protocol remains the *global* planning factor. Learning-based planners are excluded by design (parameters live in
trained weights: retrain-or-drop decision).

## Statistical analysis (supervisor protocol)

```bash
python analysis/stats_models.py --results results
```

Writes `results/stats/`: binomial GLMM for success (odds ratios + 95% CIs
against a reference algorithm, variance components for seed / map / task),
linear mixed models for time and path length on successful runs, a failure
taxonomy table + figure (goal / collision / max_time / stalled /
global_plan_failed), and a seed-bootstrap ranking-stability table + figure
(rank 95% intervals and P(top-1) -- the quantitative basis of the
ranking-instability claim). Pipeline validated by recovering injected
effects from synthetic data.

## Maps

* `map1`–`map4` are built fully natively (netconvert, pedestrian phases in
  the traffic-light programs): `python sim/build_maps.py` rebuilds them
  deterministically. `map4_london` defines two named routes
  (`path1`, `path2`).
* `map5_ucl` was imported from an OpenStreetMap export of the UCL /
  Bloomsbury area: `python sim/osm_import.py --osm map.osm --name map5_ucl
  --lefthand --ped-period 0.5`. Its default route
  (Tavistock Square junction → Upper Woburn Place → full Gower Street →
  UCL main gate, 1366 m, ~6 signalised junctions) is stored in
  `maps/map5_ucl/map_spec.json`.
* Route preview figures: `docs/previews/`.

## Design notes and known limitations

* **Evaluation unit (combination design, supervisor decision 2026-08-07).**
  Every experiment runs a complete global-local stack. The global-planning
  factor (`--global-planner`) has levels **dijkstra / astar / rrt**, all
  planning on the same walkable geometry (sidewalks, crossings, walking
  areas); the local planner under test then drives inside the 2 m corridor
  of the produced route. Validated properties: A\* and Dijkstra return
  point-identical optimal routes on this sparse graph (reported as a
  result, not assumed); RRT is stochastic by design -- its seeded route
  (workspace RRT, corridor-informed sampling, up to three deterministic
  restarts, ~30-130 s planning per run) is part of the experimental unit.
  `fixed` replays a map's stored waypoints verbatim and preserves
  bit-compatibility with legacy runs (map2 reference: 102.59 m).
* **Robot body (`--robot-radius`, `--robot-height`).** The robot is a 2-D disc.
  `--robot-radius` (default 0.25 m, the historical value) drives the collision
  threshold `max(0.42, r_robot + r_ped)`, the JuPedSim agent radius under
  `--robot-in-jps`, and the GUI marker; it is recorded per run as
  `robot_radius_m` / `collision_radius_m`. `--robot-height` is recorded but does
  not enter the 2-D dynamics. A sidewalk delivery robot is closer to 0.30–0.35 m
  radius than the default. Measured effect on map2/mixed/orca/seed 1 with
  `--reactive-peds jupedsim --robot-in-jps`: minimum pedestrian distance rises
  0.89 → 0.91 → 0.94 → 1.55 → 1.75 m for radii 0.20 / 0.25 / 0.35 / 0.50 / 0.70 m.
* **Pedestrian reactivity (`--reactive-peds jupedsim`).** Pedestrians inside the
  same interaction bubble are driven by **JuPedSim 1.4.2** (Jülich Pedestrian
  Simulator) instead of the in-repo SFM, selectable model via `--jps-model`
  (`collision_free_speed` — the one SUMO documents as extensively tested —
  `social_force`, `anticipation_velocity`). Capture/release geometry is
  identical to the `sfm` layer, so the two are levels of one factor rather than
  two different experiments. JuPedSim runs at its own 0.01 s timestep (50
  iterations per 0.5 s SUMO step); measured cost is ~12 s per 6000-step episode
  at 10 agents in the bubble and ~41 s at 30.
  Two API constraints are handled explicitly: the accessible area must be a
  single *connected* polygon (the walkable union is joined with the route
  corridor and the component containing the route is kept — `jps_area_kept_frac`
  records the fraction retained), and every steering target must lie inside it.
  **`--robot-in-jps` is opt-in** and injects the robot as a JuPedSim agent so
  pedestrians see and deflect around it. It is off by default, preserving the
  fairness rule that the robot does all the avoiding. The direct steering stage
  bypasses JuPedSim's strategic and tactical levels but *not* its operational
  one, so JuPedSim also nudges the robot; the residual is reported per run as
  `jps_robot_track_err_{mean,p95,max}_m` (measured 0.013 / 0.020 / 0.025 m —
  negligible, but recorded rather than assumed).
* **Pedestrian reactivity (`--reactive-peds sfm`).** SUMO keeps macroscopic
  demand and long-range walking; inside an interaction bubble around the
  robot (capture 12 m / release 18 m) pedestrians are driven by a Social
  Force Model that includes the robot as a repulsive agent (Helbing-Molnar
  circular specification; parameters follow PySocialForce). Standing-robot
  validation on map2 (oncoming stream, 260 ped/h, identical placement and
  seed): striping-only = 9 pass-throughs (<0.30 m), mean lateral clearance
  0.49 m; SFM layer = 1 pass-through, 1.43 m -- pedestrians deflect and
  re-merge. Reproduce with `python sim/validate_reactive.py --layer off|sfm
  --mode opposite`. Known limits: pedestrians outside the bubble do not
  see controlled ones (asymmetry at the bubble edge); standing (static)
  pedestrians are not captured; standing pedestrians (statics) are never captured (capturing would
  assign them a walking speed) but repel controlled pedestrians by
  construction via the neighbour set; SUMO's striping model does not
  avoid near-stationary pedestrians (measured baseline: 39% of close
  encounters pass through, median clearance 0.65 m), so the SFM layer
  additionally opens a capture bubble around EVERY standing pedestrian
  (2.5 m capture / 4.5 m release): walkers approaching a static are
  briefly SFM-controlled and deflect around them. Measured after the
  fix: 2% pass-through, median clearance 1.36 m. Pedestrian-side
  metrics count only pedestrians that actually came near the robot, so
  static-bubble passers-by do not dilute the robot-imposed-cost means; junction cores (walkingarea aprons) are
  excluded from SFM control -- remote persons mapped onto junction-
  internal lanes can crash SUMO's person state machine, so reactivity
  applies on sidewalk segments and crossings, with striping retained in
  the junction core (stress-tested across all five pedestrian modes). Default is `off` (bit-compatible legacy);
  the main protocol runs with `sfm`.
* Robot–pedestrian collision: centre distance < 0.42 m (3 s spawn grace).
  Failure taxonomy recorded per run: `collision`, `max_time`, `stalled`,
  `goal`.

## Provenance

Planner implementations (`sim/planners/`) are the author's own code and are
used unmodified. Map construction and demand use the official SUMO toolchain
(netconvert, randomTrips). The benchmark harness (map pipeline, signal
gate, runner, batch driver, analysis) was developed with AI assistance
(Anthropic Claude) under the author's direction; see the university's
academic-integrity guidance for the corresponding declaration.

On thesis submission, archive this repository together with the `results/`
directory on Zenodo and cite the DOI.
