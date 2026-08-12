# Sidewalk-Robot Social-Navigation Benchmark (SUMO)

A benchmark for comparing robot navigation algorithms on **sidewalks with
signalised crossings**, on four controlled synthetic maps and one real urban
map (UCL / Bloomsbury, imported from OpenStreetMap). Algorithms are evaluated
as the **local-planning component** of a fixed global–local stack.

**18 local planners are registered**, in two groups. Which group an algorithm
belongs to is a first-class fact about the result, not a footnote:

| | id | backing |
|---|---|---|
| **Published implementations** | `orca` | RVO2, UNC GAMMA (van den Berg et al., ISRR 2009) via Python-RVO2 |
| | `mpc_dompc` | do-mpc 5.1.1 on CasADi 3.7.2 with Ipopt |
| | `teb_upstream` | teb_local_planner 0.9.1 (Rösmann et al.) via pybind11 |
| | `sarl_upstream` `cadrl_upstream` `lstm_rl_upstream` | CrowdNav (Chen et al., ICRA 2019), upstream networks + this repo's checkpoints |
| | `crowdnav_dsrnn` | CrowdNav\_DSRNN (Liu et al., ICRA 2021) + its published checkpoint |
| | `crowdnav_attngraph` | CrowdNav\_Prediction\_AttnGraph (Liu et al., ICRA 2023) + GST predictor (Huang et al., RA-L 2022) |
| **In-repo implementations** | `dwa` `astar` `dijkstra` `rrt` `mpc` `teb` | the author's own code (A\*/Dijkstra/RRT as receding-horizon local variants) |
| | `orca_heuristic` | the author's former `orca` — see the note below |
| | `sarl` `cadrl` `lstm_rl` | the author's CrowdNav adaptations |

`sim/benchmark_adapters.py::PUBLISHED_IMPL` is the authoritative citation list.

> **`orca` changed meaning.** An audit found the previous in-repo ORCA contains
> no velocity-obstacle half-planes, no linear program and no reciprocity
> factor — it is a reciprocal-force heuristic rather than ORCA. `orca` is now
> the published RVO2 solver; the original is preserved and runnable as
> `orca_heuristic` so both can be reported. **Any result labelled `orca` from
> before this change is `orca_heuristic`.**

Repository layout:

```
sim/            runner, batch driver, protocol driver, map builders, demand,
                signal gate, planner adapters, planners/ (+ pretrained models)
sim/third_party/ vendored upstream sources, each with LICENSE, COMMIT and
                PATCHES.md: crowdnav/, crowdnav_dsrnn/, crowdnav_attngraph/,
                pyteb/, plus build_rvo2.py
analysis/       plotting / analysis (kept separate from simulation code)
maps/           map1_straight … map4_london (synthetic), map5_ucl (OSM import)
configs/        seed lists, Option-B task lists, tuning spaces + tuned params
docs/           code_audit.md and map/route preview figures
results/        generated outputs (git-ignored; layout defined by
                sim/run_layout.py, the single source of truth)
```

## Install

```bash
pip install -r requirements.txt          # pinned; includes SUMO and JuPedSim
# or: docker build -t swbench .
```

A standalone SUMO **1.27.1** installation (with `SUMO_HOME` set) can replace
the pip SUMO. All commands below are run from the repository root.

**Three components are not covered by `requirements.txt`.** Skip a step and the
matching algorithm fails at import — loudly, never silently:

```bash
# 1. ORCA. RVO2 is a compiled C++ extension and is NOT on PyPI.
pip install cython cmake
python sim/third_party/build_rvo2.py      # prints the RVO2_PATH to export
#    Windows: the build destination must be a SHORT path (MSBuild's
#    FileTracker fails past MAX_PATH); the script's default already is.

# 2. mpc_dompc
pip install do-mpc==5.1.1 casadi==3.7.2

# 3. --reactive-peds pysf
pip install pysocialforce==1.1.2 socialforce==0.2.3
```

`teb_upstream` uses prebuilt bindings in `sim/third_party/pyteb/prebuilt/`
(CPython 3.12 and 3.13, Windows x64). On any other platform or Python version,
build from `sim/third_party/pyteb/` — see its `PATCHES.md`.

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
> level in the statistical models. The full envelope is now propagated. Seven
> fields changed value; only the first three are read by `dwa_control` and
> therefore move the benchmarked trajectory:
>
> | field | was (DWAConfig) | now (PlannerConfig) | affects runs? |
> |---|---|---|---|
> | `max_speed` | 0.95 m/s | 1.00 m/s | **yes** |
> | `max_accel` | 0.80 m/s² | 0.50 m/s² | **yes** |
> | `max_yaw_rate` | 80 °/s | 120 °/s | **yes** |
> | `safe_distance` | 0.20 m | 0.42 m | no |
> | `social_distance` | 0.80 m | 0.85 m | no |
> | `sensor_range` | 12.0 m | 11.0 m | no |
> | `goal_tolerance` | 0.25 m | 0.35 m | no |
>
> The last four are used only by the planner file's own standalone
> run/metrics path, not by `dwa_control`; they are now consistent with the
> other planners but inert in the benchmark. Note the acceleration limit moved
> **down**: DWA loses an acceleration advantage at the same time as it gains
> speed and yaw headroom. Net effect on the reference run: `avg_speed_mps`
> 0.933→0.986, and on an isolated 60 m leg the goal is reached in 165 instead
> of 183 steps. See `docs/code_audit.md` §0. Results produced before this date
> are not comparable for DWA.

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

Use **`sim/run_protocol.py`**. It is the only entry point that runs the
complete factorial *and then proves the tree is complete* — `benchmark_batch.py`
crosses every factor except `--reactive-peds`, and never checks afterwards that
the runs it was asked for actually exist. On a sweep this size a cell can vanish
because a runner crashed or a resume skipped it, and silence looks exactly like
success.

```bash
# 1. What will this cost? Enumerate the design; run nothing.
python sim/run_protocol.py --dry-run --preset paired --jobs 24

# 2. Run it. --skip-existing is ON by default, so this is resumable.
python sim/run_protocol.py --preset paired --jobs 24 --out-root results

# 3. Prove every cell exists. Exits NON-ZERO and names what is missing.
python sim/run_protocol.py --verify-only --preset paired --out-root results

# 4. Analyse.
python analysis/benchmark_plots.py --results results/peds_sfm
python analysis/stats_models.py   --results results/peds_sfm
```

`--preset` names the algorithm set, so the choice is a word rather than an
18-item list:

| preset | planners | runs | the question it answers |
|---|---|---|---|
| `readme` | 7 | 10,500 | continuity with the original documented design |
| `published` | 8 | 12,000 | the defensible headline table |
| `classical` | 10 | 15,000 | no learned weights; reproducible from source alone |
| `paired` | 12 | 18,000 | does each reimplementation behave like the algorithm it is named after? |
| `all` | 18 | 27,000 | everything registered (the default) |

Run counts are 5 maps × 10 tasks × 3 global planners × 10 seeds × 1 mode ×
1 pedestrian layer. Crossing `--reactive-peds off sfm jupedsim pysf` multiplies
by four; each level is written to a sibling `results/peds_<level>/` root.

`--algorithms` still overrides `--preset`. `--skip-existing` resumes without
recomputation; Ctrl-C kills child SUMO processes rather than orphaning them.
Per-run raw logs (`robot_trace.csv`, `robot_metrics.json`, `scenario.json`,
including every sampled demand parameter) are written for every cell; the
directory layout is defined once in `sim/run_layout.py`.

### How long it takes

Measured per-run wall time on this repo's reference machine (**Intel Core
Ultra 7 155H**, 22 threads, mobile), one run at `--max-time 300` scaled to the
protocol's 3000 s episodes, `--reactive-peds sfm`:

| planner | min/run | | planner | min/run |
|---|---|---|---|---|
| `orca` (RVO2) | 0.48 | | `sarl` | 0.94 |
| `orca_heuristic` | 0.62 | | `lstm_rl` | 1.00 |
| `teb` | 0.72 | | `cadrl` | 1.35 |
| `mpc` | 0.87 | | `mpc_dompc` | 2.98 |
| `teb_upstream` | 0.97 | | `lstm_rl_upstream` | 4.10 |
| `crowdnav_dsrnn` | 3.0 | | `cadrl_upstream` | 4.20 |
| `crowdnav_attngraph` | 7.1 | | `sarl_upstream` | 5.88 |

`paired` mean **2.0 min/run**; a real protocol cell (map5_ucl + task + global
planner) costs a further **1.25–1.5×** over the light map2 cell these were
measured on. So budget ≈ **2.7 min/run**, i.e. **~34 CPU-days for the 18,000-run
`paired` sweep**, ~2 wall-days at `--jobs 16` on that machine.

**On a desktop Core i9 (24 cores): roughly 1–1.5 wall-days** for `paired` at
`--jobs 22`, and **~1.5–2.5 wall-days** for the 27,000-run `all` preset. That
assumes ~1.3–1.5× the per-core throughput of the mobile part and no thermal
throttling; expect sublinear scaling past ~16 concurrent runs, since each run is
a SUMO process plus a Python planner competing for memory bandwidth.

> **A 5090 will not help, and `--device cuda` may make it slower.** The learning
> planners are latency-bound, not FLOP-bound: upstream CrowdNav evaluates 81
> *batch-of-1* forward passes per control step over at most 5 humans. Measured
> here, giving one planner more CPU threads makes it **worse** — 105.9 ms/step
> at 1 thread vs 128.7 ms at 8 — which is the signature of dispatch overhead
> dominating. Moving hundreds of tiny ops per step onto a GPU adds kernel-launch
> latency on top. Everything else (SUMO, the classical planners, the shapely
> geometry) is single-threaded CPU. **Buy cores and RAM, not GPU.**

### Splitting the sweep across several machines

`--shard K/N` runs shard `K` of `N`. Every machine enumerates the *same*
deterministic design and takes `combos[K::N]`, so there is **no coordination,
no shared state and no combo run twice**. Run the same command on each machine,
changing only `K`:

```bash
# PC 1 of 3
python sim/run_protocol.py --preset paired --shard 0/3 --jobs 22 --out-root //share/results
# PC 2 of 3
python sim/run_protocol.py --preset paired --shard 1/3 --jobs 22 --out-root //share/results
# PC 3 of 3
python sim/run_protocol.py --preset paired --shard 2/3 --jobs 22 --out-root //share/results
```

Then, from any one machine, audit the **merged** tree — omitting `--shard`
audits the whole design:

```bash
python sim/run_protocol.py --preset paired --verify-only --out-root //share/results
```

Point every machine at one shared `--out-root`, or give each a local root and
merge the trees afterwards (`robocopy` / `rsync`); run directories are disjoint
by construction, so a merge cannot collide. Every machine needs the same
install, including the RVO2 build.

It strides rather than blocks deliberately: the design enumerates seeds
fastest, so consecutive cells share an algorithm. Blocking would hand one
machine every `sarl_upstream` cell — roughly 12× the cost of an `orca` cell —
while another finished early. Verified on the real 18,000-run `paired` design:

| N | shard sizes | `*_upstream` cells each | cost index |
|---|---|---|---|
| 2 | 9000 / 9000 | 3000 / 3000 | 347 / 347 |
| 3 | 6000 / 6000 / 6000 | 1950 / 1950 / 2100 | 232 / 232 / 232 |

so **3 PCs ≈ 8–12 wall-hours** for `paired` at `--jobs 22` each, against
~1–1.5 wall-days on one. Verified end to end on a 24-cell design: the three
shards are exactly disjoint, cover the design, each reports COMPLETE, the
merged audit reports 24/24 — and deleting one machine's output makes the merged
audit report `MISSING 2`, name the cell, and exit non-zero.

Sanity-check the estimate on the target machine before committing days to it:

```bash
python sim/run_protocol.py --preset paired --maps map2_crossing --tasks none \
    --global-planners fixed --seeds 1 2 --max-time 60 --jobs 8 \
    --out-root /tmp/validate
```

That is 24 runs and finishes in well under a minute; it exercises all 12 paired
planners end to end and ends with the coverage audit.

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
--globals 3 --locals 7 --modes 1` -> 10,500 runs. That is the ORIGINAL
7-planner design (`--preset readme`). With all 18 registered planners the
full cross is 27,000 runs; `python sim/run_protocol.py --dry-run` prints
the size and cost of whichever preset you choose.

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

`--unit` defaults to `both`, so this writes **`results/stats/`** (local
algorithm as the unit) **and `results/stats_combo/`** (global+local combination
as the unit); `analysis/benchmark_plots.py` likewise writes `plots/` and
`plots_combo/`. Contents: a **variational-Bayes** binomial mixed GLM for
success (`BinomialBayesMixedGLM.fit_vb`, so the reported intervals are
posterior mean ± 1.96·posterior SD — credible intervals, not frequentist CIs —
and variance components are `exp(posterior mean of log-sd)`), with crossed
variance components for seed / map / task; linear mixed models on successful
runs for time and path length, and additionally for
`min_pedestrian_distance_m`, `ped_delay_s_mean`, `ped_deflection_m_mean`,
`ped_personal_space_s_total`, `social_work`, `social_force_on_agents` and
`social_force_on_robot`; a failure
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

* **Sidewalk containment is NOT uniform across planners — declare this.** The
  in-repo learning planners and the published ones differ by more than
  "reimplementation vs. upstream": one group knows the sidewalk exists and the
  other does not.

  The in-repo SARL scores every candidate action with a band-containment term
  ([`sarl_sumo_robot_unified.py:435`](sim/planners/sarl_sumo_robot_unified.py#L435),
  commented "SUMO adapter addition: prevent the model from leaving the
  sidewalk"): a candidate whose predicted pose leaves
  `[sidewalk_x_min, x_max] × [y_min, y_max]` returns `-0.25` immediately — the
  *same magnitude as the collision penalty* — so leaving the band is scored as
  badly as hitting a pedestrian, and the policy actively steers away from the
  kerb.

  Every published planner has no such term. Upstream CrowdNav's
  `MultiHumanRL.compute_reward`
  ([`multi_human_rl.py:65`](sim/third_party/crowdnav/crowd_nav/policy/multi_human_rl.py#L65))
  scores collision, goal-reaching and discomfort distance only; grepping
  `crowd_nav/` and `crowd_sim/envs/utils/` for *sidewalk*, *band*, *wall*,
  *kerb* or *curb* returns nothing at all. The same is true of DS-RNN and
  CrowdNav++ — all three upstream arenas are open space with no static
  geometry. Nothing was added to their observations to fake a kerb, because
  that would be inventing an input the networks never saw in training.

  Three consequences for reading the results table:

  1. `sarl` vs `sarl_upstream` is **not** a clean like-for-like comparison of
     the same algorithm. They differ in the reward used for action selection.
  2. For the published planners, band-violation and `walkable_clamped_steps`
     measure **the runner clamping the robot**, not the planner avoiding a
     kerb. Read them as "how often the policy tried to leave the sidewalk".
  3. It is the main reason the published learning planners drift laterally on
     a long straight leg. That is mitigated — not removed — by the carrot local
     goal and O(2) action averaging described in
     [`docs/code_audit.md`](docs/code_audit.md); see
     `crowdnav_dsrnn_planner.py` for the measurements.

  The classical planners sit in between: DWA, MPC and TEB carry their own
  explicit band constraints, whereas `orca` (published RVO2) is given the band
  as RVO2 line obstacles, which *is* part of that published model.

* **Fallback coverage (`planner_active_frac`).** Some planners spend a large
  share of an episode in a fallback rather than in the algorithm under test,
  so every run records a `planner_status_counts` histogram and
  `planner_active_frac`, the fraction of control steps that ran the actual
  algorithm. The published CrowdNav wrappers return straight-line
  `goal_direct` motion whenever no pedestrian is inside `--sensor-range`,
  because upstream's value networks take a `(batch, #humans, 13)` tensor and
  cannot be called with zero humans. Measured on
  `map2_crossing/mixed/seed 1`: `sarl_upstream` 0.983, `cadrl_upstream` 0.950,
  **`lstm_rl_upstream` 0.467** — over half that episode was not CrowdNav.
  Removing the range filter is not a fix and is much worse: upstream trains in
  a ±6 m arena, and a pedestrian 200 m away swings the commanded direction by
  up to 135°. **Any comparison across learning planners should report, or
  condition on, `planner_active_frac`.**

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
  bit-compatibility with legacy runs (map2 reference: 138.5 m; see the
  re-baseline note above).
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
* **Pedestrian reactivity (`--reactive-peds pysf`).** The PUBLISHED Social
  Force Model, as a fourth level of the same factor: PySocialForce 1.1.2
  (default backend) and socialforce 0.2.3, both installed from PyPI unmodified,
  with the same capture/release bubble as `sfm` and `jupedsim`. Worth knowing
  before comparing it with `sfm`: the in-repo layer's docstring says its
  parameters "follow PySocialForce", and that holds for exactly ONE of the four
  constants it names — `TAU` 0.5 s matches; `A_PED` 4.5, `B_PED` 0.35 and
  `LAMBDA` 0.30 have no counterpart in either published package, and far-field
  robot repulsion differs by ~18× at 2 m. Caveat for robot-size studies: in
  `robot_as="agent"` mode the robot's RADIUS does not affect the force, because
  neither published package supports per-agent radii in its pedestrian
  interaction term (measured: radius 0.20 vs 0.60 changes trajectories by
  0.000 m). Use `robot_as="obstacle"`, or the `jupedsim` layer, where agent
  radius is genuinely part of the model.
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
* Robot–pedestrian collision: centre distance < `max(0.42, r_robot + 0.15)`,
  i.e. 0.42 m at the default `--robot-radius 0.25` and wider for a larger
  robot (3 s spawn grace).
  Failure taxonomy recorded per run: `collision`, `max_time`, `stalled`,
  `goal`.

## Provenance

`sim/planners/` holds two different kinds of code, and the distinction matters
for the academic-integrity declaration:

* **The author's own implementations**, used unmodified: `dwa`, `astar`,
  `dijkstra`, `rrt`, `mpc`, `teb`, `orca_heuristic`, and the CrowdNav
  adaptations `sarl`, `cadrl`, `lstm_rl`.
* **Third-party published implementations**, used as dependencies or vendored
  verbatim, with the benchmark supplying only a thin adapter to the planner
  interface. `sim/benchmark_adapters.py::PUBLISHED_IMPL` is the authoritative
  citation list:

  | source | licence | how it is used |
  |---|---|---|
  | RVO2 / Python-RVO2 (van den Berg et al., ISRR 2009) | Apache-2.0 | built by `sim/third_party/build_rvo2.py`; drives `orca` |
  | do-mpc 5.1.1 + CasADi 3.7.2 + Ipopt | LGPL-3.0 / EPL-2.0 | pip dependencies; drive `mpc_dompc` |
  | teb_local_planner 0.9.1 (Rösmann et al.) | BSD-3-Clause | published binary via new pybind11 glue in `sim/third_party/pyteb/`; drives `teb_upstream` |
  | CrowdNav (Chen et al., ICRA 2019) | MIT | vendored at `sim/third_party/crowdnav/` |
  | CrowdNav\_DSRNN (Liu et al., ICRA 2021) | MIT | vendored at `sim/third_party/crowdnav_dsrnn/` |
  | CrowdNav\_Prediction\_AttnGraph (Liu et al., ICRA 2023) + GST predictor (Huang et al., RA-L 2022) | MIT | vendored at `sim/third_party/crowdnav_attngraph/` |
  | JuPedSim 1.4.2 | LGPL-3.0 | pip dependency; `--reactive-peds jupedsim` |
  | PySocialForce 1.1.2 / socialforce 0.2.3 | MIT | pip dependencies; `--reactive-peds pysf` |

  Every vendored tree carries its upstream `LICENSE`, a `COMMIT` file pinning
  the exact revision (and checkpoint SHA-256 where weights are included), and a
  `PATCHES.md`. Patches are mechanical only — import guards for packages that
  no longer install on current Python, and build-system fixes for modern CMake
  and MSVC. **No algorithm, constant, network definition or config value was
  changed in any vendored source**; `diff -r` against a fresh clone at the
  pinned commit is byte-identical apart from the documented files.

  Learning-based planners run on **published pretrained checkpoints**, not
  retrained weights. The three CrowdNav checkpoints shipped with this
  repository load into the unmodified upstream networks with
  `load_state_dict(strict=True)`; DS-RNN and CrowdNav++ use the checkpoints
  their own repositories publish.

Map construction and demand use the official SUMO toolchain (netconvert,
randomTrips). The benchmark harness (map pipeline, signal gate, runner, batch
driver, protocol driver, analysis) and the third-party adapter layer were
developed with AI assistance (Anthropic Claude) under the author's direction;
see the university's academic-integrity guidance for the corresponding
declaration. `docs/code_audit.md` records the audit that produced the current
state, including which defects were found and what remains open.

On thesis submission, archive this repository together with the `results/`
directory on Zenodo and cite the DOI.
