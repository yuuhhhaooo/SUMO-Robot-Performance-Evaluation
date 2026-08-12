# Code audit — status report

Date: 2026-08-12. Scope: all Python under `sim/`, `sim/planners/`, `analysis/`.

**Method.** 11 parallel area reviews, each finding then adversarially re-verified
against the real code by an independent pass whose default position was that the
finding is wrong. 181 candidates raised, 32 refuted and dropped, 149 confirmed,
merged here to 78 distinct defects. Severities below are post-verification.

---

## 0. What was fixed in this pass

Applied to the working tree, **not committed** — review before committing.
These are also marked **[FIXED IN TREE]** inline below.

| # | File | Defect |
|---|------|--------|
| 1 | `benchmark_runner.py` | `sfm` read ~200 lines before assignment: the entire `global_plan_failed` recovery path was dead code (`UnboundLocalError`). Confirmed by `pyflakes`. |
| 2 | `benchmark_runner.py` | `global_plan_time_s` duplicated in one dict literal; first value silently discarded. Confirmed by AST scan. |
| 3 | `benchmark_runner.py` | `min_pedestrian_distance_m` emitted as `float('inf')` → `json.dumps` writes bare `Infinity`, which is invalid JSON. Now `null`; same for the trace CSV. |
| 4 | `benchmark_adapters.py` | `leg_config` made the leg goal infeasible for every classical planner (box ended exactly at the goal; planners reject within 0.02–0.06 m of the boundary). Added `LEG_GOAL_MARGIN = 0.5`. |
| 5 | `benchmark_runner.py`, `benchmark_batch.py` | `fixed` and `dijkstra` wrote to the same results directory and overwrote each other. |
| 6 | `benchmark_batch.py` | `--waypoints` made the batch read metrics from a path the runner never writes → `FileNotFoundError` killing the sweep. |
| 7 | `benchmark_batch.py` | A runner exiting 0 without metrics crashed the sweep on `json.loads`. |
| 8 | `benchmark_batch.py` | `batch_summary.json` key ignored **task** and **global planner**, pooling distinct cells into one mean. |
| 9 | `stats_models.py` | Bootstrap ranking broke exact ties by array order (i.e. alphabetically), fabricating `P(top-1)=1.0` and zero-width rank CIs. Now average ranks with shared top-1 credit. |
| 10 | `stats_models.py` | Reference algorithm forced as baseline even with zero rows after subsetting → rank-deficient design, contrasts against an empty cell. |
| 11 | `stats_models.py` | Failure-taxonomy figure omitted `planner_error:*`, so bars did not sum to 100% of runs. |

**Verified by test** (fixes 9–11): perfectly-tied algorithms now split top-1 credit
1/3 each at average rank 2.0; a genuine winner still gets `P(top-1)=1.0`; two tied
leaders split 0.5/0.5; taxonomy percentages sum to 100; an absent reference falls
back to a present level and says so.

### Re-baselining required

Fix 4 changes the leg box for **every** planner, so trajectories change: the
README's cross-machine regression number (`path_length_m: 102.59`) **must be
re-measured and the README updated**. Fix 5 renames result directories, so
existing `results/` trees under a `dijkstra` global planner must be re-run or
renamed before `--skip-existing` can be trusted.

### Two latent bugs that fix 4 activates

Local RRT could never plan, so these never ran. They will now:
`shortcut()` splices every 5th tree node **with no collision check**
(`rrt_…:103-113`), and the replan guard `or not self.path` (`rrt_…:143`) stops
firing every step. Check both before trusting new RRT results.

---

## 1. Crashes & correctness bugs

### 1.1 Global-plan-failure path raised `UnboundLocalError` on `sfm` — **[FIXED IN TREE]**
`sim/benchmark_runner.py:786` (HEAD 781)

The plan-failure metrics dict read `sfm.controlled_steps` / `sfm.capture_events` / `**sfm.ped_metrics()` ~200 lines before `sfm` was first bound, so the entire documented `termination_reason="global_plan_failed"` recording path was dead: the process died with exit 1, no `robot_metrics.json` was written, and `benchmark_batch` printed "runner failed" and continued. Reachable only via `--global-planner rrt` (`_rrt_route` at :323 is the sole `GlobalPlanFailure` raise site; dijkstra/astar `sys.exit` instead — see 1.3). Every RRT plan failure therefore vanished from the denominator of the global-planner success rate — survivorship bias, not random dropout. The tree now hardcodes `0`/`0`, drops the duplicate `global_plan_time_s` key, and writes `min_pedestrian_distance_m: None` instead of `inf`. **Action: re-run any RRT global-planner cells produced before this fix.**

### 1.2 RRT local planner could never produce a path — **[FIXED IN TREE, results stale]**
`sim/benchmark_adapters.py:47` / `sim/planners/rrt_sidewalk_robot_random_stop_collision.py:50`

`leg_config` set `sidewalk_x_max = max(leg_len, 1.0)` while the leg goal is exactly `(leg_len, off)`, and `point_safe` rejects anything within 0.04 m of the boundary. Both `segment_safe(start, goal)` (:117) and the goal-connect test (:129) sample `r=1.0`, so `plan()` returned `[]` on **every call, every map, every seed, even with zero obstacles** (reproduced by two independent verifiers). `compute_command` then took the `no_path` branch at :146–149: a centreline creeper at full `max_speed` with zero obstacle avoidance. The published "RRT" local-planner row is not RRT, the `rrt` entry in `configs/tuning_spaces.json` tuned a search that never succeeded, and `tune.py:24-27` archives a wrong diagnosis ("fallback-dominated in crowded corridors") as protocol evidence. The tree adds `LEG_GOAL_MARGIN = 0.5`. **Action: every `rrt` local-planner result and `configs/rrt.json` must be regenerated.** Two latent defects become live once it plans: `shortcut()` splices every 5th tree node with no collision check (rrt:103-113), and the replan guard `or not self.path` (rrt:143) no longer fires every step.

### 1.3 dijkstra/astar routing failure calls `sys.exit`, silently deleting the run
`sim/benchmark_runner.py:545` (also :366, :419)

`auto_route` terminates the process on an unroutable pair, while `_rrt_route` raises `GlobalPlanFailure` and gets recorded as `success=False`. So the two levels of the same experimental factor have different missing-data semantics: an RRT failure is a data point, a dijkstra failure is an absent cell. `sample_tasks.py:196` catching `SystemExit` proves the path fires for real point pairs. All ten shipped map5_ucl tasks route fine (tasks are pre-screened by the same router), so exposure is mainly `--waypoints`/`--route` users.
**Fix:** `raise GlobalPlanFailure(...)` at :419 and :545; keep `sys.exit` only for the missing-shapely error at :366. Update `sample_tasks.py:196` to catch both.

### 1.4 DWA's "emergency brake" commands forward motion into the obstacle
`sim/planners/dwa_sidewalk_robot_random_stop_collision.py:234`

When every `(v,w)` in the dynamic window collides or leaves the sidewalk, the fallback commands `brake_v = max(min_speed, v - max_accel*dt)` **forward** with no collision check of its own. Reproduced: `v=0.95`, one pedestrian 0.6 m ahead → returns `(0.55, 0.0)`, which travels 0.275 m and closes the gap to 0.325 m, below `COLLIDE_R = 0.42`. The brake produces the collision it exists to prevent, and DWA's collision rate is partly an artefact of this branch.
**Fix:** `best_u = (0.0, 0.0)` when the trajectory set is empty, or explicitly search braking trajectories across the yaw window and pick the one maximising min clearance.

### 1.5 A*/Dijkstra/RRT no-path fallback steers straight at the blockage
`astar_…:131`, `dijkstra_…:131`, `rrt_…:148`

All three abandon obstacle avoidance on search failure and drive at the goal (or the centreline) with no obstacle term consulted — but search failure is *caused* by obstacles. Verified: a 10-pedestrian wall at x=20 makes `plan()` return `[]` and `compute_command` return `(0.3, 0.0, {'status':'no_path'})`, due-east into the wall. Not rare: with `clearance = 0.55 m` against a 0.25 m y-resolution, 2 pedestrians abreast seal a 2 m band and 3 seal a 3 m band — and `make_legs` defaults to `W = 2.0`. Collisions attributed to A*/Dijkstra are produced by the fallback, not the search.
**Fix:** hold position, or score a handful of unit directions by `min_distance_to_obstacles` and reject any whose next pose is inside `self.clearance`. Retry once with reduced clearance before declaring failure.

### 1.6 TEB has no velocity decision variable and cannot slow or stop
`sim/planners/teb_sidewalk_robot_random_stop_collision.py:148`

Only band `y` coordinates are optimised; speed is derived geometrically as `min(max_speed, max(0.25, dist/dt))` where `dist` is ~2 m, so it saturates at `max_speed` everywhere except within ~2.5 m of the leg goal, and the `0.25` floor forbids stopping. Measured: `|v| = 1.000` bit-identically for an empty band, a pedestrian at 2 m, at 1 m, and a 10-pedestrian wall across the full band at 1 m — where the symmetric optimum is the centreline, so it drives head-on at full speed. ORCA on the same scenes: 1.000/0.833/0.500/0.667/0.000. The module docstring documents the fixed-x simplification, but the benchmark consequence stands: TEB's collision rate measures a missing speed channel, not TEB.
**Fix:** add per-pose time intervals `dT_k` (or an explicit speed) to the decision vector costed by `w_short_time`; at minimum drop the 0.25 floor and hard-brake when predicted clearance along the band falls below `self.clearance`.

### 1.7 SARL is fed the raw multi-hundred-metre leg goal, with a reward that is identically zero
`sim/planners/sarl_sumo_robot_unified.py:464`, reward at :424-452

`predict` uses `robot.gx/gy` verbatim and puts `dg = ‖goal − pos‖` into `mlp1` feature 0. The adapter passes the far end of the current leg — 20–300 m — against CrowdNav training data with `dg ∈ [0, 8]`. Both siblings clip (CADRL 6 m lookahead at cadrl:120-132, LSTM-RL 8 m at lstm_rl:193-200); SARL does not. Worse, `_reward` returns exactly 0.0 for every in-sidewalk, non-colliding candidate, so the argmax over 81 actions rests entirely on an out-of-distribution value net. Reproduced with the real checkpoint: same obstacle field, varying only goal distance, chosen heading went +45° (dg=2) → +22.5° (dg=6) → **−112.5°, i.e. backward** (dg=20) → +67.5° (dg=55) → 0° (dg=113); 8 of 99 sampled goal distances chose |heading| > 90°.
**Fix:** add `goal_lookahead = 8.0` and clip the goal used in `_propagate_robot`/`_rotate`/`_reward`; add a progress term to `_reward` matching cadrl:158 / lstm_rl:288-289.

### 1.8 `--waypoints` made the batch read metrics at a path the runner never wrote — **[FIXED IN TREE]**
`sim/benchmark_batch.py:139`

The batch built `mlabel` without the `__custom` suffix the runner uses (`benchmark_runner.py:804-813`), so the unguarded `json.loads(mfile.read_text())` raised `FileNotFoundError` and killed the sweep after the first successful run. The tree adds the `custom` label **and** a general `if not mfile.exists(): continue` guard so no single missing file can abort a multi-thousand-run sweep.

### 1.9 `batch_summary.json` write crashes when no run ever created `out_root`
`sim/benchmark_batch.py:254`

`benchmark_batch` never `mkdir`s `out_root` — only the runner does. With an empty combo list (all routes filtered out, or a `--tasks` value matching nothing) the glob returns nothing, `rows` is `[]`, the guarded CSV write is skipped, and the unconditional `write_text` raises `FileNotFoundError` instead of "nothing to do".
**Fix:** `out_root.mkdir(parents=True, exist_ok=True)` right after `out_root = Path(args.out_root)`, and short-circuit with a message when `combos` is empty.

### 1.10 A crashed tuning study still writes `configs/<algo>.json`
`sim/tune.py:68`

`subprocess.run(..., capture_output=True)` then `if res.returncode != 0: return 0.0`, with a bare `except Exception: return 0.0` around the stdout parse and `res.stderr` never read. A missing `SUMO_HOME`, an import error, or a TraCI failure is indistinguishable from a navigation failure. Lines 210-219 then unconditionally write `study.best_params` — an arbitrary TPE sample from an all-zero study — into the file that drives the published evaluation of that planner.
**Fix:** on nonzero return or parse failure, print `res.stderr[-2000:]` and `raise optuna.TrialPruned()`; refuse to write `configs/<algo>.json` when `study.best_value <= 0.0`.

### 1.11 Latent crashes in `auto_route` helpers
- **`attach()` bare `StopIteration`** — `benchmark_runner.py:506`. With a cached `_graph`, `pieces` is re-parsed under the *query's* bbox while `ends` comes from the cache; `next(n for n in ends if nodes[n] == P[0])` then raises uncaught. Reproduced by building the graph around one anchor and routing 500 m away. Unreachable today only because the sole caller passes `pts=None`. Fix: build a `{coord: node}` dict alongside the cached graph.
- **`_rdp` RecursionError** — `benchmark_runner.py:166`. O(n) recursion depth, no guard; a 3000-point polyline exceeds the 1000 default. Latent for the shipped `--geometry.remove` nets. Fix: rewrite iteratively with an explicit index stack.
- **`_rdp` collapses closed polylines** — `benchmark_runner.py:164`. When `points[0] == points[-1]` all perpendicular distances are 0, so the whole loop is discarded; `make_legs` returns `[]` and `Frame(legs[0])` raises `IndexError`. No shipped route is a loop; a user-supplied `--waypoints "x,y;a,b;x,y"` triggers it. Fix: measure distance to `points[0]` when the chord is degenerate.

### 1.12 MPC/TEB solver-failure fallbacks are unreachable dead code
`mpc_…:157`, `teb_…:137`

`if result.success or result.x is not None:` — L-BFGS-B always populates `.x`, so `mpc_fallback` (:171-178) and `teb_fallback` (:156-160) can never execute. A genuinely failed solve commits the stopped iterate as `self.last_solution` / `self.last_y`, which is reused as the next warm start, so staleness compounds; the only signal is a status string the runner discards. All solves converged under stress testing, so this is latent.
**Fix:** branch on `result.success`, and do **not** cache the solution on the fallback path. Surface `mpc_optimized`/`mpc_partial`/`mpc_fallback` counts into `robot_metrics.json`.

### 1.13 `time_to_ped_green` indexes `ph.state[li]` without the length guard
`sim/native_signal_gate.py:81`

`__init__:39` guards the same access with `li < len(ph.state)`; this method does not. Unreachable today: every shipped `<tlLogic>` has uniform state lengths, and grep shows `time_to_ped_green` and `ped_state_at` have **no callers anywhere in the repo**.
**Fix:** add the guard, or delete both dead methods.

---

## 2. Silent result corruption

*These change published numbers without any error, warning, or trace signal.*

### 2.1 DWA runs the whole benchmark at `max_speed = 0.95` while all six other planners run at 1.00
`sim/benchmark_adapters.py:66-71`

`DWAAdapter.__init__` passes only `dt`, `max_time` and the five sidewalk geometry fields into `DWAConfig`, so the dataclass defaults survive: `max_speed=0.95` (dwa:42), `max_accel=0.80` (dwa:46), `safe_distance=0.20` (dwa:64), versus `PlannerConfig` 1.00 / 0.50 / 0.42. The runner caps with `min(getattr(planner,'cfg',cfg).max_speed, HARD_SPEED_CAP)` at :1105-1108 → **0.95 for DWA, 1.00 for everything else**, and `max_speed` is in no tuning space, so nothing restores it. Every `sim_time_s`, `avg_speed_mps`, normalised-time and within-`--max-time` success comparison is biased ~5% against DWA across ~10,500 runs — and DWA is the reference algorithm in `stats_models.py`.
**Fix:**
```python
self.cfg = self.mod.DWAConfig(
    dt=cfg.dt, max_time=cfg.max_time,
    max_speed=cfg.max_speed, max_accel=cfg.max_accel,
    max_yaw_rate=cfg.max_yaw_rate, safe_distance=cfg.safe_distance,
    social_distance=cfg.social_distance, sensor_range=cfg.sensor_range,
    sidewalk_x_min=..., ...)
```
Then decide explicitly whether the acceleration limit applies to all planners or none (see 2.11).

### 2.2 `make_legs` takes the sidewalk band width from a whole-lane bounding box
`sim/benchmark_runner.py:588`

For legs within 2.56° of an axis, `make_legs` takes the **first** `spec["sidewalks"]` rect containing `w0` and sets `W = hi − lo`. On OSM specs those rects are `osm_import.lane_rect()` bounding boxes of entire curved lanes: 750 of map5_ucl's 1787 rects have a cross dimension > 8 m (max 144.71 m) against a true width of 2.00 m. Verified on real task t01: `W = 60.91`, giving `sidewalk_center_y = 30.45` — an 18.8 m lateral pull. Every centreline-seeking planner (dwa:213, mpc:123, orca:65, cadrl:164/252, lstm_rl:290) is then told to steer to a centreline off the sidewalk; the lateral clamps at :1094/:1133 become no-ops; the A* grid grows from 8 to 244 rows; and `crossing_ahead`'s `min(lys) > leg["W"] + 0.3` gate at :690 widens to match, corrupting `time_waiting_at_light_s`. Selection is order-dependent (first matching rect in JSON order wins). Only a subset of legs is affected, but that run's `path_length_m` / `avg_speed_mps` / `walkable_clamped_steps` are silently wrong.
**Fix:** store the true lane `width` in `map_spec` and use it. Failing that, require `1.0 <= W <= 8.0` and pick the *smallest* containing rect, falling back to `W, off = 2.0, 1.0`. Add the missing lower bound at the same time — 602 map5_ucl rects have a cross extent below 0.2 m, and `W < 0.16` inverts the clamp `min(max(v, 0.08), W-0.08)` into a constant, pinning the robot's lateral coordinate for the whole leg.

### 2.3 Collision detection and `min_pedestrian_distance_m` compare the robot at *t−dt* with pedestrians at *t*
`sim/benchmark_runner.py:1069`

`traci.simulationStep()` advances pedestrians to *t*; the observation loop then measures `hypot(px − x, py − y)` using `x, y` last assigned at :1148 of the **previous** iteration. The robot's own position error is up to 0.8 m at `HARD_SPEED_CAP`; against an oncoming pedestrian the relative error reaches ~1.55 m — several times `COLLIDE_R = 0.42` and `SOCIAL_R = 0.85`. This biases `min_pedestrian_distance_m`, `close_encounter_steps`, and the binary `collision` outcome that drives the success GLMM, in both directions. The same trace row at :1160 pairs the *post*-integration pose with this stale distance, so `robot_trace.csv` is internally inconsistent.

Two compounding half-step errors in the reactive arm only:
- **`sfm.step()` runs at :1159, after the measurement.** Safety metrics and planner input are SUMO's striping positions, which the SFM immediately overwrites via `moveToXY`. Since the SFM is precisely what pushes pedestrians laterally away from the robot while striping walks them through it, the two differ systematically — and only in the `--reactive-peds sfm` arm, i.e. exactly across the comparison the benchmark exists to make.
- **`validate_reactive.py:119` has the identical ordering bug**, in the script whose sole purpose is certifying that the SFM layer works.

**Fix:** move `sfm.step(...)` to immediately after `simulationStep()` in both files, and recompute `step_min` from the cached `(px, py)` list against `(nx, ny)` before the collision test and before `rows.append`.

### 2.4 A robot already stopped at a red light is never marked `held`
`sim/benchmark_runner.py:1121`

`held = True` is set only inside `if spd > max_sp:` → `if max_sp < 0.05:`. Once the robot has crept to the stop line, `step_room → 0` and `max_sp → 0.0`; if the planner independently commands exactly `(0,0)` — routine when a pedestrian queue blocks the crossing — then `0.0 > 0.0` is False, the whole clamp block is skipped, `wait_light` is not incremented, and the stall watchdog at :1163-1169 (`if hypot(vx,vy) < 0.05 and not held`) starts counting toward a spurious `termination_reason="stalled"` after 45 s of a legitimate red. Algorithms that decelerate smoothly are penalised; those that charge the stop line are credited. `time_waiting_at_light_s` is not comparable across algorithms.
**Fix:**
```python
if blk is not None and (max_sp < 0.05 or math.hypot(wvx, wvy) < 0.05):
    wvx = wvy = 0.0
    held = True
```
placed before `if held: wait_light += dt`. `held_prev` at :1126 is assigned and never read — delete it.

### 2.5 `same`/`opposite` are inverted on map3 and map4 vertical legs
`sim/generate_demand.py:126`

`s_ = +1 if r['axis'] == 'h' else -1`, so on vertical roads `same` means hi→lo. But the robot traverses map3's road V1 from y=54.2 to y=154.2 (lo→hi) and map4's `bridge` from y=44.2 to y=144.2 (lo→hi). On those legs `mode='same'` sends pedestrians **head-on** and `opposite` sends them with the robot — while `osm_mode_demand` is genuinely robot-relative (its docstring at osm_import.py:229-235, echoed by benchmark_runner.py:823-824, asserts parity that does not exist). `stats_models.py:107` enters `C(mode)` as a pooled fixed effect across maps, so one factor level denotes with-flow on map1/map2/map5 and counter-flow on half of map3/map4. The published mode main effect is a blend of two opposite treatments.
**Fix:** derive column directions from the robot's traversal direction of each road (available from `spec['robot']['waypoints']`) rather than from `r['axis']`. Until then, do not pool `C(mode)` across map families.

### 2.6 OSM mode demand creates a permanent 1:2:3 density ramp along the robot's route
`sim/osm_import.py:261`

`add_dir_flows` builds a 4-anchor chain and draws a random **start** anchor for *every* person at *every* departure time (`j = rng.randrange(0, len(edge_seq)-1)`), not just a warm-up cohort. The first third of the corridor is traversed by 1/3 of persons, the middle by 2/3, the last by all — for the entire episode. The inline comment at :249-252 claiming "steady-state density immediately" is false. Pedestrian density is the study's primary independent variable. (For `mixed`/`all` the reversed bwd call makes *total* flux uniform and only the directional composition ramps; `same`/`opposite` get the full total-density ramp.)
**Fix:** `j = rng.randrange(0, len(edge_seq)-1) if t < warmup else 0`, sizing the warm-up cohort as `per_hour * traversal_time / 3600`.

### 2.7 `obstacle_force_on_robot` / `social_work` only integrate on steps with a captured pedestrian
`sim/social_pedestrians.py:233`

`if not self.ctl: return` sits ~190 lines before the wall-force accumulation at :425-434, which is the only place `of_on_robot` is incremented. That quantity is a pure robot-vs-wall integral with nothing to do with pedestrians, yet it advances only while the robot has someone inside its 12 m bubble — so it scales with pedestrian density, the study's independent variable. It is emitted as `obstacle_force_on_robot` and summed into the HuNavSim `social_work` metric (:452-454, :467-470), both of which land in `robot_metrics.json`. Any regression of `social_work` on flow rate is confounded by construction. Secondary: the `.exterior`/`.boundary` branch at :428-430 measures single-Polygon and MultiPolygon unions on different definitions (holes contribute no wall force in the former).
**Fix:** move the `of_on_robot` block above the `if not self.ctl: return` guard; use `self.union.boundary` unconditionally (cached and prepared).

### 2.8 Pedestrian-cost metrics average over a wider population than their numerators cover
`sim/social_pedestrians.py:403`

Three population defects in metrics documented at :58-60 as "cost imposed on pedestrians BY the robot":
1. `st["nr"]` — which admits a record into the metric — is set at `dr < 12 m` (:403), but `delay` only accumulates at `dr < 6.0` (:405). `ped_metrics()` divides by the 12 m population (:457), so every pedestrian that came within 12 m but not 6 m contributes a hard 0 to the numerator and 1 to the denominator. The understatement scales with density.
2. `defl` (:407-410) is a running max off the capture-time line with **no distance-to-robot gate at all**, while capture also happens via `near_static()` at 2.5 m (:224) and the bypass steering at :275-300 deliberately pushes pedestrians 1.1 m around *standing pedestrians*. Static-avoidance detours are recorded as robot-imposed deflection.
3. `_done` is appended to at :142/:207/:248 with no pid keying, so a released-and-recaptured pedestrian yields two records — `ped_affected_n` counts episodes, not pedestrians.

**Fix:** gate `nr`, `delay`, `defl` and `swork` on one radius (6 m, matching the `NEIGH_R + 2.0` robot force cutoff at :342); measure `defl` from the position at which that threshold was crossed; key `_done` by pid.

### 2.9 SFM is integrated at `dt = 0.5 s = TAU`, collapsing the relaxation dynamics
`sim/social_pedestrians.py:319`

With `TAU = 0.5` (:33) and `dt = args.step_length` (default 0.5), `vx += ((vd*gex − vx)/TAU)*dt` reduces algebraically to `vx = vd*gex` — the velocity is *overwritten* with the desired velocity every step, so repulsion impulses never persist into the next tick. The layer degenerates from a second-order force model to a one-step velocity field, and per-step displacement is bounded only by `vmax*dt ≈ 0.8 m`. `ped_deflection_m_*`, `min_pedestrian_distance_m` and `social_force_on_agents` are properties of the 0.5 s discretisation rather than of the cited Helbing–Molnár model. (Note: the original claim that reference implementations use dt ≈ 0.01–0.1 s is wrong — `socialforce`'s default is `delta_t=0.4, tau=0.5`, essentially the same regime — so this is the dead-beat boundary case, not an unstable one.)
**Fix:** substep inside `SocialForceLayer.step`: `n = max(1, ceil(dt/0.05))` inner iterations of `h = dt/n`, recomputing forces each time and issuing `moveToXY` once at the end.

### 2.10 OSM statics are named `static_*` but every consumer matches `stand_*`
`sim/osm_import.py:291` vs `sim/benchmark_runner.py:1006`, `sim/social_pedestrians.py:184-189`

`osm_mode_demand` emits `id=f"static_{k}"`; `generate_demand.py:152` emits `stand_{...}`. Three consumers filter on `stand_` only, so on map5_ucl — the only real-world map — (a) `_scatter_statics()` is a no-op and every static stands on one lateral stripe, and (b) the SFM `statics` list is empty, disabling the static capture bubble, `clear_of_statics()` release gating, and the entire bypass-steering block at :265-300 that exists to stop walkers deadlocking through statics. The `static` and `all` mode cells on map5 therefore measure a different scenario from the same modes on map1–map4, recorded under the same `"mode"` label. `validate_reactive.py` runs on map2_crossing and cannot detect it. Compounding: the caller sets `_statics_scattered = True` unconditionally after the first step (`benchmark_runner.py:1056-1058`), before any static with `depart > 0.5` exists.
**Fix:** one module-level `STATIC_PREFIX = "stand_"` imported by `osm_import.py`, `benchmark_runner.py` and `social_pedestrians.py`; and only set `_statics_scattered` once the function reports it moved something.

### 2.11 Acceleration limits are bypassed for six of seven planners
`sim/benchmark_runner.py:1132`

`sidewalk_robot_common.apply_velocity` — the only acceleration limiter in the codebase, whose own comment reads "Acceleration limit for fairer comparison with DWA" — is called from exactly one place: the legacy standalone runner (`sidewalk_robot_common.py:493`). `benchmark_runner` integrates the raw command (`nlx = plx + pvx*dt`) with only a magnitude cap. DWA is rate-limited internally by `calc_dynamic_window`; A*, Dijkstra, RRT, ORCA, MPC and TEB jump 0 → 1.0 m/s in one 0.5 s step. Free instantaneous stop/go both lowers their collision rate (perfect emergency braking) and raises their average speed, repeated at every gate and every avoidance manoeuvre.
**Fix:** apply one shared kinematic filter in `benchmark_runner` before integration for every algorithm, sourcing `max_accel` from the shared `PlannerConfig`.

### 2.12 DWA treats every intermediate route waypoint as a wall and decelerates through it
`sim/planners/dwa_sidewalk_robot_random_stop_collision.py:202`

The hard feasibility test rejects any candidate whose rollout leaves `sidewalk_x_max` — which in the benchmark is the *leg length*, an interior waypoint of a longer route. Measured on an `x_max=80` leg at `v=0.95`: x=40 → 0.950, x=78 → 0.650, x=79.2 → 0.550. Leg switching happens at `--leg-switch-dist 0.8`, so the deceleration zone is ~2 m at every waypoint of every route. No other planner does this (ORCA checks one dt, MPC uses a soft penalty, TEB bounds only y, A*/Dijkstra exempt the goal cell), so it is a per-waypoint speed penalty on DWA alone whose magnitude varies by map — i.e. an interaction with the map factor. Compounded by `predict_trajectory` simulating `predict_time + dt` (dwa:148: 6 rows = 3.0 s for a nominal 2.5 s horizon), which widens the zone by 20%.
**Fix:** `sidewalk_x_max = leg_len + max_speed*predict_time + margin` in `leg_config`, or hand DWA a lookahead goal on the next leg. Fix the horizon with an explicit `n = round(predict_time/dt)` step count.

### 2.13 `DWAAdapter` plans from its own dead-reckoned velocity, which the runner routinely overrides
`sim/benchmark_adapters.py:77`

`st = RobotState(x=state.x, y=state.y, yaw=self.yaw, v=self.v, w=self.w)` — position is authoritative, but `yaw`/`v` are updated only from the command DWA *issued* (:81-82). The runner zeroes the command at a red light (:1119-1121), scales it at a gate (:1123), clamps laterally (:1133) and can pin the pose under `--strict-sidewalk` (:1144); none is fed back, even though the runner already computes correct values at :1095-1096. With `self.v` pinned near 0.95 through a red phase, `calc_dynamic_window` yields `[0.55, 0.95]` — **DWA cannot represent standing still** — and its rollouts integrate from a speed the robot does not have. Every other classical planner receives `v` and `yaw` recomputed from executed motion.
**Fix:** `st = RobotState(x=state.x, y=state.y, yaw=state.yaw, v=state.v, w=self.w)` and re-anchor `self.yaw = state.yaw` each call.

### 2.14 `crossing_ahead` returns the first matching zone in spec order, and `in_rect` aborts the whole scan
`sim/benchmark_runner.py:677-697` (`return None` at :685)

Two ordering defects in the function documented as returning the "nearest red crossing". (a) It returns on the first zone in `zone_states()` order — `map_spec.json` order, unrelated to distance — with no `min` over `gap`; the caller uses that gap as the entire braking budget at :1115-1116, so a farther zone under-constrains the speed and the robot enters a nearer red crossing. (b) `if in_rect(x, y, r): return None` aborts the *entire* scan while the robot stands inside any zone rect. Verified on the shipped map5_ucl net: 5 pairs of surviving TLS crossing rects strictly overlap (up to 8.59 m²) after the never-green filter, so being inside one while another is red ahead is geometrically realisable. `time_waiting_at_light_s` is under-counted and there is no red-light-violation counter to reveal it. (Contrary to the original report, map3/map4 crossing rects only touch at corners — this is map5-specific. The `return None` also has a defensible safety rationale: never halt mid-crossing.)
**Fix:** collect candidates and return the minimum-`gap` red one. Keep `return None` for the zone the robot is actually inside, but move the `in_rect` test *below* the leg-band filter so unrelated junctions cannot suppress it.

### 2.15 Bootstrap ranking table broke ties alphabetically — **[FIXED IN TREE]**
`analysis/stats_models.py:378`

`(-m).argsort(kind="stable")` with `algos = sorted(...)` assigned unique ranks, breaking every exact tie toward the alphabetically first algorithm in all B replicates, and `top1[order[0]] += 1` gave it the entire P(top-1) mass. Reproduced: five algorithms all at success 1.0 (zero ranking information) → `astar P_top1=1.000 ci=[1,1]`, `cadrl 0.000 [2,2]`, `dwa 0.000 [3,3]`… Ties are the common case here (binary outcome, small cells). The README calls this table "the quantitative basis of the ranking-instability claim" — the headline stability result was manufactured by argsort ordering. The tree now uses `rankdata(..., method="average")`, shares top-1 credit among ties, ranks absent algorithms last, and reports fractional ranks (also fixing the `int()` truncation that floored the *upper* CI bound). **Action: regenerate `ranking_stability.csv`/`.png`.**

### 2.16 The mixed models have no seed random intercept and nest map/task inside seed
`analysis/stats_models.py:247`

`fit_lmm` passes `vc_formula={"map": ..., "task": ...}` with `groups=d["seed"]` and never `re_formula`. Two documented statsmodels behaviours apply: variance components are processed **per group**, so `map` and `task` become map-within-seed and task-within-seed effects rather than crossed main effects; and with `re_formula=None` and `vc_formula` set, `exog_re` is dropped and the seed random intercept is silently omitted (confirmed: emitted `lmm_*.csv` has `map Var` and `task Var` rows and no `Group Var`). The module docstring and README both claim crossed seed/map/task effects. Seed-level correlation is unmodelled, so every standard error and p-value on every algorithm contrast is wrong, typically too small.
**Fix:** `groups=np.ones(len(d))` with `vc_formula={"seed": "0 + C(seed)", "map": "0 + C(cell_map)", "task": "0 + C(task)"}`, or keep `groups=seed`, pass `re_formula="1"`, and correct the docstring/README.

### 2.17 Time and path-length bars average failed runs, including 0.0 sentinels and timeouts
`analysis/benchmark_plots.py:553`

The `sim_time_s` and `path_length_m` panels are computed over every run in `algod[a]` with no success filter. Global-plan failures write `path_length_m: 0.0, sim_time_s: 0.0` as sentinels; timeouts record `sim_time_s` at the `max_time` cap. So "time to finish" is a mixture of goal times, censored caps and hard zeros, with the zero bias largest for exactly the algorithms that fail most — while `stats_models.py:445` fits the same two outcomes with `subset_success=True`. The bar chart and the LMM in the same thesis describe different populations.
**Fix:** `src = [m for m in algod[a] if m.get('success')] if key in ('sim_time_s','path_length_m') else algod[a]`, and say so in the panel title.

### 2.18 `agg()` averages failure placeholders and reports an `n` that does not match the denominator
`sim/benchmark_batch.py:231-242`

`vals` drops missing/None/non-finite entries per metric, but `num["n"] = len(sel)` reports the unfiltered cell size and is shared by every metric. The 0.0 sentinels from global-plan failures are finite and get averaged into `sim_time_s_mean` / `path_length_m_mean` / `avg_speed_mps_mean`; `min_pedestrian_distance_m` is dropped whenever it is null/inf, so its mean is over fewer runs than the printed `n` with no indication. Contained: `batch_summary.json` has no consumer in the repo, and `benchmark_plots.py` recomputes its own aggregates.
**Fix:** compute timing/geometry means over `[r for r in sel if r.get("success")]`, and emit `num[f"{k}_n"] = len(vals)` next to each mean.

### 2.19 `batch_summary.json` pooled all global-planner levels and all tasks — **[FIXED IN TREE]**
`sim/benchmark_batch.py:251`

The summary key was `(map, route, mode, algorithm)`, and `r["map"]` is the *bare* map name, not the `map_label` carrying `__<task>__g-<gp>`. On the README's own protocol (`--global-planners dijkstra astar rrt`) all three levels collapsed into one entry with the merged `n` printed as the seed count. The tree adds a `_cell()` helper keying on `(map, route, task, global_planner, mode, algorithm)` and tags the summary key accordingly.

### 2.20 `fixed` and `dijkstra` wrote to the same directory — **[FIXED IN TREE]**
`sim/benchmark_batch.py:145` / `sim/benchmark_runner.py:768`

Both files suppressed the `__g-<gp>` suffix for *both* levels, so the two behaviourally distinct conditions shared one output path: without `--skip-existing` the second silently overwrote the first; with it, the `dijkstra` cell was filled by the `fixed` result. Triggered by the README's own sequence (pilot at `--out-root results`, protocol into the same tree). The tree now suffixes every non-`fixed` level in both files. **Note the residual:** with `--task`, the runner still forces `args.auto_route = True` at :741-742 and records `gp="dijkstra"`, so a requested `fixed` level executes and is labelled as dijkstra. Fix by honouring an explicit `--global-planner fixed`.

### 2.21 Trajectory figures project every seed onto the first seed's route
`analysis/benchmark_plots.py:302`

`envelope_figure` builds the arc-length frame from `items[0]` and projects all runs onto it, but for `--global-planner rrt` the route is re-planned per seed (`rng_seed=args.seed` into `_rrt_route`, whose attempts seed from `(rng_seed << 8) + 97*attempt`). Lateral deviations are then measured against a route the run never followed, so the 10–90% band reflects inter-route geometry rather than local-planner behaviour — in the figure the README advertises as showing "WHERE trajectories diverge". `legs_for(items[0][1], spec)` at :453/:498 has the same problem.
Related: **`--unit combo` folds the global planner into the algorithm name but not the grouping label** (`benchmark_plots.py:396-401` — the `label += f"{{{gpl}}}"` is in the `elif`), so `dijkstra+dwa` and `rrt+dwa` land in one group and traces are transformed through the wrong leg frames; `--unit both` is the default. And **the median/envelope's far end can rest on 3 of N runs** (`keep = n_ok >= 3` at :316) while the title says "over {len(items)} seeds" — the band narrows toward the goal because of dropout, not consistency.
**Fix:** assert `all(it[1]['waypoints'] == wps for it in items)` and fall back to the overlay path otherwise; append `{gpl}` to `label` in combo mode too; shade or annotate the region where `n_ok < len(items)`.

### 2.22 Missing per-task geometry features are silently mean-imputed across maps
`analysis/stats_models.py:71-74` (repeated at :97)

`df[col].fillna(df[col].mean())` uses the grand mean over **all maps** whenever any value is present, with no warning and no indicator column, and the imputed columns are then entered as `standardize(...)` fixed effects that the README presents as identifying "WHICH topology properties drive ranking changes". A trigger is built in: `load_rows` defaults a missing `task` to the literal `'t0'` while committed task files use `'t01'..'t10'`, so any results tree mixing default-route and task-sampled runs assigns the cross-map grand mean to every default-route run — and shifts the standardisation for all the others.
**Fix:** warn (or raise) on `df[col].isna().any()`, print the affected `(map, task)` pairs, and add a missing-indicator column instead of substituting.

### 2.23 Statistical reporting issues (lower magnitude, same section)
- **`min_pedestrian_distance_m` can be `inf` in the plot loop** — `analysis/benchmark_plots.py:553` has no `isfinite` filter, so one `inf` makes the bar `inf` and the sd `nan`; matplotlib saves the PNG with the bar silently missing and the other bars rescaled. The runner now writes `None` for the empty-crowd case, which the plot loop also mishandles (`float(None)` raises). **Fix:** `xs = [v for v in (...) if v is not None and math.isfinite(v)]`, matching `stats_models.py:212`.
- **Proportion error bars use a symmetric Wilson half-width** — `benchmark_plots.py:563` and `:704`. The Wilson interval is asymmetric about p̂; plotting only the half-width gives an upper bound of 1.139 at p̂=1, n=10 (true bound: 1.0), clipped invisibly by `set_ylim(0, 1.05)`. Both sites also pool 10 seeds × 10 tasks as i.i.d. Bernoulli draws, understating the clustered CI (same assumption at `stats_models.py:177-178`). **Fix:** plot explicit `lo`/`hi` from the shifted centre as a 2-row `yerr`, computed at the seed level.
- **GLMM intervals are mean-field VB posterior SDs labelled "95% CI"** — `stats_models.py:129`. `fit_vb()`'s factorised posterior systematically understates marginal variances, so the odds-ratio intervals are anti-conservative; and `sd_posterior_mean` at :139-141 is `exp(E[log sd])`, i.e. the posterior *median*. The scale handling (interval on log-odds, then exponentiate) is correct. **Fix:** use `fit_map()` with Laplace SEs, or caveat the axis label; rename the column `sd_posterior_median`.
- **No multiplicity adjustment** — `stats_models.py:268`. Nine endpoints × 6 contrasts × 2 units, raw p-values only, with no q-value column for post-hoc correction. **Fix:** add `p_bh` via `multipletests(..., method="fdr_bh")` over the algorithm-contrast rows and state the family in `model_summaries.txt`.
- **Failure-taxonomy figure omitted `planner_error:*`** — **[FIXED IN TREE]** (`stats_models.py:317`; bars now sum to 100% and the exception type is collapsed).
- **Reference algorithm forced into categorical levels with zero rows** — **[FIXED IN TREE]** (`stats_models.py:83`; `_effective_reference` now re-picks after the success/finite subsetting, preventing a rank-deficient design whose pinv-based OLS fallback silently emitted finite, highly significant contrasts against an empty baseline).

### 2.24 TEB predicts pedestrians at half the correct time offsets
`sim/planners/teb_sidewalk_robot_random_stop_collision.py:69`

`_make_x_band` lays 11 poses over a 10 m span (1.0 m spacing) while the objective stamps pose *k* with `t = (k+1)*band_dt` where `band_dt = dt = 0.5` — an implied traversal speed of 2.0 m/s against `max_speed = 1.0`. Every obstacle along the band is evaluated at half the time the robot actually needs; a pedestrian crossing at 1.3 m/s is placed 6.5 m short at the far end. `_initial_y` (:74) likewise shifts the warm start by 1.0 m per step while the robot advances 0.5 m. Combined with 1.6, TEB systematically under-anticipates crossing pedestrians.
**Fix:** `band_dt = (x_band[1] - x_band[0]) / cfg.max_speed`, or set `lookahead_distance = horizon * max_speed * dt`.

### 2.25 Validation script over-weights the pedestrians the SFM slows down
`sim/validate_reactive.py:126`

`lateral_at_robot` is appended once per pedestrian **per step** inside a 1 m band, and the mean divides by sample count, not pedestrian count — so a pedestrian's weight is its dwell time, which the SFM arm systematically increases (that is the effect under test). The `off` arm contributes ~1 sample per pedestrian. Same structure in the `min_gap` scan (:122-124): at 1.2 m/s a pedestrian moves 0.6 m per step and can straddle the 0.30 m pass-through threshold without being sampled inside it — undersampling that is stronger in the faster `off` arm, biasing the headline `pass_throughs(<0.30m)` contrast toward the hypothesis.
**Fix:** accumulate `lat_by_pid.setdefault(pid, []).append(...)` and average one value per pedestrian; interpolate `min_gap` between consecutive samples per pid.

### 2.26 Tuning-space dimensions with literally zero effect pass the audit as "matched"
`sim/benchmark_adapters.py:150` and `sim/tune.py:143`

`apply_params` marks a key as a hit if the name is reachable on the planner, its config, or any class in a bound module — never that it is *read*. `configs/tuning_spaces.json` gives `astar` and `dijkstra` a `social_distance` dimension (both derive `self.clearance` from radii in `__init__` and reference `social_distance` nowhere), and `orca` a `safe_distance` dimension (it uses `collision_clearance` and `cfg.social_distance`). Verified behaviourally: low vs high values give bit-identical commands. `python sim/tune.py --algorithm astar --check` prints "4/4 matched; unmatched: []", and README:177 claims "audited: 7/7 fully matched". Optuna burns an equal-budget on a pure-noise dimension in a protocol whose fairness claim rests on that budget being meaningful, and `configs/astar.json` publishes a tuned value for a parameter nothing reads.
**Fix:** have `apply_params` report *where* each key landed and assert it appears in `inspect.getsource(type(pl))`; remove the inert dimensions from `tuning_spaces.json`. Also delete the module/class walk at :142-149 — its comment claims class-attribute writes reach future instances, which is provably false for `@dataclass` configs, and it is an unguarded name-matched `setattr` across every class the module imported (`pathlib.Path` is in the blast radius).

### 2.27 The median pruner can kill a trial before the second tuning map runs
`sim/tune.py:198`

The episode list is map-major (`for mp in args.maps for sd in args.seeds`), and `trial.report` + `should_prune` fire after every episode under `MedianPruner(n_warmup_steps=1)`. With the default `--maps map1_straight map3_grid --seeds 1000 1001 1002`, pruning becomes active after two map1 episodes, so a parameter set that is mediocre on the straight corridor but excellent on the grid is pruned before map3 is ever evaluated. `configs/<algo>.json` — used for every evaluated map — is biased toward whichever map is listed first. (Mitigated for the first 5 trials by `n_startup_trials`.)
**Fix:** build the list seed-major (`for sd in args.seeds for mp in args.maps`) and set `n_warmup_steps >= len(args.maps)`.

---

## 3. Reproducibility

### 3.1 Pedestrian behaviour depends on `hash()` of a Python `str`
`sim/social_pedestrians.py:308` (also :282)

`jx = ((hash(pid) % 1000)/1000.0 - 0.5) * 0.4` sets each pedestrian's merge-back lateral offset, and `pid` is a SUMO id string whose hash CPython salts per process. `grep -rn PYTHONHASHSEED` over the repo returns nothing, and `benchmark_batch.py:200` spawns each episode with a bare `subprocess.run(cmd)` inheriting the ambient environment — so all ~10,500 subprocesses get a different salt. Line 308 is on the common path (the `else` branch taken whenever no static blocks the walking line, which on OSM maps is *always*, per 2.10) and feeds the desired direction every step. Everything else in the layer is seed-driven, making this the single non-reproducible input. This is the `--reactive-peds sfm` condition — the one the README calls the main protocol and on which all social-navigation claims rest. Re-running a published seed gives different `ped_deflection_m_mean`, `min_pedestrian_distance_m` and possibly a different `termination_reason`; seed-matched paired comparisons between algorithms are invalid because the pedestrian environment is not held fixed. (Line 282's `side` contribution is minor — overridden by the walkable-side check and a sticky memo.)
**Fix:**
```python
import hashlib
def _pid_hash(pid, seed):
    return int.from_bytes(hashlib.blake2b(f"{seed}:{pid}".encode(),
                                          digest_size=8).digest(), "little")
```
used at both sites, with the run seed passed into `SocialForceLayer.__init__`.

### 3.2 `--skip-existing` reuses metrics produced under completely different parameters
`sim/benchmark_batch.py:149`

Only file *existence* is tested. None of `--max-time`, `--flow-min/max`, `--veh-scale`, `--params-file`, `--reactive-peds` or `--device` is compared, and `--max-time` is recorded nowhere in `robot_metrics.json`, so the mismatch is undetectable after the fact. The README's own sequence triggers it: the pilot at `--max-time 420` without SFM writes into `results/`, then the protocol resumes with `--max-time 3000 --reactive-peds sfm --skip-existing` and silently accepts those truncated non-SFM cells as protocol runs. Amplified by the fixed/dijkstra path aliasing (now fixed). Likewise, tuning with a new `--params-file` plus `--skip-existing` produces a summary computed entirely from the old parameters.
**Fix:** add `max_time`, `flow_min/max`, `veh_scale`, `params_file`, `reactive_peds` to the metrics dict, and skip only when the stored stamp equals the current one.

### 3.3 OSM demand-generation failure silently substitutes fixed-seed, mode-agnostic demand
`sim/benchmark_runner.py:865`

`except Exception: ... rou = <map>_base.rou.xml` falls back to the randomTrips file written once at import time with the import-time seed and no mode structure. The run still writes `robot_metrics.json` with no fallback marker and prints success, and `scenario.json` still records `osm_mode_flow_ph` as if the mode demand had applied. The row lands in the results table with a meaningless `mode` level and seed-invariant demand, undetectable by any aggregation step. (Trigger surface is narrower than reported — `osm_import.run()` `sys.exit`s on subprocess failure and those are re-raised — but `ET.ParseError` at osm_import.py:303 and write `OSError`s are swallowed.)
**Fix:** record `demand_fallback: <exception str>` in `robot_metrics.json` and either re-raise or have `benchmark_batch` drop flagged rows.

### 3.4 `--params-file` is silently ignored for sarl/cadrl/lstm_rl while metrics claim it was applied
`sim/benchmark_adapters.py:176-193`

The learning branches return without ever calling `apply_params` and without a warning, while `benchmark_runner` passes `params=_tuned_params` for every algorithm, documents the flag as "applied to every per-leg planner instance", and writes `"params_file": args.params_file` unconditionally. A sweep including a learning planner ships runs whose provenance is false. (Narrow exposure: `tune.py`'s `TUNABLE` excludes these three, and the README's templated `configs/{algo}.json` form makes the batch `sys.exit` on a missing file — the falsehood needs a non-templated shared params file.)
**Fix:** call `apply_params` on the learning adapters too, or print a warning and write `params_file: null` when the algorithm ignores it.

### 3.5 The implemented tuning objective is not the published one
`sim/tune.py:78`

Code: `0.0` on failure, else `1.0 - 0.15*t_norm - 0.05*soc`. README:176: "Objective: 0.8*success + 0.2*(1 − normalised time)". Different weights, a social term the README omits, and opposite rankings of a slow success against a fast failure. The module docstring matches the code, so the README is the wrong document — but a reader reproducing the tuning from it gets different best parameters for every planner.
**Fix:** correct README:176 to the implemented formula and emit the formula string into `configs/tuning_history/<algo>_trials.csv` metadata.

### 3.6 `--reverse` changes the route but not the output path and is not recorded
`sim/benchmark_batch.py:169` / `sim/benchmark_runner.py:716`

The runner genuinely reverses the itinerary but leaves `route_name`/`map_label` untouched and writes no `reverse` field, so a reversed run overwrites the forward run's `robot_metrics.json`, `robot_trace.csv` and `scenario.json` at the identical path. (The factor is recoverable post hoc from the `waypoints` list, which is stored reversed — but the forward run's data is gone.) Not used in any published protocol command.
**Fix:** `if args.reverse: mlabel += "__rev"` in both files, and add `"reverse": bool(args.reverse)` to the metrics dict.

### 3.7 Global-RRT provenance can disagree with what actually ran
- `RRT_MAX_ITERS` env var overrides the budget inside `_grow` (`benchmark_runner.py:272`) while metrics record `dict(GLOBAL_RRT_PARAMS)` — and `max_iters` decides whether a segment raises `GlobalPlanFailure`, i.e. which runs exist. **Fix:** apply the override into `GLOBAL_RRT_PARAMS` at startup next to the `--global-rrt-params` handling, or delete the hook.
- `--global-rrt-params` does a blind `dict.update` (`:723`), so a typo (`max_iter` for `max_iters`) is accepted, has no effect, and is written into the run's provenance implying a manipulation that never happened. **Fix:** `unknown = set(override) - set(GLOBAL_RRT_PARAMS); if unknown: sys.exit(...)`.
- The failure message hardcodes `"after 3 restarts"` (`:324`) while the count is the configurable `restarts`, and that string is stored verbatim as `global_plan_error`. **Fix:** interpolate `GLOBAL_RRT_PARAMS['restarts']`.

### 3.8 Committed task lists are not the equal-count stratified sample the README claims
`sim/sample_tasks.py:172`

The loop terminates on the **total** count with only an upper per-bin cap, then truncates. Recomputed from the committed artefacts' own `path_length_m` and `length_bins_m`: map1 `[2,4,4]`, map2 `[4,4,2]`, map3 `[4,4,2]`, map4 `[4,4,2]`, map5 `[4,3,3]` — while every file records `per_bin_target: 4` and the `screening` dict reports no imbalance. `stats_models.py` enters `task_path_length_m` as a standardized fixed effect, so the length stratum is an unbalanced factor nobody declared. (With `n_tasks=10, bins=3` exact equality is impossible; the achievable optimum is `[4,3,3]`, which map5 reaches — so four of five maps are one task off optimum, not 2× under-represented.)
**Fix:** `while any(len(b) < per_bin for b in bins) and attempts < max_attempts`, record the achieved counts in the output JSON, and warn when any bin is under target. Regenerate the five task files or amend the README.

### 3.9 Two demand-sampling couplings that break cross-batch comparability
- **Conditional RNG draws shift the stream** (`benchmark_runner.py:826/835`): supplying `--ped-period-min/max` inserts one `rng.uniform` before `osm_flow` and `osm_statics`, so the same seed yields a different crowd. `--ped-period` alone does *not* consume the RNG, so the two forms diverge at the same seed for reasons unrelated to the period. Within one batch the flags are constant, so the protocol's blocking property holds; it only bites someone pooling across batches. **Fix:** draw all sampled quantities unconditionally and discard unused values.
- **`--max-time` changes the sampled OSM crowd** (`benchmark_runner.py:852`, `end=args.max_time + 100.0`): `osm_mode_demand` uses `end` both to size `n` and to scale `t = rng.uniform(0, end)`, so the pilot at `--max-time 420` and the protocol at 3000 have different pedestrian streams from the same seed. The built-in maps are immune (`--end` is never passed). The rate is invariant, so these are different realisations rather than different scenarios — but the README's two examples are not the comparison a reader expects. **Fix:** use a fixed protocol horizon independent of `--max-time` and record it in `scenario.json`.

### 3.10 Silently reclassified crossings are recorded nowhere
`sim/native_signal_gate.py:42`

`skipped_never_green` is computed and never read anywhere in the repo — not in `robot_metrics.json`, not in `scenario.json`. Verified against the shipped data: map3 16/16 and map4 32/32 kept, but **7 of map5_ucl's 61 signalised crossings are dropped** and treated as unsignalised, with the operator given no signal. The runner then re-implements the identical filter at `:892-905` over an already-filtered list, so `_served` always equals `gate.crossings` and its diagnostic `print` is unreachable dead code.
**Fix:** emit `gate.skipped_never_green` and the dropped ids into `scenario.json`; delete the dead re-filter.

---

## 4. Performance

*None of these change a number. Ranked by recoverable CPU-hours across the ~10,500-run protocol.*

### 4.1 The walkable graph is rebuilt from scratch on every run — ~45–55 s each
`sim/benchmark_runner.py:453`

`for g in warea:` buffers and `prep()`s every walkingarea, then scans **every** piece endpoint: `members = [n for n in ends if gp.covers(Point(nodes[n]))]`. On map5_ucl that is 1254 walkingareas × 3030 endpoints ≈ 3.8 M prepared `covers` calls; cProfile attributes 44.7 of the 44.9 s build to this loop (the XML parse is 0.2 s). Measured end-to-end per task: 16–49 s. `build_walk_graph`/`_graph` exists to amortise this, but `main()` calls `auto_route` at :752 with `_graph=None` and every run is a separate subprocess — and the `--global-planner fixed` OSM path pays it too (`:852` calls `auto_route` for demand route edges). **~160 CPU-hours across the protocol** for a purely static, deterministic artefact.

Three related leaks in the same function:
- **The cache is defeated even when supplied** (`:383`): `pieces, warea` unpacked from `_graph` are overwritten by `[], []` and the `.net.xml` is re-parsed with ~1254 shapely `Polygon`/`buffer(0)` reconstructions on every cached call (~0.17 s). `sample_tasks.py` pays this up to `--max-attempts` (400) times per map.
- **The same net is parsed 3–4× per process** (`:384` auto_route, `:640` load_walkable, `social_pedestrians.py:89` `_build_zone`, `osm_import.py:279` statics) with no shared cache — ~1 s/run.
- `attach()` (`:483`) scans every segment of every piece with no index, then resolves endpoints by two linear scans over 3030 nodes; `_rrt_route`'s edge attribution (`:337`) is O(waypoints × total piece points). Both are milliseconds per run — fix them opportunistically, not first.

**Fix (both halves):**
```python
from shapely import STRtree
tree = STRtree([Point(nodes[n]) for n in ends])
members = [ends[i] for i in tree.query(gb)]   # replaces the O(W×E) scan
```
and pickle `(pieces, warea, nodes, adj, ends)` to `maps/<map>/<map>.walkgraph.pkl` keyed on the net file's mtime+size, loaded by `main()` — turning 45 s into a ~0.3 s unpickle. Also guard the reparse behind `if not _skip_build:` (preserving the bbox filter in memory) and add a module-level `@lru_cache` `_parse_net(net_file)` shared by all four consumers.

### 4.2 Learning planners run 81 separate forward passes per control step
`sarl_sumo_robot_unified.py:484`, `cadrl_…:226`, `lstm_rl_…:373`

All three loop over the 81-action space and, per action, rebuild an action-*independent* propagated-human list, materialise a fresh `torch.tensor`, rotate, and run a batch-of-1 forward with a `.item()` host sync. All three networks accept an arbitrary leading batch dimension. Measured on CPU: SARL 118–124 ms/step, LSTM-RL 110–114 ms/step, CADRL 71–80 ms/step — pure Python/dispatch overhead on tensors of ≤ 5 rows. **A verifier implemented the batched SARL variant: identical values (max abs diff 1.9e-6) at 3.7 ms/step, a ~32× speedup.** At the shipped default `--max-time 900` that is ~200 s → ~7 s of inference per run, across ~4,500 runs.
**Fix:** hoist the human propagation out of the loop; build one `(81*N, 14)` raw tensor, rotate once, `reshape(81, N, 13)`, one forward. **Important:** SARL must keep the `(81, N, ·)` shape so its attention softmax stays per-scene (and `self.model.attention_weights` must be indexed, not overwritten); the `min`-over-humans reduction is correct only for CADRL. Also truncate `humans` to `cfg.max_humans` in `SarlPolicy.predict` — the cap at :634 lives in a function the adapter never calls — and `torch.set_num_threads(1)` at construction (81 tiny GEMMs oversubscribe a 16-thread pool).

### 4.3 Global-RRT nearest-neighbour scans O(r²) cells per ring
`sim/benchmark_runner.py:245`

`_nearest` expands a square ring but iterates the full `(2r+1)²` square and discards everything with `max(|dx|,|dy|) != r`, so reaching radius R costs ~(4/3)R³ dict probes instead of ~4R². With `CELL=14.0` and a sampler that draws 70% of samples on centrelines anywhere in a >1 km map, far queries are routine: measured 969 inner iterations at 100 m, 16,215 at 300 m, 113,564 at 600 m, 877,975 at 1200 m (19–56 ms per query). Runs once per iteration for up to `max_iters=40000` × `restarts=3`; a failing map5_ucl segment burned 47.8 s. Fixing it produces **bit-identical routes** (the iteration budget is fixed) — it only makes them resolve faster.
**Fix:** iterate the perimeter only —
```python
ring = ([(d, -r) for d in range(-r, r+1)] + [(d, r) for d in range(-r, r+1)]
        + [(-r, d) for d in range(-r+1, r)] + [(r, d) for d in range(-r+1, r)])
```
— measured 19× on the worst case. Better: `scipy.spatial.cKDTree` rebuilt every ~256 insertions plus a linear tail scan; and bound `r` by the occupied-cell bounding box so an out-of-extent query terminates in one step instead of growing to r=300.

### 4.4 TraCI is polled with O(N) blocking round-trips per step; no subscriptions anywhere
`sim/benchmark_runner.py:1068`, `sim/social_pedestrians.py:221`, `sim/native_signal_gate.py:48`

Per control step the runner issues `getIDList` + `getPosition` for **every** person in the network + `getSpeed`/`getAngle` for those in range; `NativeSignalGate.step` adds one `getRedYellowGreenState` per TLS (17 on map5_ucl, unconditionally, whether or not a signal is near); and under `--reactive-peds sfm` the layer independently re-fetches `getPosition` for the same population (plus `getRoadID`/`getStage` per candidate and `moveToXY` per controlled pedestrian), roughly doubling the per-person polls. `grep subscribe` over `sim/` and `analysis/` returns **zero hits**. At ~200 round-trips/step × 6000 steps that is ~1.2 M synchronous socket exchanges per run; at a conservative 20–50 µs each, ~25–60 s per run.
**Fix:** one context subscription in `benchmark_runner`, shared with the SFM layer:
```python
traci.person.subscribeContext("robot0", tc.CMD_GET_PERSON_VARIABLE,
                              args.sensor_range,
                              [tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ANGLE])
```
plus a plain `person.subscribe` for the global min-distance and `trafficlight.subscribe` for the TLS states, read once per step from `getAllSubscriptionResults()`. Pass the resulting `{pid: (x,y,speed,angle)}` dict into `sfm.step(...)` so pedestrian state is fetched **once per step in total**. This is metric-equivalent (same values, same instant — the SFM's `moveToXY` calls happen after the sweep).

### 4.5 DWA's rollout and obstacle cost are pure-Python triple loops
`sim/planners/dwa_sidewalk_robot_random_stop_collision.py:155`

`calc_obstacle_cost` loops rows × obstacles calling `math.hypot`, once per (v,w) candidate; `predict_trajectory` builds a Python list and calls `np.array` per candidate (~336 small allocations/step); the sidewalk test at :202 iterates numpy rows in a generator; and :182 allocates an unused default `best_traj` every call. Measured 20.6 / 32.6 / 38.3 ms per `dwa_control` at 10 / 30 / 60 obstacles, every step (no replan interval). ~140 s of planner CPU per 6000-step episode — the single largest planner cost, in the protocol's reference algorithm.
**Fix:** stack the candidate set into `(n_v, n_w, T, 2)` with numpy broadcasting; `d = np.hypot(P[...,None,0]-ox, P[...,None,1]-oy)`; replace the :202 generator with `((traj[:,0] < x_min+m) | ...).any()`. Note the 20 ms floor at 10 obstacles means all three must be vectorised, not just the obstacle loop. Do the obstacle time-propagation (see 5.6) in the same pass at no extra cost.

### 4.6 A*/Dijkstra re-scan every pedestrian for every edge expansion
`astar_…:59`, `dijkstra_…:62`

`is_blocked` loops all obstacles per candidate cell and is called once per *edge* (up to 8× per cell), with no occupancy grid, no memoisation, and `idx_to_xy(current)` recomputed inside the neighbour loop. Measured on a 250 m × 2 m band at 10 obstacles: A* 28.6 ms, Dijkstra 45.3 ms per replan, every second control step (~85–135 s per run). Amplified on failure: `or not self.path` makes a failed search re-run every step while the corridor stays blocked. (The dense case is *cheaper*, not worse — 30 discs seal the corridor and the frontier exhausts immediately, which is its own problem, see 1.5.)
**Fix:** stamp the clearance discs onto the grid once per `plan()` (`~9 cells × n_obs ≈ 270 ops`) and make `is_blocked` a `idx in blocked` set lookup. Hoist `idx_to_xy(current)`. Add a failure cooldown so a failed search does not re-plan every step.

### 4.7 RRT re-runs its full 700-iteration search every control step
`sim/planners/rrt_sidewalk_robot_random_stop_collision.py:143`

Because `plan()` never succeeded (1.2), `self.path` was permanently `[]`, so `or not self.path` fired every step and the 1.0 s replan interval was dead — 59.5 ms of discarded tree search per step. Mostly resolved by the `LEG_GOAL_MARGIN` fix; the residual defects are the missing failure cooldown and `nearest_index`'s O(n) linear scan (:77-85), which is O(max_iter²/2) ≈ 245k distance evaluations per call.
**Fix:** cooldown instead of `or not self.path`; vectorised `argmin` over stacked node coordinates; bound the initial `segment_safe(start, goal)` check to sensor range rather than the whole leg.

### 4.8 The sweep runs strictly serially while the protocol calculator assumes 32 workers
`sim/benchmark_batch.py:200`

`res = subprocess.run(cmd)` blocks per run, with no pool anywhere in the repo (`grep multiprocessing|concurrent.futures|joblib|n_jobs` → nothing), while `analysis/factor_table.py:24` computes `wall_days = cpu_days / 32` and prints "~1.4 wall-days at 32 parallel workers" for what is actually 43.75 single-core days. Runs write to disjoint directories, so nothing blocks concurrency — and the sweep *is* shardable today via `--maps`/`--seeds`/`--algorithms` plus `--skip-existing`. This is a tooling/documentation mismatch, not a bug.
**Fix:** add `--jobs` and a `ThreadPoolExecutor` around the `subprocess.run` calls (threads suffice — the work is in children), consuming `as_completed`, with the aggregation after all futures.

### 4.9 Smaller performance items
| Item | Location | Cost / note |
|---|---|---|
| SFM force loop is O(\|ctl\|×N) over the whole network; `near_states` stores every person unconditionally; `near_static` is O(N×S) | `social_pedestrians.py:223, 322` | Bin `pos_all` into `NEIGH_R` cells; restrict `near_states` to `CAPTURE_R + NEIGH_R`. Falls out of the 4.4 subscription fix. |
| Unprepared `nearest_points(self.union, ...)` fallback and once-per-step unprepared `boundary.distance` | `social_pedestrians.py:388, 425` | Cache a prepared boundary in `__init__`; STRtree over union components. (The `__import__`/in-function imports at :163/:386 are ~1 µs dict lookups — cosmetic.) |
| Strip figure fully built, `tight_layout`ed, then discarded whenever ≥3 seeds (the normal case); traces re-parsed by `envelope_figure` | `benchmark_plots.py:498, 515` | Hoist the `len(items) >= 3` test above figure construction; pass parsed traces into `envelope_figure`. |
| `_project_traj` loops per trace sample; `plot_strip_trace` transforms every row `2 × n_legs` times | `benchmark_plots.py:275, 197` | Broadcast the projection to an `(N, nseg)` matrix; bucket rows by leg once and call `to_local` once per row. Minutes, plotting-stage only. |
| Aggregation re-filters all rows per cell — O(cells × runs) with a function call per comparison; every metrics file read twice | `benchmark_batch.py:256, 216` | `defaultdict(list)` single-pass grouping (11 M `_cell()` calls → 10.5 k). Keep the re-glob (intentionally incremental), drop the in-loop `rows.append`. |
| cadrl/lstm_rl re-read their `.pth` and rebuild networks on every leg switch; MPC/TEB warm starts discarded; RRT reseeds identically per leg | `benchmark_adapters.py:179, 187` | ~3 ms/leg at shipped checkpoint sizes. Cache like `_SARL_CACHE`; add a `retarget(cfg)` so per-leg geometry changes do not destroy warm state. |
| Signal gate rebuilds a dict per crossing per step; `crossing_ahead` projects all zones with no distance prefilter | `native_signal_gate.py:50`, `benchmark_runner.py:685` | Persistent zone dicts mutated in place; cheap `hypot` prefilter before the four `to_local` calls. |

---

## 5. Robustness / lower priority

### 5.1 Reported velocity is the *commanded* value, so a physically frozen robot never trips the watchdog
`sim/benchmark_runner.py:1148`

`vx, vy = wvx, wvy` stores the command after `path_len` has already accumulated the *realized* displacement. The lateral clamp (:1133) and the strict-sidewalk boundary snap (:1142) can zero the realized motion while the command stays at 1.0 m/s, so `frozen_since` resets every step and the 45 s `stalled` watchdog can never fire — the episode burns the full `--max-time` and is recorded as `max_time` instead of `stalled`. `robot_trace.csv`'s `vx`/`vy` columns also do not match the finite differences of its own `x`/`y`.
**Fix:** `vx, vy = (nx - x)/dt, (ny - y)/dt` before `x, y = nx, ny`, and drive the watchdog off that.

### 5.2 Per-run demand file is never deleted (absolute path compared to a relative one)
`sim/benchmark_runner.py:1241`

`rou` is built with `.resolve()` while `run_dir` derives from the relative default `--out-root results`, so `rou.parent == run_dir` is never true and the cleanup block is dead. Every run leaves `demand.rou.xml` + `demand.rou.scenario.json` on disk, and `--keep-demand` is a no-op.
**Fix:** `rou.parent == run_dir.resolve()`.

### 5.3 The collision/planner-error step is missing from `robot_trace.csv`
`sim/benchmark_runner.py:1083` / `:1102`

Both break before `rows.append` at :1160, while `metrics["sim_time_s"]` records the current `t`. For collision and `planner_error` runs the final trace row is one control step (up to 0.8 m) before the recorded termination time; for goal/stalled/max_time it is at it. Any per-run duration or terminal pose reconstructed from the trace is inconsistent between termination reasons. (`path_length_m` is correct — that step never happened.)
**Fix:** append the row before breaking.

### 5.4 `--tasks` matching nothing on a map drops that map and blames routes
`sim/benchmark_batch.py:122`

`map_tasks[mp]` becomes `[]`, the comprehension yields zero combos for that map, and the only diagnostic is "(skipping N combos: route not defined on that map)". The runner does this correctly (`benchmark_runner.py:735-737` `sys.exit`s with the available ids).
**Fix:** validate `map_tasks[mp]` after building it and exit with the available ids. Separately, the "skipping N combos" count itself (`:132-135`) omits the task dimension from both sides and is understated by the per-map task count — cosmetic, but misleads compute budgeting.

### 5.5 Tuning leaks temp directories and can hang forever
`sim/tune.py:56`, `:187`, `:68` (and `benchmark_batch.py:200`)

`tempfile.mkdtemp()` per episode (receiving the full run output) and per trial, never removed on any exit path — `shutil` is not even imported. Neither `subprocess.run` passes `timeout=`, so one wedged SUMO stalls a multi-day sweep, and in `tune.py` a hang loses the entire study because `trials_dataframe()` is written only after `study.optimize` returns.
**Fix:** `with tempfile.TemporaryDirectory() as td:` around both sites; dump `study.trials_dataframe()` in a `finally` block. A blanket timeout risks killing legitimately slow 3000 s runs — use `timeout=max(600, 4*max_time)` and convert `TimeoutExpired` into a recorded failure / `TrialPruned`.

### 5.6 DWA scores a 3 s rollout against a frozen obstacle snapshot
`sim/planners/dwa_sidewalk_robot_random_stop_collision.py:169`

`Obstacle.vx/vy` are populated but never used: `d = hypot(px - obs.x, py - obs.y)`. Reproduced: bit-identical output whether a pedestrian 3 m ahead is static or closing at 1.4 m/s — which covers 4.2 m over the horizon. MPC (:127-128) and TEB (:108-109) both propagate, so the DWA/MPC collision-rate difference partly measures this gap. **Documented as intentional** (docstring at :158-159, and faithful to textbook DWA), so this belongs in the method section rather than the bug list — but propagate anyway as part of the 4.5 vectorisation, where it is free.

### 5.7 Legacy standalone-runner defects (do not affect the ~10,500-run protocol)
`sim/planners/sidewalk_robot_common.py` + `dwa_…` standalone `main()`. `benchmark_runner` bypasses all of this; these matter only if anyone pools the standalone `robot_metrics.json` outputs.

- **Different collision/hazard thresholds** — dwa:743 uses `0.40`, common:469 uses `max(safe_distance, radii) = 0.42`; `hazard_steps` 0.80 vs 0.85, `social_steps` 12.0 vs 11.0, `goal_tolerance` 0.25 vs 0.35. The two metrics families are on different scales. **Fix:** have the DWA script reuse `sidewalk_robot_common.compute_metrics`.
- **Different `--flow-mode` default** — dwa:843 `probability` vs common:397 `personsPerHour`, so `--seed 7` builds a different scenario for DWA than for the other six; any per-seed paired test across standalone scripts is invalid. **Fix:** call `add_common_arguments` from the DWA script.
- **`apply_velocity` allows instantaneous stops** — `common:104`: for a zero command, `scale = 0.0` and `speed` is recomputed *after* scaling, so the robot halts from 1.0 m/s in one step (−2.0 m/s², 4× the 0.5 limit) while acceleration is limited. ORCA emits `(0,0)` as an explicit candidate. A 0.001 m/s change in the command changes the step displacement by 0.375 m. **Fix:** assign `speed = limited` and carry the heading: `vx, vy = limited*cos(state.yaw), limited*sin(state.yaw)`.
- **Yaw limit applied after position integration** — `common:109`: position uses the raw commanded vector, then yaw is clamped, so the reported `yaw`/`w` describe a different motion than the trace's own `x`/`y`, and `mean_abs_yaw_rate_radps` is hard-censored at `max_yaw_rate` by construction.
- **`min_person_dist` pairs the post-step robot with pre-step pedestrians** — `common:498`; bias bounded by `ped_speed × dt ≈ 0.8 m`.
- **Outputs written outside the `try/finally`** — `common:535`: any exception discards the entire run including already-collected rows. `benchmark_runner` guards its planner call (`reason = "planner_error:..."`); the standalone path does not.
- **Relative route path built before `os.chdir`** — `common:448`: `cfg_path` is `.resolve()`d at :433 precisely to survive the chdir, the route file is not, so SUMO resolves it against the `.sumocfg` parent. Loud failure in practice. **Fix:** `.resolve()` it too.
- **No per-seed output directory without `--random-scenario`** — `common:168`: every seed overwrites the same `robot_trace.csv`/`robot_metrics.json`, and the metrics dict has no `seed` field; with `--seed` omitted a fresh random seed is invented, passed to the planner but not to SUMO, and recorded nowhere. **Fix:** always create `seed_<k>/` and add `metrics["seed"] = seed`.

### 5.8 Documentation and latent-data issues
- **`departPos` docstring is false** — `generate_demand.py:7` claims "random along the first edge"; `walk_flow:92` uses a fixed 0.3 m offset from one road end, so the corridor starts empty and fills over one traversal time with no warm-up. The false claim propagated into `osm_import.py:248-251`'s design rationale for the (defective) ramp in 2.6. Arrival processes also differ between map families (Poisson `exp(rate)` vs a fixed uniform count).
- **Vehicle flows hardcode `begin="0.00"`** — `generate_demand.py:171` ignores `--begin` while pedestrian flows honour it. Latent: nothing in the tree passes `--begin`.
- **`getAllProgramLogics(t)[0]` assumes the active program** — `native_signal_gate.py:30`. `getPhase()` indices from the *active* program are then applied to program 0's phase table, and the never-green filter evaluates the wrong table. Every shipped net has exactly one `<tlLogic>` per TLS, so this is latent. **Fix:** `next((p for p in progs if p.programID == getProgram(t)), progs[0])`.
- **`validate_reactive.py:135` reports the upper middle element as the "median"** and emits bare `NaN` on an empty sample, which `json.dumps` writes as invalid JSON — now inconsistent with the runner, which was just fixed to emit `None`. **Fix:** `statistics.median`, `None` instead of NaN, and report the sample counts.
- **Duplicate `global_plan_time_s` key** in the plan-failure dict — **[FIXED IN TREE]**.
- **`time_waiting_at_light_s` undercount when a pedestrian independently stops the robot at a red** — the narrow residual of 2.4; same one-line fix.

### 5.9 Unverified findings
- **Pedestrians on internal/empty road ids can never be released** — `sim/social_pedestrians.py:212`. **Unverified — needs a run to confirm.** The code facts hold: the release test requires `not self._on_internal(pid)`, which returns True for a road id starting with `":"`, equal to `""`, *or* on any `TraCIException`; there is no timeout or forced drop; `_build_zone` deliberately leaves crossings capturable; and `moveToXY(..., keepRoute=2)` places persons without re-mapping onto the route. If SUMO persistently reports an internal id for such a person, it is driven on its frozen capture-time heading for the rest of the episode with striping disabled — blocking the crossing for everyone else. Whether `getRoadID` actually behaves that way, and whether the walking-stage sentinel eventually fires, requires running SUMO. **Suggested guard regardless:** record `st["t_far"]` when a pedestrian first exceeds `RELEASE_R` and force-drop it from `self.ctl` after a few seconds, and refresh `st["edir"]` from `getAngle` periodically.
- **The tuning objective's social term may be pinned at its cap** — `sim/tune.py:79`. **Unverified — needs a run to confirm.** `soc = min(ped_personal_space_s_total / 60.0, 1.0)` divides an unbounded cross-pedestrian sum by a fixed 60 s. Whether it saturates depends on realised crowd density: `PS_R` is only 1.2 m and the default flows put ~13–58 pedestrians past the robot in a 600 s episode, so saturation is plausible only in the densest cells. Even at full saturation it shifts every score by the same 0.05 constant and changes no ranking — it removes signal rather than corrupting results. **Fix regardless:** normalise per affected pedestrian or per second, and log both raw and normalised values into the trial history so saturation is visible.

---

## Fix these first

1. **Re-run everything the two working-tree fixes invalidate.** The `LEG_GOAL_MARGIN` fix (1.2) means every `rrt` local-planner row and `configs/rrt.json` currently in `results/` measures a centreline creeper, not RRT; the `sfm` fix (1.1) means every prior RRT *global*-planner failure was dropped from the denominator rather than recorded. Regenerate both cells, plus `ranking_stability.*` and `failure_taxonomy.*` (2.15, 2.23).
2. **Propagate the full kinematic envelope into `DWAConfig`** — `sim/benchmark_adapters.py:66`. A one-block change removes a flat ~5% speed handicap from every time- and speed-based comparison in the thesis, in the algorithm that is also the statistical reference level (2.1). Decide the acceleration-limit question (2.11) in the same commit.
3. **Stop deriving the sidewalk band width from lane bounding boxes** — `sim/benchmark_runner.py:588`. Silently corrupts `path_length_m`, `avg_speed_mps` and `time_waiting_at_light_s` on affected legs and drags every centreline-seeking planner off the walkable surface (2.2). Add the missing lower bound at the same time.
4. **Fix the measurement half-step desync and the light-hold flag** — `sim/benchmark_runner.py:1069`, `:1159`, `:1121`. These three edits together correct `min_pedestrian_distance_m`, `close_encounter_steps`, the binary `collision` outcome driving the success GLMM, `time_waiting_at_light_s`, and the spurious `stalled` terminations (2.3, 2.4). Apply the same reordering to `validate_reactive.py:128`.
5. **Replace `hash(pid)` with a stable seeded hash** — `sim/social_pedestrians.py:308`. Four lines. Without it the main protocol's published seeds do not reproduce and seed-matched paired comparisons are invalid (3.1).
6. **Resolve the `same`/`opposite` semantics inversion before pooling `C(mode)`** — `sim/generate_demand.py:126`. On map3 and map4 vertical legs the factor level means the opposite of what it means elsewhere, so the published mode main effect blends two opposite treatments (2.5). Either fix the generator or stop pooling across map families.

*Highest-value performance work, once correctness is settled: the walkable-graph STRtree + on-disk cache (4.1, ~160 CPU-hours) and the batched learning-planner forward pass (4.2, ~32× measured).*