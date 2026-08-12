#!/usr/bin/env python3
"""Nonlinear MPC for the SUMO sidewalk robot, solved by the PUBLISHED do-mpc / CasADi stack.

Provenance of the optimisation code
-----------------------------------
Nothing in this file implements an optimiser.  The NLP is transcribed and solved by:

    do-mpc   5.1.1   (do-mpc/do-mpc, S. Lucia / F. Fiedler, TU Dortmund / TU Berlin)
                     pip package ``do-mpc``, LGPL-3.0.
                     ``do_mpc.model.Model(model_type='discrete')`` +
                     ``do_mpc.controller.MPC`` -> multiple-shooting NLP
                     (states AND inputs are decision variables, the model
                     equation enters as an equality constraint per stage).
    CasADi   3.7.2   (casadi/casadi, Andersson et al. 2019), LGPL-3.0.
                     Symbolic AD + ``nlpsol``.
    Ipopt    (shipped inside the CasADi wheel), EPL-2.0.  Interior-point NLP solver.

Both are installed as ordinary pip dependencies (`pip install do-mpc`, which pulls
casadi), i.e. the upstream published packages, unmodified.  No vendoring and no
patch was needed on Windows / CPython 3.13.

Relation to ``mpc_sidewalk_robot_random_stop_collision.py`` (the in-repo hand-rolled MPC)
----------------------------------------------------------------------------------------
SAME control problem:
  * single-integrator (holonomic velocity) robot,  p_{k+1} = p_k + u_k * dt
  * receding horizon, first input applied, replan every step
  * constant-velocity prediction of pedestrians over the horizon
  * the same objective terms and the same weights (w_terminal_goal, w_goal_path,
    w_speed_ref, w_lateral, w_centerline, w_control_smooth, w_social) and the same
  * input box (vx in [0, v_max], vy in [-0.75 v_max, 0.75 v_max]).

DIFFERENT (this is the point of the exercise) -- everything the hand-rolled version
expressed as a *penalty on a single-shooting rollout under L-BFGS-B* is now an
explicit constraint of a real NLP:
  * multiple shooting: p_0..p_N and u_0..u_{N-1} are all decision variables and
    the dynamics are equality constraints (the hand-rolled one substituted the
    rollout into the cost -> single shooting, N inputs only).
  * sidewalk band  y in [y_min+margin, y_max-margin]  and  x in [x_min, x_max]
    are HARD state bounds (were a quadratic penalty, w_boundary).
  * speed limit  ||u_k||^2 <= v_max^2  is a HARD nonlinear constraint
    (was a quadratic penalty above v_max plus a post-hoc rescale of u_0).
  * acceleration limit  | ||u_k|| - ||u_{k-1}|| | <= a_max*dt  is a HARD nonlinear
    constraint, with u_{-1} carried in the state so the limit also binds on the
    first applied input (the hand-rolled MPC had NO acceleration limit at all --
    it only got one afterwards, from the plant in ``apply_velocity``).
  * collision avoidance  ||p_k - o_j(k)||^2 >= clearance^2  is a nonlinear
    constraint, softened by do-mpc's own slack mechanism (``soft_constraint=True``)
    with a bounded maximum violation, so the NLP stays feasible when a pedestrian
    is already inside the clearance disc at k=0 (was a piecewise quadratic penalty).
  * ``w_backward`` is gone: it is implied by the hard input bound vx >= 0.

Interface
---------
Standard benchmark planner interface:
    DoMPCPlanner(cfg: PlannerConfig, seed: int)
    compute_command(state, goal, obstacles, sim_time) -> (vx, vy, info)
in the LEG-LOCAL frame (sidewalk along +x, y in [0, band_width]).

``info`` additionally carries ``solve_ms`` (wall time of the NLP solve) and
``iters`` so the per-step cost is measurable from the trace.
"""
from __future__ import annotations

import argparse
import math
import time
from typing import List, Sequence, Tuple

import numpy as np

from sidewalk_robot_common import (
    Obstacle,
    PlannerConfig,
    RobotState,
    add_common_arguments,
    run_traci_with_planner,
)

# do-mpc emits a UserWarning about the optional OPC-UA extra at import time.
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    import casadi as ca
    import do_mpc


# Far-away parking spot for unused obstacle slots (metres).  Keeps every slot's
# constraint trivially satisfied and its social cost exactly zero without making
# the numbers large enough to hurt the Jacobian conditioning.
_PARKED = 1.0e3


class DoMPCPlanner:
    """Receding-horizon NLP-MPC.  Transcription + solve by do-mpc / CasADi / Ipopt."""

    def __init__(self, cfg: PlannerConfig, seed: int,
                 horizon: int = 8, n_obstacle_slots: int = 12,
                 collision_slack_penalty: float = 2.0e4):
        self.cfg = cfg
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.n_obs = int(n_obstacle_slots)

        self.margin = 0.06
        self.clearance = max(cfg.safe_distance,
                             cfg.robot_radius + cfg.pedestrian_radius + 0.05)
        self.social_clearance = max(cfg.social_distance, self.clearance + 0.15)

        # ---- weights: identical to the in-repo hand-rolled MPC -------------
        self.w_terminal_goal = 9.0
        self.w_goal_path = 0.08
        self.w_social = 0.8
        self.w_speed_ref = 0.45
        self.w_control_smooth = 0.55
        self.w_centerline = 0.03
        self.w_lateral = 0.08
        # penalty on do-mpc's collision slack (replaces w_obstacle*100 of the
        # hand-rolled penalty, now attached to a real constraint violation)
        # Penalty on do-mpc's collision slack.  MUST be given to the
        # constructor: it is baked into the NLP by _build(), so assigning the
        # attribute afterwards (e.g. via benchmark_adapters.apply_params) has
        # no effect.  Higher = the robot would rather stop than squeeze past;
        # 1e5 gives no clearance violation even in a fully blocked band but
        # costs ~2x the solve time (measured).
        self.w_collision_slack = float(collision_slack_penalty)

        # A first input carrying less than stall_vx*v_max of forward speed is
        # treated as "the NLP stalled" and triggers the multi-start (below).
        self.stall_vx = 0.25

        # previously APPLIED velocity, used as u_{-1} for the acceleration bound
        self._u_prev = np.zeros(2, dtype=float)
        self._solves = 0
        self._nlp_calls = 0
        self._fail = 0
        self._t_total = 0.0

        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        cfg = self.cfg
        dt = float(cfg.dt)

        model = do_mpc.model.Model("discrete")

        pos = model.set_variable("_x", "pos", shape=(2, 1))       # p_k
        u_prev = model.set_variable("_x", "u_prev", shape=(2, 1))  # u_{k-1}
        u = model.set_variable("_u", "u", shape=(2, 1))            # u_k

        goal = model.set_variable("_tvp", "goal", shape=(2, 1))
        obs = model.set_variable("_tvp", "obs", shape=(2 * self.n_obs, 1))

        # multiple-shooting dynamics (enters the NLP as an equality constraint)
        model.set_rhs("pos", pos + u * dt)
        model.set_rhs("u_prev", u)

        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = self.horizon
        mpc.settings.t_step = dt
        mpc.settings.n_robust = 0
        mpc.settings.store_full_solution = False
        mpc.settings.store_lagr_multiplier = False
        mpc.settings.store_solver_stats = ["success", "iter_count", "t_wall_total"]
        mpc.settings.nlpsol_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 80,
            # adaptive barrier update: measured ~15% faster per solve than the
            # monotone default on this problem, same trajectories.
            "ipopt.mu_strategy": "adaptive",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-6,
            "ipopt.warm_start_mult_bound_push": 1e-6,
            "ipopt.mu_init": 1e-3,
        }

        eps = 1e-9
        gx, gy = goal[0], goal[1]
        px, py = pos[0], pos[1]
        goal_sq = (gx - px) ** 2 + (gy - py) ** 2

        speed = ca.sqrt(u[0] ** 2 + u[1] ** 2 + eps)
        soc_sq = self.social_clearance ** 2

        lterm = (self.w_goal_path * goal_sq
                 + self.w_speed_ref * (speed - 0.85 * cfg.max_speed) ** 2
                 + self.w_lateral * u[1] ** 2
                 + self.w_centerline * (py - cfg.sidewalk_center_y) ** 2)
        for j in range(self.n_obs):
            d_sq = (px - obs[2 * j]) ** 2 + (py - obs[2 * j + 1]) ** 2
            # smooth-enough hinge: fmax(0, .)^2 is C1
            lterm = lterm + self.w_social * ca.fmax(0.0, soc_sq - d_sq) ** 2

        mterm = self.w_terminal_goal * goal_sq
        mpc.set_objective(mterm=mterm, lterm=lterm)
        # Delta-u penalty == the hand-rolled w_control_smooth term
        mpc.set_rterm(u=np.array([self.w_control_smooth, self.w_control_smooth]))

        # ---- HARD state bounds: the sidewalk band -------------------------
        # The plant clamps the pose to [min+0.03, max-0.03], and the benchmark
        # runner to [0.02, len] x [0.06, W-0.06], so x_0 always satisfies these.
        m = self.margin
        band_lo = cfg.sidewalk_y_min + m
        band_hi = cfg.sidewalk_y_max - m
        x_lo = cfg.sidewalk_x_min
        x_hi = cfg.sidewalk_x_max
        mpc.bounds["lower", "_x", "pos"] = np.array([[x_lo], [band_lo]])
        mpc.bounds["upper", "_x", "pos"] = np.array([[x_hi], [band_hi]])

        # ---- HARD input bounds -------------------------------------------
        mpc.bounds["lower", "_u", "u"] = np.array([[0.0], [-0.75 * cfg.max_speed]])
        mpc.bounds["upper", "_u", "u"] = np.array([[cfg.max_speed], [0.75 * cfg.max_speed]])

        # ---- HARD nonlinear constraints: speed magnitude and acceleration --
        mpc.set_nl_cons("speed_limit",
                        u[0] ** 2 + u[1] ** 2 - cfg.max_speed ** 2, ub=0.0)
        # ||u_k - u_{k-1}|| <= a_max*dt.  Convex (a disc), and by the reverse
        # triangle inequality it implies | ||u_k|| - ||u_{k-1}|| | <= a_max*dt,
        # which is exactly the limiter the plant applies in apply_velocity --
        # so every trajectory the NLP plans is realisable by the simulator.
        dv = cfg.max_accel * dt
        mpc.set_nl_cons("accel_limit",
                        (u[0] - u_prev[0]) ** 2 + (u[1] - u_prev[1]) ** 2 - dv ** 2,
                        ub=0.0)

        # ---- collision avoidance: nonlinear constraint, do-mpc slack -------
        cl_sq = self.clearance ** 2
        for j in range(self.n_obs):
            d_sq = (px - obs[2 * j]) ** 2 + (py - obs[2 * j + 1]) ** 2
            mpc.set_nl_cons(f"collision_{j}", cl_sq - d_sq, ub=0.0,
                            soft_constraint=True,
                            penalty_term_cons=self.w_collision_slack,
                            maximum_violation=cl_sq)

        # ---- time-varying parameters (goal + predicted pedestrians) --------
        self._tvp_tmpl = mpc.get_tvp_template()
        for k in range(self.horizon + 1):
            self._tvp_tmpl["_tvp", k, "goal"] = np.zeros((2, 1))
            self._tvp_tmpl["_tvp", k, "obs"] = np.full((2 * self.n_obs, 1), _PARKED)

        def tvp_fun(_t_now):
            return self._tvp_tmpl

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()

        self.model = model
        self.mpc = mpc
        self._x0 = np.zeros((4, 1))
        self._initialised = False
        # Exact NLP objective of do-mpc's own transcription, so alternative
        # multi-start solutions can be ranked without re-running the solver.
        self._obj_fun = ca.Function("nlp_obj",
                                    [mpc.nlp["x"], mpc.nlp["p"]],
                                    [mpc.nlp["f"]])

    # ------------------------------------------------------------- per-step
    def _relevance(self, o: Obstacle, state: RobotState,
                   nom: Tuple[float, float]) -> float:
        """Predicted closest approach over the horizon under a nominal robot motion.

        The NLP has a fixed number of obstacle slots, so when more pedestrians
        are in sensor range than there are slots they have to be ranked.
        Ranking on the CURRENT distance drops pedestrians that are far now but
        will be hit inside the horizon; ranking on predicted closest approach
        does not.
        """
        dt = self.cfg.dt
        best = float("inf")
        for k in range(self.horizon + 1):
            t = k * dt
            d = math.hypot(state.x + nom[0] * t - (o.x + o.vx * t),
                           state.y + nom[1] * t - (o.y + o.vy * t))
            if d < best:
                best = d
        return best

    def _fill_tvp(self, goal: Tuple[float, float],
                  obstacles: Sequence[Obstacle], state: RobotState) -> int:
        """Constant-velocity prediction of the n_obs most relevant pedestrians."""
        cfg = self.cfg
        in_range = [o for o in obstacles
                    if math.hypot(o.x - state.x, o.y - state.y) <= cfg.sensor_range]
        if len(in_range) > self.n_obs:
            if state.v > 0.1:
                nom = (state.v * math.cos(state.yaw), state.v * math.sin(state.yaw))
            else:
                gx, gy = goal[0] - state.x, goal[1] - state.y
                n = max(math.hypot(gx, gy), 1e-9)
                nom = (cfg.max_speed * gx / n, cfg.max_speed * gy / n)
            in_range.sort(key=lambda o: self._relevance(o, state, nom))
        near: List[Obstacle] = in_range[: self.n_obs]

        g = np.array([[goal[0]], [goal[1]]], dtype=float)
        buf = np.full((2 * self.n_obs, 1), _PARKED, dtype=float)
        for k in range(self.horizon + 1):
            t = k * cfg.dt
            buf[:] = _PARKED
            for j, o in enumerate(near):
                buf[2 * j, 0] = o.x + o.vx * t
                buf[2 * j + 1, 0] = o.y + o.vy * t
            self._tvp_tmpl["_tvp", k, "goal"] = g
            self._tvp_tmpl["_tvp", k, "obs"] = buf.copy()
        return len(near)

    # ------------------------------------------------------------- NLP solve
    def _set_guess(self, y_target: float, v_ref: float) -> None:
        """Write a straight-line rollout into do-mpc's initial-guess vector.

        Only the STARTING POINT of Ipopt is set here; do-mpc's transcription and
        Ipopt's iterations are untouched.  Used for the multi-start below.
        """
        cfg = self.cfg
        dt = cfg.dt
        x, y, upx, upy = (float(v) for v in self._x0[:, 0])
        T = max(self.horizon * dt, 1e-9)
        vy = float(np.clip((y_target - y) / T, -0.75 * cfg.max_speed,
                           0.75 * cfg.max_speed))
        vx = float(np.clip(math.sqrt(max(v_ref ** 2 - vy ** 2, 0.0)),
                           0.0, cfg.max_speed))
        ox = self.mpc.opt_x_num
        px, py = x, y
        pux, puy = upx, upy
        for k in range(self.horizon):
            ox["_x", k, 0, -1] = np.array([[px], [py], [pux], [puy]])
            ox["_u", k, 0] = np.array([[vx], [vy]])
            px += vx * dt
            py = float(np.clip(py + vy * dt,
                               cfg.sidewalk_y_min + self.margin,
                               cfg.sidewalk_y_max - self.margin))
            pux, puy = vx, vy
        ox["_x", self.horizon, 0, -1] = np.array([[px], [py], [pux], [puy]])

    def _solve_once(self) -> Tuple[bool, float, np.ndarray]:
        """One do-mpc/Ipopt solve from whatever is currently in opt_x_num."""
        self.mpc.solve()
        stats = self.mpc.solver_stats or {}
        ok = bool(stats.get("success", False))
        obj = float(self._obj_fun(self.mpc.opt_x_num, self.mpc.opt_p_num))
        u0 = np.asarray(self.mpc.opt_x_num["_u", 0, 0]).reshape(-1).astype(float)
        return ok, obj, u0

    def compute_command(self, state: RobotState, goal: Tuple[float, float],
                        obstacles: Sequence[Obstacle], sim_time: float):
        cfg = self.cfg
        n_near = self._fill_tvp(goal, obstacles, state)

        # u_{-1}: the velocity the plant actually applied last step.  state.v /
        # state.yaw is the authoritative record of it in both the standalone
        # runner and the benchmark runner, so read it back from there.
        upx = state.v * math.cos(state.yaw)
        upy = state.v * math.sin(state.yaw)
        if state.v <= 1e-9:
            upx = upy = 0.0
        self._u_prev[:] = (upx, upy)

        # keep x0 strictly inside the hard band bounds so the NLP is feasible
        x0 = float(np.clip(state.x, cfg.sidewalk_x_min, cfg.sidewalk_x_max))
        y0 = float(np.clip(state.y,
                           cfg.sidewalk_y_min + self.margin,
                           cfg.sidewalk_y_max - self.margin))
        self._x0[:, 0] = (x0, y0, upx, upy)

        # Feed the current problem instance straight into do-mpc's own parameter
        # struct and call MPC.solve().  This is the documented alternative to
        # make_step() and skips only do-mpc's per-step Data logging (which uses
        # np.append and would be O(n^2) over a 6000-step episode).
        mpc = self.mpc
        mpc.x0 = self._x0
        mpc.u0 = self._u_prev.reshape(2, 1)
        mpc.opt_p_num["_x0"] = self._x0
        mpc.opt_p_num["_u_prev"] = self._u_prev.reshape(2, 1)
        mpc.opt_p_num["_tvp"] = mpc.tvp_fun(sim_time)["_tvp"]
        mpc.opt_p_num["_p"] = mpc.p_fun(sim_time)["_p"]
        if not self._initialised:
            mpc.set_initial_guess()
            self._initialised = True

        t_start = time.perf_counter()
        n_solves = 0
        best_ok, best_obj, best_u = False, float("inf"), np.zeros(2)
        try:
            best_ok, best_obj, best_u = self._solve_once()
            n_solves = 1
            # -------- multi-start on stall ---------------------------------
            # A single local NLP solve warm-started from its own previous
            # solution can sit in a "stand still" local minimum in front of a
            # static pedestrian pair.  When the first input carries essentially
            # no forward progress, restart Ipopt from two alternative straight
            # -line guesses (pass high / pass low).  This changes only the
            # initial guess -- the objective, the constraints and the solver
            # are do-mpc's.  It is the same remedy teb_local_planner applies
            # with its parallel homotopy-class planning.
            if (not best_ok) or best_u[0] < self.stall_vx * cfg.max_speed:
                band_lo = cfg.sidewalk_y_min + self.margin
                band_hi = cfg.sidewalk_y_max - self.margin
                best_vec = ca.DM(mpc.opt_x_num.master)
                for y_t in (band_hi, band_lo):
                    self._set_guess(y_t, 0.85 * cfg.max_speed)
                    ok, obj, u = self._solve_once()
                    n_solves += 1
                    better = (ok and not best_ok) or \
                             (ok == best_ok and obj < best_obj - 1e-9)
                    if better:
                        best_ok, best_obj, best_u = ok, obj, u
                        best_vec = ca.DM(mpc.opt_x_num.master)
                # leave opt_x_num holding the winner so the NEXT step warm
                # starts from the trajectory that was actually applied
                mpc.opt_x_num.master = best_vec
        except Exception:
            best_ok = False
        solve_ms = (time.perf_counter() - t_start) * 1e3

        stats = mpc.solver_stats or {}
        iters = int(stats.get("iter_count", 0) or 0)

        self._solves += 1
        self._nlp_calls += n_solves
        self._t_total += solve_ms
        if not best_ok:
            self._fail += 1

        if best_ok and np.all(np.isfinite(best_u)):
            vx, vy = float(best_u[0]), float(best_u[1])
            sp = math.hypot(vx, vy)
            if sp > cfg.max_speed:                     # numerical guard only
                vx *= cfg.max_speed / sp
                vy *= cfg.max_speed / sp
            status = "mpc_dompc"
        else:
            # Ipopt gave up: hold the previous command, decelerated within the
            # acceleration limit.  Never a re-implementation of the planner.
            sp_prev = math.hypot(upx, upy)
            sp = max(0.0, sp_prev - cfg.max_accel * cfg.dt)
            if sp_prev > 1e-9:
                vx, vy = upx / sp_prev * sp, upy / sp_prev * sp
            else:
                vx = vy = 0.0
            status = "mpc_dompc_infeasible"

        return vx, vy, {
            "status": status,
            "cost": best_obj if math.isfinite(best_obj) else float("nan"),
            "solve_ms": round(solve_ms, 3),
            "nlp_solves": n_solves,
            "iters": iters,
            "n_obs_used": n_near,
        }

    # ---------------------------------------------------------------- report
    def timing_summary(self) -> dict:
        n = max(self._solves, 1)
        return {"solves": self._solves,
                "nlp_calls": self._nlp_calls,
                "failures": self._fail,
                "mean_solve_ms": self._t_total / n,
                "total_solve_s": self._t_total / 1e3}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="do-mpc / CasADi NLP-MPC robot-as-pedestrian controller for a SUMO sidewalk scene")
    add_common_arguments(p)
    p.add_argument("--mpc-horizon", type=int, default=8)
    p.add_argument("--mpc-obstacle-slots", type=int, default=12)
    p.add_argument("--mpc-collision-slack-penalty", type=float, default=2.0e4)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_traci_with_planner(
        args,
        lambda cfg, seed: DoMPCPlanner(
            cfg, seed,
            horizon=args.mpc_horizon,
            n_obstacle_slots=args.mpc_obstacle_slots,
            collision_slack_penalty=args.mpc_collision_slack_penalty),
        "mpc_dompc")


if __name__ == "__main__":
    main()
