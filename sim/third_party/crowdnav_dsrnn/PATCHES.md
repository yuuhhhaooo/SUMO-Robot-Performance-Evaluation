# Vendoring notes: CrowdNav_DSRNN @ 91fb53c0

Upstream, licence and checkpoint hashes: see `COMMIT` and `LICENSE`.

**No algorithm, network, hyper-parameter or numerical constant was changed.**
`srnn_model.py`, `model.py`, `distributions.py` and `configs/config.py` are
byte-identical to upstream. Exactly one file carries a patch, and it is an
import guard.

## 1. Source patch (one file, one hunk)

### `pytorchBaselines/a2c_ppo_acktr/utils.py`

    -from pytorchBaselines.a2c_ppo_acktr.envs import VecNormalize
    +try:
    +    from pytorchBaselines.a2c_ppo_acktr.envs import VecNormalize
    +except ImportError:
    +    VecNormalize = None

Why: `envs.py` imports OpenAI `baselines` and `gym`, i.e. the training/vectorised
-environment stack, which is not vendored (see "not vendored" below) and does not
install on Python 3.13. The policy network reaches this module only for `init()`
and `AddBias()`, neither of which touches `VecNormalize`; the only consumer is
`get_vec_normalize()`, which the benchmark never calls. Without the guard,
`import pytorchBaselines.a2c_ppo_acktr.model` raises `ModuleNotFoundError` before
any weight is loaded.

## 2. Layout change (one directory renamed, contents unchanged)

    upstream  crowd_nav/configs/config.py
    here      configs/config.py            (byte-identical)

Why: a second CrowdNav is vendored in this benchmark under
`sim/third_party/crowdnav/`, and it also has a top-level `crowd_nav` package. The
adapter puts this directory on `sys.path`, so a `crowd_nav` package here would
shadow the other one depending on import order. The names this tree exposes on
`sys.path` are therefore only `pytorchBaselines`, `configs` and `data`, none of
which collides.

`data/example_model/configs/config.py` and
`data/example_model_unicycle/configs/config.py` keep their upstream paths; they
are the configs upstream saved next to each checkpoint and are what
`sim/planners/crowdnav_dsrnn_planner.py` actually loads (upstream's `test.py`
does the same). They differ from `configs/config.py` only in comments and in
`training.load_path`.

## 3. Not vendored

Only the inference path is here. Omitted, because nothing on the path from the
adapter to the weights imports them:

* `train.py`, `test.py`, `plot.py`, `requirements.txt`, `figures/`
* `pytorchBaselines/a2c_ppo_acktr/{envs,storage,shmem_vec_env}.py`,
  `pytorchBaselines/a2c_ppo_acktr/algo/` (PPO), `pytorchBaselines/evaluation.py`
  -- the training / vec-env stack (needs `gym`, `baselines`)
* `crowd_sim/` and `crowd_nav/policy/` -- the simulator and the ORCA / social
  force baselines (need `gym`, `rvo2`). The benchmark supplies its own
  pedestrians, and ORCA is already a separate benchmark algorithm.

Copies of the files that *define the observation and the evaluation loop* are
kept, unmodified, under `_ref/` purely so the adapter's mapping can be audited
against them. **Nothing under `_ref/` is ever imported** -- `_ref` is not a
package name and never goes on `sys.path`. Some of those files would not even
import under numpy 2 (`np.bool` was removed). They are documentation.

## 4. Compatibility handled on the adapter side (no upstream edit)

These are done in `sim/planners/crowdnav_dsrnn_planner.py` so upstream stays
clean:

* **gym -> gymnasium.** Upstream builds `gym.spaces` objects and passes them to
  `Policy`. `Policy` only reads `action_space.__class__.__name__` and
  `action_space.shape[0]`, and `SRNN.__init__` ignores the observation-space dict
  entirely (it sizes itself from `config`). The adapter builds the same spaces
  with `gymnasium`, whose `Box` has the same class name and shape API. `gym`
  itself does not install on Python 3.13.
* **`config.training.cuda`** is set to match the requested device. It is
  upstream's own flag, read by `SRNN.forward` to pick where to allocate the edge
  hidden-state buffer.
* **`torch.set_num_threads(1)`**, exactly as upstream's `test.py` does. Measured
  here: 96.5 ms -> 3.6 ms per control step, because the tensors are tiny and the
  default 16-thread pool costs more than the arithmetic.

## 5. Verified

`torch 2.12.0+cu132`, `numpy 2.4.6`, Python 3.13.12, Windows:

* `data/example_model/checkpoints/27776.pt` loads into
  `Policy(base='srnn', base_kwargs=Config())` with `strict=True`; all 47 keys
  match and every tensor shape matches the upstream module definition.
* On upstream's own circle-crossing scenario (5 humans, circle radius 6, robot
  (0,-6) -> (0,+6), time step 0.25 s) the loaded policy reached the goal in
  12/12 seeds, mean 16.1 s, mean closest-pedestrian distance 0.89 m -- with
  NON-reactive humans, which is harder than upstream's ORCA humans.
