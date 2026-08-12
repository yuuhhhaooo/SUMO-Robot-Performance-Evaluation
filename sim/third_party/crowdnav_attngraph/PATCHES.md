# Vendoring notes: CrowdNav_Prediction_AttnGraph (CrowdNav++) @ 39077313

Upstream, licences and checkpoint hashes: see `COMMIT`, `LICENSE` and
`gst_updated/LICENSE`.

**No algorithm, network, hyper-parameter or numerical constant was changed.**
`arguments.py`, `rl/networks/{model,srnn_model,selfAttn_srnn_temp_node,
distributions}.py`, `configs/config.py`, the whole Gumbel Social Transformer
(`gst_updated/src/gumbel_social_transformer/*.py`) and the predictor wrapper
`gst_updated/scripts/wrapper/crowd_nav_interface_parallel.py` are byte-identical
to upstream. Exactly one file carries a patch, and it is an import guard.

## 1. Source patch (one file, one hunk)

### `rl/networks/network_utils.py`

    -from rl.networks.envs import VecNormalize
    +try:
    +    from rl.networks.envs import VecNormalize
    +except ImportError:
    +    VecNormalize = None

Why: `rl/networks/envs.py` imports OpenAI `baselines` and `gym`, i.e. the
training/vectorised-environment stack, which is not vendored and does not install
on Python 3.13. `srnn_model.py` and `distributions.py` reach this module only for
`init()` and `AddBias()`; the only consumer of `VecNormalize` is
`get_vec_normalize()`, which the benchmark never calls. Without the guard,
`import rl.networks.model` raises `ModuleNotFoundError` before any weight is
loaded.

## 2. Layout change (one directory renamed, contents unchanged)

    upstream  crowd_nav/configs/config.py
    here      configs/config.py            (byte-identical)

Why: a second CrowdNav is vendored in this benchmark under
`sim/third_party/crowdnav/`, and it also has a top-level `crowd_nav` package. The
adapter puts this directory on `sys.path`, so a `crowd_nav` package here would
shadow the other one depending on import order. The names this tree exposes on
`sys.path` are `rl`, `gst_updated`, `configs`, `trained_models` and `arguments.py`
-- and `arguments` is deliberately NOT imported as a top-level module: the adapter
loads it by file path and registers it in `sys.modules['arguments']` only for the
few milliseconds `configs/config.py` needs it (upstream's config imports it at
class-body time), then removes it again.

`trained_models/*/arguments.py` and `trained_models/*/configs/config.py` keep
their upstream paths and contents.

## 3. Upstream inconsistency, resolved in favour of the weights

`trained_models/GST_predictor_non_rand/` and `.../GST_predictor_rand/` are the
directories upstream's `test.py` loads `arguments.py` and `configs/config.py`
from. Both copies are STALE with respect to the checkpoints sitting next to them:
they carry the repository defaults

    arguments.py:  --env-name  default 'CrowdSimVarNum-v0'
    config.py:     sim.predict_method = 'none'

which build a 2-dimensional spatial edge
(`selfAttn_srnn_temp_node.py::SpatialEdgeSelfAttn.__init__`). The shipped weights
have `base.spatial_attn.embedding_layer.0.weight` of shape **(128, 12)**, which
that same code produces only for `env_name in {CrowdSimPred-v0,
CrowdSimPredRealGST-v0}`; 12 = `2*(1 + sim.predict_steps)` with
`predict_steps = 5`. The model directories are also literally named
`GST_predictor_*`, and the repository root `crowd_nav/configs/config.py` has
`predict_method = 'inferred'`.

The adapter therefore uses `--env-name CrowdSimPredRealGST-v0` and the root
config (vendored as `configs/config.py`), and the strict `load_state_dict`
against the resulting module is the check that this reading is right. Loading
under the stale settings fails with a shape mismatch, which is how the question
was settled rather than assumed.

## 4. Not vendored

Only the inference path is here:

* `train.py`, `test.py`, `plot.py`, `collect_data.py`, `requirements.txt`,
  `figures/`
* `rl/ppo/`, `rl/networks/{envs,storage,shmem_vec_env,dummy_vec_env}.py`,
  `rl/vec_env/`, `rl/evaluation.py` -- training / vec-env stack (needs `gym`,
  `baselines`)
* `crowd_sim/` and `crowd_nav/policy/` -- simulator and ORCA / social force
  baselines (need `gym`, `rvo2`)
* `gst_updated/{scripts/experiments,src/mgnn,src/pec_net,run,tuning,datasets}`
  -- GST training and dataset tooling. Upstream states this repo does not
  contain code to train a predictor anyway.
* the TensorBoard event files under `gst_updated/results/*/sj/` (kept: only
  `checkpoint/args.pickle` and `checkpoint/epoch_100.pt`, which is all
  `CrowdNavPredInterfaceMultiEnv` reads).

Copies of the files that *define the observation* -- `crowd_sim/envs/crowd_sim.py`,
`crowd_sim_var_num.py`, `crowd_sim_pred.py`, `crowd_sim_pred_real_gst.py`, the
agent/state utils, `rl/vec_env/vec_pretext_normalize.py` and `rl/evaluation.py`
-- are kept unmodified under `_ref/` purely so the adapter's observation
assembly can be audited line by line against them. **Nothing under `_ref/` is
ever imported**; `_ref` is not a package name and never goes on `sys.path`.
Several of those files would not import under numpy 2 anyway (`np.bool` was
removed). They are documentation.

The two `*_ghost.py` files under `gst_updated/src/gumbel_social_transformer/`
ARE vendored, unchanged, including their upstream-broken `from src....` imports.
They are unreachable: the shipped `args.pickle` has `ghost = False`, so
`GumbelSocialTransformer.__init__` imports the `*_no_ghost.py` pair instead. They
are kept so the tree matches upstream rather than being quietly pruned.

## 5. Compatibility handled on the adapter side (no upstream edit)

Done in `sim/planners/crowdnav_attngraph_planner.py`:

* **`torch.load` weights-only default.** Since torch 2.6 `torch.load` defaults to
  `weights_only=True`. The policy checkpoints are plain tensor state_dicts and
  load fine, but the GST checkpoint `epoch_100.pt` is a full training checkpoint
  and carries numpy scalars (`train_loss_epoch` etc.) alongside the tensors, so
  upstream's un-flagged `torch.load` inside `crowd_nav_interface_parallel.py`
  raises `UnpicklingError`. The adapter wraps the constructor in
  `torch.serialization.safe_globals([...])` allowlisting
  `numpy.core.multiarray.scalar`, `numpy.dtype` and the concrete numpy dtype
  classes. This keeps `weights_only=True` -- it does **not** fall back to
  arbitrary-code unpickling -- and needs no edit to upstream.
* **gym -> gymnasium.** Upstream passes `gym.spaces` objects to `Policy`;
  `Policy` reads `action_space.__class__.__name__` and `.shape[0]`, and
  `selfAttn_merge_SRNN.__init__` reads `obs_space_dict['spatial_edges'].shape`.
  `gymnasium` provides the same. `gym` does not install on Python 3.13.
* **`--no-cuda`** is passed to upstream's own argument parser when the device is
  CPU, rather than poking `args.no_cuda` afterwards, because `args.cuda` and
  `config.training.device` are derived from it.
* **`torch.set_num_threads(1)`**, exactly as upstream's `test.py` does.

## 6. Verified

`torch 2.12.0+cu132`, `numpy 2.4.6`, Python 3.13.12, Windows:

* `trained_models/GST_predictor_non_rand/checkpoints/41200.pt` loads into
  `Policy(base='selfAttn_merge_srnn')` with `strict=True`; all 47 keys match and
  every tensor shape matches the upstream module definition.
* `gst_updated/results/100-...-seed_1000/sj/checkpoint/epoch_100.pt` loads into
  `st_model` through upstream's own `CrowdNavPredInterfaceMultiEnv`, and the
  predictor returns valid predictions for tracked pedestrians (confirmed via
  `info['n_predicted']`).
* On upstream's own circle-crossing scenario (20 humans, circle radius 6*sqrt(2),
  robot (0,-R) -> (0,+R), time step 0.25 s) the loaded policy reached the goal in
  12/12 seeds, mean 23.5 s, mean closest-pedestrian distance 0.78 m -- with
  NON-reactive humans, which is harder than upstream's ORCA humans.
