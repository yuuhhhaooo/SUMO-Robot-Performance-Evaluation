# Vendored CrowdNav — provenance and patches

## Provenance

| item | value |
|---|---|
| Upstream | https://github.com/vita-epfl/CrowdNav |
| Commit | `20d678085c06831e658a65b9e20c8bb6f6ecdc10` (2021-10-02, `Update links`, tip of `master`) |
| Paper | C. Chen, Y. Liu, S. Kreiss, A. Alahi, *"Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning"*, ICRA 2019 |
| License | MIT, © 2018 VITA lab at EPFL — see `LICENSE` (unmodified) |
| Exact hash also recorded in | `COMMIT` |

The full upstream `crowd_nav/` and `crowd_sim/` packages are vendored verbatim
except for the two patches below. `crowd_nav/policy/{cadrl,multi_human_rl,sarl,lstm_rl}.py`
— i.e. every line of the algorithms and the network definitions — are
**byte-for-byte upstream**. So are `crowd_nav/configs/policy.config`,
`crowd_nav/configs/env.config`, `crowd_sim/envs/policy/policy.py` and
`crowd_sim/envs/utils/{state,action}.py`.

Verify with:

```
git clone https://github.com/vita-epfl/CrowdNav /tmp/CrowdNav
cd /tmp/CrowdNav && git checkout 20d678085c06831e658a65b9e20c8bb6f6ecdc10
diff -r /tmp/CrowdNav/crowd_nav <this dir>/crowd_nav   # only PATCHES.md/COMMIT absent upstream
diff -r /tmp/CrowdNav/crowd_sim <this dir>/crowd_sim   # only the two __init__.py below differ
```

## Environment it had to be made to run on

Python 3.13.12, torch 2.12.0+cu132, numpy 2.4.6, Windows 11.

The codebase is from 2019–2021 and its `setup.py` pins `gym`, `torchvision`,
`gitpython`, plus (via `crowd_sim/envs/policy/orca.py`) `rvo2` from
[Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2), a Cython extension
with no wheels that does not build against Python 3.13.

**Everything needed for the three learning policies imports cleanly on this
stack after two patches.** No numpy-2 or torch-2 API breakage was hit at all in
the policy/network code: no `np.float`/`np.int`/`np.bool` aliases, no
`torch.Tensor` deprecations, no removed `nn` kwargs. `nn.LSTM`, `nn.Linear`,
`nn.Sequential`, `torch.exp`, masked-softmax indexing and `np.isin` all behave
identically. The only breakage was package-level eager imports.

## Patch 1 of 2 — `crowd_sim/__init__.py`

**Why:** upstream unconditionally calls `gym.envs.registration.register(...)` at
package-import time. That makes even `import crowd_sim.envs.utils.state` (three
plain data classes, no dependencies) raise `ModuleNotFoundError: No module named
'gym'`. `gym` is unmaintained and does not install on Python 3.13.

**Change:** the `import` + `register(...)` call is wrapped in
`try/except ImportError`. Nothing else. When `gym` IS installed the registration
happens exactly as upstream.

## Patch 2 of 2 — `crowd_sim/envs/__init__.py`

**Why:** upstream is `from .crowd_sim import CrowdSim`, which transitively pulls
in `gym`, `matplotlib` and `crowd_sim.envs.policy.orca` → `rvo2`. `rvo2` has no
Python 3.13 build. Same problem: it blocks importing the pure-python state and
policy modules.

**Change:** the import is wrapped in `try/except ImportError`, with
`CrowdSim = None` on failure. `crowd_sim.envs.CrowdSim` still resolves
identically whenever the dependencies are present.

## Mathematics changed

**None.** Neither patch touches an expression, a constant, a network
definition, a config value, or any line inside `crowd_nav/policy/`. Both are
import-guard wrappers in `__init__.py` files.

## Deliberately NOT patched

* `crowd_nav/policy/lstm_rl.py` `ValueNetwork1.forward` / `ValueNetwork2.forward`
  allocate the LSTM initial state as `torch.zeros(1, size[0], hidden)` with no
  `device=` argument. On a CUDA device this raises a device-mismatch
  `RuntimeError`. Rather than patch upstream, `sim/planners/crowdnav_upstream.py`
  refuses a non-CPU device for LSTM-RL with an explicit message. The two other
  policies run on CUDA fine.
* `crowd_sim/envs/crowd_sim.py`, `crowd_sim/envs/policy/orca.py`,
  `crowd_nav/train.py`, `crowd_nav/utils/*` are vendored unmodified but are not
  importable here (they need `gym` / `rvo2` / `gitpython`). They are kept only
  so the vendored tree is a faithful copy and so the training code is on hand if
  the networks are ever retrained.

## Shipped checkpoints

`sim/planners/models/{sarl,cadrl,lstm_rl}_rl_model.pth` load into the
**unmodified upstream** network classes with `load_state_dict(..., strict=True)`
and byte-identical key sets and tensor shapes, under upstream's own default
`crowd_nav/configs/policy.config`. See the header of
`sim/planners/crowdnav_upstream.py` for the full key-by-key comparison and for
the self-check you can run (`python sim/planners/crowdnav_upstream.py`).
