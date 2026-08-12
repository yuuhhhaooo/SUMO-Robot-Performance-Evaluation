# --- BENCHMARK PATCH 2/2 (see PATCHES.md) -------------------------------
# Upstream eagerly imports the CrowdSim environment, which pulls in gym,
# matplotlib and `rvo2` (Python-RVO2, a Cython extension that has no wheel and
# does not build against Python 3.13).  Importing
# `crowd_sim.envs.utils.state` / `crowd_sim.envs.policy.policy` must not
# require any of those.  The import is made lazy; `crowd_sim.envs.CrowdSim`
# still resolves identically when the dependencies are present.
try:
    from .crowd_sim import CrowdSim
except ImportError:  # pragma: no cover - gym / rvo2 / matplotlib not installed
    CrowdSim = None
# --- END BENCHMARK PATCH 2/2 --------------------------------------------
