# --- BENCHMARK PATCH 1/2 (see PATCHES.md) -------------------------------
# Upstream unconditionally registers the CrowdSim gym environment here, which
# makes `import crowd_sim.envs.utils.state` (a pure dataclass module) hard-fail
# when `gym` is absent.  `gym` is unmaintained and does not install on
# Python 3.13.  The registration is wrapped so that importing the *policy* and
# *state* modules works without gym; when gym IS installed the behaviour is
# byte-for-byte the upstream behaviour.  No algorithm code is touched.
try:
    from gym.envs.registration import register

    register(
        id='CrowdSim-v0',
        entry_point='crowd_sim.envs:CrowdSim',
    )
except ImportError:  # pragma: no cover - gym not installed
    pass
# --- END BENCHMARK PATCH 1/2 --------------------------------------------
