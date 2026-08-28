"""Single source of algorithm display names for every figure script.

Supervisor item 7: rows labelled TEB, ORCA, SARL, CADRL and so on name
published algorithms that the self-implemented variants do not fully
implement, so every self-implemented local planner is marked (ours).
Upstream/library variants and the global planners keep their bare names.
stats_models.py, benchmark_plots.py, and map_sensitivity_figs.py all
import display()/full() from here when this module is present.
"""

DISPLAY = {
    # self-implemented local planners -> marked (ours)
    "dwa": "DWA (ours)",
    "mpc": "MPC (ours)",
    "teb": "TEB (ours)",
    "sarl": "SARL (ours)",
    "cadrl": "CADRL (ours)",
    "lstm_rl": "LSTM-RL (ours)",
    "orca_heuristic": "ORCA (heuristic, ours)",
    # classical searches: the LOCAL receding-horizon variants are also
    # self-implemented, so display() marks them (ours); the global side
    # uses global_display() below and stays bare
    "astar": "A* (ours)",
    "dijkstra": "Dijkstra (ours)",
    "rrt": "RRT (ours)",
    # upstream / library variants
    "orca": "ORCA (RVO2)",
    "mpc_dompc": "MPC (do-mpc)",
    "teb_upstream": "TEB (upstream)",
    "sarl_upstream": "SARL (upstream)",
    "cadrl_upstream": "CADRL (upstream)",
    "lstm_rl_upstream": "LSTM-RL (upstream)",
    "crowdnav_dsrnn": "CrowdNav DS-RNN",
    "crowdnav_attngraph": "CrowdNav AttnGraph",
}

FULL = {
    "dwa": "Dynamic Window Approach (DWA, ours)",
    "astar": "A* (search)",
    "dijkstra": "Dijkstra (search)",
    "rrt": "Rapidly-exploring Random Tree (RRT)",
    "orca_heuristic": "Optimal Reciprocal Collision Avoidance "
                      "(heuristic, ours)",
    "orca": "Optimal Reciprocal Collision Avoidance (RVO2)",
    "mpc": "Model Predictive Control (MPC, ours)",
    "mpc_dompc": "Model Predictive Control (do-mpc)",
    "teb": "Timed Elastic Band (TEB, ours)",
    "teb_upstream": "Timed Elastic Band (upstream)",
    "sarl": "Socially Attentive RL (SARL, ours)",
    "sarl_upstream": "Socially Attentive RL (upstream)",
    "cadrl": "Collision Avoidance with Deep RL (CADRL, ours)",
    "cadrl_upstream": "Collision Avoidance with Deep RL (upstream)",
    "lstm_rl": "LSTM-based RL (LSTM-RL, ours)",
    "lstm_rl_upstream": "LSTM-based RL (upstream)",
    "crowdnav_dsrnn": "CrowdNav Decentralized Structural RNN (DS-RNN)",
    "crowdnav_attngraph": "CrowdNav Attention Graph (AttnGraph)",
}


GLOBAL_DISPLAY = {"astar": "A*", "dijkstra": "Dijkstra", "rrt": "RRT"}


def display(u):
    return DISPLAY.get(str(u), str(u))


def global_display(u):
    """Name for the GLOBAL half of a combination (textbook searches)."""
    return GLOBAL_DISPLAY.get(str(u), display(u))


def full(u):
    return FULL.get(str(u), display(u))
