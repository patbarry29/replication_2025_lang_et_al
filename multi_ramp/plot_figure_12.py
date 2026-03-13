import traci
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
SUMO_BINARY = "sumo"
SUMO_PATH = "simulation.sumocfg"
STEP_LENGTH = 1
SIM_END = 3600
FLOW_WINDOW = 300  # seconds for rolling average

# First downstream detectors after each on-ramp merge point.
# Area 1 is the most upstream ramp, Area 4 the most downstream.
# det_a2_dn1 is on the 5-lane merge edge, so we skip lane 0 (range(1,5)).
ramp_detectors = {
    "(4) Inner Ring South": [f"det_a1_dn1_{i}" for i in range(4)],
    "(3) Shiyang":          [f"det_a2_dn1_{i}" for i in range(1, 5)],
    "(1) Maquan":           [f"det_a3_dn1_{i}" for i in range(4)],
    "(2) Shaungqi":         [f"det_a4_dn1_{i}" for i in range(4)],
}

# =========================
# SIMULATION
# =========================
traci.start([SUMO_BINARY, "-c", SUMO_PATH])

times = []
raw_counts = {name: [] for name in ramp_detectors}

while traci.simulation.getTime() < SIM_END:
    traci.simulationStep()
    t = traci.simulation.getTime()
    times.append(t)
    for name, det_ids in ramp_detectors.items():
        step_count = sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in det_ids)
        raw_counts[name].append(step_count)

traci.close()

times = np.array(times)

# =========================
# POST-PROCESSING
# =========================
def rolling_flow_vehph(raw, window):
    """Rolling-average flow in veh/h using a centred window."""
    raw = np.array(raw, dtype=float)
    cumsum = np.zeros(len(raw) + 1)
    cumsum[1:] = np.cumsum(raw)
    flow = np.zeros(len(raw))
    half = window // 2
    for i in range(len(raw)):
        s = max(0, i - half)
        e = min(len(raw), i + half + 1)
        flow[i] = (cumsum[e] - cumsum[s]) / (e - s) * 3600
    return flow

# =========================
# PLOTTING
# =========================
fig, ax = plt.subplots(figsize=(7, 5))

line_styles = {
    "(4) Inner Ring South": dict(color="#d62728", marker="x", markersize=5, linewidth=1.2),
    "(3) Shiyang":          dict(color="#ff7f0e", marker="+", markersize=5, linewidth=1.2),
    "(1) Maquan":           dict(color="#2ca02c", marker="o", markersize=5, linewidth=1.2),
    "(2) Shaungqi":         dict(color="#1f77b4", marker="^", markersize=4, linewidth=1.2),
}

MARKER_EVERY = 300  # ~every 200 s to keep the plot readable

for name in ramp_detectors:
    flow = rolling_flow_vehph(raw_counts[name], FLOW_WINDOW)
    mask = times >= 600
    t_plot = times[mask]
    f_plot = flow[mask]
    ax.plot(t_plot[::MARKER_EVERY], f_plot[::MARKER_EVERY],
            label=name, **line_styles[name])

ax.set_xlabel("Simulation time (s)", fontsize=11)
ax.set_ylabel("On ramps downstream flow (veh/h)", fontsize=11)

# Two-column legend below the axes, matching the paper's layout
ax.legend(
    ncols=2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    frameon=False,
    fontsize=10,
)

plt.tight_layout()
plt.savefig("figure_12.png", dpi=150, bbox_inches="tight")
plt.show()
