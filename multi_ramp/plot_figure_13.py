import traci
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIGURATION
# =========================
SUMO_BINARY = "sumo"
SUMO_PATH = "simulation.sumocfg"
STEP_LENGTH = 1
SIM_END = 4500
FREE_FLOW_SPEED = 27.78  # m/s

detector_ids = ["det_ramp_queue_0", "det_ramp_queue_1"]

# Background flow from paper
q0 = 8100 / 3600.0  # veh/sec

# Define structure for the 4 subplots
plots_config = {
    "Area 4": {
        "title": "(a)",
        "loc": (0, 0),
        "detectors": {
            "+6822 m": {"ids": [f"det_a4_up_{i}" for i in range(4)], "dist": 0},
            "+7335 m": {"ids": [f"det_a4_dn1_{i}" for i in range(4)], "dist": 513},
            "+7635 m": {"ids": [f"det_a4_dn2_{i}" for i in range(4)], "dist": 513 + 300}
        }
    },
    "Area 2": {
        "title": "(b)",
        "loc": (0, 1),
        "detectors": {
            "+3916 m": {"ids": [f"det_a2_up_{i}" for i in range(4)], "dist": 0},
            "+4254 m": {"ids": [f"det_a2_dn1_{i}" for i in range(5)], "dist": 338},
            "+4404 m": {"ids": [f"det_a2_dn2_{i}" for i in range(4)], "dist": 338 + 250}
        }
    },
    "Area 3": {
        "title": "(c)",
        "loc": (1, 0),
        "detectors": {
            "+5221 m": {"ids": [f"det_a3_up_{i}" for i in range(4)], "dist": 0},
            "+5470 m": {"ids": [f"det_a3_dn1_{i}" for i in range(4)], "dist": 249},
            "+5620 m": {"ids": [f"det_a3_dn2_{i}" for i in range(5)], "dist": 249 + 250}
        }
    },
    "Area 1": {
        "title": "(d)",
        "loc": (1, 1),
        "detectors": {
            "+955 m": {"ids": [f"det_a1_up_{i}" for i in range(4)], "dist": 0},
            "+1317 m": {"ids": [f"det_a1_dn1_{i}" for i in range(4)], "dist": 362},
            "+1652 m": {"ids": [f"det_a1_dn2_{i}" for i in range(4)], "dist": 362 + 335}
        }
    }
}

# =========================
# INITIALIZATION
# =========================
traci.start([SUMO_BINARY, "-c", SUMO_PATH])

times = []
records = {}

for area, config in plots_config.items():
    records[area] = {}
    for label in config["detectors"]:
        records[area][label] = {"cumulative": 0, "history": []}

# =========================
# SIMULATION LOOP
# =========================
while traci.simulation.getTime() < SIM_END:
    traci.simulationStep()
    t = traci.simulation.getTime()
    times.append(t)

    for area, config in plots_config.items():
        for label, det_info in config["detectors"].items():
            step_count = 0
            for det in det_info["ids"]:
                step_count += traci.inductionloop.getLastStepVehicleNumber(det)

            records[area][label]["cumulative"] += step_count
            records[area][label]["history"].append(records[area][label]["cumulative"])

traci.close()

# =========================
# POST-PROCESSING & PLOTTING
# =========================
times = np.array(times)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

styles = [
    {'color': '#d62728', 'marker': '+', 'markersize': 4, 'linewidth': 1},
    {'color': '#ffbf0e', 'marker': '.', 'markersize': 4, 'linewidth': 1},
    {'color': '#1f77b4', 'marker': 'x', 'markersize': 4, 'linewidth': 1}
]

marker_interval = 15

for area, config in plots_config.items():
    row, col = config["loc"]
    ax = axs[row, col]

    for idx, (label, det_info) in enumerate(config["detectors"].items()):
        N = np.array(records[area][label]["history"])
        lane_count = len(det_info["ids"])

        # Convert to per-lane cumulative
        N_lane = N / lane_count

        # Calculate background flow offset based on uniform lane assumption
        q0_lane = q0 / 4.0
        N_mod = N_lane - (q0_lane * times)

        # Apply time shift based on relative distance and free-flow speed
        shift_seconds = det_info["dist"] / FREE_FLOW_SPEED
        shift_steps = int(shift_seconds / STEP_LENGTH)

        if shift_steps > 0:
            N_mod = N_mod[shift_steps:]
            t_mod = times[:-shift_steps]
        else:
            t_mod = times

        # Filter out 0 points or negative initialization artifacts to mimic plot start times
        mask = t_mod > 1000

        ax.plot(t_mod[mask][::marker_interval],
                N_mod[mask][::marker_interval],
                label=label,
                **styles[idx])

    ax.set_xlabel("Simulation time (s)", fontsize=11)
    ax.set_ylabel("$N'(x,t) = N(x,t) - q_0 \\times t$", fontsize=11)
    ax.set_title(config["title"], y=-0.2, fontsize=12)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.4))

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()