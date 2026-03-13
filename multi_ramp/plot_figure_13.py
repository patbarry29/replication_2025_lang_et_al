from tqdm import tqdm
import traci
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

SUMO_BINARY = "sumo"
SUMO_CONFIG = "simulation.sumocfg"
STEP_LENGTH = 1
SIM_END = 4500
FREE_FLOW_SPEED = 27.78

# BACKGROUND_FLOW = 8100 / 3600.0
LANES = 4
MARKER_EVERY = 15
X_PADDING = 100

DETECTOR_STYLES = [
    {"color": "#d62728", "marker": "^", "linewidth": 1},
    {"color": "#ffbf0e", "marker": ".", "linewidth": 1},
    {"color": "#1f77b4", "marker": "x", "linewidth": 1},
]

RAMPS = {
    "Ramp 1": {
        "loc": (1, 1),
        "x_range": (1400, 3600),
        "bg_flow": 10356,
        "detectors": {
            # ADD YOUR RAMP DETECTOR ID TO THE UPSTREAM LIST
            "+955 m":  {"ids": [f"det_a1_up_{i}"  for i in range(4)], "shift": 0},
            "+1317 m": {"ids": [f"det_a1_dn1_{i}" for i in range(4)],   "shift": 30},
            "+1652 m": {"ids": [f"det_a1_dn2_{i}" for i in range(4)],   "shift": 58},
        },
    },
    "Ramp 2": {
        "loc": (0, 1),
        "x_range": (2100, 4600),
        "bg_flow": 9432,
        "detectors": {
            "+3916 m": {"ids": [f"det_a2_up_{i}"  for i in range(4)], "shift": 0},
            "+4254 m": {"ids": [f"det_a2_dn1_{i}" for i in range(1, 5)], "shift": 29},
            "+4404 m": {"ids": [f"det_a2_dn2_{i}" for i in range(4)],   "shift": 51},
        },
    },
    "Ramp 3": {
        "loc": (1, 0),
        "x_range": (1500, 3900),
        "bg_flow": 8472,
        "detectors": {
            "+5221 m": {"ids": [f"det_a3_up_{i}"  for i in range(4)]
                        , "shift": 0},
            "+5470 m": {"ids": [f"det_a3_dn1_{i}" for i in range(4)],   "shift": 22},
            "+5620 m": {"ids": [f"det_a3_dn2_{i}" for i in range(1, 5)], "shift": 43},
        },
    },
    "Ramp 4": {
        "loc": (0, 0),
        "x_range": (1000, 3900),
        "bg_flow": 6204,
        "detectors": {
            "+6822 m": {"ids": [f"det_a4_up_{i}"  for i in range(4)], "shift": 0},
            "+7335 m": {"ids": [f"det_a4_dn1_{i}" for i in range(4)],   "shift": 39},
            "+7635 m": {"ids": [f"det_a4_dn2_{i}" for i in range(4)],   "shift": 62},
        },
    },
}


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation():
    """Run SUMO and collect cumulative vehicle counts for every detector."""
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    times = []
    # Structure: counts[ramp][label] = running cumulative count
    counts = {
        ramp: {label: 0 for label in cfg["detectors"]}
        for ramp, cfg in RAMPS.items()
    }
    history = {
        ramp: {label: [] for label in cfg["detectors"]}
        for ramp, cfg in RAMPS.items()
    }

    for i in tqdm(range(SIM_END)):
        traci.simulationStep()
        times.append(i)

        for ramp, cfg in RAMPS.items():
            for label, det in cfg["detectors"].items():
                step_count = sum(
                    traci.inductionloop.getLastStepVehicleNumber(d)
                    for d in det["ids"]
                )
                counts[ramp][label] += step_count
                history[ramp][label].append(counts[ramp][label])

    traci.close()
    return np.array(times), history


# =============================================================================
# PLOTTING
# =============================================================================

def compute_modified_counts(raw_counts, times, shift, lane_count, bg_flow):
    q0_veh_per_sec_per_lane = (bg_flow / 3600.0) / lane_count

    N = (np.array(raw_counts) / lane_count) - (q0_veh_per_sec_per_lane * times)
    t_shifted = times - shift

    return t_shifted, N

def plot_ramp(ax, ramp_name, cfg, times, history):
    x_min, x_max = cfg["x_range"]

    for idx, (label, det) in enumerate(cfg["detectors"].items()):
        # Force normalisation by mainline lanes only
        lane_count = LANES

        t, N = compute_modified_counts(
            history[ramp_name][label],
            times,
            det["shift"],
            lane_count,
            cfg["bg_flow"]
        )

        in_range = (t >= x_min) & (t <= x_max)
        t_plot = t[in_range][::MARKER_EVERY]
        N_plot = N[in_range][::MARKER_EVERY]

        ax.plot(t_plot, N_plot, label=label, **DETECTOR_STYLES[idx])

    ax.set_xlim(x_min - X_PADDING, x_max + X_PADDING)
    ax.margins(y=0.05)
    ax.set_xlabel("Simulation time (s)", fontsize=11)
    ax.set_ylabel("$N'(x,t) = N(x,t) - q_0 \\times t$", fontsize=11)
    ax.set_title(ramp_name, y=-0.2, fontsize=12)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.4))


def make_figure(times, history):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    times = np.asarray(times)

    for ramp_name, cfg in RAMPS.items():
        row, col = cfg["loc"]
        plot_ramp(axs[row, col], ramp_name, cfg, times, history)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    times, history = run_simulation()
    make_figure(times, history)