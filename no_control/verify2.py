import traci
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

SUMO_BINARY = "sumo"     # or sumo-gui
SUMO_CONFIG = "simulation.sumocfg"

STEP_LENGTH = 1
SIM_END = 2250

# background flow from paper
q0 = 8160 / 3600.0   # veh/sec

# detector groups
detector_groups = {
    "+920 m (upstream)": [
        "det_loc1_0","det_loc1_1","det_loc1_2","det_loc1_3","det_loc1_4"
    ],
    "+1225 m (downstream1)": [
        "det_loc2_0","det_loc2_1","det_loc2_2","det_loc2_3"
    ],
    "+1475 m (downstream2)": [
        "det_loc3_0","det_loc3_1","det_loc3_2","det_loc3_3"
    ]
}

# downstream travel time shifts from paper
time_shift = {
    "+920 m (upstream)": 0,
    "+1225 m (downstream1)": 16,
    "+1475 m (downstream2)": 25
}

# =========================
# START SUMO
# =========================

traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

times = []
cumulative = {k: 0 for k in detector_groups}
records = {k: [] for k in detector_groups}

# =========================
# SIM LOOP
# =========================

cumulative = {k: 0 for k in detector_groups}
records = {k: [] for k in detector_groups}
times = []

while traci.simulation.getTime() < SIM_END:
    traci.simulationStep()
    t = traci.simulation.getTime()
    times.append(t)

    for loc, dets in detector_groups.items():
        step_count = 0
        for det in dets:
            # FIX: Only count vehicles that have completely passed to avoid double-counting
            step_count += traci.inductionloop.getLastStepVehicleNumber(det) 
            # Note: If values are still high, use: len(traci.inductionloop.getPassedVehiclesID(det))
        
        cumulative[loc] += step_count
        records[loc].append(cumulative[loc])

# =========================
# POST PROCESS
# =========================

times = np.array(times)

modified_curves = {}

lane_count = {
    "+920 m (upstream)": 5,
    "+1225 m (downstream1)": 4,
    "+1475 m (downstream2)": 4
}

for loc in records:

    N = np.array(records[loc])

    # convert to per-lane cumulative
    N = N / lane_count[loc]

    # subtract background flow PER LANE
    q0_lane = (8160/3600.0) / 4.0   # assuming 4 mainline lanes
    N_mod = N - q0_lane * times


    # shift downstream curves left
    shift_steps = int(time_shift[loc] / STEP_LENGTH)

    if shift_steps > 0:
        N_mod = N_mod[shift_steps:]
        t_mod = times[:-shift_steps]
    else:
        t_mod = times

    modified_curves[loc] = (t_mod, N_mod)
# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 7))
skip_steps = 400 
marker_interval = 15  # Matches the paper's control step length

styles = {
    "+920 m (upstream)":     {'color': '#d62728', 'marker': '^', 'markersize': 5, 'linewidth': 1}, 
    "+1225 m (downstream1)": {'color': '#ff7f0e', 'marker': '.', 'markersize': 5, 'linewidth': 1}, 
    "+1475 m (downstream2)": {'color': '#1f77b4', 'marker': 'x', 'markersize': 5, 'linewidth': 1}  
}

for loc, (t, data) in modified_curves.items():
    mask = t > skip_steps
    # Slice using [::marker_interval] to show markers every 15 seconds
    plt.plot(t[mask][::marker_interval], 
             data[mask][::marker_interval], 
             label=loc.split(' ')[0], 
             **styles[loc])

# Vertical dashed lines for capacity drop phases (approx 600s and 950s)
plt.axvline(x=600, color='lightgray', linestyle='--')
plt.axvline(x=950, color='lightgray', linestyle='--')

plt.xlabel("Simulation time (s)", fontsize=12)
plt.ylabel("$N'(x,t) = N(x,t) - q_0 \\times t$", fontsize=12)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()