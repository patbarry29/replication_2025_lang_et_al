import traci
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "simulation.sumocfg"
SIM_END = 4500
CONTROL_STEP = 15

NUM_LANES = 4
RAMPS = {
    "Ramp 1": {
        "loc": (1, 1),
        "x_range": (1400, 3600),
        "q0": 10356,
        "detectors": {
            "up":  {"ids": [f"det_a1_up_{i}"  for i in range(4)], "shift": 0},
            "down1": {"ids": [f"det_a1_dn1_{i}" for i in range(4)],   "shift": 30},
            "down2": {"ids": [f"det_a1_dn2_{i}" for i in range(4)],   "shift": 58},
        },
    },
    "Ramp 2": {
        "loc": (0, 1),
        "x_range": (2100, 4600),
        "q0": 9432,
        "detectors": {
            "up": {"ids": [f"det_a2_up_{i}"  for i in range(4)], "shift": 0},
            "down1": {"ids": [f"det_a2_dn1_{i}" for i in range(1, 5)], "shift": 29},
            "down2": {"ids": [f"det_a2_dn2_{i}" for i in range(4)],   "shift": 51},
        },
    },
    "Ramp 3": {
        "loc": (1, 0),
        "x_range": (1500, 3900),
        "q0": 8472,
        "detectors": {
            "up": {"ids": [f"det_a3_up_{i}"  for i in range(4)]
                        , "shift": 0},
            "down1": {"ids": [f"det_a3_dn1_{i}" for i in range(4)],   "shift": 22},
            "down2": {"ids": [f"det_a3_dn2_{i}" for i in range(1, 5)], "shift": 43},
        },
    },
    "Ramp 4": {
        "loc": (0, 0),
        "x_range": (1000, 3900),
        "q0": 6204,
        "detectors": {
            "up": {"ids": [f"det_a4_up_{i}"  for i in range(4)], "shift": 0},
            "down1": {"ids": [f"det_a4_dn1_{i}" for i in range(4)],   "shift": 39},
            "down2": {"ids": [f"det_a4_dn2_{i}" for i in range(4)],   "shift": 62},
        },
    },
}

times = []

history = {
    "Ramp 1": {
            "up": [], "down1": [], "down2": []
        },
    "Ramp 2": {
            "up": [], "down1": [], "down2": []
        },
    "Ramp 3": {
            "up": [], "down1": [], "down2": []
        },
    "Ramp 4": {
            "up": [], "down1": [], "down2": []
        },
}

def getAllVehCounts(history):
    for ramp_name, item in RAMPS.items():
        up_ids = item['detectors']['up']['ids']
        down1_ids = item['detectors']['down1']['ids']
        down2_ids = item['detectors']['down2']['ids']

        up_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in up_ids])/4
        down1_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in down1_ids])/4
        down2_count = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in down2_ids])/4

        history[ramp_name]['up'].append(up_count)
        history[ramp_name]['down1'].append(down1_count)
        history[ramp_name]['down2'].append(down2_count)

    return history

ramp_times = [0] + list(np.arange(600, 3601, step=300)) + [SIM_END]

ramp1_demand = [9450, 9450, 9500, 9500, 9500, 9500, 9500, 9450, 9400, 9400, 8900, 8250, 8250]
ramp2_demand = [8250, 8250, 8350, 8350, 8350, 8350, 8350, 8250, 8150, 8150, 8150, 8050, 8050]
ramp3_demand = [8100, 8100, 8175, 8175, 8175, 8175, 8175, 8175, 8100, 8100, 8100, 8100, 8100]
ramp4_demand = [7925, 7925, 8000, 8000, 8000, 8000, 8000, 7900, 7775, 7775, 7775, 7775, 7775]

def get_ramp1_flow(t):
    return np.interp(t, ramp_times, ramp1_demand)

def get_ramp2_flow(t):
    return np.interp(t, ramp_times, ramp2_demand)

def get_ramp3_flow(t):
    return np.interp(t, ramp_times, ramp3_demand)

def get_ramp4_flow(t):
    return np.interp(t, ramp_times, ramp4_demand)

routes = {
    1:  ["ramp1_to_end", "ramp1_to_off2", "ramp1_to_off3", "ramp1_to_off4"],
    2:  ["ramp2_to_end", "ramp2_to_off3", "ramp2_to_off4"],
    3:  ["ramp3_to_end", "ramp3_to_off4"],
    4:  ["ramp4_to_end"]
}

probs = {
    1:  [0.7, 0.1, 0.1, 0.1],
    2:  [0.6, 0.2, 0.2],
    3:  [0.8, 0.2],
    4:  [1.0]
}

def insertRampVehicles(rampIndex, t):
    fncs = {1: get_ramp1_flow, 2: get_ramp2_flow, 3: get_ramp3_flow, 4: get_ramp4_flow}

    get_flow_fnc = fncs[rampIndex]

    V = get_flow_fnc(t)
    p = V / (3600 * 2)

    for lane in range(2):
        if np.random.random() < p:
            route = np.random.choice(routes[rampIndex], p=probs[rampIndex])

            traci.vehicle.add(f"ramp{rampIndex}_{t}_{lane}", route, typeID="car_ramp",
                departLane="best", departPos="free", departSpeed="random")

def insertVehicles():
    t = traci.simulation.getTime()

    for i in range(4):
        insertRampVehicles(i+1, t)
traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

# ----------- SIMULATION -----------
for i in tqdm(range(SIM_END)):
    insertVehicles()

    # if i % CONTROL_STEP == 0 and i != 0:
    history = getAllVehCounts(history)

    times.append(i)

    traci.simulationStep()

traci.close()

def getPlotValues(cfg, ramp_history, times):
    SHIFT_DN1 = cfg['detectors']['down1']['shift']
    SHIFT_DN2 = cfg['detectors']['down2']['shift']

    plot_values = {
        'up': {'x': times, 'y': []},
        'down1': {'x': times-SHIFT_DN1, 'y': []},
        'down2': {'x': times-SHIFT_DN2, 'y': []},
    }

    q0 = cfg['q0'] / 3600.0 / NUM_LANES

    up_cumsum = np.cumsum(np.asarray(ramp_history['up']))
    down1_cumsum = np.cumsum(np.asarray(ramp_history['down1']))
    down2_cumsum = np.cumsum(np.asarray(ramp_history['down2']))

    # Calculate modified cumulative arrivals N'(x,t)
    plot_values['up']['y'] = up_cumsum - (q0 * times)
    plot_values['down1']['y'] = down1_cumsum - (q0 * times)
    plot_values['down2']['y'] = down2_cumsum - (q0 * times)

    return plot_values

times = np.asarray(times)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for ramp_name, cfg in RAMPS.items():
    row, col = cfg["loc"]

    ax = axs[row,col]

    plot_values = getPlotValues(cfg, history[ramp_name], times)

    # Shift the downstream curves to the left (backward in time)
    ax.plot(plot_values['up']['x'], plot_values['up']['y'], color="r", label="Upstream")
    ax.plot(plot_values['down1']['x'], plot_values['down1']['y'], color="orange", label="Downstream 1")
    ax.plot(plot_values['down2']['x'], plot_values['down2']['y'], color="b", label="Downstream 2")

    ax.set_xlim(cfg['x_range'])
    ax.set_title(ramp_name)

plt.legend()

plt.savefig(os.path.join("plot", "fig_13.png"))
plt.show()
