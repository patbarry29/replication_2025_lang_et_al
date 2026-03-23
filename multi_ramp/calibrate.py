import numpy as np

import traci

SUMO_BINARY = "sumo"
SUMO_CONFIG = "simulation.sumocfg"
WARMUP_START = 500
WARMUP_END = 600
SIM_END = 4500

AREAS = {
    "Area 1": {
        "loc": (1, 1),
        "x_range": (1400, 3600),
        "bg_flow": 8100,
        "detectors": {
            "+955 m":  {"ids": [f"det_a1_up_{i}"  for i in range(4)],   "dist": 0},
            "+1317 m": {"ids": [f"det_a1_dn1_{i}" for i in range(4)],   "dist": 362},
            "+1652 m": {"ids": [f"det_a1_dn2_{i}" for i in range(4)],   "dist": 362 + 335},
        },
    },
    "Area 2": {
        "loc": (0, 1),
        "x_range": (2100, 4600),
        "bg_flow": 8100,
        "detectors": {
            "+3916 m": {"ids": [f"det_a2_up_{i}"  for i in range(4)],   "dist": 0},
            "+4254 m": {"ids": [f"det_a2_dn1_{i}" for i in range(1, 5)], "dist": 338},
            "+4404 m": {"ids": [f"det_a2_dn2_{i}" for i in range(4)],   "dist": 338 + 250},
        },
    },
    "Area 3": {
        "loc": (1, 0),
        "x_range": (1500, 3900),
        "bg_flow": 8100,
        "detectors": {
            "+5221 m": {"ids": [f"det_a3_up_{i}"  for i in range(4)],   "dist": 0},
            "+5470 m": {"ids": [f"det_a3_dn1_{i}" for i in range(4)],   "dist": 249},
            "+5620 m": {"ids": [f"det_a3_dn2_{i}" for i in range(1, 5)], "dist": 249 + 250},
        },
    },
    "Area 4": {
        "loc": (0, 0),
        "x_range": (1000, 3900),
        "bg_flow": 8100,
        "detectors": {
            "+6822 m": {"ids": [f"det_a4_up_{i}"  for i in range(4)],   "dist": 0},
            "+7335 m": {"ids": [f"det_a4_dn1_{i}" for i in range(4)],   "dist": 513},
            "+7635 m": {"ids": [f"det_a4_dn2_{i}" for i in range(4)],   "dist": 513 + 300},
        },
    },
}

ramp_times = [0] + list(np.arange(600, 3601, step=300)) + [SIM_END]

ramp_demands = {
    1:  [1700, 1700, 2175, 2175, 2175, 2175, 2175, 2175, 1400, 1400, 1400, 1400, 1400],
    2:  [1030, 1030, 1243, 1243, 1243, 1243, 1243, 1080, 960, 960, 960, 960, 960],
    3:  [1103, 1103, 1292, 1292, 1292, 1292, 1292, 1103, 1086, 1086, 1086, 1046, 1046],
    4:  [1313, 1313, 2221, 2221, 2221, 2221, 2221, 1313, 1300, 1300, 1275, 1250, 1250]
}

ramp_demands = {
    1:  [2175]*13,
    2:  [1678]*13,
    3:  [1503]*13,
    4:  [2300]*13
}

def get_ramp1_flow(t):
    return np.interp(t, ramp_times, ramp_demands[1])

def get_ramp2_flow(t):
    return np.interp(t, ramp_times, ramp_demands[2])

def get_ramp3_flow(t):
    return np.interp(t, ramp_times, ramp_demands[3])

def get_ramp4_flow(t):
    return np.interp(t, ramp_times, ramp_demands[4])
routes = {
    1:  ["ramp1_to_end", "ramp1_to_off1", "ramp1_to_off2", "ramp1_to_off3", "ramp1_to_off4"],
    2:  ["ramp2_to_end", "ramp2_to_off3", "ramp2_to_off4"],
    3:  ["ramp3_to_end", "ramp3_to_off4"],
    4:  ["ramp4_to_end"]
}

# probs = {
#     1:  [0.6, 0.1, 0.1, 0.1, 0.1],
#     2:  [0.8, 0.1, 0.1],
#     3:  [0.9, 0.1],
#     4:  [1.0]
# }

probs = {
    1:  [0.5, 0.2, 0.1, 0.1, 0.1],
    2:  [0.6, 0.2, 0.2],
    3:  [0.7, 0.3],
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


def calibrate():
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    counts = {area: 0 for area in AREAS}
    speeds = {area: [] for area in AREAS}

    while traci.simulation.getTime() < WARMUP_END:
        insertVehicles()

        traci.simulationStep()
        current_time = traci.simulation.getTime()

        if WARMUP_START <= current_time < WARMUP_END:
            for area, cfg in AREAS.items():
                area_speed = 0
                det_count = 0
                for label, det in cfg["detectors"].items():
                    for d in det["ids"]:
                        # Tally counts
                        counts[area] += traci.inductionloop.getLastStepVehicleNumber(d)
                        # Average speeds
                        speed = traci.inductionloop.getLastStepMeanSpeed(d)
                        if speed >= 0:  # -1 means no vehicle crossed
                            area_speed += speed
                            det_count += 1
                if det_count > 0:
                    speeds[area].append(area_speed / det_count)

    traci.close()

    time_window_s = WARMUP_END - WARMUP_START

    for area, cfg in AREAS.items():
        num_locs = len(cfg["detectors"])
        bg_flow = (counts[area] / num_locs) / time_window_s * 3600
        avg_speed = sum(speeds[area]) / len(speeds[area]) if speeds[area] else 27.78

        print(f"--- {area} ---")
        print(f"Calibrated bg_flow: {round(bg_flow)} veh/h")
        print(f"Average speed: {round(avg_speed, 2)} m/s")

        for label, det in cfg["detectors"].items():
            if det["dist"] > 0:
                shift = round(det["dist"] / avg_speed)
                print(f"Time shift for {label}: {shift} s")

if __name__ == "__main__":
    calibrate()