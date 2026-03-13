import traci
import numpy as np

MAIN_T = [0, 600, 600.1, 3300, 3600, 4200]
MAIN_VEH = [7400, 7400, 7900, 7900, 4000, 4000]

RAMP_T = [0, 600, 600.1, 3300, 3600, 4200]
RAMP_VEH = [600, 1000, 1300, 1300, 500, 500]

def get_mainline_flow(t):
    return np.interp(t, MAIN_T, MAIN_VEH)

def get_ramp_flow(t):
    return np.interp(t, RAMP_T, RAMP_VEH)

def insertVehicles():
    t = traci.simulation.getTime()

    V_main = get_mainline_flow(t)
    p_main = V_main / (3600 * 4)

    for lane in range(4):
        if np.random.random() < p_main:
            traci.vehicle.add(f"main_{t}_{lane}", "route_main", typeID="car_main",
                departLane="best", departPos="free", departSpeed="random")

    # Process Ramp
    V_ramp = get_ramp_flow(t)
    p_ramp = V_ramp / (3600 * 2)

    for lane in range(2):
        if np.random.random() < p_ramp:
            traci.vehicle.add(f"ramp_{t}_{lane}", "route_ramp", typeID="car_ramp",
                departLane="best", departPos="free", departSpeed="random")

