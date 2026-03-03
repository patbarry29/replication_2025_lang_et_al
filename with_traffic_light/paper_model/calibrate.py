import os
import traci
import numpy as np
import torch

from config import (MAX_SPEED, UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS,
                    SUMO_PATH, CONTROL_STEPS_PER_EPISODE, TLS_ID, SIM_STEPS_PER_CONTROL)

def get_aggregated_stats():
    """Helper to fetch and process induction loop data."""
    def _process(detectors):
        occ = np.mean([traci.inductionloop.getLastIntervalOccupancy(d) for d in detectors])
        raw_speeds = [traci.inductionloop.getLastIntervalMeanSpeed(d) for d in detectors]
        speeds = [s if s >= 0 else MAX_SPEED for s in raw_speeds]
        speed = np.mean(speeds)
        veh_total = np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in detectors])
        return occ, speed, (veh_total / SIM_STEPS_PER_CONTROL)

    up_occ, up_spd, up_v = _process(UPSTREAM_DETS)
    dn_occ, dn_spd, dn_v = _process(DOWNSTREAM_DETS)

    arr_v = np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in RAMP_ARR_DETS]) / SIM_STEPS_PER_CONTROL
    dep_v = np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in RAMP_DEP_DETS]) / SIM_STEPS_PER_CONTROL

    return up_occ, up_spd, up_v, dn_occ, dn_spd, dn_v, arr_v, dep_v

def run_calibration_episode():
    traci.start(["sumo", "-c", SUMO_PATH, "--no-step-log", "true"])

    tts_per_step = []
    state_vectors = []

    print("Starting Baseline Calibration (No-Control)...")

    for step in range(CONTROL_STEPS_PER_EPISODE):
        # 1. Action: Baseline (Ramp always open/Green)
        traci.trafficlight.setRedYellowGreenState(TLS_ID, "gg")

        current_step_tts = 0
        for _ in range(int(SIM_STEPS_PER_CONTROL)):
            traci.simulationStep()
            current_step_tts += traci.vehicle.getIDCount()

        # 2. Collect Measurements
        u_occ, u_spd, u_v, d_occ, d_spd, d_v, r_arr, r_dep = get_aggregated_stats()
        q_len = traci.edge.getLastStepVehicleNumber("edge_ramp_2")
        last_green = SIM_STEPS_PER_CONTROL # (1.0 ratio * SIM_STEPS_PER_CONTROL)

        # 3. Build State Vector (Matches your STATE_DIM = 10)
        state = [u_occ, u_spd, u_v, d_occ, d_spd, d_v, q_len, r_arr, r_dep, last_green]

        tts_per_step.append(current_step_tts)
        state_vectors.append(state)

    traci.close()
    return np.array(tts_per_step), np.array(state_vectors)

if __name__ == "__main__":
    tts_data, state_data = run_calibration_episode()

    # CALCULATE REWARD CONSTANTS
    alpha = np.max(tts_data)
    beta = np.mean(tts_data)

    # CALCULATE STATE NORMALIZATION
    means = np.mean(state_data, axis=0)
    stds = np.std(state_data, axis=0)
    stds[stds == 0] = 1e-8 # Prevent nan

    print(f"\n--- Calibration Results ---")
    print(f"REWARD_ALPHA (Max TTS): {alpha:.2f}")
    print(f"REWARD_BETA (Avg TTS): {beta:.2f}")
    print(f"STATE_MEANS: {means.round(4).tolist()}")
    print(f"STATE_STDS: {stds.round(4).tolist()}")