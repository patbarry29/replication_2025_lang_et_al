import os
import pickle
import time

import numpy as np
import torch
import traci

from model import SharedActorCritic
from train import get_traffic_state

# --- Hyperparameters ---
STATE_DIM = 10
CONTROL_STEPS_PER_EPISODE = 240
SIM_STEPS_PER_CONTROL = 15

SUMO_PATH = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","sumo_network","data","simulation.sumocfg")

# Define detector ID lists matching your XML configuration
UPSTREAM_DETS = [f"det_upstream_{i}" for i in range(4)]
DOWNSTREAM_DETS = [f"det_loc2_{i}" for i in range(4)] + [f"det_loc3_{i}" for i in range(4)]
RAMP_ARR_DETS = [f"det_ramp_arr_{i}" for i in range(2)]
RAMP_DEP_DETS = [f"det_ramp_dep_{i}" for i in range(2)]

TLS_ID = "1494194482"

def evaluate_model(model_path, state_tracker):
    agent = SharedActorCritic(STATE_DIM)
    agent.load_state_dict(torch.load(model_path))
    agent.eval() # Set to deterministic evaluation mode

    # Use sumo-gui for visual inspection
    sumo_cmd = ["sumo-gui", "-c", SUMO_PATH, "--start"]
    traci.start(sumo_cmd)

    time.sleep(5)

    last_green_duration = 0
    raw_state = get_traffic_state(last_green_duration)

    # Must use the exact mean/std tracker saved from the training run
    state = normalize_state_eval(raw_state, state_tracker)

    total_tts = 0
    max_queue = 0

    for step in range(CONTROL_STEPS_PER_EPISODE):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            dist, _ = agent(state_tensor)
            # Use the mean of the distribution for deterministic evaluation, do not sample
            action = dist.mean
            action = torch.clamp(action, 0.0, 1.0)

        green_duration = int(action.item() * 15)
        red_duration = 15 - green_duration

        # Execute Green Phase
        if green_duration > 0:
            traci.trafficlight.setRedYellowGreenState(TLS_ID, "GGGGGG")
            for _ in range(green_duration):
                traci.simulationStep()
                total_tts += traci.vehicle.getIDCount()
                current_queue = traci.edge.getLastStepVehicleNumber("edge_ramp")
                max_queue = max(max_queue, current_queue)

        # Execute Red Phase
        if red_duration > 0:
            traci.trafficlight.setRedYellowGreenState(TLS_ID, "GGGGrr")
            for _ in range(red_duration):
                traci.simulationStep()
                total_tts += traci.vehicle.getIDCount()
                current_queue = traci.edge.getLastStepVehicleNumber("edge_ramp")
                max_queue = max(max_queue, current_queue)

        raw_next_state = get_traffic_state(green_duration)
        state = normalize_state_eval(raw_next_state, state_tracker)

    traci.close()
    print(f"Evaluation Complete. Total TTS: {total_tts}, Max Queue: {max_queue}")

def normalize_state_eval(raw_state, tracker):
    # During evaluation, do not push new states to the tracker
    # Only normalize using the frozen mean and std from training
    raw_state = np.array(raw_state)
    std = tracker.std()
    std[std == 0] = 1e-8
    return (raw_state - tracker.mean) / std

if __name__ == "__main__":
    MODEL_PATH = os.path.join("models","model_ep90.pth")
    TRACKER_PATH = os.path.join("models","state_tracker_ep90.pkl")

    with open(TRACKER_PATH, "rb") as f:
        state_tracker = pickle.load(f)

    evaluate_model(MODEL_PATH, state_tracker)