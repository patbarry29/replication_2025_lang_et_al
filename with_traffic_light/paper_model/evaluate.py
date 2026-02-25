import os
import pickle
import time

import numpy as np
import torch
import traci

from model import SharedActorCritic
from train import get_traffic_state, apply_action_and_get_reward, UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS
from action_replacement import calculate_lower_bound

# --- Hyperparameters ---
STATE_DIM = 10
CONTROL_STEPS_PER_EPISODE = 240
SIM_STEPS_PER_CONTROL = 15

SUMO_PATH = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","sumo_network","data","simulation.sumocfg")

TLS_ID = "1494194482"

def evaluate_model(model_path, state_tracker):
    agent = SharedActorCritic(STATE_DIM)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()

    sumo_cmd = ["sumo-gui", "-c", SUMO_PATH, "--start"]
    traci.start(sumo_cmd)

    time.sleep(5)

    # Bootstrap: run 15 sim steps to get initial aggregated detector readings
    last_green_duration = 0
    init_ramp_arr, init_ramp_dep = 0, 0
    init_up = {'occ': 0.0, 'speed': 0.0, 'veh': 0.0}
    init_down = {'occ': 0.0, 'speed': 0.0, 'veh': 0.0}
    for _ in range(SIM_STEPS_PER_CONTROL):
        traci.simulationStep()
        init_ramp_arr += np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in RAMP_ARR_DETS])
        init_ramp_dep += np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in RAMP_DEP_DETS])
        init_up['occ'] += np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in UPSTREAM_DETS])
        init_up['speed'] += np.mean([traci.inductionloop.getLastStepMeanSpeed(d) for d in UPSTREAM_DETS])
        init_up['veh'] += np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in UPSTREAM_DETS])
        init_down['occ'] += np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in DOWNSTREAM_DETS])
        init_down['speed'] += np.mean([traci.inductionloop.getLastStepMeanSpeed(d) for d in DOWNSTREAM_DETS])
        init_down['veh'] += np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in DOWNSTREAM_DETS])
    init_up['occ'] /= SIM_STEPS_PER_CONTROL
    init_up['speed'] /= SIM_STEPS_PER_CONTROL
    init_up['veh'] /= SIM_STEPS_PER_CONTROL
    init_down['occ'] /= SIM_STEPS_PER_CONTROL
    init_down['speed'] /= SIM_STEPS_PER_CONTROL
    init_down['veh'] /= SIM_STEPS_PER_CONTROL
    init_ramp_arr /= SIM_STEPS_PER_CONTROL
    init_ramp_dep /= SIM_STEPS_PER_CONTROL

    raw_state = get_traffic_state(last_green_duration, init_ramp_arr, init_ramp_dep, init_up, init_down)
    state = normalize_state_eval(raw_state, state_tracker)

    prev_demand = raw_state[7]

    total_tts = 0
    max_queue = 0

    for step in range(CONTROL_STEPS_PER_EPISODE):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            dist, _ = agent(state_tensor)
            action = dist.mean
            env_action = torch.clamp(action, 0.0, 1.0)

        curr_queue = raw_state[6]
        curr_demand = raw_state[7]

        # Action replacement
        lower_bound = calculate_lower_bound(prev_demand, curr_queue)
        replaced = False
        if env_action.item() < lower_bound:
            env_action = torch.tensor(lower_bound)
            replaced = True

        print(f"Eval Step {step} | Raw Action: {action.item():.3f} | Replaced: {replaced} (LB: {lower_bound:.3f}) | Exec Green: {int(env_action.item() * 15)}s | Queue: {curr_queue} | Demand/s: {curr_demand:.2f}")

        # Environment step with aggregated polling
        reward, agg_ramp_arr, agg_ramp_dep, agg_up, agg_down = apply_action_and_get_reward(env_action, TLS_ID, 4470.0, 3556.64)

        # Track TTS and max queue from the post-step snapshot
        current_queue = traci.edge.getLastStepVehicleNumber("edge_ramp")
        max_queue = max(max_queue, current_queue)

        green_duration = int(env_action.item() * 15)
        raw_next_state = get_traffic_state(green_duration, agg_ramp_arr, agg_ramp_dep, agg_up, agg_down)
        state = normalize_state_eval(raw_next_state, state_tracker)

        raw_state = raw_next_state
        prev_demand = curr_demand

    traci.close()
    print(f"Evaluation Complete. Max Queue: {max_queue}")

def normalize_state_eval(raw_state, tracker):
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