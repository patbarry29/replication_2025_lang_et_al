import os
import pickle
import numpy as np
import torch
import traci

from model import SharedActorCritic
from train import get_traffic_state, apply_action_and_get_reward
from action_replacement import calculate_lower_bound

# --- Hyperparameters ---
STATE_DIM = 10
CONTROL_STEPS_PER_EPISODE = 280
SIM_STEPS_PER_CONTROL = 15

SUMO_PATH = os.path.join(r"C:\Users", "pbarry", "Documents", "2025_yang_dqn", "with_traffic_light", "sumo_network", "data", "simulation.sumocfg")
TLS_ID = "junction_ramp"

def run_evaluation(model_path=None, tracker_path=None, use_replacement=False, no_control=False):
    agent = None
    state_tracker = None

    if not no_control:
        agent = SharedActorCritic(STATE_DIM)
        agent.load_state_dict(torch.load(model_path))
        agent.eval()
        with open(tracker_path, "rb") as f:
            state_tracker = pickle.load(f)

    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]
    traci.start(sumo_cmd)

    # Bootstrap step
    _, _, avg_ra, avg_rd, agg_up, agg_down = apply_action_and_get_reward(action_ratio=1.0)

    last_green_duration = int(1.0 * 15) if no_control else int(0.0 * 15)
    raw_state = get_traffic_state(last_green_duration, avg_ra, avg_rd, agg_up, agg_down)

    if not no_control:
        state = normalize_state_eval(raw_state, state_tracker)

    prev_demand = raw_state[7]

    total_tts_seconds = 0
    max_queue = 0
    spillback_occurred = False

    for step in range(CONTROL_STEPS_PER_EPISODE):
        curr_queue = raw_state[6]
        curr_demand = raw_state[7]

        if no_control:
            env_action = 1.0  # Always green
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                dist, _ = agent(state_tensor)
                # Use deterministic action (mean) for evaluation, not sampling
                action = dist.mean
                env_action = torch.clamp(action, 0.0, 1.0).item()

            if use_replacement:
                lower_bound = calculate_lower_bound(prev_demand, curr_queue)
                if env_action < lower_bound:
                    env_action = lower_bound

        # Step environment
        step_tts, _, agg_ramp_arr, agg_ramp_dep, agg_up, agg_down = apply_action_and_get_reward(env_action)

        total_tts_seconds += step_tts

        current_queue = traci.edge.getLastStepVehicleNumber("edge_ramp_2")
        max_queue = max(max_queue, current_queue)

        if current_queue > 42 * 0.9:
            spillback_occurred = True

        green_duration = int(env_action * 15)
        raw_next_state = get_traffic_state(green_duration, agg_ramp_arr, agg_ramp_dep, agg_up, agg_down)

        if not no_control:
            state = normalize_state_eval(raw_next_state, state_tracker)

        raw_state = raw_next_state
        prev_demand = curr_demand

    traci.close()

    # Calculate TTS in hours
    tts_hours = total_tts_seconds / 3600.0
    return tts_hours, max_queue, spillback_occurred

def normalize_state_eval(raw_state, tracker):
    raw_state = np.array(raw_state)
    std = tracker.std()
    std[std == 0] = 1e-8
    return (raw_state - tracker.mean) / std

if __name__ == "__main__":
    print("--- Single Ramp Evaluation ---")

    # 1. No Control
    print("\nRunning No-Control Baseline...")
    tts_nc, mq_nc, sb_nc = run_evaluation(no_control=True)
    print(f"No-Control -> TTS: {tts_nc:.2f} h | Max Queue: {mq_nc} | Spillback: {sb_nc}")

    # 2. RL-based (Unrestrained)
    print("\nRunning RL-Based (Unrestrained)...")
    base_model = os.path.join("models", "model_ep100.pth")
    base_tracker = os.path.join("models", "state_tracker_ep100.pkl")
    tts_base, mq_base, sb_base = run_evaluation(model_path=base_model, tracker_path=base_tracker, use_replacement=False)
    print(f"RL-Based   -> TTS: {tts_base:.2f} h | Max Queue: {mq_base} | Spillback: {sb_base}")

    # 3. RL-based with Action Replacement
    print("\nRunning RL-Based (With Replacement)...")
    rep_model = os.path.join("models_replacement", "model_ep100.pth")
    rep_tracker = os.path.join("models_replacement", "state_tracker_ep100.pkl")
    tts_rep, mq_rep, sb_rep = run_evaluation(model_path=rep_model, tracker_path=rep_tracker, use_replacement=True)
    print(f"RL+Replace -> TTS: {tts_rep:.2f} h | Max Queue: {mq_rep} | Spillback: {sb_rep}")

    print("\n--- Target values from paper ---")
    print("No-control: 595.77 h")
    print("ALINEA:     578.27 h")
    print("RL-based:   564.98 h")