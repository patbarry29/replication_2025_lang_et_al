import os
import pickle
import time

import numpy as np
import torch
import traci
import matplotlib.pyplot as plt

from model import SharedActorCritic
from train import get_traffic_state, apply_action_and_get_reward
from action_replacement import calculate_lower_bound
from config import *

def evaluate_model(model_path, state_tracker):
    agent = SharedActorCritic(STATE_DIM)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()

    sumo_cmd = ["sumo-gui", "-c", SUMO_PATH, "--start"]
    traci.start(sumo_cmd)

    time.sleep(5)

    # Bootstrap: run 15 sim steps to get initial aggregated detector readings
    _, _, avg_ra, avg_rd, agg_up, agg_down = apply_action_and_get_reward(
        action_ratio=0.0,   # neutral metering
    )

    last_green_duration = int(0.0 * 15)

    raw_state = get_traffic_state(
        last_green_duration,
        avg_ra,
        avg_rd,
        agg_up,
        agg_down
    )

    state = normalize_state_eval(raw_state, state_tracker)
    prev_demand = raw_state[7]

    total_tts = 0
    max_queue = 0

    # History tracking
    history = {
        "step": [],
        "green_times": [],
        "lower_bounds": [],
        "queues": [],
        "downstream_speeds": [],
        "replacements": []
    }

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

        green_duration = int(env_action.item() * 15)
        lb_duration = int(lower_bound * 15)

        print(f"Eval Step {step} | Raw Action: {action.item():.3f} | Replaced: {replaced} (LB: {lower_bound:.3f}) | Exec Green: {green_duration}s | Queue: {curr_queue} | Demand/s: {curr_demand:.2f}")

        # Environment step with aggregated polling
        _, reward, agg_ramp_arr, agg_ramp_dep, agg_up, agg_down = apply_action_and_get_reward(env_action)

        # Track TTS and max queue from the post-step snapshot
        current_queue = traci.edge.getLastStepVehicleNumber("edge_ramp_2")
        max_queue = max(max_queue, current_queue)

        # Record metrics for plotting
        history["step"].append(step)
        history["green_times"].append(green_duration)
        history["lower_bounds"].append(lb_duration)
        history["queues"].append(current_queue)
        history["downstream_speeds"].append(agg_down["speed"])
        history["replacements"].append(replaced)

        raw_next_state = get_traffic_state(green_duration, agg_ramp_arr, agg_ramp_dep, agg_up, agg_down)
        state = normalize_state_eval(raw_next_state, state_tracker)

        raw_state = raw_next_state
        prev_demand = curr_demand

    traci.close()
    print(f"Evaluation Complete. Max Queue: {max_queue}")

    plot_evaluation(history)

def plot_evaluation(hist):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot 1: Green Light Duration & Action Replacement bounds
    axs[0].plot(hist["step"], hist["green_times"], label="Executed Green Duration", drawstyle="steps-mid")
    axs[0].plot(hist["step"], hist["lower_bounds"], label="Lower Bound Constraint", linestyle="--", alpha=0.7, drawstyle="steps-mid")

    # Highlight points where action was replaced
    replacements_idx = [i for i, r in enumerate(hist["replacements"]) if r]
    replacements_y = [hist["green_times"][i] for i in replacements_idx]
    axs[0].scatter(replacements_idx, replacements_y, color="red", zorder=5, label="Action Replaced")

    axs[0].set_ylabel("Green Time (s)")
    axs[0].set_title("Agent Actions vs Safety Bounds")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Ramp Queue length
    axs[1].plot(hist["step"], hist["queues"], color="red", label="Ramp Queue")
    axs[1].axhline(y=42, color="black", linestyle="--", label="Max Capacity (42)")
    axs[1].set_ylabel("Number of Vehicles")
    axs[1].set_title("Ramp Storage Status")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Mainline Downstream Speed
    axs[2].plot(hist["step"], hist["downstream_speeds"], color="green", label="Downstream Speed")
    axs[2].set_xlabel("Control Step")
    axs[2].set_ylabel("Speed (m/s)")
    axs[2].set_title("Mainline Traffic State (Capacity Drop Monitor)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def normalize_state_eval(raw_state, tracker):
    raw_state = np.array(raw_state)
    std = tracker.std()
    std[std == 0] = 1e-8
    return (raw_state - tracker.mean) / std

if __name__ == "__main__":
    MODEL_PATH = os.path.join("models_replacement","model_ep100.pth")
    TRACKER_PATH = os.path.join("models_replacement","state_tracker_ep100.pkl")
    # MODEL_PATH = os.path.join("models","model_ep100.pth")
    # TRACKER_PATH = os.path.join("models","state_tracker_ep100.pkl")

    with open(TRACKER_PATH, "rb") as f:
        state_tracker = pickle.load(f)

    evaluate_model(MODEL_PATH, state_tracker)