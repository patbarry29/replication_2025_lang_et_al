import os
import pickle
import time
import numpy as np
import torch
import matplotlib.subplots as plt

from config import (
    STATE_DIM, SUMO_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS, TLS_ID
)
from env import RampMeterEnv
from controllers import RLController
from utils import normalize_static, format_state_vector
from model import SharedActorCritic
from action_replacement import calculate_lower_bound

def visualise(model_path, tracker_path):
    sumo_cmd = ["sumo-gui", "-c", SUMO_PATH, "--start"]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd, tls_id=TLS_ID, upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS, ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS, ramp_edge="edge_ramp_2"
    )

    agent = SharedActorCritic(STATE_DIM)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()

    with open(tracker_path, "rb") as f:
        state_tracker = pickle.load(f)

    controller = RLController(
        agent=agent, state_tracker=state_tracker,
        normalize_fnc=normalize_static, use_replacement=True
    )

    env.start()
    time.sleep(5)

    # Bootstrap phase
    env.apply_action_and_get_tts(0, SIM_STEPS_PER_CONTROL)
    raw_state_dict = env.get_traffic_state(SIM_STEPS_PER_CONTROL)
    raw_state = format_state_vector(raw_state_dict, 0)

    history = {
        "step": [], "green_times": [], "lower_bounds": [],
        "queues": [], "downstream_speeds": [], "replacements": []
    }
    max_queue = 0

    for step in range(CONTROL_STEPS_PER_EPISODE):
        prev_demand = raw_state[7]
        curr_queue = raw_state[6]

        action_ratio, _, _, _ = controller.execute_control(raw_state, is_training=False)

        lower_bound = calculate_lower_bound(prev_demand, curr_queue)
        replaced = action_ratio == lower_bound and lower_bound > 0.0

        green_duration = int(action_ratio * SIM_STEPS_PER_CONTROL)
        red_duration = SIM_STEPS_PER_CONTROL - green_duration

        print(f"Eval Step {step} | Exec Green: {green_duration}s | Queue: {curr_queue} | Demand/s: {prev_demand:.2f}")

        env.apply_action_and_get_tts(green_duration, red_duration)

        next_state_dict = env.get_traffic_state(SIM_STEPS_PER_CONTROL)
        next_raw_state = format_state_vector(next_state_dict, green_duration)

        current_queue = next_raw_state[6]
        max_queue = max(max_queue, current_queue)

        history["step"].append(step)
        history["green_times"].append(green_duration)
        history["lower_bounds"].append(int(lower_bound * SIM_STEPS_PER_CONTROL))
        history["queues"].append(current_queue)
        history["downstream_speeds"].append(next_state_dict["downstream"]["speed"])
        history["replacements"].append(replaced)

        raw_state = next_raw_state

    env.close()
    print(f"Evaluation Complete. Max Queue: {max_queue}")
    plot_evaluation(history)

def plot_evaluation(hist):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(hist["step"], hist["green_times"], label="Executed Green Duration", drawstyle="steps-mid")
    axs[0].plot(hist["step"], hist["lower_bounds"], label="Lower Bound Constraint", linestyle="--", alpha=0.7, drawstyle="steps-mid")

    replacements_idx = [i for i, r in enumerate(hist["replacements"]) if r]
    replacements_y = [hist["green_times"][i] for i in replacements_idx]
    axs[0].scatter(replacements_idx, replacements_y, color="red", zorder=5, label="Action Replaced")

    axs[0].set_ylabel("Green Time (s)")
    axs[0].set_title("Agent Actions vs Safety Bounds")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(hist["step"], hist["queues"], color="red", label="Ramp Queue")
    axs[1].axhline(y=42, color="black", linestyle="--", label="Max Capacity (42)")
    axs[1].set_ylabel("Number of Vehicles")
    axs[1].set_title("Ramp Storage Status")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(hist["step"], hist["downstream_speeds"], color="green", label="Downstream Speed")
    axs[2].set_xlabel("Control Step")
    axs[2].set_ylabel("Speed (m/s)")
    axs[2].set_title("Mainline Traffic State (Capacity Drop Monitor)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = os.path.join("models_replacement","model_ep100.pth")
    tracker_path = os.path.join("models_replacement","state_tracker_ep100.pkl")
    visualise(model_path, tracker_path)