import torch
import traci
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn_model.dqn_agent import Agent
from dqn_model.live_plot import init_plot, update_live_plot

# Constants
SUMO_PATH = os.path.join("sumo_network", "data", "simulation.sumocfg")

# ALPHA and BETA were discovered using the calibrate.py file
ALPHA = 5601.0
BETA = 4238.26

CONTROL_INTERVAL = 15
MAX_STEPS = 3600
MAX_SPEED, MAX_OCC, MAX_QUEUE, MAX_FLOW = 30.0, 100.0, 20.0, 2000.0
TL_ID = "1494194482"

# Hyperparameters
EPISODES = 50
BATCH_SIZE = 32

def get_detector_data(detector_prefix, num_lanes):
    speeds, occs, flows = [], [], []
    for i in range(num_lanes):
        det_id = f"{detector_prefix}_{i}"
        speeds.append(traci.inductionloop.getLastStepMeanSpeed(det_id))
        occs.append(traci.inductionloop.getLastStepOccupancy(det_id))
        # Flow = Vehicle count in the last step
        flows.append(traci.inductionloop.getLastStepVehicleNumber(det_id))

    # Normalize: Flow is usually 0, 1, or 2 vehicles per step.
    avg_flow = np.sum(flows) * 3600 / num_lanes

    return np.mean(speeds), np.mean(occs), avg_flow

def run_simulation(agent):
    traci.start(["sumo", "-c", SUMO_PATH, "--no-warnings", "--no-step-log"])

    step = 0
    total_score = 0
    cumulative_veh = 0
    prev_state, prev_action = None, 0
    action = 0

    histories = {
        'speed15m': [], 'speed225m': [], 'speed475m': [],
        'occ15m': [], 'occ225m': [], 'occ475m': [],
        'flow15m': [], 'flow225m': [], 'flow475m': [],
        'queue': [], 'reward': []
    }
    buffer = {k: [] for k in histories.keys() if k != 'reward'}

    for step in tqdm(range(MAX_STEPS), desc=f"Ep /", leave=False):
        # 1. Collect Data
        s15, o15, f15 = get_detector_data("det_loc1", 5)
        s225, o225, f225 = get_detector_data("det_loc2", 4)
        s475, o475, f475 = get_detector_data("det_loc3", 4)
        current_q = (traci.lane.getLastStepHaltingNumber("edge_ramp_0") +
                     traci.lane.getLastStepHaltingNumber("edge_ramp_1")) / 2

        buffer['speed15m'].append(s15); buffer['occ15m'].append(o15); buffer['flow15m'].append(f15)
        buffer['speed225m'].append(s225); buffer['occ225m'].append(o225); buffer['flow225m'].append(f225)
        buffer['speed475m'].append(s475); buffer['occ475m'].append(o475); buffer['flow475m'].append(f475)
        buffer['queue'].append(current_q)

        # 3. Agent Control (Every 30s)
        if step % CONTROL_INTERVAL == 0 and step > 0:
            avg_vals = {k: np.mean(v) for k, v in buffer.items()}
            for k in buffer:
                histories[k].append(avg_vals[k])

            state = [
                avg_vals['speed15m'] / MAX_SPEED,
                avg_vals['occ15m'] / MAX_OCC,
                avg_vals['speed225m'] / MAX_SPEED,
                avg_vals['occ225m'] / MAX_OCC,
                avg_vals['speed475m'] / MAX_SPEED,
                avg_vals['occ475m'] / MAX_OCC,
                avg_vals['flow15m'] / MAX_FLOW, # Normalize flow (approx max 2000)
                avg_vals['flow225m'] / MAX_FLOW,
                avg_vals['flow475m'] / MAX_FLOW,
                avg_vals['queue'] / MAX_QUEUE
            ]


            current_tts = cumulative_veh
            reward = (ALPHA - current_tts) / BETA
            reward = np.clip(reward, -1.0, 1.0)
            total_score += reward
            histories['reward'].append(total_score)

            if prev_state is not None:
                agent.remember(prev_state, prev_action, reward, state, False)
                agent.replay()

            action = agent.act(state)

            prev_state, prev_action = state, action
            cumulative_veh = 0
            for k in buffer: buffer[k] = []

        if action == 0:
            # FREE FLOW
            traci.trafficlight.setPhase(TL_ID, 0) # Green
        else:
            # METERING (The "Smart Red")
            # E.g., 2 seconds Green, 4 seconds Red cycle
            cycle_position = step % 6
            if cycle_position < 2:
                traci.trafficlight.setPhase(TL_ID, 0) # Green (Release 1 car)
            else:
                traci.trafficlight.setPhase(TL_ID, 2) # Red (Hold back)


        cumulative_veh += traci.simulation.getMinExpectedNumber()
        traci.simulationStep()

    traci.close()

    master_agent.decay_epsilon()

    return histories, total_score

def plot_results(h):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    ctrl_time = np.arange(len(h['queue'])) * CONTROL_INTERVAL
    rew_time = np.arange(len(h['reward'])) * CONTROL_INTERVAL

    # Speed
    for loc in ['15m', '225m', '475m']:
        axs[0, 0].plot(ctrl_time, h[f'speed{loc}'], label=loc)
    axs[0, 0].set_title("Speed (m/s)"); axs[0, 0].legend()

    # Occupancy
    for loc in ['15m', '225m', '475m']:
        axs[0, 1].plot(ctrl_time, h[f'occ{loc}'], label=loc)
    axs[0, 1].set_title("Occupancy (%)"); axs[0, 1].legend()

    # Queue
    axs[1, 0].plot(ctrl_time, h['queue'], color='tab:red')
    axs[1, 0].set_title("Queue Length"); axs[1, 0].set_xlabel("Time (s)")

    # Cumulative Reward
    axs[1, 1].plot(rew_time, h['reward'], color='tab:green', linewidth=2)
    axs[1, 1].set_title("Cumulative Reward"); axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    master_agent = Agent(state_size=10, action_size=2)
    episode_scores = []

    # Track the best score (initialize with a very low number)
    best_score = -float('inf')

    # Ensure directory exists
    os.makedirs('models', exist_ok=True)

    line, ax, fig = init_plot()

    for e in tqdm(range(EPISODES), desc="Training Progress", unit="ep"):
        histories, score = run_simulation(master_agent)
        episode_scores.append(score)

        update_live_plot(episode_scores, line, ax, fig)

        # 1. Check if this is the best score so far
        if score > best_score:
            best_score = score
            save_path = os.path.join('models', 'best_model.pth')
            torch.save(master_agent.model.state_dict(), save_path)
            tqdm.write(f"--> New Best Score: {score:.2f}! Model saved to {save_path}")

        # 2. Regular checkpoint (optional, every 10 eps)
        # if (e + 1) % 10 == 0:
        #     torch.save(master_agent.model.state_dict(), os.path.join('models', f"model_ep{e+1}.pth"))

        print(f"\nEpisode {e+1}/{EPISODES} | Score: {score:.2f} | Epsilon: {master_agent.epsilon:.2f}\n")

    plt.ioff()
    plt.show()
    plot_results(histories)