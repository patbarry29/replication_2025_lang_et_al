import os
import pickle
import argparse
from matplotlib import pyplot as plt
import torch
import traci
import numpy as np

from ppo_loss import compute_gae, ppo_update
from model import SharedActorCritic
from stats import RunningStat
from live_plot import init_plot, update_live_plot
from action_replacement import calculate_lower_bound, calculate_penalty

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

def normalize_state(raw_state, tracker):
    raw_state = np.array(raw_state)
    tracker.push(raw_state)
    std = tracker.std()
    std[std == 0] = 1e-8 # Prevent division by zero
    return (raw_state - tracker.mean) / std

def get_traffic_state(last_green_duration):
    state = []

    # 1. Mainline State: Upstream
    up_occ = np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in UPSTREAM_DETS])
    up_speed = np.mean([traci.inductionloop.getLastStepMeanSpeed(d) for d in UPSTREAM_DETS])
    up_veh = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in UPSTREAM_DETS])
    state.extend([up_occ, up_speed, up_veh])

    # 2. Mainline State: Downstream
    down_occ = np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in DOWNSTREAM_DETS])
    down_speed = np.mean([traci.inductionloop.getLastStepMeanSpeed(d) for d in DOWNSTREAM_DETS])
    down_veh = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in DOWNSTREAM_DETS])
    state.extend([down_occ, down_speed, down_veh])

    # 3. Ramp State
    ramp_arr = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in RAMP_ARR_DETS])
    ramp_dep = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in RAMP_DEP_DETS])

    # Queue length is calculated by counting halting vehicles (speed < 0.1 m/s) on the ramp edge
    queue_length = traci.edge.getLastStepHaltingNumber("edge_ramp")

    state.extend([queue_length, ramp_arr, ramp_dep, last_green_duration])

    return state

def apply_action_and_get_reward(action_ratio, tls_id, max_tts, avg_tts):
    # Action is a continuous value in [0, 1]
    # Total control step is 15 simulation seconds
    green_duration = int(action_ratio * 15)
    red_duration = 15 - green_duration

    tts = 0

    # 1. Execute the Green Phase
    if green_duration > 0:
        # "GG" assumes a 2-lane ramp. Replace with your exact SUMO phase string.
        traci.trafficlight.setRedYellowGreenState(tls_id, "GGGGGG")
        for _ in range(green_duration):
            traci.simulationStep()
            tts += traci.vehicle.getIDCount()

    # 2. Execute the Red Phase
    if red_duration > 0:
        traci.trafficlight.setRedYellowGreenState(tls_id, "GGGGrr")
        for _ in range(red_duration):
            traci.simulationStep()
            tts += traci.vehicle.getIDCount()

    # 3. Calculate Reward
    reward = (max_tts - tts) / avg_tts

    return reward

def train(use_replacement=False):
    agent = SharedActorCritic(STATE_DIM)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]
    traci.start(sumo_cmd)

    # Replace with your calibrated values
    MAX_TTS = 4470.0
    AVG_TTS = 3556.64
    TLS_ID = "1494194482"

    line, ax, fig = init_plot()
    all_scores = []

    for episode in range(100):
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        traci.load(["-c", SUMO_PATH])

        # Initialize tracking variable for the first step
        last_green_duration = 0
        raw_state = get_traffic_state(last_green_duration)
        state = normalize_state(raw_state, state_tracker)

        prev_demand = raw_state[7]

        for step in range(CONTROL_STEPS_PER_EPISODE):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            curr_queue = raw_state[6]
            curr_demand = raw_state[7]

            with torch.no_grad():
                dist, state_value = agent(state_tensor)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action)

                env_action = torch.clamp(raw_action, 0.0, 1.0)

            step_penalty = 0.0

            if use_replacement:
                lower_bound = calculate_lower_bound(prev_demand, curr_queue)

                # If the agent's action violates the constraint, override it
                if env_action < lower_bound:
                    env_action = lower_bound
                    step_penalty = calculate_penalty(curr_queue, curr_demand, env_action)

            # Environment Step using the potentially replaced action
            base_reward = apply_action_and_get_reward(env_action, TLS_ID, MAX_TTS, AVG_TTS)

            # Apply the penalty to the environment reward
            reward = base_reward - step_penalty

            current_green_duration = int(env_action * 15)
            raw_next_state = get_traffic_state(current_green_duration)
            next_state = normalize_state(raw_next_state, state_tracker)
            done = (step == CONTROL_STEPS_PER_EPISODE - 1)

            states.append(state_tensor)
            actions.append(raw_action)
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            raw_state = raw_next_state
            last_green_duration = current_green_duration

            # Update previous demand for the next control step calculation
            prev_demand = curr_demand

        with torch.no_grad():
            _, next_value = agent(torch.FloatTensor(state).unsqueeze(0))

        states_tensor = torch.cat(states)
        actions_tensor = torch.cat(actions)
        log_probs_tensor = torch.cat(log_probs)

        returns, advantages = compute_gae(rewards, values, next_value.item(), dones)

        ppo_update(agent, optimizer, states_tensor, actions_tensor, log_probs_tensor, returns, advantages)

        total_episode_reward = sum(rewards)
        all_scores.append(total_episode_reward)
        update_live_plot(all_scores, line, ax, fig)

        print(f"\n\nEpisode {episode} Complete. Total Reward: {total_episode_reward}\n")

        if episode % 10 == 0:
            save_path = os.path.join("models", f"model_ep{episode}.pth")
            torch.save(agent.state_dict(), save_path)

            with open(os.path.join("models", f"state_tracker_ep{episode}.pkl"), "wb") as f:
                pickle.dump(state_tracker, f)

    plt.ioff()
    plt.show()
    traci.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Ramp Metering")
    parser.add_argument("--use_replacement", action="store_true", help="Enable action replacement module")
    args = parser.parse_args()

    state_tracker = RunningStat(shape=(10,))
    train(use_replacement=args.use_replacement)