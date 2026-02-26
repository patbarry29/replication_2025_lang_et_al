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
NUM_EPISODES = 100
MAX_SPEED = 27.78

MAX_TTS = 4119.0
AVG_TTS = 3405.64
TLS_ID = "junction_ramp"
SUMO_PATH = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","sumo_network","data","simulation.sumocfg")

# Define detector ID lists matching your XML configuration
UPSTREAM_DETS = [f"det_upstream_{i}" for i in range(4)]
DOWNSTREAM_DETS = [f"det_loc2_{i}" for i in range(4)] + [f"det_loc3_{i}" for i in range(4)]
RAMP_ARR_DETS = [f"det_ramp_arr_{i}" for i in range(2)]
RAMP_DEP_DETS = [f"det_ramp_dep_{i}" for i in range(2)]

def get_traffic_state(last_green_duration, ramp_arr, ramp_dep, upstream, downstream):
    """Build the state vector using pre-aggregated (averaged) detector readings
    from the previous 15-second control interval.
    """
    state = []
    state.extend([upstream['occ'], upstream['speed'], upstream['veh']])
    state.extend([downstream['occ'], downstream['speed'], downstream['veh']])

    # Queue is an instantaneous snapshot (not averaged)
    queue_length = traci.edge.getLastStepVehicleNumber("edge_ramp_2")
    state.extend([queue_length, ramp_arr, ramp_dep, last_green_duration])

    return state

def normalize_state(raw_state, tracker):
    raw_state = np.array(raw_state)
    tracker.push(raw_state)
    std = tracker.std()
    std[std == 0] = 1e-8 # Prevent division by zero
    return (raw_state - tracker.mean) / std

def apply_action_and_get_reward(action_ratio):
    """Run one 15s control step and return reward + averaged detector stats."""

    green = int(action_ratio * 15)
    red = 15 - green

    tts = 0
    n_steps = 0

    unique_arr_ids = set()
    unique_dep_ids = set()

    # Accumulators
    stats = {
        "up":   {"occ": 0.0, "speed": 0.0, "veh": 0.0},
        "down": {"occ": 0.0, "speed": 0.0, "veh": 0.0},
    }

    def aggregate(detectors):
        occ = np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in detectors])

        raw_speeds = [traci.inductionloop.getLastStepMeanSpeed(d) for d in detectors]
        speeds = [s if s != -1.0 else MAX_SPEED for s in raw_speeds]
        speed = np.mean(speeds)

        veh = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in detectors])
        return occ, speed, veh

    def poll():
        nonlocal n_steps

        # ramp vehicles (unique IDs)
        for d in RAMP_ARR_DETS:
            unique_arr_ids.update(traci.inductionloop.getLastStepVehicleIDs(d))
        for d in RAMP_DEP_DETS:
            unique_dep_ids.update(traci.inductionloop.getLastStepVehicleIDs(d))

        # upstream / downstream
        for name, dets in [("up", UPSTREAM_DETS), ("down", DOWNSTREAM_DETS)]:
            occ, speed, veh = aggregate(dets)
            stats[name]["occ"]   += occ
            stats[name]["speed"] += speed
            stats[name]["veh"]   += veh

        n_steps += 1

    # --- Run phases ---
    for duration, state in [(green, "GGGGGG"), (red, "GGGGrr")]:
        if duration <= 0:
            continue

        traci.trafficlight.setRedYellowGreenState(TLS_ID, state)
        for _ in range(duration):
            traci.simulationStep()
            tts += traci.vehicle.getIDCount()
            poll()

    # --- Compute outputs ---
    n = max(n_steps, 1)

    reward = (MAX_TTS - tts) / AVG_TTS

    agg_ramp_arr = len(unique_arr_ids) / n
    agg_ramp_dep = len(unique_dep_ids) / n

    for location in stats:
        for key in stats[location]:
            stats[location][key] /= n

    return reward, agg_ramp_arr, agg_ramp_dep, stats["up"], stats["down"]

def train(use_replacement=False):
    agent = SharedActorCritic(STATE_DIM)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]
    traci.start(sumo_cmd)

    line, ax, fig = init_plot()
    all_scores = []

    for episode in range(1, NUM_EPISODES+1):
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

        traci.load(["-c", SUMO_PATH])

        # Bootstrap: run 15 sim steps to seed the initial state with averaged detector readings
        _, _ra, _rd, _up, _dn = apply_action_and_get_reward(
            action_ratio=1.0,
        )

        last_green_duration = int(1.0 * 15)

        raw_state = get_traffic_state(last_green_duration, _ra, _rd, _up, _dn)
        state = normalize_state(raw_state, state_tracker)

        prev_demand = raw_state[7]

        for step in range(CONTROL_STEPS_PER_EPISODE):
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

            curr_queue  = raw_state[6]
            curr_demand = raw_state[7]

            # ---- Policy ----
            with torch.no_grad():
                dist, value = agent(state_tensor)
                raw_action = dist.sample()
                log_prob = dist.log_prob(raw_action)

            action = torch.clamp(raw_action, 0.0, 1.0).item()
            penalty = 0.0

            # ---- Replacement logic ----
            if use_replacement:
                lower_bound = calculate_lower_bound(prev_demand, curr_queue)

                if action < lower_bound:
                    penalty = calculate_penalty(curr_queue, curr_demand, action)
                    print(
                        f"[Replacement] Step {step} | "
                        f"Agent: {action:.3f} | "
                        f"LowerBound: {lower_bound:.3f} | "
                        f"Penalty: {penalty:.3f}"
                    )
                    action = lower_bound

            # ---- Environment step ----
            base_reward, avg_ra, avg_rd, agg_up, agg_down = \
                apply_action_and_get_reward(action)

            reward = base_reward - penalty

            # ---- Next state ----
            green_duration = int(action * 15)

            raw_next_state = get_traffic_state(
                green_duration, avg_ra, avg_rd, agg_up, agg_down
            )
            next_state = normalize_state(raw_next_state, state_tracker)

            final_queue = raw_next_state[6]
            spillback = final_queue > 0.9 * 42.0

            # ---- Termination logic ----
            if spillback:
                reward -= 10.0
                done = True
                print(
                    f"[FAILURE] Step {step} | "
                    f"Queue {final_queue:.0f} | "
                    f"Reward {reward:.3f}"
                )
            else:
                done = (step == CONTROL_STEPS_PER_EPISODE - 1)
                print(
                    f"Step {step} | "
                    f"Q {curr_queue:.0f} → {final_queue:.0f} | "
                    f"Demand {curr_demand:.2f} | "
                    f"Reward {reward:.3f}"
                )

            # ---- Store trajectory ----
            states.append(state_tensor)
            actions.append(raw_action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            # ---- Advance ----
            state = next_state
            raw_state = raw_next_state
            prev_demand = curr_demand

            if done:
                break

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
            os.makedirs("models", exist_ok=True)
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