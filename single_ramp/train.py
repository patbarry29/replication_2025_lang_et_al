import os
import pickle
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    STATE_DIM, SUMO_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS, TLS_ID
)
from env import RampMeterEnv
from controllers import RLController
from runner import run_episode
from utils import normalize_static
from model import SharedActorCritic
from ppo_loss import compute_gae, ppo_update
from stats import RunningStat
from live_plot import init_plot, update_live_plot


NUM_EPISODES = 250

def train(use_replacement, seed):
    model_dir = "models_replacement" if use_replacement else "models"
    os.makedirs(model_dir, exist_ok=True)

    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true", "--seed", str(seed)]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd,
        tls_id=TLS_ID,
        upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS,
        ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS,
        ramp_edge="edge_ramp_2"
    )

    agent = SharedActorCritic(STATE_DIM)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    state_tracker = RunningStat(shape=(STATE_DIM,))

    controller = RLController(
        agent=agent,
        state_tracker=state_tracker,
        normalize_fnc=normalize_static,
        use_replacement=use_replacement
    )

    line, ax, fig = init_plot(use_replacement)
    all_scores, history_steps, history_lengths, history_tts = [], [], [], []
    cumulative_steps = 0

    for episode in range(1, NUM_EPISODES+1):
        env.start()

        # Execute the episode using the runner
        trajectory, history, _ = run_episode(
            env=env,
            controller=controller,
            control_steps=CONTROL_STEPS_PER_EPISODE,
            sim_steps_per_control=SIM_STEPS_PER_CONTROL,
            is_training=True
        )

        env.close()

        # Extract trajectory for PPO update
        states = torch.cat(trajectory["states"])
        actions = torch.cat(trajectory["actions"])
        log_probs = torch.cat(trajectory["log_probs"])
        values = trajectory["values"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]

        # Calculate advantages and update policy
        with torch.no_grad():
            _, next_value = agent(states[-1].unsqueeze(0))

        returns, advantages = compute_gae(rewards, values, next_value.item(), dones)
        ppo_update(agent, optimizer, states, actions, log_probs, returns, advantages)

        # Logging
        total_reward = sum(rewards)
        all_scores.append(total_reward)
        cumulative_steps += len(rewards)

        history_steps.append(cumulative_steps)
        history_lengths.append(len(rewards))
        history_tts.append(history["tts_total"])

        update_live_plot(all_scores, line, ax, fig)
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Steps: {len(rewards)} | TTS: {history['tts_total']:.0f}")

        # Save checkpoints
        if episode % 10 == 0:
            torch.save(agent.state_dict(), os.path.join(model_dir, f"model_ep{episode}.pth"))
            with open(os.path.join(model_dir, f"state_tracker_ep{episode}.pkl"), "wb") as f:
                pickle.dump(state_tracker, f)

    # Save final metrics
    file_prefix = "replacement" if use_replacement else "baseline"
    with open(os.path.join(model_dir, f"training_history_{file_prefix}_seed{seed}.pkl"), "wb") as f:
        pickle.dump({
            "scores": all_scores, "steps": history_steps,
            "lengths": history_lengths, "tts": history_tts
        }, f)

    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_replacement", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args.use_replacement, args.seed)