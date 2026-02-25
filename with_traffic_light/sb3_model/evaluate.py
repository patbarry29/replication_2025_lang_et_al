"""
Evaluate a trained SB3 PPO ramp metering agent.
Loads the saved model + VecNormalize stats and runs one episode with sumo-gui.

Usage:
    python evaluate.py
    python evaluate.py --model models/ppo_ramp_final --vecnorm models/vec_normalize.pkl
    python evaluate.py --no_replacement   # Disable action replacement during eval
"""
import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sumo_env import SumoRampMeteringEnv
from action_replacement import ActionReplacementWrapper, calculate_lower_bound
from config import CONTROL_STEPS_PER_EPISODE


def evaluate(model_path: str, vecnorm_path: str, use_replacement: bool = True):
    # --- Build env with GUI ---
    def make_env():
        env = SumoRampMeteringEnv(use_gui=True)
        if use_replacement:
            env = ActionReplacementWrapper(env)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False    # Don't update running stats
    vec_env.norm_reward = False # Don't normalize reward during eval

    model = PPO.load(model_path, env=vec_env)

    # --- Run one episode ---
    obs = vec_env.reset()
    time.sleep(3)  # Give sumo-gui time to render

    total_reward = 0.0
    max_queue = 0.0

    for step in range(CONTROL_STEPS_PER_EPISODE):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = vec_env.step(action)
        info = info[0]  # Unwrap from vectorized env

        total_reward += reward[0]
        queue = info.get("queue", 0)
        max_queue = max(max_queue, queue)

        replaced = info.get("replaced", False)
        lb = info.get("lower_bound", 0.0)

        print(
            f"Step {step:3d} | "
            f"Action: {float(action[0][0]):.3f} | "
            f"Replaced: {replaced} (LB: {lb:.3f}) | "
            f"Queue: {queue:.0f} | "
            f"Reward: {float(reward[0]):.3f}"
        )

        if done[0]:
            print(f"[TERMINATED] Spillback at step {step}")
            break

    vec_env.close()
    print(f"\nEvaluation Complete.")
    print(f"  Total Reward:  {total_reward:.2f}")
    print(f"  Max Queue:     {max_queue:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SB3 PPO Ramp Metering")
    parser.add_argument("--model", type=str, default="models/ppo_ramp_final",
                        help="Path to saved SB3 model (without .zip)")
    parser.add_argument("--vecnorm", type=str, default="models/vec_normalize.pkl",
                        help="Path to VecNormalize stats")
    parser.add_argument("--no_replacement", action="store_true",
                        help="Disable action replacement during eval")
    args = parser.parse_args()

    evaluate(args.model, args.vecnorm, use_replacement=not args.no_replacement)
