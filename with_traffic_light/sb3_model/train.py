"""
Train a PPO agent for ramp metering using Stable Baselines 3.
Replicates the paper's methodology with SB3's PPO implementation.

Usage:
    python train.py                         # Train without action replacement
    python train.py --use_replacement       # Train with action replacement
    python train.py --episodes 500          # Custom episode count
"""
import os
import argparse

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from sumo_env import SumoRampMeteringEnv
from action_replacement import ActionReplacementWrapper
from config import CONTROL_STEPS_PER_EPISODE


class EpisodeLogCallback(BaseCallback):
    """Prints a summary after each episode (rollout)."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._episode = 0

    def _on_rollout_end(self) -> None:
        self._episode += 1
        rewards = self.model.rollout_buffer.rewards.sum()
        print(f"\nEpisode {self._episode} Complete. Total Reward: {rewards:.2f}\n")

    def _on_step(self) -> bool:
        return True


def make_env(use_replacement: bool):
    """Factory that returns a function creating the (optionally wrapped) env."""
    def _init():
        env = SumoRampMeteringEnv()
        if use_replacement:
            env = ActionReplacementWrapper(env)
        return env
    return _init


def train(use_replacement: bool = False, total_episodes: int = 200):
    # --- Environment ---
    vec_env = DummyVecEnv([make_env(use_replacement)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # --- Policy network matching the paper ---
    # Paper: 3×64 shared layers → 2×64 actor + 2×64 critic, all Tanh
    # SB3 CombinedExtractor with share_features_extractor=True
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh,
    )

    # --- PPO hyperparameters (matching paper_model) ---
    total_timesteps = total_episodes * CONTROL_STEPS_PER_EPISODE

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=CONTROL_STEPS_PER_EPISODE,   # Collect one full episode per rollout
        batch_size=CONTROL_STEPS_PER_EPISODE, # Update on the full episode
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    # --- Callbacks ---
    os.makedirs("models", exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=10 * CONTROL_STEPS_PER_EPISODE,   # Every 10 episodes
        save_path="models/",
        name_prefix="ppo_ramp",
    )

    episode_log_cb = EpisodeLogCallback()

    # --- Train ---
    print(f"Training for {total_episodes} episodes ({total_timesteps} timesteps)")
    print(f"Action replacement: {'ENABLED' if use_replacement else 'DISABLED'}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, episode_log_cb],
    )

    # --- Save final model + normalizer ---
    model.save("models/ppo_ramp_final")
    vec_env.save("models/vec_normalize.pkl")

    print("Training complete. Model saved to models/")
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SB3 PPO Ramp Metering")
    parser.add_argument("--use_replacement", action="store_true",
                        help="Enable action replacement module")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes")
    args = parser.parse_args()

    train(use_replacement=args.use_replacement, total_episodes=args.episodes)
