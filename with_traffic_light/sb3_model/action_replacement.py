"""
Action Replacement wrapper for the SB3 ramp metering environment.
Implements the store-and-forward safety constraint from the paper.
"""
import gymnasium as gym
import numpy as np

from config import (
    W_MAX, ALPHA, R_MIN_RATIO, CAPACITY_PER_STEP, PENALTY_SCALING,
)


def calculate_lower_bound(demand_prev, current_queue):
    """Minimum green ratio to prevent spillback (store-and-forward model)."""
    required_discharge = demand_prev - (ALPHA * W_MAX - current_queue)
    r_lb = required_discharge / CAPACITY_PER_STEP
    r_lb = max(R_MIN_RATIO, r_lb)
    r_lb = min(1.0, r_lb)
    return r_lb


def calculate_penalty(current_queue, demand_current, action_ratio):
    """Penalty proportional to how quickly spillback would occur."""
    discharge = action_ratio * CAPACITY_PER_STEP
    net_accumulation = demand_current - discharge

    if net_accumulation <= 0:
        return 0.0

    remaining_capacity = (ALPHA * W_MAX) - current_queue
    steps_to_spillback = max(1.0, remaining_capacity / net_accumulation)
    penalty = current_queue / (steps_to_spillback + 1.0)
    return PENALTY_SCALING * penalty


class ActionReplacementWrapper(gym.Wrapper):
    """
    Wraps a SumoRampMeteringEnv to enforce the action replacement constraint.

    On each step:
      1. Read prev_demand and current_queue from the underlying env.
      2. Compute the lower-bound green ratio.
      3. If the agent's action is below the bound, replace it and add a
         penalty to the reward so the agent learns to avoid unsafe actions.
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_demand = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # After reset the env has already run a bootstrap step,
        # so demand (index 7) is available.
        self._prev_demand = float(obs[7])
        return obs, info

    def step(self, action):
        action_ratio = float(np.clip(action[0], 0.0, 1.0))

        # Read current env state
        curr_queue = self.unwrapped.current_queue
        curr_demand = self.unwrapped.current_demand

        # Lower-bound check
        lower_bound = calculate_lower_bound(self._prev_demand, curr_queue)
        penalty = 0.0

        if action_ratio < lower_bound:
            penalty = calculate_penalty(curr_queue, curr_demand, action_ratio)
            action_ratio = lower_bound

        # Pass the (possibly replaced) action to the real env
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([action_ratio], dtype=np.float32)
        )

        reward -= penalty

        # Record replacement info
        info["replaced"] = (action_ratio != float(np.clip(action[0], 0.0, 1.0)))
        info["lower_bound"] = lower_bound
        info["penalty"] = penalty

        # Update for next step
        self._prev_demand = curr_demand

        return obs, reward, terminated, truncated, info
