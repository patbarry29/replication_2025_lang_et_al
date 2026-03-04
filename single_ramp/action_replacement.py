import math

from config import W_MAX, ALPHA, R_MIN_RATIO, CAPACITY_PER_STEP

def calculate_lower_bound(demand_prev, current_queue):
    # Calculate the minimum required discharge in vehicles per step
    # Formula: d(k-1) - (alpha * w_max - w(k))
    required_discharge_veh = demand_prev - (ALPHA * W_MAX - current_queue)

    # Convert the required discharge into an action ratio [0, 1]
    r_lb_raw = required_discharge_veh / CAPACITY_PER_STEP

    # Apply the absolute minimum rate to prevent complete closure
    r_lb = max(R_MIN_RATIO, r_lb_raw)

    # Cap the lower bound at 1.0 (100% green time)
    r_lb = min(1.0, r_lb)

    return r_lb

def calculate_penalty(current_queue, demand_current, action_ratio, penalty_scaling_factor=2.0):
    # Predict step of spillback (k_sp) based on current dynamics
    discharge_current = action_ratio * CAPACITY_PER_STEP
    net_accumulation = demand_current - discharge_current

    if net_accumulation <= 0:
        # Queue is shrinking or stable, spillback won't occur under these conditions
        return 0.0

    remaining_capacity = (ALPHA * W_MAX) - current_queue
    steps_to_spillback = max(1.0, remaining_capacity / net_accumulation)

    # Penalty Formula: w(k) / (k_sp - k + 1)
    # Since we calculate steps_to_spillback directly, it replaces (k_sp - k)
    penalty = current_queue / (steps_to_spillback + 1.0)

    return penalty_scaling_factor * penalty