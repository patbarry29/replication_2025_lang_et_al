import math

from config import RAMP_CAPACITY_VEH_S, SIM_STEPS_PER_CONTROL, W_MAX, ALPHA, R_MIN_RATIO

def calculate_lower_bound(demand_prev_veh_s, current_queue):
    # Calculate maximum allowed vehicles based on adjustment coefficient
    max_allowed_queue = ALPHA * W_MAX

    # Calculate how many more vehicles the ramp can hold
    available_storage = max_allowed_queue - current_queue

    # Convert available storage into an absorption flow rate (veh/s)
    absorption_rate_veh_s = available_storage / SIM_STEPS_PER_CONTROL

    # Calculate required discharge rate (veh/s) using Equation 13
    required_discharge_veh_s = demand_prev_veh_s - absorption_rate_veh_s

    # Convert the required discharge rate into an action ratio [0, 1]
    r_lb_raw = required_discharge_veh_s / RAMP_CAPACITY_VEH_S

    # Apply the absolute minimum rate
    r_lb = max(R_MIN_RATIO, r_lb_raw)

    # Cap the lower bound at 1.0
    r_lb = min(1.0, r_lb)

    return r_lb

def calculate_penalty(current_queue, demand_current, action_ratio, penalty_scaling_factor=2.0):
    # Predict step of spillback (k_sp) based on current dynamics
    discharge_current = action_ratio * RAMP_CAPACITY_VEH_S
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