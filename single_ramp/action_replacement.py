from config import RAMP_CAPACITY_VEH_S, SIM_STEPS_PER_CONTROL, W_MAX, ALPHA, R_MIN_RATIO

def calculate_lower_bound(demand_prev_veh_s, current_queue):
    # Calculate maximum allowed vehicles
    max_allowed_queue = ALPHA * W_MAX

    # Calculate how many more vehicles the ramp can hold
    available_storage = max_allowed_queue - current_queue

    # Convert available storage into an absorption flow rate (veh/s)
    absorption_rate_veh_s = available_storage / SIM_STEPS_PER_CONTROL

    # Calculate required discharge rate (veh/s) using Equation 13
    required_discharge_veh_s = demand_prev_veh_s - absorption_rate_veh_s

    r_lb_raw = required_discharge_veh_s / RAMP_CAPACITY_VEH_S

    # Convert the required discharge rate into an action ratio [0, 1]
    r_lb = min(1.0, max(R_MIN_RATIO, r_lb_raw))

    return r_lb


def calculate_penalty(current_queue, demand_current, action_ratio, penalty_scaling_factor=0.5):
    # Predict step of spillback (k_sp) based on current dynamics
    discharge_current = action_ratio * RAMP_CAPACITY_VEH_S
    net_accumulation_veh_s = demand_current - discharge_current

    # Multiply by control step length (T) to get accumulation per control step
    net_accumulation_per_step = net_accumulation_veh_s * SIM_STEPS_PER_CONTROL

    if net_accumulation_per_step <= 0:
        return 0.0

    remaining_capacity = (ALPHA * W_MAX) - current_queue

    # (k_sp - k)
    steps_to_spillback = max(1.0, remaining_capacity / net_accumulation_per_step)

    # Penalty Formula: w(k) / (k_sp - k + 1)
    penalty = current_queue / (steps_to_spillback + 1.0)

    return penalty_scaling_factor * penalty