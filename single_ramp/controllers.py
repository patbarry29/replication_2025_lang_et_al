import torch
from action_replacement import calculate_lower_bound

class BaseController:
    def execute_control(self, raw_state_list, is_training=False):
        raise NotImplementedError

class NoControlBaseline(BaseController):
    def execute_control(self, raw_state_list, is_training=False):
        return 1.0, None, None, None, None, False  # 100% green ratio

class PiAlineaController(BaseController):
    def __init__(self, target_occ=14.0, k_r=90.0, k_p=10.0, min_ratio=0.1, max_ratio=1.0, sim_steps_per_control=15.0):
        self.target_occ = target_occ
        self.k_r = k_r
        self.k_p = k_p
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.sim_steps_per_control = sim_steps_per_control
        self.prev_dn_occ = None

    def execute_control(self, raw_state_list, is_training=False):
        dn_occ = raw_state_list[3]        # Downstream occupancy
        last_green = raw_state_list[9]    # Last green duration in steps

        # Initialize previous occupancy on the first step to prevent large jumps
        if self.prev_dn_occ is None:
            self.prev_dn_occ = dn_occ

        # Calculate previous ratio
        prev_ratio = last_green / self.sim_steps_per_control

        # PI-ALINEA feedback equation
        action_ratio = (prev_ratio +
                        self.k_r * (self.target_occ - dn_occ) -
                        self.k_p * (dn_occ - self.prev_dn_occ))

        # Update previous occupancy for the next step
        self.prev_dn_occ = dn_occ

        # Clamp between minimum and maximum allowed ratios
        action_ratio = max(self.min_ratio, min(self.max_ratio, action_ratio))

        return action_ratio, None, None, None, None, False

class RLController(BaseController):
    def __init__(self, agent, state_tracker, normalize_fnc, use_replacement=False):
        self.agent = agent
        self.state_tracker = state_tracker
        self.normalize_fnc = normalize_fnc
        self.use_replacement = use_replacement

    def execute_control(self, raw_state_list, is_training=False):
        state = self.normalize_fnc(raw_state_list, self.state_tracker)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        replaced = 0

        with torch.no_grad():
            dist, value = self.agent(state_tensor)
            action = dist.sample() if is_training else dist.mean
            log_prob = dist.log_prob(action)
            action_ratio = torch.clamp(action, 0.0, 1.0).item()

        if self.use_replacement:
            prev_demand = raw_state_list[7]  # index 7 is ramp_arr
            curr_queue = raw_state_list[6]   # index 6 is queue length
            lower_bound = calculate_lower_bound(prev_demand, curr_queue)
            replaced = int(action_ratio < lower_bound)
            action_ratio = max(action_ratio, lower_bound)

        return action_ratio, log_prob, value, action, state_tensor, replaced