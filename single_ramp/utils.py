import numpy as np
from config import STATE_MEANS, STATE_STDS

def normalize_static(raw_state, tracker=None):
    """Uses pre-calculated means and standard deviations from config."""
    raw_state = np.asarray(raw_state)
    return (raw_state - STATE_MEANS) / STATE_STDS

def normalize_dynamic(raw_state, tracker):
    """Uses the dynamic running statistics tracker."""
    raw_state = np.array(raw_state)
    std = tracker.std()
    std[std == 0] = 1e-8
    return (raw_state - tracker.mean) / std

def format_state_vector(state_dict, last_green_duration):
    """Converts the environment state dictionary into the 10-element list expected by the model."""
    return [
        state_dict["upstream"]["occ"], state_dict["upstream"]["speed"], state_dict["upstream"]["veh"],
        state_dict["downstream"]["occ"], state_dict["downstream"]["speed"], state_dict["downstream"]["veh"],
        state_dict["queue"], state_dict["ramp_arr"], state_dict["ramp_dep"], last_green_duration
    ]