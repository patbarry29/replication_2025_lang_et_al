from action_replacement import calculate_penalty
from utils import format_state_vector
from config import MAX_TTS, AVG_TTS

def run_episode(env, controller, control_steps, sim_steps_per_control, is_training=False):
    trajectory = {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
    history = {"green_times": [], "queues": [], "downstream_speeds": [], "tts_total": 0, "lower_bound": []}
    num_replacements = 0

    # Bootstrap initial state
    env.apply_action_and_get_tts(sim_steps_per_control, 0)
    last_green = sim_steps_per_control

    raw_state_dict = env.get_traffic_state(sim_steps_per_control)
    raw_state = format_state_vector(raw_state_dict, last_green)

    for step in range(control_steps):
        # 1. Get action from controller
        action_ratio, log_prob, value, raw_action, state_tensor, extras = controller.execute_control(raw_state, is_training)
        replaced = extras[0]

        green_duration = int(action_ratio * sim_steps_per_control)
        red_duration = sim_steps_per_control - green_duration

        # 2. Step environment
        step_tts = env.apply_action_and_get_tts(green_duration, red_duration)
        reward = (MAX_TTS - step_tts) / AVG_TTS
        current_queue = raw_state[6]

        use_replacement = getattr(controller, 'use_replacement', False)
        if use_replacement and replaced:
            penalty = calculate_penalty(current_queue, raw_state[7], action_ratio)
            reward = reward - penalty
        elif not use_replacement:
            reward = reward - (current_queue*0.01)

        # 3. Get next state
        next_state_dict = env.get_traffic_state(sim_steps_per_control)
        next_raw_state = format_state_vector(next_state_dict, green_duration)

        # 4. Check termination
        final_queue = next_raw_state[6]
        spillback = final_queue > 0.9 * 42.0
        done = spillback or (step == control_steps - 1)

        # 5. Record data
        history["tts_total"] += step_tts
        history["green_times"].append(action_ratio)
        history["queues"].append(final_queue)
        history["downstream_speeds"].append(next_state_dict["downstream"]["speed"])
        history["lower_bound"].append(extras[1])
        num_replacements += replaced

        if is_training:
            trajectory["states"].append(state_tensor)
            trajectory["actions"].append(raw_action)
            trajectory["log_probs"].append(log_prob)
            trajectory["values"].append(value)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

        raw_state = next_raw_state

        if done and is_training:
            break

    # mean_reward = np.mean(trajectory["rewards"])
    num_steps = len(history["green_times"])
    # print(f"\n\n AVERAGE REWARD ({num_steps}): {mean_reward}\n")

    return trajectory, history, raw_state, (num_replacements/num_steps)*100