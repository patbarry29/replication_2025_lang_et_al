import numpy as np
from config import (
    RAMP_DETS, SUMO_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS, TLS_ID
)
from env import RampMeterEnv
from utils import format_state_vector

def run_calibration_episode():
    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd, tls_id=TLS_ID, upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS, ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS, ramp_detector=RAMP_DETS
    )

    env.start()
    tts_per_step = []
    state_vectors = []

    print("Starting Baseline Calibration (No-Control)...")

    for step in range(CONTROL_STEPS_PER_EPISODE):
        green_duration = SIM_STEPS_PER_CONTROL

        current_step_tts = env.apply_action_and_get_tts(green_duration, 0)

        state_dict = env.get_traffic_state(SIM_STEPS_PER_CONTROL)
        state = format_state_vector(state_dict, green_duration)

        tts_per_step.append(current_step_tts)
        state_vectors.append(state)

    env.close()
    return np.array(tts_per_step), np.array(state_vectors)

if __name__ == "__main__":
    tts_data, state_data = run_calibration_episode()

    alpha = np.max(tts_data)
    beta = np.mean(tts_data)

    means = np.mean(state_data, axis=0)
    stds = np.std(state_data, axis=0)
    stds[stds == 0] = 1e-8

    print("\n--- Calibration Results ---")
    print(f"REWARD_ALPHA (Max TTS): {alpha:.2f}")
    print(f"REWARD_BETA (Avg TTS):  {beta:.2f}")
    print(f"STATE_MEANS: {means.round(4).tolist()}")
    print(f"STATE_STDS:  {stds.round(4).tolist()}")