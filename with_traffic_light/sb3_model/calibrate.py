"""
Calibration utility — runs one SUMO episode with no agent to measure
MAX_TTS and AVG_TTS for the reward function.

Usage:
    python calibrate.py
"""
import os
import numpy as np
import traci

from config import SUMO_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL


def calibrate_tts():
    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]
    traci.start(sumo_cmd)

    tts_records = []

    for control_step in range(CONTROL_STEPS_PER_EPISODE):
        step_tts = 0
        for _ in range(SIM_STEPS_PER_CONTROL):
            traci.simulationStep()
            step_tts += traci.vehicle.getIDCount()
        tts_records.append(step_tts)

    traci.close()

    max_tts = np.max(tts_records)
    avg_tts = np.mean(tts_records)

    print("Calibration Complete.")
    print(f"  MAX_TTS: {max_tts}")
    print(f"  AVG_TTS: {avg_tts:.2f}")
    print(f"\nUpdate these values in config.py:")
    print(f'  MAX_TTS = {max_tts}')
    print(f'  AVG_TTS = {avg_tts:.2f}')


if __name__ == "__main__":
    calibrate_tts()
