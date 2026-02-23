import traci
import numpy as np
import os

def calibrate_tts():
    # "C:\Users\pbarry\Documents\2025_yang_dqn\with_traffic_light\sumo_network\data\simulation.sumocfg"
    SUMO_PATH = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","sumo_network","data","simulation.sumocfg")
    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]
    traci.start(sumo_cmd)

    tts_records = []

    # Run a full episode (240 control steps * 15 simulation seconds)
    for control_step in range(240):
        step_tts = 0

        # Advance 15 seconds, representing one control step
        for _ in range(15):
            traci.simulationStep()
            # Sum all vehicles currently in the network
            step_tts += traci.vehicle.getIDCount()

        tts_records.append(step_tts)

    traci.close()

    max_tts = np.max(tts_records)
    avg_tts = np.mean(tts_records)

    print(f"Calibration Complete.")
    print(f"MAX_TTS: {max_tts}")
    print(f"AVG_TTS: {avg_tts}")

if __name__ == "__main__":
    calibrate_tts()