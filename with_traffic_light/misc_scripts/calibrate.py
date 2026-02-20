import os
import traci
import numpy as np
import sys

# CONFIGURATION
sumoBinary = "sumo" # Use command line for speed
SUMO_PATH = os.path.join("sumo_network", "data", "simulation.sumocfg")
sumoCmd = [sumoBinary, "-c", SUMO_PATH, "--start"]

CONTROL_INTERVAL = 15
MAX_STEPS = 3600

def run_calibration():
    # 1. Start Simulation
    traci.start(sumoCmd)

    step = 0
    cumulative_veh = 0
    tts_history = [] # Store TTS for every 30s interval

    print("Running Calibration (No Control)...")

    while step < MAX_STEPS:
        # A. Count vehicles (Road + Backlog)
        # This represents the "Instantaneous TTS" for this second
        n_vehicles = traci.simulation.getMinExpectedNumber()
        cumulative_veh += n_vehicles

        # B. End of Control Interval
        if step % CONTROL_INTERVAL == 0 and step > 0:
            # 1. Store the Total Time Spent for this specific 30s chunk
            tts_history.append(cumulative_veh)

            # 2. Reset counter
            cumulative_veh = 0

            # 3. Force Green Light (No Control Strategy)
            # Replace "1494194482" with your actual Traffic Light ID
            traci.trafficlight.setPhase("1494194482", 0)

        # C. Step
        traci.simulationStep()
        step += 1

    traci.close()

    # 2. Calculate Alpha and Beta
    # Paper: "replace a and B with the maximum TTS and average TTS... respectively"
    alpha_val = np.max(tts_history)
    beta_val = np.mean(tts_history)

    print("\n" + "="*30)
    print("CALIBRATION RESULTS")
    print("="*30)
    print(f"Alpha (Max TTS): {alpha_val}")
    print(f"Beta (Avg TTS):  {beta_val}")
    print("="*30)
    print(f"Update your Agent with:")
    print(f"ALPHA = {alpha_val}")
    print(f"BETA = {beta_val}")

if __name__ == "__main__":
    run_calibration()