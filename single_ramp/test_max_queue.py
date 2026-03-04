import os
import traci
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (SIM_STEPS_PER_CONTROL, SUMO_PATH, CONTROL_STEPS_PER_EPISODE, TLS_ID)

SUMO_PATH = os.path.join("..", "..", "sumo_network", "data", "simulation.sumocfg")
EDGE_ID = "edge_ramp_2"

def evaluate_max_queue():
    # Start SUMO with GUI
    sumo_cmd = ["sumo-gui", "-c", SUMO_PATH, "--start"]
    traci.start(sumo_cmd)

    max_queue = 0

    print(f"Starting queue capacity test on '{EDGE_ID}'...")

    for step in range(CONTROL_STEPS_PER_EPISODE):
        # Force the traffic light to stay red every step
        # "rr" matches the 2-lane connection we set up in the XML
        traci.trafficlight.setRedYellowGreenState(TLS_ID, "rr")

        # Advance the simulation
        traci.simulationStep()

        # Check queue every SIM_STEPS_PER_CONTROL steps
        if step % SIM_STEPS_PER_CONTROL == 0:
            current_queue = traci.edge.getLastStepVehicleNumber(EDGE_ID)

            # Update max queue
            if current_queue > max_queue:
                max_queue = current_queue

            print(f"Sim Step {step:04d} | Current Queue: {current_queue:02d} | Max Queue so far: {max_queue:02d}")

    traci.close()
    print("-" * 40)
    print(f"Test Complete. Absolute Maximum Queue Reached: {max_queue}")

if __name__ == "__main__":
    evaluate_max_queue()