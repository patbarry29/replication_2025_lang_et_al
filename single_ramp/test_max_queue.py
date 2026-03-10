import traci

from config import (SIM_STEPS_PER_CONTROL, SUMO_PATH, TLS_ID)

detector_ids = ["det_ramp_queue_0", "det_ramp_queue_1"]

def evaluate_max_queue():
    sumo_cmd = ["sumo", "-c", SUMO_PATH]
    print(SUMO_PATH)
    traci.start(sumo_cmd)

    max_queue = 0

    for step in range(400):
        # Force the traffic light to stay red every step
        traci.trafficlight.setRedYellowGreenState(TLS_ID, "rr")

        # Advance the simulation
        traci.simulationStep()

        if step % SIM_STEPS_PER_CONTROL == 0:
            current_queue = sum(traci.lanearea.getJamLengthVehicle(det) for det in detector_ids)

            # Update max queue
            if current_queue > max_queue:
                max_queue = current_queue

            print(f"Sim Step {step:04d} | Current Queue: {current_queue:02d} | Max Queue so far: {max_queue:02d}")

    traci.close()
    print("-" * 40)
    print(f"Test Complete. Absolute Maximum Queue Reached: {max_queue}")

if __name__ == "__main__":
    evaluate_max_queue()