import os
import traci
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
SUMO_PATH = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","redesign","simulation.sumocfg")
TLS_ID = "junction_ramp"
TOTAL_SIM_STEPS = 4000
AGGREGATION_INTERVAL = 15
MAX_SPEED = 27.78

# Detector groupings
DETS_UPSTREAM = [f"det_upstream_{i}" for i in range(4)]
DETS_PRE_MERGE = [f"det_pre_merge_{i}" for i in range(4)]
DETS_MERGE = [f"det_loc1_{i}" for i in range(5)]
DETS_DOWNSTREAM = [f"det_loc2_{i}" for i in range(4)]

def get_interval_metrics(detectors):
    speeds = [traci.inductionloop.getLastIntervalMeanSpeed(d) for d in detectors]
    # Handle intervals where no vehicles passed (-1.0)
    valid_speeds = [s if s >= 0 else MAX_SPEED for s in speeds]
    avg_speed = np.mean(valid_speeds)

    # Total flow across all lanes in the group (vehicles per hour)
    veh_count = np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in detectors])
    flow_vph = (veh_count / AGGREGATION_INTERVAL) * 3600

    return avg_speed, flow_vph

def run_debug_simulation():
    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]
    traci.start(sumo_cmd)

    # Data tracking
    time_steps = []
    history = {
        "speed_up": [], "speed_pre": [], "speed_merge": [], "speed_down": [],
        "flow_up": [], "flow_down": [],
        "ramp_queue": [],
        "speed_merge_L0": [],
        "speed_merge_L1": []
    }

    for step in range(TOTAL_SIM_STEPS):
        # Force "no-control" state
        traci.trafficlight.setRedYellowGreenState(TLS_ID, "GG")
        traci.simulationStep()

        if step > 0 and step % AGGREGATION_INTERVAL == 0:
            time_steps.append(step)

            # Poll detectors
            s_up, f_up = get_interval_metrics(DETS_UPSTREAM)
            s_pre, _ = get_interval_metrics(DETS_PRE_MERGE)
            s_merge, _ = get_interval_metrics(DETS_MERGE)
            s_down, f_down = get_interval_metrics(DETS_DOWNSTREAM)

            queue = traci.edge.getLastStepVehicleNumber("edge_ramp_2") + \
                    traci.edge.getLastStepVehicleNumber("edge_ramp_out")

            l0_speed = traci.inductionloop.getLastIntervalMeanSpeed("det_loc1_0")
            l1_speed = traci.inductionloop.getLastIntervalMeanSpeed("det_loc1_1")


            # Store data
            history["speed_up"].append(s_up)
            history["speed_pre"].append(s_pre)
            history["speed_merge"].append(s_merge)
            history["speed_down"].append(s_down)
            history["flow_up"].append(f_up)
            history["flow_down"].append(f_down)
            history["ramp_queue"].append(queue)
            history["speed_merge_L0"].append(l0_speed if l0_speed >= 0 else MAX_SPEED)
            history["speed_merge_L1"].append(l1_speed if l1_speed >= 0 else MAX_SPEED)

    traci.close()
    plot_results(time_steps, history)

def plot_results(times, hist):
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True) # Changed to 4 subplots

    # 1. Speed Comparison
    axs[0].plot(times, hist["speed_up"], label="Upstream", alpha=0.7)
    axs[0].plot(times, hist["speed_pre"], label="Pre-Merge (Bottleneck)", linewidth=2)
    axs[0].plot(times, hist["speed_merge"], label="Merge Area", alpha=0.7)
    axs[0].plot(times, hist["speed_down"], label="Downstream", alpha=0.7)
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].set_title("Network Speeds Over Time")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # 2. Flow Breakdown
    axs[1].plot(times, hist["flow_up"], label="Input Flow (Upstream)")
    axs[1].plot(times, hist["flow_down"], label="Discharge Flow (Downstream)")
    axs[1].set_ylabel("Flow (veh/h)")
    axs[1].set_title("Capacity Drop Observation")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # 3. Ramp Queue
    axs[2].plot(times, hist["ramp_queue"], label="Ramp Queue", color="red")
    axs[2].axhline(y=42, color="black", linestyle="--", label="Max Capacity (42)")
    axs[2].set_ylabel("Vehicles")
    axs[2].set_xlabel("Simulation Time (s)")
    axs[2].set_title("Ramp Storage Status")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(times, hist["speed_merge_L0"], label="Merge Lane 0 (Ramp Accel)", color="red")
    axs[3].plot(times, hist["speed_merge_L1"], label="Merge Lane 1 (Mainline Right)", color="orange")
    axs[3].plot(times, hist["speed_merge"], label="Merge Area Average", color="green", linestyle="--", alpha=0.5)
    axs[3].set_ylabel("Speed (m/s)")
    axs[3].set_xlabel("Simulation Time (s)")
    axs[3].set_title("Merge Friction: Lane 0 vs Lane 1")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_debug_simulation()