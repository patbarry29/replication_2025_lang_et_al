import traci
import time
import torch
import numpy as np
import sys
import os

# Import your existing Agent class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn_model.dqn_agent import Agent

# CONFIGURATION
MODEL_PATH = os.path.join("models", "best_model.pth")
SUMO_PATH = os.path.join("sumo_network", "data", "simulation.sumocfg")
SUMO_CMD = ["sumo-gui", "-c", SUMO_PATH, "--start"]
CONTROL_INTERVAL = 15
MAX_STEPS = 3600

# Normalization Constants (Must match training!)
MAX_SPEED = 30.0
MAX_OCC = 100.0
MAX_QUEUE = 20.0
MAX_FLOW = 2000.0

def get_detector_data(detector_prefix, num_lanes):
    """Returns average speed, average occupancy, and TOTAL flow across lanes."""
    speeds, occs, flows = [], [], []
    for i in range(num_lanes):
        det_id = f"{detector_prefix}_{i}"
        s = traci.inductionloop.getLastStepMeanSpeed(det_id)
        o = traci.inductionloop.getLastStepOccupancy(det_id)
        f = traci.inductionloop.getLastStepVehicleNumber(det_id)

        if s != -1: speeds.append(s)
        occs.append(o)
        flows.append(f)

    return (np.mean(speeds) if speeds else 0.0), \
           (np.mean(occs) if occs else 0.0), \
           np.sum(flows) * 3600 / num_lanes


def visualize():
    # 1. Start SUMO-GUI
    print("Starting SUMO-GUI...")
    traci.start(SUMO_CMD)
    time.sleep(2)

    # 2. Initialize the Agent
    # MUST match the state_size/action_size used during training
    agent = Agent(state_size=10, action_size=2)

    # 3. Load the Trained Weights
    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Successfully loaded model: {MODEL_PATH}")
    else:
        print(f"Error: Could not find model file '{MODEL_PATH}'")
        traci.close()
        return

    # 4. Set to Evaluation Mode
    agent.epsilon = 0.0       # No random actions (Pure Exploitation)
    agent.model.eval()        # PyTorch eval mode

    # Buffers for the 15s control interval
    buffer = {
        'speed15m': [], 'occ15m': [], 'flow15m': [],
        'speed225m': [], 'occ225m': [], 'flow225m': [],
        'speed475m': [], 'occ475m': [], 'flow475m': [],
        'queue': []
    }

    step = 0

    # --- SIMULATION LOOP ---
    while step < MAX_STEPS:

        # A. Collect Sensor Data
        s15, o15, f15 = get_detector_data("det_loc1", 5)
        s225, o225, f225 = get_detector_data("det_loc2", 4)
        s475, o475, f475 = get_detector_data("det_loc3", 4)

        # Get Ramp Queue (Lanes 0 and 1)
        q0 = traci.lane.getLastStepHaltingNumber("edge_ramp_0")
        q1 = traci.lane.getLastStepHaltingNumber("edge_ramp_1")
        current_q = (q0 + q1) / 2.0

        buffer['speed15m'].append(s15); buffer['occ15m'].append(o15); buffer['flow15m'].append(f15)
        buffer['speed225m'].append(s225); buffer['occ225m'].append(o225); buffer['flow225m'].append(f225)
        buffer['speed475m'].append(s475); buffer['occ475m'].append(o475); buffer['flow475m'].append(f475)
        buffer['queue'].append(current_q)

        # B. Agent Thinks (Every 30 seconds)
        if step % CONTROL_INTERVAL == 0 and step > 0:

            # 1. Average the buffer
            avg_vals = {k: np.mean(v) for k, v in buffer.items()}

            # 2. Create State Vector (Normalized)
            state = [
                avg_vals['speed15m'] / MAX_SPEED,
                avg_vals['occ15m'] / MAX_OCC,
                avg_vals['flow15m'] / MAX_FLOW, # New

                avg_vals['speed225m'] / MAX_SPEED,
                avg_vals['occ225m'] / MAX_OCC,
                avg_vals['flow225m'] / MAX_FLOW, # New

                avg_vals['speed475m'] / MAX_SPEED,
                avg_vals['occ475m'] / MAX_OCC,
                avg_vals['flow475m'] / MAX_FLOW, # New

                avg_vals['queue'] / MAX_QUEUE
            ]
            # 3. Choose Action (Using the Brain)
            action = agent.act(state)

            # 4. Print decision to console so you can watch
            decision = "GREEN" if action == 1 else "RED"
            print(f"Time {step}s | Queue: {avg_vals['queue']:.1f} | Speed: {avg_vals['speed15m']:.1f} | Action: {decision}")

            # 5. Execute Action
            # Replace with your actual Traffic Light ID
            traci.trafficlight.setPhase("1494194482", 0 if action == 1 else 2)

            # Reset buffers
            for k in buffer: buffer[k] = []

        # C. Advance Simulation
        traci.simulationStep()

        # Optional: Slow down visualization if it's too fast
        # time.sleep(0.01)

        step += 1
    traci.close()

if __name__ == "__main__":
    visualize()