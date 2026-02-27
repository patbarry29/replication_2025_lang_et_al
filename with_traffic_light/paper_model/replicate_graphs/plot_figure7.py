import os
import traci
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
SUMO_NET_FILE = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","sumo_network","data", "network.net.xml") # Update if needed
SUMO_ROU_FILE = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","sumo_network","data", "dynamic_routes.rou.xml")
SUMO_CFG_FILE = os.path.join(r"C:\Users","pbarry","Documents","2025_yang_dqn","with_traffic_light","sumo_network","data", "simulation.sumocfg")

# Grid Search Parameters (from the paper)
MAINLINE_FLOWS = [7400, 7800, 8200, 8600, 9000, 9400]
RAMP_FLOWS = [600, 800, 1000, 1200, 1400, 1600]

# Simulation parameters
SIM_STEPS = 3600
WARMUP_STEPS = 1800  # Discard early data before steady-state is reached
TLS_ID = "junction_ramp"
DOWNSTREAM_DETS = [f"det_loc2_{i}" for i in range(4)]

def generate_route_file(main_flow, ramp_flow):
    """Dynamically generates a route file for the specific flow combination."""
    # Ensure the car-following parameters match the paper's genetic algorithm calibration
    vtype = """<vType id="paper_car" length="5.0" minGap="2.0" accel="2.6" decel="4.5" sigma="0.3" tau="1.1"
                      lcCooperative="1.0" lcSpeedGain="2.5" lcImpatience="1.0"
                      lcOvertakeRight="0.3" lcLookaheadLeft="0.5" lcAssertive="3.0" lcStrategic="0.8"/>"""

    # Define the physical routes through your updated network
    routes = """
    <route id="route_main" edges="edge_virtual_main edge_mainline edge_merge edge_downstream"/>
    <route id="route_ramp" edges="edge_virtual_ramp edge_ramp_2 edge_ramp_out edge_ramp_1 edge_merge edge_downstream"/>
    """

    # Define continuous flows for 1 hour
    flows = f"""
    <flow id="flow_main" type="paper_car" route="route_main" begin="0" end="{SIM_STEPS}" vehsPerHour="{main_flow}" departLane="best" departSpeed="max"/>
    <flow id="flow_ramp" type="paper_car" route="route_ramp" begin="0" end="{SIM_STEPS}" vehsPerHour="{ramp_flow}" departLane="best" departSpeed="max"/>
    """

    with open(SUMO_ROU_FILE, "w") as f:
        f.write(f'<routes>\n{vtype}\n{routes}\n{flows}\n</routes>')

def run_simulation_and_get_occupancy():
    """Runs a single SUMO instance and returns the steady-state downstream occupancy."""

    # Use the config file to load additionals, override the route file, and disable teleporting
    sumo_cmd = [
        "sumo",
        "-c", SUMO_CFG_FILE,          # Loads your net and additionals
        "-r", SUMO_ROU_FILE,          # Overrides the routes with our dynamic ones
        "--no-step-log", "true",
        "--time-to-teleport", "-1",   # Stops SUMO from deleting stuck vehicles
        "--collision.action", "none"  # Stops SUMO from deleting vehicles that bump each other
    ]

    traci.start(sumo_cmd)

    occupancies = []

    for step in range(SIM_STEPS):
        traci.trafficlight.setRedYellowGreenState(TLS_ID, "GG")
        traci.simulationStep()

        # Only collect data after warmup period to ensure queue has stabilized
        if step > WARMUP_STEPS and step % 15 == 0:
            occ = np.mean([traci.inductionloop.getLastIntervalOccupancy(d) for d in DOWNSTREAM_DETS])
            if occ >= 0:  # Ignore -1.0 readings
                occupancies.append(occ)

    traci.close()

    # Return average occupancy across the steady-state window
    return np.mean(occupancies) if occupancies else 0.0

def generate_heatmap():
    results = np.zeros((len(RAMP_FLOWS), len(MAINLINE_FLOWS)))

    print("Starting Grid Search for Figure 7...")
    for i, r_flow in enumerate(RAMP_FLOWS):
        for j, m_flow in enumerate(MAINLINE_FLOWS):
            print(f"Testing: Mainline={m_flow} veh/h, Ramp={r_flow} veh/h", end="", flush=True)

            generate_route_file(m_flow, r_flow)
            avg_occ = run_simulation_and_get_occupancy()

            results[i, j] = avg_occ
            print(f" -> Occupancy: {avg_occ:.2f}%")

    df = pd.DataFrame(results, index=RAMP_FLOWS, columns=MAINLINE_FLOWS)
    df = df.iloc[::-1]

    plt.figure(figsize=(8, 6))

    # REMOVED vmin and vmax so it scales dynamically to your data
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu_r", cbar_kws={'label': 'Occupancy (%)'},
                     linewidths=.5)

    plt.title("Variation of the downstream occupancy")
    plt.xlabel("Mainline flow (veh/h)")
    plt.ylabel("On-ramp flow (veh/h)")
    ax.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_heatmap()