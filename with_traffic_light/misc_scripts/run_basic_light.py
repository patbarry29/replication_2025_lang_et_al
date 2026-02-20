import traci
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration
sumoBinary = "sumo" # Use "sumo" for command line only (faster)
sumoCmd = [sumoBinary, "-c", "simulation.sumocfg"] # Replace with your config name

# 2. Start the simulation
traci.start(sumoCmd)

queue_history = []
speed15m_history = []
speed225m_history = []
speed475m_history = []
occ15m_history = []
occ225m_history = []
occ475m_history = []

speed15m_15s = []
speed225m_15s = []
speed475m_15s = []
occ15m_15s = []
occ225m_15s = []
occ475m_15s = []

queue_15s = []

# 3. The Control Loop
step = 0
control_interval = 30
cumulative_vehicle_count = 0


while step < 3600: # Run for 1 hour (3600 seconds)

    # --- PHASE 1: GET STATE ---
    # Replace 'loop_15m', 'loop_225m' with your actual detector IDs from your .add.xml file

    # Get speed (m/s) and occupancy (%) from the merge detector
    speed_15m_0 = traci.inductionloop.getLastStepMeanSpeed("det_loc1_0")
    speed_15m_1 = traci.inductionloop.getLastStepMeanSpeed("det_loc1_1")
    speed_15m_2 = traci.inductionloop.getLastStepMeanSpeed("det_loc1_2")
    speed_15m_3 = traci.inductionloop.getLastStepMeanSpeed("det_loc1_3")
    speed_15m_4 = traci.inductionloop.getLastStepMeanSpeed("det_loc1_4")
    speed_15m = (speed_15m_0+speed_15m_1+speed_15m_2+speed_15m_3+speed_15m_4)/5

    occ_15m_0 = traci.inductionloop.getLastStepOccupancy("det_loc1_0")
    occ_15m_1 = traci.inductionloop.getLastStepOccupancy("det_loc1_1")
    occ_15m_2 = traci.inductionloop.getLastStepOccupancy("det_loc1_2")
    occ_15m_3 = traci.inductionloop.getLastStepOccupancy("det_loc1_3")
    occ_15m_4 = traci.inductionloop.getLastStepOccupancy("det_loc1_4")
    occ_15m = (occ_15m_0+occ_15m_1+occ_15m_2+occ_15m_3+occ_15m_4)/5

    # Get data from 225m detector
    speed_225m_0 = traci.inductionloop.getLastStepMeanSpeed("det_loc2_0")
    speed_225m_1 = traci.inductionloop.getLastStepMeanSpeed("det_loc2_1")
    speed_225m_2 = traci.inductionloop.getLastStepMeanSpeed("det_loc2_2")
    speed_225m_3 = traci.inductionloop.getLastStepMeanSpeed("det_loc2_3")
    speed_225m = (speed_225m_0+speed_225m_1+speed_225m_2+speed_225m_3)/4

    occ_225m_0 = traci.inductionloop.getLastStepOccupancy("det_loc2_0")
    occ_225m_1 = traci.inductionloop.getLastStepOccupancy("det_loc2_1")
    occ_225m_2 = traci.inductionloop.getLastStepOccupancy("det_loc2_2")
    occ_225m_3 = traci.inductionloop.getLastStepOccupancy("det_loc2_3")
    occ_225m = (occ_225m_0+occ_225m_1+occ_225m_2+occ_225m_3)/4

    # Get data from 475m detector
    speed_475m_0 = traci.inductionloop.getLastStepMeanSpeed("det_loc3_0")
    speed_475m_1 = traci.inductionloop.getLastStepMeanSpeed("det_loc3_1")
    speed_475m_2 = traci.inductionloop.getLastStepMeanSpeed("det_loc3_2")
    speed_475m_3 = traci.inductionloop.getLastStepMeanSpeed("det_loc3_3")
    speed_475m = (speed_475m_0+speed_475m_1+speed_475m_2+speed_475m_3)/4

    occ_475m_0 = traci.inductionloop.getLastStepOccupancy("det_loc3_0")
    occ_475m_1 = traci.inductionloop.getLastStepOccupancy("det_loc3_1")
    occ_475m_2 = traci.inductionloop.getLastStepOccupancy("det_loc3_2")
    occ_475m_3 = traci.inductionloop.getLastStepOccupancy("det_loc3_3")
    occ_475m = (occ_475m_0+occ_475m_1+occ_475m_2+occ_475m_3)/4

    if speed_15m != -1 and step>0:
        speed15m_15s.append(speed_15m)
    if speed_225m != -1 and step>0:
        speed225m_15s.append(speed_225m)
    if speed_475m != -1 and step>0:
        speed475m_15s.append(speed_475m)

    occ15m_15s.append(occ_15m)
    occ225m_15s.append(occ_225m)
    occ475m_15s.append(occ_475m)


    # The paper also uses Ramp Queue Length.
    # A simple way to get this is checking the lane directly:
    # 'ramp_lane_id' is the ID of your on-ramp lane (e.g., "ramp_0")
    queue_length_0 = traci.lane.getLastStepHaltingNumber("edge_ramp_0")
    queue_length_1 = traci.lane.getLastStepHaltingNumber("edge_ramp_1")
    queue_length = (queue_length_0+queue_length_1) / 2

    queue_15s.append(queue_length)

    if step % control_interval == 0 and step > 0:
        avg_speed_15m = np.mean(speed15m_15s) if speed15m_15s else 0.0
        avg_occ_15m = np.mean(occ15m_15s)
        avg_speed_225m = np.mean(speed225m_15s) if speed225m_15s else 0.0
        avg_occ_225m = np.mean(occ225m_15s)
        avg_speed_475m = np.mean(speed475m_15s) if speed475m_15s else 0.0
        avg_occ_475m = np.mean(occ475m_15s)

        speed15m_15s = []
        occ15m_15s = []
        speed225m_15s = []
        occ225m_15s = []
        speed475m_15s = []
        occ475m_15s = []

        speed15m_history.append(avg_speed_15m)
        speed225m_history.append(avg_speed_225m)
        speed475m_history.append(avg_speed_475m)
        occ15m_history.append(avg_occ_15m)
        occ225m_history.append(avg_occ_225m)
        occ475m_history.append(avg_occ_475m)

        queue_length = np.mean(queue_15s)

        queue_15s = []

        queue_history.append(queue_length)

    # --- PHASE 2: TAKE ACTION (The Hands) ---
    n_vehicles = traci.simulation.getMinExpectedNumber()

    cumulative_vehicle_count += n_vehicles

    reward = -cumulative_vehicle_count
    # action_green = np.random.choice([True, False])

    # # C. APPLY ACTION
    # tls_id = "1494194482"
    # if action_green:
    #     traci.trafficlight.setPhase(tls_id, 0) # Green
    # else:
    #     traci.trafficlight.setPhase(tls_id, 2) # Red

    # cumulative_vehicle_count = 0
    # --- PHASE 3: ADVANCE SIMULATION ---
    traci.simulationStep() # Tell SUMO to move 1 second forward
    step += 1

# 4. Close
traci.close()


print(f"\n\n\n {cumulative_vehicle_count} \n\n") # 535125


# Setup for a 2x2 grid to allow the bottom plot to be centered or spans
plt.figure(figsize=(14, 10))

# First subplot: Speed (Top Left)
plt.subplot(2, 2, 1)
plt.plot(np.arange(len(speed15m_history)) * control_interval, speed15m_history, label="15m")
plt.plot(np.arange(len(speed225m_history)) * control_interval, speed225m_history, label="225m")
plt.plot(np.arange(len(speed475m_history)) * control_interval, speed475m_history, label="475m")
plt.title("Speed")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.legend()

# Second subplot: Occupancy (Top Right)
plt.subplot(2, 2, 2)
plt.plot(np.arange(len(occ15m_history)) * control_interval, occ15m_history, label="15m")
plt.plot(np.arange(len(occ225m_history)) * control_interval, occ225m_history, label="225m")
plt.plot(np.arange(len(occ475m_history)) * control_interval, occ475m_history, label="475m")
plt.title("Occupancy")
plt.xlabel("Time")
plt.ylabel("Occupancy")
plt.legend()

# Third subplot: Queue History (Bottom Center)
# Using a specific gridspec-like approach to center it in the bottom row
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(queue_history)) * control_interval, queue_history, color='tab:red')
plt.title("Queue History")
plt.xlabel("Time")
plt.ylabel("Queue Length")

plt.tight_layout()
plt.show()