import traci
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SUMO_CMD = ["sumo", "-c", "simulation.sumocfg"]

# Detector IDs (Indices mapped to your 4-lane/5-lane sections)
DETECTORS_1 = ["det_loc1_0", "det_loc1_1", "det_loc1_2", "det_loc1_3", "det_loc1_4"] # +920m
DETECTORS_2 = ["det_loc2_0", "det_loc2_1", "det_loc2_2", "det_loc2_3"]               # +1225m
DETECTORS_3 = ["det_loc3_0", "det_loc3_1", "det_loc3_2", "det_loc3_3"]               # +1475m

# Paper Constants
Q0_HOUR = 8160.0           # Background flow (veh/h) [cite: 264]
STEP_LENGTH = 0.5          # Your SUMO step length
Q0_SEC = Q0_HOUR / 3600.0  # Background flow per second (~2.267 veh/s)
SIM_TOTAL_TIME = 2400      # Duration matching Figure 6 [cite: 346]

def run_replication():
    traci.start(SUMO_CMD)
    
    # Cumulative vehicle counters (N)
    cum_1 = 0; cum_2 = 0; cum_3 = 0
    
    # Data storage for N'(x,t) sampled every 1 second
    n_prime_1 = []; n_prime_2 = []; n_prime_3 = []
    time_points = []
    
    step = 0
    max_steps = int(SIM_TOTAL_TIME / STEP_LENGTH)
    
    while step < max_steps:
        traci.simulationStep()
        
        # 1. Update cumulative vehicle counts N(x,t)
        # sum() aggregates vehicles passing the detectors in this 0.5s step
        cum_1 += sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in DETECTORS_1)
        cum_2 += sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in DETECTORS_2)
        cum_3 += sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in DETECTORS_3)
        
        # 2. Sample data every 1 second (every 2 steps) to match paper resolution
        if step % 2 == 0:
            current_time = step * STEP_LENGTH
            
            # Formula: N'(x,t) = N(x,t) - q0 * t [cite: 79]
            np1 = cum_1 - (Q0_SEC * current_time)
            np2 = cum_2 - (Q0_SEC * current_time)
            np3 = cum_3 - (Q0_SEC * current_time)
            
            n_prime_1.append(np1)
            n_prime_2.append(np2)
            n_prime_3.append(np3)
            time_points.append(current_time)
            
        step += 1

    traci.close()
    plot_figure_6(time_points, n_prime_1, n_prime_2, n_prime_3)

def plot_figure_6(t, data1, data2, data3):
    plt.figure(figsize=(8, 6))
    
    # Travel time shifts from paper (seconds) [cite: 264]
    # These align downstream observations back to the upstream bottleneck entry
    shift_1225 = 16 
    shift_1475 = 25 
    
    # --- PLOTTING WITH PAPER STYLING ---
    # Line 1 (+920m): Upstream reference (Red)
    plt.plot(t[:-shift_1475], data1[:-shift_1475], 
             label='+920 m', color='firebrick', linewidth=1.2)
    
    # Line 2 (+1225m): Orange, shifted 16s left
    plt.plot(t[:-shift_1475], data2[shift_1225 : -(shift_1475 - shift_1225)], 
             label='+1225 m', color='orange', linewidth=1.2)

    # Line 3 (+1475m): Blue, shifted 25s left, with 'x' markers 
    plt.plot(t[:-shift_1475], data3[shift_1475:], 
             label='+1475 m', color='steelblue', linewidth=1.0, 
             marker='x', markevery=60, markersize=4)
    
    # Matching y-axis label and scale from Figure 6
    plt.ylabel("$N'(x, t) = N(x, t) - q_0 \\times t$", fontsize=12)
    plt.xlabel("Simulation time (s)", fontsize=12)
    
    # Formatting
    plt.legend(loc='lower left', frameon=False, fontsize=10)
    plt.grid(True, linestyle='-', alpha=0.2)
    
    # Congestion onset markers mentioned in Section 4.1 [cite: 265-267]
    plt.axvline(x=600, color='lightgray', linestyle='--', linewidth=0.8)
    plt.axvline(x=950, color='lightgray', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_replication()