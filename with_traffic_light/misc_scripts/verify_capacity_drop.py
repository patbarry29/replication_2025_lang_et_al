import os
import traci
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SUMO_PATH = os.path.join("sumo_network", "data", "simulation.sumocfg")
SUMO_CMD = ["sumo", "-c", SUMO_PATH, "--start"]

# Define the 3 detector groups
DETECTORS_1 = ["det_loc1_0", "det_loc1_1", "det_loc1_2", "det_loc1_3", "det_loc1_4"]
DETECTORS_2 = ["det_loc2_0", "det_loc2_1", "det_loc2_2", "det_loc2_3"]
DETECTORS_3 = ["det_loc3_0", "det_loc3_1", "det_loc3_2", "det_loc3_3"]

# Paper Constants
Q0_HOUR = 8160.0
STEP_LENGTH = 1 # Matches your sumocfg
Q0_STEP = (Q0_HOUR / 3600.0) * STEP_LENGTH # Subtract per step, not per second

def run_replication():
    traci.start(SUMO_CMD)

    # Cumulative counters
    cum_1 = 0; cum_2 = 0; cum_3 = 0

    # Lists to store N'(x,t)
    n_prime_1 = []; n_prime_2 = []; n_prime_3 = []

    step = 0
    while step < 4200:
        traci.simulationStep()

        # 1. Sum counts for each location
        c1 = sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in DETECTORS_1)
        c2 = sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in DETECTORS_2)
        c3 = sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in DETECTORS_3)

        # 2. Update cumulative totals
        cum_1 += c1
        cum_2 += c2
        cum_3 += c3

        # 3. Calculate N'(x, t) = N(x, t) - q0 * t
        np1 = cum_1 - (Q0_STEP * step)
        np2 = cum_2 - (Q0_STEP * step)
        np3 = cum_3 - (Q0_STEP * step)

        n_prime_1.append(np1)
        n_prime_2.append(np2)
        n_prime_3.append(np3)

        step += 1

    traci.close()
    plot_figure_6(n_prime_1, n_prime_2, n_prime_3)

def plot_figure_6(data1, data2, data3):
    plt.figure(figsize=(10, 6))

    # Time axis
    t = [i * STEP_LENGTH for i in range(len(data1))]

    # --- SHIFTING LOGIC ---
    # The paper shifts downstream curves to the left by travel time.
    # "16s and 25s, respectively"
    # Shift 1 (+920m): 0s (Reference)
    # Shift 2 (+1225m): -16s
    # Shift 3 (+1475m): -25s

    shift_2 = 16
    shift_3 = 25

    # Plotting
    # We slice the arrays to align them visually
    # Line 1 (Red in paper): +920m
    plt.plot(t[:-shift_3], data1[:-shift_3], label='+920m', color='firebrick', linewidth=1.5)

    # Line 2 (Orange in paper): +1225m (Shifted left by 16s)
    # This means data2[16] aligns with data1[0]
    plt.plot(t[:-shift_3], data2[shift_2:-(shift_3-shift_2)], label='+1225m', color='orange', linewidth=1.5, linestyle='--')

    # Line 3 (Blue in paper): +1475m (Shifted left by 25s)
    plt.plot(t[:-shift_3], data3[shift_3:], label='+1475m', color='steelblue', linewidth=1.5, linestyle='-.')

    plt.title("Replication of Figure 6: Capacity Drop Analysis")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("N'(x,t) = N(x,t) - q0*t")
    plt.legend()
    plt.grid(True)

    # Congestion markers from paper (approximate)
    plt.axvline(x=600, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=950, color='gray', linestyle=':', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    run_replication()