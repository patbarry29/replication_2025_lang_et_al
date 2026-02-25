"""
Shared constants and hyperparameters for the SB3 ramp metering model.
Mirrors the values used in paper_model/.
"""
import os

# --- Simulation ---
STATE_DIM = 10
CONTROL_STEPS_PER_EPISODE = 240
SIM_STEPS_PER_CONTROL = 15
MAX_SPEED = 27.78  # m/s (100 km/h)

# --- Reward calibration ---
MAX_TTS = 4470.0
AVG_TTS = 3556.64

# --- SUMO paths ---
TLS_ID = "1494194482"
SUMO_PATH = os.path.join(
    r"C:\Users", "pbarry", "Documents", "2025_yang_dqn",
    "with_traffic_light", "sumo_network", "data", "simulation.sumocfg"
)

# --- Detector ID lists (must match detectors.add.xml) ---
UPSTREAM_DETS = [f"det_upstream_{i}" for i in range(4)]
DOWNSTREAM_DETS = (
    [f"det_loc2_{i}" for i in range(4)]
    + [f"det_loc3_{i}" for i in range(4)]
)
RAMP_ARR_DETS = [f"det_ramp_arr_{i}" for i in range(2)]
RAMP_DEP_DETS = [f"det_ramp_dep_{i}" for i in range(2)]

# --- Action replacement (store-and-forward) ---
W_MAX = 42.0           # Max queue capacity (vehicles)
ALPHA = 0.9            # Spillback safety factor
R_MIN_RATIO = 0.1      # Absolute minimum green ratio
CAPACITY_PER_STEP = 8.04  # Max discharge per 15s step (vehicles)
PENALTY_SCALING = 2.0  # Penalty multiplier
