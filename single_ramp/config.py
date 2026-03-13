import os
import numpy as np

# Constants for the Store-and-Forward Model
W_MAX = 42.0
ALPHA = 0.9
R_MIN_RATIO = 0.1
RAMP_CAPACITY_VEH_S = 1930.0 / 3600.0

MAX_SPEED = 27.78
MAX_TTS = 4098.0
AVG_TTS = 3457.55
TLS_ID = "junction_ramp"
SUMO_PATH = os.path.join("sumo_network", "data", "simulation.sumocfg")

STATE_DIM = 10
CONTROL_STEPS_PER_EPISODE = 240
SIM_STEPS_PER_CONTROL = 15.0

# Define detector ID lists matching your XML configuration
UPSTREAM_DETS = [f"det_upstream_{i}" for i in range(4)]
DOWNSTREAM_DETS = [f"det_loc2_{i}" for i in range(4)] + [f"det_loc3_{i}" for i in range(4)]
RAMP_ARR_DETS = [f"det_ramp_arr_{i}" for i in range(2)]
RAMP_DEP_DETS = [f"det_ramp_dep_{i}" for i in range(2)]
RAMP_DETS = ["det_ramp_queue_0", "det_ramp_queue_1"]

STATE_MEANS = np.asarray([20.3471, 18.1585, 1.9217, 8.4116, 26.6432, 4.4006, 3.4375, 0.3303, 0.33,7.5])
STATE_STDS = np.asarray([6.9003, 4.5178, 0.1949, 1.289, 0.2667, 0.5689, 0.9812, 0.0775, 0.079, 7.5])

REPLACEMENT_HISTORY_PATH = os.path.join("..", "models_replacement", "training_history_replacement_seed42.pkl")
BASELINE_HISTORY_PATH = os.path.join("..", "models", "training_history_baseline_seed42.pkl")

MAIN_T = [0, 600, 600.1, 3300, 3600, 4200]
MAIN_VEH = [7400, 7400, 7900, 7900, 4000, 4000]

RAMP_T = [0, 600, 600.1, 3300, 3600, 4200]
RAMP_VEH = [600, 1000, 1300, 1300, 500, 500]
