import os
import numpy as np

# Constants for the Store-and-Forward Model
W_MAX = 42.0
ALPHA = 0.85
R_MIN_RATIO = 0.1
RAMP_CAPACITY_VEH_S = 1930.0 / 3600.0

MAX_SPEED = 27.78
MAX_TTS = 4242.0
AVG_TTS = 3560.33
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

STATE_MEANS = np.asarray([20.554, 18.0392, 1.9333, 8.6758, 26.1541, 4.4272, 0.1375, 0.3375, 0.3394, 7.5])
STATE_STDS = np.asarray([7.0051, 4.3702, 0.2053, 1.3078, 0.3666, 0.5736, 0.7316, 0.1372, 0.1332, 7.5])

REPLACEMENT_HISTORY_PATH = os.path.join("..", "models_replacement", "training_history_replacement_seed42.pkl")
BASELINE_HISTORY_PATH = os.path.join("..", "models", "v1_training_history_baseline_seed42.pkl")

MAIN_T = [0, 600, 600.1, 3300, 3301, 4200]
MAIN_VEH = [7400, 7400, 7900, 7900, 4000, 4000]

RAMP_T = [0, 600, 600.1, 3300, 3600, 4200]
RAMP_VEH = [600, 1000, 1300, 1300, 500, 500]
