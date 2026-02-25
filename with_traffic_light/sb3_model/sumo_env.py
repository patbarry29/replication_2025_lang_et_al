"""
Gymnasium environment wrapping SUMO/TraCI for ramp metering control.
Replicates the logic from paper_model/train.py as a proper Gym env.
"""
import gymnasium as gym
import numpy as np
import traci

from config import (
    STATE_DIM, CONTROL_STEPS_PER_EPISODE, MAX_SPEED,
    MAX_TTS, AVG_TTS, TLS_ID, SUMO_PATH,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS,
    W_MAX, ALPHA,
)


class SumoRampMeteringEnv(gym.Env):
    """
    Observation (10-dim):
        [up_occ, up_speed, up_veh,
         down_occ, down_speed, down_veh,
         queue, ramp_arr, ramp_dep, last_green_duration]

    Action (continuous, 1-dim):
        Green-time ratio in [0, 1].  green_seconds = action * 15.

    Reward:
        (MAX_TTS - step_tts) / AVG_TTS
        Spillback penalty of -10 when queue > 0.9 * W_MAX.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, use_gui=False):
        super().__init__()

        self.render_mode = render_mode
        self.use_gui = use_gui
        self._sumo_running = False

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(1,), dtype=np.float32,
        )

        # Internal state
        self._step_count = 0
        self._last_green_duration = 0
        self._raw_state = np.zeros(STATE_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._sumo_running:
            traci.load(["-c", SUMO_PATH])
        else:
            exe = "sumo-gui" if self.use_gui else "sumo"
            sumo_cmd = [exe, "-c", SUMO_PATH, "--no-step-log", "true"]
            if self.use_gui:
                sumo_cmd.append("--start")
            traci.start(sumo_cmd)
            self._sumo_running = True

        self._step_count = 0

        # Bootstrap: run one 15s interval with no green (neutral start)
        _, ramp_arr, ramp_dep, up, down = self._run_control_step(0.0)
        self._last_green_duration = 0

        obs = self._build_state(self._last_green_duration, ramp_arr, ramp_dep, up, down)
        self._raw_state = obs.copy()

        return obs, {}

    def step(self, action):
        action_ratio = float(np.clip(action[0], 0.0, 1.0))

        # Execute the 15s control step in SUMO
        tts, ramp_arr, ramp_dep, up, down = self._run_control_step(action_ratio)

        # Reward
        reward = (MAX_TTS - tts) / AVG_TTS

        # Next observation
        green_dur = int(action_ratio * 15)
        obs = self._build_state(green_dur, ramp_arr, ramp_dep, up, down)
        self._last_green_duration = green_dur
        self._raw_state = obs.copy()

        # Termination: spillback
        queue = obs[6]
        spillback = queue > 0.9 * W_MAX

        if spillback:
            reward -= 10.0

        self._step_count += 1
        terminated = spillback
        truncated = (self._step_count >= CONTROL_STEPS_PER_EPISODE)

        info = {
            "tts": tts,
            "queue": queue,
            "spillback": spillback,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

    # ------------------------------------------------------------------
    # Properties exposed for the ActionReplacementWrapper
    # ------------------------------------------------------------------

    @property
    def current_queue(self):
        return self._raw_state[6]

    @property
    def current_demand(self):
        return self._raw_state[7]

    # ------------------------------------------------------------------
    # Internal helpers (ported from paper_model/train.py)
    # ------------------------------------------------------------------

    def _run_control_step(self, action_ratio):
        """Run one 15s control step. Returns (tts, ramp_arr, ramp_dep, up_dict, down_dict)."""
        green = int(action_ratio * 15)
        red = 15 - green

        tts = 0
        n_steps = 0

        unique_arr = set()
        unique_dep = set()

        stats = {
            "up":   {"occ": 0.0, "speed": 0.0, "veh": 0.0},
            "down": {"occ": 0.0, "speed": 0.0, "veh": 0.0},
        }

        def poll():
            nonlocal n_steps

            for d in RAMP_ARR_DETS:
                unique_arr.update(traci.inductionloop.getLastStepVehicleIDs(d))
            for d in RAMP_DEP_DETS:
                unique_dep.update(traci.inductionloop.getLastStepVehicleIDs(d))

            for name, dets in [("up", UPSTREAM_DETS), ("down", DOWNSTREAM_DETS)]:
                occ = np.mean([traci.inductionloop.getLastStepOccupancy(d) for d in dets])
                raw_speeds = [traci.inductionloop.getLastStepMeanSpeed(d) for d in dets]
                speeds = [s if s != -1.0 else MAX_SPEED for s in raw_speeds]
                speed = np.mean(speeds)
                veh = np.sum([traci.inductionloop.getLastStepVehicleNumber(d) for d in dets])
                stats[name]["occ"] += occ
                stats[name]["speed"] += speed
                stats[name]["veh"] += veh

            n_steps += 1

        # Execute green then red phases
        for duration, tls_state in [(green, "GGGGGG"), (red, "GGGGrr")]:
            if duration <= 0:
                continue
            traci.trafficlight.setRedYellowGreenState(TLS_ID, tls_state)
            for _ in range(duration):
                traci.simulationStep()
                tts += traci.vehicle.getIDCount()
                poll()

        n = max(n_steps, 1)
        ramp_arr = len(unique_arr) / n
        ramp_dep = len(unique_dep) / n

        for loc in stats:
            for key in stats[loc]:
                stats[loc][key] /= n

        return tts, ramp_arr, ramp_dep, stats["up"], stats["down"]

    def _build_state(self, last_green_dur, ramp_arr, ramp_dep, up, down):
        """Assemble the 10-dim observation vector."""
        queue = traci.edge.getLastStepVehicleNumber("edge_ramp")
        obs = np.array([
            up["occ"], up["speed"], up["veh"],
            down["occ"], down["speed"], down["veh"],
            queue, ramp_arr, ramp_dep, last_green_dur,
        ], dtype=np.float32)
        return obs
