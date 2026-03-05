import os
import pickle
import torch

from config import (
    STATE_DIM, SUMO_PATH, CONTROL_STEPS_PER_EPISODE, SIM_STEPS_PER_CONTROL,
    UPSTREAM_DETS, DOWNSTREAM_DETS, RAMP_ARR_DETS, RAMP_DEP_DETS, TLS_ID
)
from env import RampMeterEnv
from controllers import RLController, NoControlBaseline, PiAlineaController
from runner import run_episode
from utils import normalize_static
from model import SharedActorCritic

def run_evaluation(model_path=None, tracker_path=None, use_replacement=False, no_control=False, alinea=False):
    sumo_cmd = ["sumo", "-c", SUMO_PATH, "--no-step-log", "true"]

    env = RampMeterEnv(
        sumo_cmd=sumo_cmd, tls_id=TLS_ID, upstream_dets=UPSTREAM_DETS,
        downstream_dets=DOWNSTREAM_DETS, ramp_arr_dets=RAMP_ARR_DETS,
        ramp_dep_dets=RAMP_DEP_DETS, ramp_edge="edge_ramp_2"
    )

    if no_control:
        controller = NoControlBaseline()
    elif alinea:
        controller = PiAlineaController()
    else:
        agent = SharedActorCritic(STATE_DIM)
        agent.load_state_dict(torch.load(model_path))
        agent.eval()

        with open(tracker_path, "rb") as f:
            state_tracker = pickle.load(f)

        controller = RLController(
            agent=agent, state_tracker=state_tracker,
            normalize_fnc=normalize_static, use_replacement=use_replacement
        )

    env.start()
    _, history, _ = run_episode(
        env=env,
        controller=controller,
        control_steps=CONTROL_STEPS_PER_EPISODE,
        sim_steps_per_control=SIM_STEPS_PER_CONTROL,
        is_training=False
    )
    env.close()

    tts_hours = history["tts_total"] / 3600.0
    max_queue = max(history["queues"])
    spillback_occurred = max_queue > (42 * 0.9)

    return tts_hours, max_queue, spillback_occurred

if __name__ == "__main__":
    print("--- Single Ramp Evaluation ---")

    tts_nc, mq_nc, sb_nc = run_evaluation(no_control=True)
    print(f"No-Control -> TTS: {tts_nc:.2f} h | Max Queue: {mq_nc} | Spillback: {sb_nc}")

    tts_alinea, mq_alinea, sb_alinea = run_evaluation(alinea=True)
    print(f"ALINEA     -> TTS: {tts_alinea:.2f} h | Max Queue: {mq_alinea} | Spillback: {sb_alinea}")

    base_model = os.path.join("models", "model_ep100.pth")
    base_tracker = os.path.join("models", "state_tracker_ep100.pkl")
    tts_base, mq_base, sb_base = run_evaluation(model_path=base_model, tracker_path=base_tracker)
    print(f"RL-Based   -> TTS: {tts_base:.2f} h | Max Queue: {mq_base} | Spillback: {sb_base}")

    # rep_model = os.path.join("models_replacement", "model_ep100.pth")
    # rep_tracker = os.path.join("models_replacement", "state_tracker_ep100.pkl")
    # tts_rep, mq_rep, sb_rep = run_evaluation(model_path=rep_model, tracker_path=rep_tracker, use_replacement=True)
    # print(f"RL+Replace -> TTS: {tts_rep:.2f} h | Max Queue: {mq_rep} | Spillback: {sb_rep}")