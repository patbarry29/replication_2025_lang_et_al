import traci

SUMO_BINARY = "sumo"
SUMO_CONFIG = "simulation.sumocfg"
WARMUP_START = 500
WARMUP_END = 600

AREAS = {
    "Area 1": {
        "loc": (1, 1),
        "x_range": (1400, 3600),
        "bg_flow": 9500,
        "detectors": {
            "+955 m":  {"ids": [f"det_a1_up_{i}"  for i in range(4)],   "dist": 0},
            "+1317 m": {"ids": [f"det_a1_dn1_{i}" for i in range(4)],   "dist": 362},
            "+1652 m": {"ids": [f"det_a1_dn2_{i}" for i in range(4)],   "dist": 362 + 335},
        },
    },
    "Area 2": {
        "loc": (0, 1),
        "x_range": (2100, 4600),
        "bg_flow": 8350,
        "detectors": {
            "+3916 m": {"ids": [f"det_a2_up_{i}"  for i in range(4)],   "dist": 0},
            "+4254 m": {"ids": [f"det_a2_dn1_{i}" for i in range(1, 5)], "dist": 338},
            "+4404 m": {"ids": [f"det_a2_dn2_{i}" for i in range(4)],   "dist": 338 + 250},
        },
    },
    "Area 3": {
        "loc": (1, 0),
        "x_range": (1500, 3900),
        "bg_flow": 8200,
        "detectors": {
            "+5221 m": {"ids": [f"det_a3_up_{i}"  for i in range(4)],   "dist": 0},
            "+5470 m": {"ids": [f"det_a3_dn1_{i}" for i in range(4)],   "dist": 249},
            "+5620 m": {"ids": [f"det_a3_dn2_{i}" for i in range(1, 5)], "dist": 249 + 250},
        },
    },
    "Area 4": {
        "loc": (0, 0),
        "x_range": (1000, 3900),
        "bg_flow": 8000,
        "detectors": {
            "+6822 m": {"ids": [f"det_a4_up_{i}"  for i in range(4)],   "dist": 0},
            "+7335 m": {"ids": [f"det_a4_dn1_{i}" for i in range(4)],   "dist": 513},
            "+7635 m": {"ids": [f"det_a4_dn2_{i}" for i in range(4)],   "dist": 513 + 300},
        },
    },
}

def calibrate():
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    counts = {area: 0 for area in AREAS}
    speeds = {area: [] for area in AREAS}

    while traci.simulation.getTime() < WARMUP_END:
        traci.simulationStep()
        current_time = traci.simulation.getTime()

        if WARMUP_START <= current_time < WARMUP_END:
            for area, cfg in AREAS.items():
                area_speed = 0
                det_count = 0
                for label, det in cfg["detectors"].items():
                    for d in det["ids"]:
                        # Tally counts
                        counts[area] += traci.inductionloop.getLastStepVehicleNumber(d)
                        # Average speeds
                        speed = traci.inductionloop.getLastStepMeanSpeed(d)
                        if speed >= 0:  # -1 means no vehicle crossed
                            area_speed += speed
                            det_count += 1
                if det_count > 0:
                    speeds[area].append(area_speed / det_count)

    traci.close()

    time_window_s = WARMUP_END - WARMUP_START

    for area, cfg in AREAS.items():
        num_locs = len(cfg["detectors"])
        bg_flow = (counts[area] / num_locs) / time_window_s * 3600
        avg_speed = sum(speeds[area]) / len(speeds[area]) if speeds[area] else 27.78

        print(f"--- {area} ---")
        print(f"Calibrated bg_flow: {round(bg_flow)} veh/h")
        print(f"Average speed: {round(avg_speed, 2)} m/s")

        for label, det in cfg["detectors"].items():
            if det["dist"] > 0:
                shift = round(det["dist"] / avg_speed)
                print(f"Time shift for {label}: {shift} s")

if __name__ == "__main__":
    calibrate()