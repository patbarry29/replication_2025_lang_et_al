import traci
import numpy as np

class RampMeterEnv:
    def __init__(self, sumo_cmd, tls_id, upstream_dets, downstream_dets, ramp_arr_dets, ramp_dep_dets, ramp_detector, max_speed=27.78):
        self.sumo_cmd = sumo_cmd
        self.tls_id = tls_id
        self.upstream_dets = upstream_dets
        self.downstream_dets = downstream_dets
        self.ramp_arr_dets = ramp_arr_dets
        self.ramp_dep_dets = ramp_dep_dets
        self.ramp_detector = ramp_detector
        self.max_speed = max_speed

    def start(self):
        traci.start(self.sumo_cmd)

    def close(self):
        traci.close()

    def _get_aggregate(self, detectors, interval_steps):
        """Helper to fetch and process induction loop data for a given interval."""
        if not detectors:
            return 0.0, 0.0, 0.0

        occ = np.mean([traci.inductionloop.getLastIntervalOccupancy(d) for d in detectors])

        raw_speeds = [traci.inductionloop.getLastIntervalMeanSpeed(d) for d in detectors]
        speeds = [s if s >= 0 else self.max_speed for s in raw_speeds]
        speed = np.mean(speeds)

        veh_total = np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in detectors])
        veh_per_sec = veh_total / interval_steps

        return occ, speed, veh_per_sec

    def get_traffic_state(self, interval_steps):
        """Returns aggregated upstream, downstream, and ramp states, plus instantaneous queue."""
        up_occ, up_speed, up_veh = self._get_aggregate(self.upstream_dets, interval_steps)
        dn_occ, dn_speed, dn_veh = self._get_aggregate(self.downstream_dets, interval_steps)

        arr_total = np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in self.ramp_arr_dets])
        dep_total = np.sum([traci.inductionloop.getLastIntervalVehicleNumber(d) for d in self.ramp_dep_dets])

        ramp_arr = arr_total / interval_steps
        ramp_dep = dep_total / interval_steps

        # Queue is an instantaneous snapshot
        queue_length = sum(traci.lanearea.getJamLengthVehicle(det) for det in self.ramp_detector)

        return {
            "upstream": {"occ": up_occ, "speed": up_speed, "veh": up_veh},
            "downstream": {"occ": dn_occ, "speed": dn_speed, "veh": dn_veh},
            "ramp_arr": ramp_arr,
            "ramp_dep": ramp_dep,
            "queue": queue_length
        }

    def apply_action_and_get_tts(self, green_duration, red_duration):
        """Executes the traffic light phases, advances the simulation, and calculates Total Time Spent (TTS)."""
        tts = 0

        # Execute phases: green then red
        for duration, state in [(green_duration, "gg"), (red_duration, "rr")]:
            if duration <= 0:
                continue

            traci.trafficlight.setRedYellowGreenState(self.tls_id, state)

            for _ in range(int(duration)):
                traci.simulationStep()
                tts += traci.vehicle.getIDCount()

        return tts