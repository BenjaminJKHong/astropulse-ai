"""
Telemetry Stream Manager for Astronaut Benjamin.
Reads simulated NASA vitals data, supports live step-by-step telemetry,
and injects dynamic anomaly scenarios (hypoxia, cardiac spike, radiation burst).
"""

import csv
import time
import os
from typing import List, Dict, Any, Optional
from schemas import TelemetryFrame

class BenjaminTelemetryStream:
    def __init__(self, csv_path: str = "nasa_vitals.csv"):
        self.csv_path = csv_path
        self.astronaut_id = "Benjamin"
        self.callsign = "AstroPulse-Lead-01"
        self.recorded_frames: List[Dict[str, Any]] = []
        self.current_index = 0
        self.active_simulation_queue: List[Dict[str, Any]] = []
        
        self.load_csv()

    def load_csv(self):
        """Loads Benjamin's NASA vitals CSV dataset."""
        self.recorded_frames = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.recorded_frames.append({
                        "timestamp": int(row["timestamp"]),
                        "heart_rate": int(row["heart_rate"]),
                        "spo2": float(row["spo2"]),
                        "blood_pressure_systolic": int(row["blood_pressure_systolic"]),
                        "blood_pressure_diastolic": int(row["blood_pressure_diastolic"]),
                        "radiation_msv": float(row["radiation_msv"]),
                        "astronaut_id": self.astronaut_id
                    })
        else:
            # Fallback baseline if file is missing
            self.recorded_frames = [
                {"timestamp": 1698820000, "heart_rate": 72, "spo2": 99.0, "blood_pressure_systolic": 120, "blood_pressure_diastolic": 80, "radiation_msv": 0.12, "astronaut_id": self.astronaut_id},
                {"timestamp": 1698820005, "heart_rate": 75, "spo2": 98.5, "blood_pressure_systolic": 122, "blood_pressure_diastolic": 81, "radiation_msv": 0.13, "astronaut_id": self.astronaut_id},
                {"timestamp": 1698820010, "heart_rate": 85, "spo2": 96.0, "blood_pressure_systolic": 130, "blood_pressure_diastolic": 85, "radiation_msv": 0.15, "astronaut_id": self.astronaut_id},
                {"timestamp": 1698820015, "heart_rate": 110, "spo2": 92.0, "blood_pressure_systolic": 145, "blood_pressure_diastolic": 95, "radiation_msv": 0.20, "astronaut_id": self.astronaut_id},
                {"timestamp": 1698820020, "heart_rate": 115, "spo2": 90.5, "blood_pressure_systolic": 150, "blood_pressure_diastolic": 98, "radiation_msv": 0.22, "astronaut_id": self.astronaut_id}
            ]

    def get_current_frame(self, advance: bool = True) -> TelemetryFrame:
        """
        Retrieves the next telemetry frame for Benjamin.
        Prioritizes injected simulation scenarios if queued; otherwise cycles recorded NASA vitals.
        """
        now = int(time.time())
        
        if self.active_simulation_queue:
            raw = self.active_simulation_queue.pop(0)
            raw["timestamp"] = now
            return TelemetryFrame(**raw)

        if not self.recorded_frames:
            # Default nominal frame
            return TelemetryFrame(
                timestamp=now,
                heart_rate=72,
                spo2=99.0,
                blood_pressure_systolic=120,
                blood_pressure_diastolic=80,
                radiation_msv=0.12,
                astronaut_id=self.astronaut_id
            )

        data = dict(self.recorded_frames[self.current_index])
        if advance:
            self.current_index = (self.current_index + 1) % len(self.recorded_frames)
        
        data["timestamp"] = now
        data["astronaut_id"] = self.astronaut_id
        return TelemetryFrame(**data)

    def trigger_hypoxia_event(self, severity: str = "severe") -> List[TelemetryFrame]:
        """
        Simulates an acute hypoxia event progression for Astronaut Benjamin.
        Simulates rapid O2 depletion, compensatory heart rate acceleration, and cardiovascular strain.
        """
        now = int(time.time())
        frames = []
        
        if severity == "critical" or severity == "severe":
            # Progression from initial drop to critical hypoxia collapse
            steps = [
                {"heart_rate": 82, "spo2": 95.0, "blood_pressure_systolic": 128, "blood_pressure_diastolic": 84, "radiation_msv": 0.14},
                {"heart_rate": 105, "spo2": 91.5, "blood_pressure_systolic": 142, "blood_pressure_diastolic": 92, "radiation_msv": 0.14},
                {"heart_rate": 122, "spo2": 87.0, "blood_pressure_systolic": 155, "blood_pressure_diastolic": 98, "radiation_msv": 0.15},
                {"heart_rate": 138, "spo2": 82.5, "blood_pressure_systolic": 168, "blood_pressure_diastolic": 104, "radiation_msv": 0.15},
                {"heart_rate": 145, "spo2": 78.0, "blood_pressure_systolic": 175, "blood_pressure_diastolic": 110, "radiation_msv": 0.16}
            ]
        else: # moderate
            steps = [
                {"heart_rate": 88, "spo2": 94.0, "blood_pressure_systolic": 130, "blood_pressure_diastolic": 86, "radiation_msv": 0.13},
                {"heart_rate": 112, "spo2": 90.5, "blood_pressure_systolic": 146, "blood_pressure_diastolic": 94, "radiation_msv": 0.14},
                {"heart_rate": 118, "spo2": 89.0, "blood_pressure_systolic": 150, "blood_pressure_diastolic": 96, "radiation_msv": 0.14}
            ]

        self.active_simulation_queue.clear()
        for idx, step in enumerate(steps):
            frame_dict = {
                "timestamp": now + (idx * 5),
                "heart_rate": step["heart_rate"],
                "spo2": step["spo2"],
                "blood_pressure_systolic": step["blood_pressure_systolic"],
                "blood_pressure_diastolic": step["blood_pressure_diastolic"],
                "radiation_msv": step["radiation_msv"],
                "astronaut_id": self.astronaut_id
            }
            self.active_simulation_queue.append(frame_dict)
            frames.append(TelemetryFrame(**frame_dict))

        return frames

    def trigger_tachycardia_event(self) -> List[TelemetryFrame]:
        """Simulates sudden extreme cardiac arrhythmia / physical exertion spike."""
        now = int(time.time())
        steps = [
            {"heart_rate": 110, "spo2": 97.0, "blood_pressure_systolic": 135, "blood_pressure_diastolic": 88, "radiation_msv": 0.13},
            {"heart_rate": 145, "spo2": 96.0, "blood_pressure_systolic": 155, "blood_pressure_diastolic": 95, "radiation_msv": 0.14},
            {"heart_rate": 165, "spo2": 95.0, "blood_pressure_systolic": 170, "blood_pressure_diastolic": 105, "radiation_msv": 0.14}
        ]
        self.active_simulation_queue.clear()
        frames = []
        for idx, step in enumerate(steps):
            f = {**step, "timestamp": now + (idx * 5), "astronaut_id": self.astronaut_id}
            self.active_simulation_queue.append(f)
            frames.append(TelemetryFrame(**f))
        return frames

    def trigger_radiation_spike(self) -> List[TelemetryFrame]:
        """Simulates Solar Particle Event radiation burst."""
        now = int(time.time())
        steps = [
            {"heart_rate": 78, "spo2": 98.0, "blood_pressure_systolic": 125, "blood_pressure_diastolic": 82, "radiation_msv": 1.25},
            {"heart_rate": 84, "spo2": 97.5, "blood_pressure_systolic": 130, "blood_pressure_diastolic": 85, "radiation_msv": 6.80},
            {"heart_rate": 92, "spo2": 97.0, "blood_pressure_systolic": 136, "blood_pressure_diastolic": 88, "radiation_msv": 12.40}
        ]
        self.active_simulation_queue.clear()
        frames = []
        for idx, step in enumerate(steps):
            f = {**step, "timestamp": now + (idx * 5), "astronaut_id": self.astronaut_id}
            self.active_simulation_queue.append(f)
            frames.append(TelemetryFrame(**f))
        return frames

    def reset_to_nominal(self):
        """Clears anomalies and resets stream to normal baseline."""
        self.active_simulation_queue.clear()
        self.current_index = 0

