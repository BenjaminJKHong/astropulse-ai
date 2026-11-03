"""
AstroPulse AI Flight Surgeon Engine
Analyzes astronaut telemetry (Heart Rate, SpO2, BP, Radiation) via Groq, Nebius,
or OpenAI-compatible LLM endpoints with high speed, structured JSON output,
and automatic deterministic aerospace medical fallback.
Supports both asynchronous (FastAPI/streaming) and synchronous (CLI/testing) execution.
"""

import os
import json
import re
import time
from collections import deque
from typing import Dict, Any, Optional, List
import httpx
from dotenv import load_dotenv

from schemas import TelemetryFrame, TriageReport, VitalStatus, TriageWarning
from prompts import FLIGHT_SURGEON_SYSTEM_PROMPT, build_telemetry_user_prompt

load_dotenv()

class AIFlightSurgeon:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        window_size: int = 10,
        request_timeout: float = 6.0
    ):
        """
        Initializes the AI Flight Surgeon.
        Supports Groq, Nebius, or custom OpenAI-compatible endpoints.
        """
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "").lower()
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.request_timeout = request_timeout
        
        # Resolve provider automatically if not explicitly given
        if not self.provider:
            if os.getenv("GROQ_API_KEY"):
                self.provider = "groq"
            elif os.getenv("NEBIUS_API_KEY"):
                self.provider = "nebius"
            elif os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
            else:
                self.provider = "fallback"

        # Set default base URLs and models for speed
        if self.provider == "groq":
            self.base_url = self.base_url or "https://api.groq.com/openai/v1"
            self.model = self.model or "llama-3.3-70b-versatile"
        elif self.provider == "nebius":
            self.base_url = self.base_url or "https://api.studio.nebius.ai/v1"
            self.model = self.model or "meta-llama/Meta-Llama-3.1-70B-Instruct"
        elif self.provider == "openai":
            self.base_url = self.base_url or "https://api.openai.com/v1"
            self.model = self.model or "gpt-4o-mini"
        elif self.provider == "fallback":
            self.model = "aerospace-clinical-expert-v1"

        # Rolling history of telemetry frames for trend & spike detection
        self.history: deque = deque(maxlen=window_size)
        self.latest_triage: Optional[TriageReport] = None

    def get_provider_status(self) -> Dict[str, Any]:
        """Returns the current operational status of the AI Flight Surgeon."""
        has_key = bool(self.api_key and len(self.api_key.strip()) > 5)
        return {
            "status": "OPERATIONAL",
            "provider": self.provider,
            "model": self.model,
            "has_api_key": has_key,
            "endpoint": self.base_url if has_key else "internal://aerospace-clinical-expert",
            "active_astronaut": "Benjamin",
            "history_buffer_len": len(self.history)
        }

    def compute_trend_deltas(self, current_frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes rate of change (deltas) over the recent window to catch sudden spikes
        (e.g., sudden desaturation drops or compensatory heart rate surges).
        """
        if not self.history:
            return {
                "delta_spo2": 0.0,
                "delta_hr": 0,
                "delta_systolic": 0,
                "delta_radiation": 0.0,
                "spike_detected": False
            }

        baseline = self.history[0]
        delta_spo2 = round(current_frame["spo2"] - baseline["spo2"], 2)
        delta_hr = current_frame["heart_rate"] - baseline["heart_rate"]
        delta_systolic = current_frame["blood_pressure_systolic"] - baseline["blood_pressure_systolic"]
        delta_radiation = round(current_frame["radiation_msv"] - baseline["radiation_msv"], 4)

        # Spike detection criteria
        spike_detected = (
            delta_spo2 <= -3.0 or  # Acute drop in SpO2
            delta_hr >= 20 or      # Acute tachycardic spike
            current_frame["spo2"] < 92.0 or # Hypoxia threshold
            delta_systolic >= 25 or
            delta_radiation >= 0.10
        )

        return {
            "delta_spo2": delta_spo2,
            "delta_hr": delta_hr,
            "delta_systolic": delta_systolic,
            "delta_radiation": delta_radiation,
            "spike_detected": spike_detected
        }

    def analyze_telemetry_sync(self, frame: TelemetryFrame) -> TriageReport:
        """
        Synchronous evaluation of telemetry frame.
        Useful for high-throughput batching, deterministic rules, or sync testing.
        """
        frame_dict = frame.model_dump()
        delta_info = self.compute_trend_deltas(frame_dict)
        
        start_time = time.perf_counter()
        triage_report: Optional[TriageReport] = None

        if self.api_key and self.provider in ("groq", "nebius", "openai"):
            try:
                triage_report = self._call_llm_sync(frame_dict, delta_info)
            except Exception as e:
                print(f"[AI Flight Surgeon] Sync LLM call error ({self.provider}): {e}. Falling back to clinical expert.")

        if triage_report is None:
            triage_report = self._evaluate_clinical_expert(frame_dict, delta_info)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        triage_report.response_time_ms = elapsed_ms

        self.history.append(frame_dict)
        self.latest_triage = triage_report
        return triage_report

    async def analyze_telemetry(self, frame: TelemetryFrame) -> TriageReport:
        """
        Asynchronous evaluation of telemetry frame.
        Called by FastAPI endpoints and real-time streaming handlers.
        """
        frame_dict = frame.model_dump()
        delta_info = self.compute_trend_deltas(frame_dict)
        
        start_time = time.perf_counter()
        triage_report: Optional[TriageReport] = None

        if self.api_key and self.provider in ("groq", "nebius", "openai"):
            try:
                triage_report = await self._call_llm_async(frame_dict, delta_info)
            except Exception as e:
                print(f"[AI Flight Surgeon] Async LLM call error ({self.provider}): {e}. Falling back to clinical expert.")

        if triage_report is None:
            triage_report = self._evaluate_clinical_expert(frame_dict, delta_info)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        triage_report.response_time_ms = elapsed_ms

        self.history.append(frame_dict)
        self.latest_triage = triage_report
        return triage_report

    def _call_llm_sync(self, frame_dict: Dict[str, Any], delta_info: Dict[str, Any]) -> TriageReport:
        """Synchronous HTTP call to LLM completion endpoint."""
        user_prompt = build_telemetry_user_prompt(frame_dict, delta_info)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": FLIGHT_SURGEON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        with httpx.Client(timeout=self.request_timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            result = resp.json()

        raw_content = result["choices"][0]["message"]["content"]
        parsed_json = self._extract_json(raw_content)
        
        if "astronaut_id" not in parsed_json:
            parsed_json["astronaut_id"] = frame_dict.get("astronaut_id", "Benjamin")
        if "timestamp" not in parsed_json:
            parsed_json["timestamp"] = frame_dict.get("timestamp", int(time.time()))
        
        parsed_json["source_model"] = f"{self.provider}:{self.model}"
        return TriageReport.model_validate(parsed_json)

    async def _call_llm_async(self, frame_dict: Dict[str, Any], delta_info: Dict[str, Any]) -> TriageReport:
        """Asynchronous HTTP call to LLM completion endpoint."""
        user_prompt = build_telemetry_user_prompt(frame_dict, delta_info)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": FLIGHT_SURGEON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            result = resp.json()

        raw_content = result["choices"][0]["message"]["content"]
        parsed_json = self._extract_json(raw_content)
        
        if "astronaut_id" not in parsed_json:
            parsed_json["astronaut_id"] = frame_dict.get("astronaut_id", "Benjamin")
        if "timestamp" not in parsed_json:
            parsed_json["timestamp"] = frame_dict.get("timestamp", int(time.time()))
        
        parsed_json["source_model"] = f"{self.provider}:{self.model}"
        return TriageReport.model_validate(parsed_json)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Strips markdown code fences and returns clean parsed JSON dict."""
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        return json.loads(text)

    def _evaluate_clinical_expert(self, frame_dict: Dict[str, Any], delta_info: Dict[str, Any]) -> TriageReport:
        """
        High-precision deterministic aerospace medical evaluation engine.
        Implements the exact same NASA Flight Surgeon threshold logic as the system prompt.
        Ensures 100% reliability, instant zero-latency processing, and offline safety.
        """
        hr = frame_dict["heart_rate"]
        spo2 = frame_dict["spo2"]
        systolic = frame_dict["blood_pressure_systolic"]
        diastolic = frame_dict["blood_pressure_diastolic"]
        radiation = frame_dict["radiation_msv"]
        astronaut_id = frame_dict.get("astronaut_id", "Benjamin")
        timestamp = frame_dict.get("timestamp", int(time.time()))

        warnings: List[TriageWarning] = []
        immediate_actions: List[str] = []
        secondary_concerns: List[str] = []
        severity_rank = {"NOMINAL": 0, "INFO": 0, "ADVISORY": 1, "WARNING": 2, "CRITICAL": 3}
        current_max_severity = "NOMINAL"

        def update_severity(sev: str):
            nonlocal current_max_severity
            if severity_rank[sev] > severity_rank[current_max_severity]:
                current_max_severity = sev

        # 1. SpO2 Analysis
        is_hypoxia = False
        if spo2 < 88.0 or (delta_info.get("delta_spo2", 0) <= -4.0 and spo2 <= 92.0):
            update_severity("CRITICAL")
            is_hypoxia = True
            warnings.append(TriageWarning(
                parameter="spo2",
                value=spo2,
                severity="CRITICAL",
                message=f"CRITICAL HYPOXIA SPIKE: SpO2 collapsed to {spo2}%. Rapid onset of cognitive degradation and loss of consciousness risk."
            ))
            immediate_actions.append("Initiate 100% emergency O2 flood via EVA/IVA life-support.")
            immediate_actions.append("Seal helmet visor, verify suit pressure integrity (ppO2 >= 3.0 psi).")
        elif spo2 < 92.0:
            update_severity("WARNING")
            is_hypoxia = True
            warnings.append(TriageWarning(
                parameter="spo2",
                value=spo2,
                severity="WARNING",
                message=f"Moderate hypoxia detected: SpO2 at {spo2}%. Oxygen desaturation ongoing."
            ))
            immediate_actions.append("Check cabin/suit O2 regulator and enrich partial oxygen flow.")
        elif spo2 < 95.0:
            update_severity("ADVISORY")
            warnings.append(TriageWarning(
                parameter="spo2",
                value=spo2,
                severity="ADVISORY",
                message=f"Mild oxygen desaturation: SpO2 at {spo2}%."
            ))

        # 2. Heart Rate Analysis
        if hr > 130:
            update_severity("CRITICAL")
            warnings.append(TriageWarning(
                parameter="heart_rate",
                value=float(hr),
                severity="CRITICAL",
                message=f"Severe tachyarrhythmia threshold exceeded: Heart Rate {hr} BPM."
            ))
            immediate_actions.append("Halt EVA/exercise immediately, place astronaut in supine position.")
        elif hr < 45:
            update_severity("CRITICAL")
            warnings.append(TriageWarning(
                parameter="heart_rate",
                value=float(hr),
                severity="CRITICAL",
                message=f"Severe bradycardia: Heart Rate {hr} BPM."
            ))
        elif hr > 110:
            sev = "WARNING"
            update_severity(sev)
            msg = f"Compensatory tachycardia ({hr} BPM) driven by oxygen deprivation." if is_hypoxia else f"Tachycardia detected: Heart Rate {hr} BPM."
            warnings.append(TriageWarning(
                parameter="heart_rate",
                value=float(hr),
                severity=sev,
                message=msg
            ))
            secondary_concerns.append("Compensatory Tachycardia")
        elif hr > 95:
            update_severity("ADVISORY")
            warnings.append(TriageWarning(
                parameter="heart_rate",
                value=float(hr),
                severity="ADVISORY",
                message=f"Mild heart rate elevation ({hr} BPM)."
            ))

        # 3. Blood Pressure Analysis
        if systolic >= 180 or diastolic >= 120:
            update_severity("CRITICAL")
            warnings.append(TriageWarning(
                parameter="blood_pressure",
                value=float(systolic),
                severity="CRITICAL",
                message=f"Hypertensive Crisis: BP {systolic}/{diastolic} mmHg."
            ))
            immediate_actions.append("Prepare emergency hypotensive protocol and monitor for intracranial pressure.")
        elif systolic < 85 or diastolic < 50:
            update_severity("CRITICAL")
            warnings.append(TriageWarning(
                parameter="blood_pressure",
                value=float(systolic),
                severity="CRITICAL",
                message=f"Cardiogenic/Hypotensive shock danger: BP {systolic}/{diastolic} mmHg."
            ))
        elif systolic >= 140 or diastolic >= 90:
            update_severity("WARNING")
            warnings.append(TriageWarning(
                parameter="blood_pressure",
                value=float(systolic),
                severity="WARNING",
                message=f"Stage 2 Hypertension: BP {systolic}/{diastolic} mmHg indicating acute physiological stress."
            ))
            secondary_concerns.append("Cardiovascular Strain")
        elif systolic > 120 or diastolic > 80:
            update_severity("ADVISORY")
            warnings.append(TriageWarning(
                parameter="blood_pressure",
                value=float(systolic),
                severity="ADVISORY",
                message=f"Pre-hypertensive vitals: BP {systolic}/{diastolic} mmHg."
            ))

        # 4. Radiation Exposure Analysis
        if radiation > 5.0:
            update_severity("CRITICAL")
            warnings.append(TriageWarning(
                parameter="radiation",
                value=radiation,
                severity="CRITICAL",
                message=f"CRITICAL RADIATION EVENT: {radiation:.2f} mSv. Solar Particle Event or shield breach."
            ))
            immediate_actions.append("Evacuate to AstroPulse storm shelter immediately; administer radioprotectants.")
        elif radiation >= 1.0:
            update_severity("WARNING")
            warnings.append(TriageWarning(
                parameter="radiation",
                value=radiation,
                severity="WARNING",
                message=f"Elevated radiation hazard: {radiation:.2f} mSv. Suspend external EVA operations."
            ))
            secondary_concerns.append("Radiation Hazard")
        elif radiation >= 0.20:
            update_severity("ADVISORY")
            warnings.append(TriageWarning(
                parameter="radiation",
                value=radiation,
                severity="ADVISORY",
                message=f"Radiation slightly elevated above cosmic baseline: {radiation:.2f} mSv."
            ))

        # Clinical Assessment & Synthesis
        if current_max_severity == "CRITICAL":
            trend = "DETERIORATING"
            intervention = True
            if is_hypoxia:
                primary = "Acute Severe Hypoxia & Respiratory Compromise"
                assessment = f"EMERGENCY: Astronaut {astronaut_id} is experiencing severe acute hypoxia (SpO2 {spo2}%) with rapid desaturation and compensatory sympathetic tachycardia ({hr} BPM). High risk of incapacitation."
            else:
                primary = "Critical Physiological Emergency"
                assessment = f"EMERGENCY: Critical biometric deviation detected for Astronaut {astronaut_id}. Immediate mission abort and clinical intervention required."
        elif current_max_severity == "WARNING":
            trend = "DETERIORATING" if delta_info.get("spike_detected") else "FLUCTUATING"
            intervention = True
            if is_hypoxia:
                primary = "Moderate Hypoxia with Compensatory Tachycardia"
                assessment = f"WARNING: Astronaut {astronaut_id} displays deteriorating oxygenation (SpO2 {spo2}%) accompanied by heart rate elevation to {hr} BPM and hypertension."
            else:
                primary = "Cardiovascular / Environmental Distress"
                assessment = f"WARNING: Abnormal physiological indicators detected for Astronaut {astronaut_id}. Close continuous telemetry monitoring required."
        elif current_max_severity == "ADVISORY":
            trend = "FLUCTUATING"
            intervention = False
            primary = "Mild Physiological Deviation"
            assessment = f"ADVISORY: Astronaut {astronaut_id} shows mild subclinical deviations in vitals (HR {hr} BPM, SpO2 {spo2}%). Continue standard tracking."
            immediate_actions.append("Maintain standard biotelemetry polling.")
        else:
            trend = "STABLE"
            intervention = False
            primary = "None - Vitals Stable"
            assessment = f"NOMINAL: Astronaut {astronaut_id}'s physiological markers are well within standard deep-space parameters (HR {hr} BPM, SpO2 {spo2}%, BP {systolic}/{diastolic})."
            immediate_actions.append("Maintain baseline telemetry log; astronaut fit for flight duty.")

        return TriageReport(
            astronaut_id=astronaut_id,
            timestamp=timestamp,
            triage_level=current_max_severity,
            vital_status=VitalStatus(
                heart_rate_bpm=hr,
                spo2_percent=spo2,
                blood_pressure=f"{systolic}/{diastolic}",
                radiation_msv=radiation
            ),
            clinical_assessment=assessment,
            primary_concern=primary,
            secondary_concerns=secondary_concerns,
            triage_warnings=warnings,
            immediate_actions=immediate_actions,
            requires_crew_intervention=intervention,
            telemetry_trend=trend,
            source_model="aerospace-clinical-expert-v1"
        )
