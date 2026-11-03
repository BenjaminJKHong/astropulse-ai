"""
Comprehensive Automated Test Suite for AstroPulse AI Flight Surgeon.
Tests:
1. Strict JSON triage warning schema compliance
2. Telemetry prompt generation with delta tracking
3. Baseline nominal vitals response
4. Sudden hypoxia event detection and triage escalation (SpO2 collapse + compensatory tachycardia)
5. Sudden tachycardia and cardiovascular stress detection
6. End-to-end integration with Benjamin's recorded telemetry stream
7. Anomaly injection pipeline and recovery
8. Sudden solar radiation burst detection
9. Raw Markdown-wrapped LLM JSON repair and validation
"""

import json
import time
from schemas import TelemetryFrame, TriageReport
from flight_surgeon import AIFlightSurgeon
from telemetry_stream import BenjaminTelemetryStream
from prompts import FLIGHT_SURGEON_SYSTEM_PROMPT, build_telemetry_user_prompt

def test_system_prompt_structure():
    """Verify system prompt directives and threshold rules are strictly present."""
    print("[TEST 1] Verifying System Prompt Directives & Threshold Rules...")
    assert "You are the AI Flight Surgeon" in FLIGHT_SURGEON_SYSTEM_PROMPT
    assert "Astronaut Benjamin" in FLIGHT_SURGEON_SYSTEM_PROMPT
    assert "OXYGEN SATURATION (SpO2 %)" in FLIGHT_SURGEON_SYSTEM_PROMPT
    assert "HEART RATE (BPM)" in FLIGHT_SURGEON_SYSTEM_PROMPT
    assert "HYPOXIA EVENT PATTERN" in FLIGHT_SURGEON_SYSTEM_PROMPT
    assert "STRICT OUTPUT ENFORCEMENT" in FLIGHT_SURGEON_SYSTEM_PROMPT
    print("  [PASSED] System prompt structure and clinical directives verified.")

def test_user_prompt_construction():
    """Verify prompt builder formats numerical telemetry and deltas."""
    print("[TEST 2] Verifying Telemetry Prompt Payload Construction...")
    frame = {
        "timestamp": 1698820000,
        "heart_rate": 115,
        "spo2": 90.5,
        "blood_pressure_systolic": 150,
        "blood_pressure_diastolic": 98,
        "radiation_msv": 0.22,
        "astronaut_id": "Benjamin"
    }
    deltas = {
        "delta_spo2": -8.5,
        "delta_hr": +43,
        "delta_systolic": +30,
        "delta_radiation": +0.10,
        "spike_detected": True
    }
    prompt = build_telemetry_user_prompt(frame, deltas)
    assert "Astronaut Benjamin" in prompt
    assert "Heart Rate: 115 BPM" in prompt
    assert "Blood Oxygen (SpO2): 90.5%" in prompt
    assert "Delta SpO2: -8.50%" in prompt
    assert "Rapid Spike Detected: True" in prompt
    print("  [PASSED] User prompt construction verified.")

def test_nominal_vitals():
    """Ensure nominal baseline vitals produce NOMINAL triage status with zero emergency warnings."""
    print("[TEST 3] Testing Nominal Baseline Vitals...")
    surgeon = AIFlightSurgeon()
    nominal_frame = TelemetryFrame(
        timestamp=int(time.time()),
        heart_rate=72,
        spo2=99.0,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        radiation_msv=0.12,
        astronaut_id="Benjamin"
    )
    report = surgeon.analyze_telemetry_sync(nominal_frame)
    assert report.triage_level == "NOMINAL", f"Expected NOMINAL, got {report.triage_level}"
    assert report.vital_status.spo2_percent == 99.0
    assert report.vital_status.heart_rate_bpm == 72
    assert report.requires_crew_intervention is False
    assert len(report.triage_warnings) == 0
    print(f"  [PASSED] Nominal status verified. Assessment: '{report.clinical_assessment}'")

def test_sudden_hypoxia_spike():
    """
    Simulate a sudden hypoxia spike:
    SpO2 collapses from 99% down to 87.5%, HR spikes to 125 bpm (compensatory response).
    Ensure the AI Flight Surgeon catches it and issues critical triage warnings.
    """
    print("[TEST 4] Testing Sudden Hypoxia Spike Detection (SpO2 Collapse & Compensatory Tachycardia)...")
    surgeon = AIFlightSurgeon()
    
    # Prime history with baseline
    baseline = TelemetryFrame(
        timestamp=int(time.time()) - 15,
        heart_rate=72,
        spo2=99.0,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        radiation_msv=0.12,
        astronaut_id="Benjamin"
    )
    surgeon.analyze_telemetry_sync(baseline)

    # Inject sudden acute hypoxia spike
    hypoxia_frame = TelemetryFrame(
        timestamp=int(time.time()),
        heart_rate=125,
        spo2=87.5,
        blood_pressure_systolic=155,
        blood_pressure_diastolic=98,
        radiation_msv=0.14,
        astronaut_id="Benjamin"
    )
    report = surgeon.analyze_telemetry_sync(hypoxia_frame)

    print(f"  -> AI Triage Level: {report.triage_level}")
    print(f"  -> Primary Concern: {report.primary_concern}")
    print(f"  -> Clinical Assessment: {report.clinical_assessment}")
    print(f"  -> Triage Warnings Count: {len(report.triage_warnings)}")

    # Assertions
    assert report.triage_level in ("CRITICAL", "WARNING"), f"Expected high alert, got {report.triage_level}"
    assert report.requires_crew_intervention is True
    assert report.telemetry_trend == "DETERIORATING"

    # Verify SpO2 warning
    spo2_warnings = [w for w in report.triage_warnings if w.parameter == "spo2"]
    assert len(spo2_warnings) > 0, "AI failed to produce a triage warning for SpO2"
    assert spo2_warnings[0].severity in ("CRITICAL", "WARNING")
    assert "hypoxia" in spo2_warnings[0].message.lower() or "desaturation" in spo2_warnings[0].message.lower()

    # Verify Heart Rate warning
    hr_warnings = [w for w in report.triage_warnings if w.parameter == "heart_rate"]
    assert len(hr_warnings) > 0, "AI failed to produce a triage warning for compensatory tachycardia"

    # Verify Immediate Actions include oxygen countermeasure
    actions_text = " ".join(report.immediate_actions).lower()
    assert "o2" in actions_text or "oxygen" in actions_text or "seal" in actions_text, "Missing oxygen intervention in actions"
    print("  [PASSED] Hypoxia event successfully caught with emergency oxygen actions ordered.")

def test_benjamin_telemetry_stream_connection():
    """
    Connect the AI pipeline to Benjamin's live telemetry stream (nasa_vitals.csv).
    Verify that reading through the stream reflects the deteriorating vitals progression.
    """
    print("[TEST 5] Testing AI Connection to Benjamin's Recorded Telemetry Stream...")
    stream = BenjaminTelemetryStream(csv_path="nasa_vitals.csv")
    surgeon = AIFlightSurgeon()
    
    triage_history = []
    
    # Process all 5 recorded frames in Benjamin's dataset
    for i in range(len(stream.recorded_frames)):
        frame = stream.get_current_frame(advance=True)
        report = surgeon.analyze_telemetry_sync(frame)
        triage_history.append((frame, report))
        print(f"  [Frame {i+1}] HR={frame.heart_rate} BPM, SpO2={frame.spo2}%, BP={frame.blood_pressure_systolic}/{frame.blood_pressure_diastolic} -> Triage: {report.triage_level} | Concern: {report.primary_concern}")

    # The 1st frame is nominal (HR 72, SpO2 99%)
    assert triage_history[0][1].triage_level == "NOMINAL"

    # By the 5th frame (HR 115, SpO2 90.5%, BP 150/98), the patient is in hypoxia & hypertension
    final_report = triage_history[-1][1]
    assert final_report.triage_level in ("WARNING", "CRITICAL"), f"Expected escalation at frame 5, got {final_report.triage_level}"
    assert len(final_report.triage_warnings) >= 2, "Expected multiple warnings (SpO2, HR, BP)"
    print("  [PASSED] Stream connection and progressive triage deterioration verified.")

def test_simulated_scenario_queuing():
    """Verify anomaly simulation queuing and reset functionality."""
    print("[TEST 6] Testing Anomaly Simulation Pipeline & Reset...")
    stream = BenjaminTelemetryStream()
    surgeon = AIFlightSurgeon()

    # Inject severe hypoxia simulation
    frames = stream.trigger_hypoxia_event(severity="severe")
    assert len(frames) == 5
    assert stream.active_simulation_queue

    # Fetch queued frame and analyze
    first_injected = stream.get_current_frame(advance=True)
    report = surgeon.analyze_telemetry_sync(first_injected)
    assert report.vital_status.spo2_percent == 95.0

    # Advance to the peak hypoxia frame (SpO2 78%, HR 145)
    for _ in range(3):
        stream.get_current_frame(advance=True)
    peak_frame = stream.get_current_frame(advance=True)
    peak_report = surgeon.analyze_telemetry_sync(peak_frame)
    assert peak_report.triage_level == "CRITICAL"
    assert peak_report.vital_status.spo2_percent == 78.0
    assert peak_report.vital_status.heart_rate_bpm == 145

    # Test stream reset to nominal
    stream.reset_to_nominal()
    assert len(stream.active_simulation_queue) == 0
    reset_frame = stream.get_current_frame(advance=False)
    assert reset_frame.heart_rate == 72
    assert reset_frame.spo2 == 99.0
    print("  [PASSED] Simulation injection, peak critical capture, and nominal reset verified.")

def test_radiation_burst():
    """Verify radiation spike detection."""
    print("[TEST 7] Testing Sudden Solar Radiation Burst Spike...")
    surgeon = AIFlightSurgeon()
    rad_frame = TelemetryFrame(
        timestamp=int(time.time()),
        heart_rate=75,
        spo2=98.5,
        blood_pressure_systolic=122,
        blood_pressure_diastolic=80,
        radiation_msv=8.45,
        astronaut_id="Benjamin"
    )
    report = surgeon.analyze_telemetry_sync(rad_frame)
    assert report.triage_level == "CRITICAL"
    rad_warnings = [w for w in report.triage_warnings if w.parameter == "radiation"]
    assert len(rad_warnings) > 0
    assert rad_warnings[0].severity == "CRITICAL"
    assert any("shelter" in act.lower() for act in report.immediate_actions)
    print("  [PASSED] Radiation spike and shelter evacuation protocol verified.")

def test_json_repair_and_validation():
    """Ensure raw LLM responses wrapped in markdown code blocks parse cleanly."""
    print("[TEST 8] Testing Markdown JSON Stripping & Pydantic Validation...")
    surgeon = AIFlightSurgeon()
    mock_llm_markdown = """```json
{
  "astronaut_id": "Benjamin",
  "timestamp": 1698820020,
  "triage_level": "WARNING",
  "vital_status": {
    "heart_rate_bpm": 115,
    "spo2_percent": 90.5,
    "blood_pressure": "150/98",
    "radiation_msv": 0.22
  },
  "clinical_assessment": "Astronaut Benjamin is exhibiting acute hypoxemic distress.",
  "primary_concern": "Acute Hypoxia",
  "secondary_concerns": ["Compensatory Tachycardia"],
  "triage_warnings": [
    {
      "parameter": "spo2",
      "value": 90.5,
      "severity": "WARNING",
      "message": "Rapid desaturation detected."
    }
  ],
  "immediate_actions": ["Switch EMU to 100% emergency O2 supply"],
  "requires_crew_intervention": true,
  "telemetry_trend": "DETERIORATING"
}
```"""
    parsed = surgeon._extract_json(mock_llm_markdown)
    report = TriageReport.model_validate(parsed)
    assert report.astronaut_id == "Benjamin"
    assert report.triage_level == "WARNING"
    assert report.vital_status.spo2_percent == 90.5
    print("  [PASSED] Markdown JSON extraction and schema validation verified.")

def run_all_tests():
    print("======================================================================")
    print("     ASTROPULSE AI FLIGHT SURGEON - COMPREHENSIVE TEST SUITE          ")
    print("======================================================================")
    test_system_prompt_structure()
    test_user_prompt_construction()
    test_json_repair_and_validation()
    test_nominal_vitals()
    test_sudden_hypoxia_spike()
    test_benjamin_telemetry_stream_connection()
    test_simulated_scenario_queuing()
    test_radiation_burst()
    print("======================================================================")
    print("       ALL 8 TEST SUITES PASSED CLEANLY (100% SUCCESS)                ")
    print("======================================================================")

if __name__ == "__main__":
    run_all_tests()
