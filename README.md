# AstroPulse AI // Flight Surgeon Operations

Autonomous, real-time predictive biotelemetry and clinical triage surveillance for deep-space astronaut safety.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Astronaut Telemetry Stream       |
                      | (nasa_vitals.csv / Live Bio-Sensors)  |
                      +---------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           AI Flight Surgeon Engine                                |
|                                                                                   |
|  - Rate of Change / Delta Analysis (Delta SpO2, Delta HR, Delta BP, Delta Rad)   |
|  - Strict Medical System Prompts & Aerospace Clinical Protocols                  |
|  - High-Speed LLM Inference (Groq / Nebius Studio / OpenAI)                       |
|  - Deterministic Aerospace Clinical Fallback Engine (Zero-Latency / Offline)      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Strict JSON Triage Report       |
                      | (Triage Level, Warnings, Directives)  |
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             [FastAPI REST / SSE]                [Mission Control GUI]
             /api/flight-surgeon/*               /dashboard
```

---

## Features

1. **AI Medical Officer Pipeline**:
   - Analyzes real-time physiological vitals: **Heart Rate (BPM)**, **Blood Oxygen (SpO2 %)**, **Blood Pressure (Systolic/Diastolic)**, and **Radiation (mSv)**.
   - Detects acute syndromic patterns (e.g., **Hypoxia with Compensatory Tachycardia**, **Hypertensive Emergencies**, and **Solar Radiation Storms**).

2. **Strict System Prompts & Structured JSON Schema**:
   - Enforces NASA Flight Surgeon directives with strict numerical thresholds.
   - Eliminates conversational chatter; produces parseable JSON matching the `TriageReport` schema.

3. **Connected to Benjamin's Telemetry Stream**:
   - Ingests `nasa_vitals.csv` representing Astronaut Benjamin's live bio-stream.
   - Tracks rolling windows to compute $\Delta SpO_2 / \Delta t$ and $\Delta HR / \Delta t$ for sudden spike detection.

4. **Sudden Spike & Anomaly Simulation**:
   - `POST /api/flight-surgeon/simulate` to trigger acute hypoxia events, tachycardia spikes, or radiation bursts.
   - Validated against rapid desaturation tests.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check & surgeon provider status |
| `GET` | `/api/telemetry` | Next raw telemetry frame from Benjamin's stream |
| `GET` | `/api/flight-surgeon/status` | Current AI Surgeon configuration & active astronaut |
| `POST` | `/api/flight-surgeon/triage` | Analyzes a provided frame or the current stream frame |
| `GET` | `/api/flight-surgeon/live-triage` | Live pipeline: pulls next frame & executes AI triage |
| `POST` | `/api/flight-surgeon/simulate` | Injects anomaly events (`hypoxia`, `tachycardia`, `radiation_spike`, `reset`) |
| `GET` | `/api/flight-surgeon/stream` | Server-Sent Events (SSE) live push |
| `GET` | `/dashboard` | Interactive Mission Control Web GUI |

---

## WebSocket Endpoints

| Endpoint | Data/Param | Description |
| `/ws/telemetry?frequency={optional-parameter}` | sleep(float, default=1.0, Optional) | Handshake with AstroPulse websocket to retrive real-time data such as telemetry and triage every 1 second by default. The call frequency are adjustable |
Sample Output (1s): /ws/telemetry
```
{
    "telemetry": {
        "timestamp": 1793801272,
        "heart_rate": 72,
        "spo2": 99.0,
        "blood_pressure_systolic": 120,
        "blood_pressure_diastolic": 80,
        "radiation_msv": 0.12,
        "astronaut_id": "Benjamin"
    },
    "triage": {
        "astronaut_id": "Benjamin",
        "timestamp": 1793801272,
        "triage_level": "NOMINAL",
        "vital_status": {
            "heart_rate_bpm": 72,
            "spo2_percent": 99.0,
            "blood_pressure": "120/80",
            "radiation_msv": 0.12
        },
        "clinical_assessment": "NOMINAL: Astronaut Benjamin's physiological markers are well within standard deep-space parameters (HR 72 BPM, SpO2 99.0%, BP 120/80).",
        "primary_concern": "None - Vitals Stable",
        "secondary_concerns": [],
        "triage_warnings": [],
        "immediate_actions": [
            "Maintain baseline telemetry log; astronaut fit for flight duty."
        ],
        "requires_crew_intervention": false,
        "telemetry_trend": "STABLE",
        "response_time_ms": 0.03,
        "source_model": "aerospace-clinical-expert-v1"
    }
}
```

## Configuration (`.env`)

To connect to Groq or Nebius:

```env
# Groq (Ultra-fast inference: Llama-3.3-70b-versatile, Llama-3.1-8b-instant)
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Nebius Studio (Alternative OpenAI-compatible API)
# NEBIUS_API_KEY=your_nebius_api_key_here
# LLM_PROVIDER=nebius
# LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
# LLM_BASE_URL=https://api.studio.nebius.ai/v1
```

*Note: If no API key is provided, the system seamlessly uses the built-in deterministic aerospace medical engine, ensuring 100% test coverage and offline reliability.*

---

## Running Tests

Execute the automated test suites:

```bash
# Core Flight Surgeon Engine & Spike Detection Tests
python test_flight_surgeon.py

# FastAPI Endpoints & Simulation API Tests
python test_api_endpoints.py
```

## Starting the Server

```bash
uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000/dashboard` to access the Mission Control Dashboard.
