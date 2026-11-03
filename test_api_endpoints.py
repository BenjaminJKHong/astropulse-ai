"""
FastAPI Endpoints Test Suite for AstroPulse AI Flight Surgeon.
Verifies REST API endpoints, simulation triggers, and dashboard serving.
"""

from starlette.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check_endpoint():
    print("[API TEST 1] Testing GET / health check...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "AstroPulse Backend is Online"
    assert data["monitored_astronaut"] == "Benjamin"
    assert "flight_surgeon" in data
    print("  [PASSED] Health check endpoint OK.")

def test_telemetry_stream_endpoint():
    print("[API TEST 2] Testing GET /api/telemetry (Benjamin's stream)...")
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "heart_rate" in data
    assert "spo2" in data
    assert data["astronaut_id"] == "Benjamin"
    print(f"  [PASSED] Telemetry frame retrieved: HR={data['heart_rate']} BPM, SpO2={data['spo2']}%.")

def test_flight_surgeon_status_endpoint():
    print("[API TEST 3] Testing GET /api/flight-surgeon/status...")
    response = client.get("/api/flight-surgeon/status")
    assert response.status_code == 200
    data = response.json()
    assert data["astronaut"]["name"] == "Benjamin"
    assert data["surgeon"]["status"] == "OPERATIONAL"
    print("  [PASSED] Flight surgeon status endpoint OK.")

def test_flight_surgeon_triage_custom_hypoxia_frame():
    print("[API TEST 4] Testing POST /api/flight-surgeon/triage with sudden hypoxia spike...")
    hypoxia_payload = {
        "timestamp": 1700000000,
        "heart_rate": 130,
        "spo2": 86.0,
        "blood_pressure_systolic": 158,
        "blood_pressure_diastolic": 100,
        "radiation_msv": 0.15,
        "astronaut_id": "Benjamin"
    }
    response = client.post("/api/flight-surgeon/triage", json=hypoxia_payload)
    assert response.status_code == 200
    report = response.json()
    assert report["triage_level"] == "CRITICAL"
    assert report["requires_crew_intervention"] is True
    assert "hypoxia" in report["primary_concern"].lower() or "hypoxia" in report["clinical_assessment"].lower()
    
    # Check that immediate actions contain O2 response
    assert any("o2" in act.lower() or "oxygen" in act.lower() for act in report["immediate_actions"])
    print(f"  [PASSED] Hypoxia triage response verified: Level={report['triage_level']}, Warnings={len(report['triage_warnings'])}.")

def test_live_triage_endpoint():
    print("[API TEST 5] Testing GET /api/flight-surgeon/live-triage...")
    response = client.get("/api/flight-surgeon/live-triage")
    assert response.status_code == 200
    report = response.json()
    assert report["astronaut_id"] == "Benjamin"
    assert report["triage_level"] in ("NOMINAL", "ADVISORY", "WARNING", "CRITICAL")
    print(f"  [PASSED] Live triage endpoint returned active triage: {report['triage_level']}.")

def test_simulation_scenario_injection():
    print("[API TEST 6] Testing POST /api/flight-surgeon/simulate (hypoxia & reset)...")
    
    # 1. Inject Hypoxia
    inject_resp = client.post("/api/flight-surgeon/simulate", json={"scenario": "hypoxia", "severity": "severe"})
    assert inject_resp.status_code == 200
    assert inject_resp.json()["status"] == "SUCCESS"
    assert inject_resp.json()["queued_frames_count"] == 5

    # 2. Consume from stream and verify hypoxia is delivered
    live_resp = client.get("/api/flight-surgeon/live-triage")
    assert live_resp.status_code == 200
    first_frame = live_resp.json()
    assert first_frame["vital_status"]["spo2_percent"] == 95.0

    # 3. Reset to nominal
    reset_resp = client.post("/api/flight-surgeon/simulate", json={"scenario": "reset"})
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "SUCCESS"
    print("  [PASSED] Scenario simulation injection and reset verified.")

def test_dashboard_endpoint():
    print("[API TEST 7] Testing GET /dashboard HTML serving...")
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "ASTROPULSE // AI FLIGHT SURGEON" in response.text
    assert "ASTRONAUT BENJAMIN" in response.text
    print("  [PASSED] Mission Control Dashboard served successfully.")

if __name__ == "__main__":
    print("======================================================================")
    print("      ASTROPULSE API & FLIGHT SURGEON ENDPOINTS TEST SUITE            ")
    print("======================================================================")
    test_health_check_endpoint()
    test_telemetry_stream_endpoint()
    test_flight_surgeon_status_endpoint()
    test_flight_surgeon_triage_custom_hypoxia_frame()
    test_live_triage_endpoint()
    test_simulation_scenario_injection()
    test_dashboard_endpoint()
    print("======================================================================")
    print("       ALL 7 API ENDPOINTS TESTED AND VERIFIED CLEANLY (100%)         ")
    print("======================================================================")

