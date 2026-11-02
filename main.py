from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
import time

app = FastAPI(title="AstroPulse AI Backend", version="1.0")

# Security: Configure CORS so the Next.js frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update this to your Vercel/Render URL later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "AstroPulse Backend is Online"}

@app.get("/api/telemetry")
async def get_simulated_vitals():
    """
    Temporary endpoint. Will be replaced by real NASA CSV data streaming.
    Outputs simulated vitals for the frontend to start testing charts.
    """
    return {
        "timestamp": int(time.time()),
        "heart_rate": random.randint(60, 100),       # BPM
        "spo2": random.uniform(95.0, 100.0),         # Oxygen percentage
        "blood_pressure_systolic": random.randint(110, 130),
        "blood_pressure_diastolic": random.randint(70, 85),
        "radiation_msv": random.uniform(0.1, 0.5)    # Millisieverts
    }