from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import csv
import time

app = FastAPI(title="AstroPulse AI Backend", version="1.0")

# Allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the NASA data into memory when the server starts
telemetry_data = []
current_index = 0

try:
    with open("nasa_vitals.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert string CSV data into proper numbers for the frontend and AI
            telemetry_data.append({
                "timestamp": int(row["timestamp"]),
                "heart_rate": int(row["heart_rate"]),
                "spo2": float(row["spo2"]),
                "blood_pressure_systolic": int(row["blood_pressure_systolic"]),
                "blood_pressure_diastolic": int(row["blood_pressure_diastolic"]),
                "radiation_msv": float(row["radiation_msv"])
            })
except FileNotFoundError:
    print("Warning: nasa_vitals.csv not found. Create it in the root directory.")

@app.get("/")
async def health_check():
    return {"status": "AstroPulse Backend is Online"}

@app.get("/api/telemetry")
async def get_simulated_vitals():
    """
    Reads the NASA CSV file row by row to simulate a live telemetry stream.
    """
    global current_index
    
    if not telemetry_data:
        return {"error": "NASA data not loaded."}
    
    # Grab the current row of data
    data = telemetry_data[current_index]
    
    # Move to the next row for the next API call, loop back if at the end
    current_index = (current_index + 1) % len(telemetry_data)
    
    # Overwrite the static CSV timestamp with the actual current time for the live charts
    data["timestamp"] = int(time.time())
    
    return data