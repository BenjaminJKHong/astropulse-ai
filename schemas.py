from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import time

class TelemetryFrame(BaseModel):
    timestamp: int = Field(default_factory=lambda: int(time.time()), description="Unix epoch timestamp in seconds")
    heart_rate: int = Field(..., ge=20, le=250, description="Heart rate in beats per minute (BPM)")
    spo2: float = Field(..., ge=50.0, le=100.0, description="Blood oxygen saturation percentage (SpO2 %)")
    blood_pressure_systolic: int = Field(..., ge=40, le=260, description="Systolic blood pressure (mmHg)")
    blood_pressure_diastolic: int = Field(..., ge=20, le=180, description="Diastolic blood pressure (mmHg)")
    radiation_msv: float = Field(..., ge=0.0, le=500.0, description="Cumulative or hourly radiation dose (mSv)")
    astronaut_id: str = Field(default="Benjamin", description="Identifier of the astronaut being monitored")

class VitalStatus(BaseModel):
    heart_rate_bpm: int
    spo2_percent: float
    blood_pressure: str
    radiation_msv: float

class TriageWarning(BaseModel):
    parameter: str = Field(..., description="Vital sign name (e.g., spo2, heart_rate, blood_pressure, radiation)")
    value: float = Field(..., description="Numerical value triggering warning")
    severity: Literal["INFO", "ADVISORY", "WARNING", "CRITICAL"] = Field(...)
    message: str = Field(..., description="Concise clinical assessment of the deviation")

class TriageReport(BaseModel):
    astronaut_id: str = Field(default="Benjamin")
    timestamp: int
    triage_level: Literal["NOMINAL", "ADVISORY", "WARNING", "CRITICAL"]
    vital_status: VitalStatus
    clinical_assessment: str
    primary_concern: str
    secondary_concerns: List[str] = Field(default_factory=list)
    triage_warnings: List[TriageWarning] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    requires_crew_intervention: bool = False
    telemetry_trend: Literal["IMPROVING", "STABLE", "FLUCTUATING", "DETERIORATING"] = "STABLE"
    response_time_ms: Optional[float] = None
    source_model: Optional[str] = None

class SimulationScenarioRequest(BaseModel):
    scenario: Literal["hypoxia", "tachycardia", "decompression", "radiation_spike", "nominal", "reset"]
    astronaut_id: Optional[str] = "Benjamin"
    severity: Optional[str] = "moderate" # mild, moderate, severe

class BlandCallRequest(BaseModel):
    phone_number: str = Field(default="+15550192834", description="Recipient phone number (E.164 format)")
    task_override: Optional[str] = Field(default=None, description="Custom instruction prompt for Bland AI voice agent")
    trigger_reason: Optional[str] = Field(default="Telemetry Critical Warning", description="Reason for initiating call")

class BlandCallResponse(BaseModel):
    call_id: str
    status: str
    phone_number: str
    task: str
    simulated: bool
    voice: str = "maya"
    created_at: int
    message: str

class SurgeonInteractionRequest(BaseModel):
    message: str = Field(..., description="Message or query from Astronaut Benjamin or Mission Controller")
    astronaut_id: Optional[str] = "Benjamin"

class SurgeonInteractionResponse(BaseModel):
    reply: str = Field(..., description="Flight Surgeon's verbal/conversational response")
    speaker: str = "AstroPulse AI Flight Surgeon"
    triage_level: str
    current_spo2: float
    current_hr: int
    clinical_concern: str
    recommended_actions: List[str] = Field(default_factory=list)


