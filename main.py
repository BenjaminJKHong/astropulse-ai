from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import json
import time
from typing import Optional

from schemas import (
    TelemetryFrame,
    TriageReport,
    SimulationScenarioRequest,
    BlandCallRequest,
    BlandCallResponse,
    SurgeonInteractionRequest,
    SurgeonInteractionResponse
)
from telemetry_stream import BenjaminTelemetryStream
from flight_surgeon import AIFlightSurgeon
from bland_service import BlandAIInteractionService
from band_agent import BandAgentService

# Initialize core services (before lifespan so they're accessible everywhere)
telemetry_stream = BenjaminTelemetryStream(csv_path="nasa_vitals.csv")
flight_surgeon = AIFlightSurgeon()
bland_service = BlandAIInteractionService()
band_service = BandAgentService(
    flight_surgeon=flight_surgeon,
    telemetry_stream=telemetry_stream,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: start BAND agent on startup, stop on shutdown."""
    await band_service.start()
    yield
    await band_service.stop()


app = FastAPI(
    title="AstroPulse AI - Flight Surgeon Medical Operations",
    description="Autonomous real-time clinical triage monitoring and conversational voice interaction for deep-space astronaut telemetry.",
    version="2.2.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Health check endpoint indicating backend, flight surgeon, Bland AI, and BAND status."""
    surgeon_status = flight_surgeon.get_provider_status()
    return {
        "status": "AstroPulse Backend is Online",
        "flight_surgeon": surgeon_status,
        "bland_ai": {
            "configured": bland_service.is_configured(),
            "default_voice": bland_service.default_voice,
            "api_endpoint": bland_service.base_url
        },
        "band_ai": {
            "enabled": band_service.enabled,
            "configured": band_service.is_configured(),
            "connected": band_service.is_connected(),
        },
        "monitored_astronaut": "Benjamin",
        "system_time": int(time.time())
    }

@app.get("/api/telemetry")
async def get_simulated_vitals():
    """
    Backward-compatible telemetry endpoint for Astronaut Benjamin.
    Returns the next telemetry frame from the live stream.
    """
    frame = telemetry_stream.get_current_frame(advance=True)
    return frame.model_dump()

@app.get("/api/flight-surgeon/status")
async def get_flight_surgeon_status():
    """Returns AI Flight Surgeon configuration, LLM provider, Bland AI and BAND AI status, and telemetry stream status."""
    return {
        "surgeon": flight_surgeon.get_provider_status(),
        "bland_ai": {
            "configured": bland_service.is_configured(),
            "default_voice": bland_service.default_voice
        },
        "band_ai": band_service.get_status(),
        "astronaut": {
            "name": "Benjamin",
            "role": "Mission Specialist / EVA Lead",
            "callsign": telemetry_stream.callsign,
            "status": "Monitored Active"
        },
        "latest_triage": flight_surgeon.latest_triage.model_dump() if flight_surgeon.latest_triage else None
    }


@app.get("/api/band/status")
async def get_band_status():
    """
    Returns BAND AI agent integration status.
    Reports: enabled flag, credentials configured, connection state,
    agent identity, collaboration activity, and last error.
    Never exposes API keys or secrets.
    """
    status = band_service.get_status()
    return status

@app.post("/api/flight-surgeon/triage", response_model=TriageReport)
async def analyze_telemetry_frame(frame: Optional[TelemetryFrame] = None):
    """
    Analyzes numerical vitals for Astronaut Benjamin using the AI Flight Surgeon.
    Accepts an explicit frame or samples the next frame from Benjamin's live stream.
    Outputs strict JSON triage warnings and recommendations.
    """
    if frame is None:
        frame = telemetry_stream.get_current_frame(advance=True)
    
    triage_report = await flight_surgeon.analyze_telemetry(frame)
    return triage_report

@app.get("/api/flight-surgeon/live-triage", response_model=TriageReport)
async def get_live_triage():
    """
    Single-call live pipeline: advances Benjamin's telemetry stream by one frame
    and immediately executes AI Flight Surgeon clinical triage.
    """
    frame = telemetry_stream.get_current_frame(advance=True)
    triage_report = await flight_surgeon.analyze_telemetry(frame)
    return triage_report

@app.post("/api/flight-surgeon/interact", response_model=SurgeonInteractionResponse)
async def interact_with_flight_surgeon(req: SurgeonInteractionRequest):
    """
    Conversational AI interaction with the AstroPulse Flight Surgeon.
    Enables voice/chat dialogue with Astronaut Benjamin or ground control,
    grounded in real-time telemetry and triage status.
    """
    triage = flight_surgeon.latest_triage
    if triage is None:
        frame = telemetry_stream.get_current_frame(advance=False)
        triage = await flight_surgeon.analyze_telemetry(frame)
    
    response = await bland_service.interact(req.message, triage, flight_surgeon)
    return response

@app.post("/api/flight-surgeon/call", response_model=BlandCallResponse)
async def dispatch_bland_call(req: BlandCallRequest):
    """
    Initiates an outbound emergency voice call via Bland AI (https://bland.ai)
    to Astronaut Benjamin or mission ground control, communicating urgent triage directives.
    """
    triage = flight_surgeon.latest_triage
    if triage is None:
        frame = telemetry_stream.get_current_frame(advance=False)
        triage = await flight_surgeon.analyze_telemetry(frame)
    
    call_res = await bland_service.initiate_call(
        phone_number=req.phone_number,
        triage=triage,
        task_override=req.task_override
    )
    return call_res

@app.post("/api/flight-surgeon/simulate")
async def simulate_event(req: SimulationScenarioRequest):
    """
    Injects sudden data spikes or anomalies into Benjamin's telemetry stream:
    - 'hypoxia': Acute oxygen desaturation with compensatory tachycardia
    - 'tachycardia': Extreme cardiac elevation
    - 'radiation_spike': Solar Particle Event spike
    - 'nominal' / 'reset': Clears queues and restores nominal baseline
    """
    if req.scenario == "hypoxia":
        frames = telemetry_stream.trigger_hypoxia_event(severity=req.severity or "severe")
        msg = f"Injected acute hypoxia event ({len(frames)} frames queued)."
    elif req.scenario == "tachycardia":
        frames = telemetry_stream.trigger_tachycardia_event()
        msg = f"Injected sudden tachycardia event ({len(frames)} frames queued)."
    elif req.scenario == "radiation_spike":
        frames = telemetry_stream.trigger_radiation_spike()
        msg = f"Injected radiation burst event ({len(frames)} frames queued)."
    elif req.scenario in ("nominal", "reset"):
        telemetry_stream.reset_to_nominal()
        msg = "Telemetry stream reset to nominal baseline."
        frames = []
    else:
        raise HTTPException(status_code=400, detail=f"Unknown scenario '{req.scenario}'")

    return {
        "status": "SUCCESS",
        "scenario": req.scenario,
        "message": msg,
        "queued_frames_count": len(frames)
    }

@app.get("/api/flight-surgeon/stream")
async def stream_live_telemetry_sse():
    """
    Server-Sent Events (SSE) stream delivering real-time telemetry paired with
    AI Flight Surgeon triage warnings every 2 seconds.
    """
    async def event_generator():
        while True:
            frame = telemetry_stream.get_current_frame(advance=True)
            triage = await flight_surgeon.analyze_telemetry(frame)
            data = {
                "telemetry": frame.model_dump(),
                "triage": triage.model_dump()
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/dashboard", response_class=HTMLResponse)
async def flight_surgeon_dashboard():
    """
    Interactive Mission Control Dashboard for Astronaut Benjamin's Telemetry,
    AI Flight Surgeon Live Triage, Spike Detection, and Bland AI Voice Interaction.
    """
    html_content = Path("dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)

# Parameter of sleep
@app.websocket("/ws/telemetry")
async def telemetry_websocket_endpoint(websocket: WebSocket, frequency: float | None = 1.0):
    """
    Dedicated WebSocket endpoint for AstroPulse live telemetry & clinical triage.
    Pushes a new NASA vitals frame + AI Flight Surgeon triage report every 1 second.
    """
    await websocket.accept()
    try:
        while True:
            # 1. Get Astronaunt data here, in this case is Benjamin's telemetry
            frame = telemetry_stream.get_current_frame(advance=True)
            
            # 2. Execute AI Flight Surgeon clinical triage evaluation
            triage = await flight_surgeon.analyze_telemetry(frame)
            
            # 3. Construct payload for frontend charts & gauges
            payload = {
                "telemetry": frame.model_dump(),
                "triage": triage.model_dump()
            }
            
            # 4. Push JSON payload down the socket
            await websocket.send_text(json.dumps(payload))
            
            # 5. Depending on frequency parameter but if it is empty, it will be 1 second by default before broadcasting the next live pulse
            await asyncio.sleep(frequency)
            
    except WebSocketDisconnect:
        print("Mission Control WebSocket client disconnected from /ws/telemetry")
    except Exception as e:
        print(f"WebSocket error on /ws/telemetry: {e}")