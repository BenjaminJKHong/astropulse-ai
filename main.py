from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
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
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AstroPulse AI | Flight Surgeon Mission Control</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --border: #1f293d;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #38bdf8;
      --critical: #ef4444;
      --warning: #f59e0b;
      --advisory: #3b82f6;
      --nominal: #10b981;
      --bland: #8b5cf6;
      --band: #06b6d4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background-color: var(--bg); color: var(--text); padding: 20px; }
    .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 20px; }
    .title-group h1 { font-size: 24px; color: #38bdf8; letter-spacing: 1px; display: flex; align-items: center; gap: 10px; }
    .badge { font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: bold; text-transform: uppercase; }
    .badge-provider { background: #1e293b; color: #38bdf8; border: 1px solid #38bdf8; }
    .badge-bland { background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid #8b5cf6; }
    .badge-band { background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid #06b6d4; }
    .triage-badge { font-size: 14px; padding: 6px 14px; border-radius: 6px; font-weight: 700; }
    .triage-NOMINAL { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .triage-ADVISORY { background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid #3b82f6; }
    .triage-WARNING { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; animation: pulse 2s infinite; }
    .triage-CRITICAL { background: rgba(239, 68, 68, 0.25); color: #ef4444; border: 1px solid #ef4444; animation: blink 1s infinite; }
    
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.02); } }

    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
    .card-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px; }
    .card-value { font-size: 28px; font-weight: bold; }
    .card-unit { font-size: 14px; color: var(--text-muted); margin-left: 4px; }

    .main-section { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
    .chart-container { height: 280px; width: 100%; }
    
    .triage-panel { display: flex; flex-direction: column; gap: 12px; }
    .panel-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
    .panel-box h3 { font-size: 14px; color: var(--accent); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }

    .warning-item { padding: 8px 10px; border-radius: 4px; margin-bottom: 6px; font-size: 13px; border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.08); }
    .warning-item.WARNING { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.08); }
    .warning-item.ADVISORY { border-left-color: #3b82f6; background: rgba(59, 130, 246, 0.08); }
    
    .actions-list { list-style: none; }
    .actions-list li { font-size: 13px; padding: 6px 0; border-bottom: 1px dashed var(--border); display: flex; align-items: flex-start; gap: 8px; }
    .actions-list li::before { content: "►"; color: var(--accent); font-size: 10px; margin-top: 3px; }

    .controls-bar { display: flex; gap: 12px; align-items: center; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; flex-wrap: wrap; }
    .btn { background: #1f2937; color: white; border: 1px solid #374151; padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
    .btn:hover { background: #374151; border-color: #4b5563; }
    .btn-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border-color: #ef4444; }
    .btn-danger:hover { background: rgba(239, 68, 68, 0.35); }
    .btn-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border-color: #f59e0b; }
    .btn-reset { background: rgba(16, 185, 129, 0.2); color: #34d399; border-color: #10b981; }
    .btn-bland { background: rgba(139, 92, 246, 0.25); color: #c084fc; border-color: #8b5cf6; }
    .btn-bland:hover { background: rgba(139, 92, 246, 0.45); }

    /* Interaction & Comms Section */
    .comms-section { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; margin-bottom: 20px; }
    .chat-box { display: flex; flex-direction: column; height: 320px; background: #0b1120; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
    .chat-messages { flex: 1; padding: 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; font-size: 13px; }
    .chat-msg { max-width: 85%; padding: 8px 12px; border-radius: 8px; line-height: 1.4; }
    .chat-msg.surgeon { background: #1e293b; color: #e2e8f0; align-self: flex-start; border-left: 3px solid #38bdf8; }
    .chat-msg.user { background: #312e81; color: #ede9fe; align-self: flex-end; }
    .chat-input-bar { display: flex; border-top: 1px solid var(--border); background: var(--card-bg); padding: 8px; gap: 8px; }
    .chat-input { flex: 1; background: #0b1120; border: 1px solid var(--border); color: white; padding: 8px 12px; border-radius: 6px; font-size: 13px; outline: none; }
    .chat-input:focus { border-color: var(--accent); }

    .quick-prompts { display: flex; gap: 6px; padding: 6px 12px; background: #0f172a; border-top: 1px solid var(--border); overflow-x: auto; }
    .quick-chip { font-size: 11px; background: #1e293b; color: #94a3b8; padding: 3px 8px; border-radius: 12px; cursor: pointer; white-space: nowrap; border: 1px solid #334155; }
    .quick-chip:hover { color: #38bdf8; border-color: #38bdf8; }

    .bland-panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between; }
    .bland-call-log { font-family: "Courier New", Courier, monospace; font-size: 11px; background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 10px; color: #c084fc; max-height: 140px; overflow-y: auto; margin-top: 10px; }

    #raw-json { font-family: "Courier New", Courier, monospace; font-size: 11px; color: #a7f3d0; background: #0b1120; padding: 12px; border-radius: 6px; max-height: 180px; overflow-y: auto; border: 1px solid var(--border); }

    /* BAND AI Multi-Agent Coordination Panel */
    .band-section { background: var(--card-bg); border: 1px solid rgba(6, 182, 212, 0.4); border-radius: 8px; padding: 18px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(6, 182, 212, 0.05); }
    .band-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 14px; }
    .band-header h3 { font-size: 15px; color: #22d3ee; display: flex; align-items: center; gap: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    .band-grid { display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 16px; }
    .band-box { background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 12px; display: flex; flex-direction: column; justify-content: space-between; }
    .band-box h4 { font-size: 11px; color: #22d3ee; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }
    .band-meta-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.04); font-size: 12px; }
    .band-meta-label { color: var(--text-muted); }
    .band-meta-val { font-family: monospace; font-weight: bold; }
    .band-event-log { font-family: "Courier New", Courier, monospace; font-size: 11px; background: #070d18; border: 1px solid var(--border); border-radius: 4px; padding: 8px; color: #67e8f9; max-height: 120px; overflow-y: auto; }
  </style>
</head>
<body>

  <div class="header">
    <div class="title-group">
      <h1>ASTROPULSE // AI FLIGHT SURGEON</h1>
      <span class="badge badge-provider" id="provider-badge">Connecting...</span>
      <span class="badge badge-bland" id="bland-badge">BLAND AI VOICE: READY</span>
      <span class="badge badge-band" id="band-badge">BAND AI: CONNECTING...</span>
      <span class="badge" style="background:#0f172a; color:#94a3b8; border:1px solid #334155;">SUBJECT: ASTRONAUT BENJAMIN</span>
    </div>
    <div id="triage-status-pill" class="triage-badge triage-NOMINAL">NOMINAL</div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="card-label">Heart Rate</div>
      <div><span class="card-value" id="val-hr">--</span><span class="card-unit">BPM</span></div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Baseline: 60 - 95 BPM</div>
    </div>
    <div class="card">
      <div class="card-label">Oxygen Saturation (SpO2)</div>
      <div><span class="card-value" id="val-spo2">--</span><span class="card-unit">%</span></div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Nominal: &ge; 95.0%</div>
    </div>
    <div class="card">
      <div class="card-label">Blood Pressure</div>
      <div><span class="card-value" id="val-bp">--/--</span><span class="card-unit">mmHg</span></div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Target: 120/80 mmHg</div>
    </div>
    <div class="card">
      <div class="card-label">Radiation Exposure</div>
      <div><span class="card-value" id="val-rad">--</span><span class="card-unit">mSv</span></div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Limit: &lt; 0.20 mSv/hr</div>
    </div>
  </div>

  <div class="main-section">
    <div class="card">
      <div class="card-label" style="display:flex; justify-content:space-between;">
        <span>Live Telemetry Trend (SpO2 & Heart Rate)</span>
        <span id="trend-indicator" style="color:var(--nominal);">TREND: STABLE</span>
      </div>
      <div class="chart-container">
        <canvas id="telemetryChart"></canvas>
      </div>
    </div>

    <div class="triage-panel">
      <div class="panel-box">
        <h3>Clinical Assessment</h3>
        <p id="assessment-text" style="font-size: 13px; line-height: 1.5; color: #e2e8f0;">Initializing clinical model...</p>
        <div style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
          <strong>Primary Concern:</strong> <span id="primary-concern" style="color: #f1f5f9;">None</span>
        </div>
      </div>

      <div class="panel-box" style="flex: 1;">
        <h3>AI Triage Warnings (<span id="warn-count">0</span>)</h3>
        <div id="warnings-container">
          <div style="font-size: 12px; color: var(--text-muted);">No active warnings. Physiological parameters nominal.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Conversational Interaction & Bland AI Voice Comms Section -->
  <div class="comms-section">
    <div class="panel-box">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h3>AI Flight Surgeon Comm-Link (Interactive Dialogue)</h3>
        <label style="font-size: 11px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; cursor: pointer;">
          <input type="checkbox" id="voice-synthesis-toggle" checked>
          <span>🔈 Voice Audio Synthesis</span>
        </label>
      </div>
      <div class="chat-box">
        <div class="chat-messages" id="chat-messages">
          <div class="chat-msg surgeon">
            <strong>Flight Surgeon:</strong> Comms link established, Astronaut Benjamin. Telemetry stream is active. What is your current status?
          </div>
        </div>
        <div class="quick-prompts">
          <span class="quick-chip" onclick="sendQuickPrompt('Surgeon, I am feeling slightly dizzy and short of breath.')">"I feel dizzy"</span>
          <span class="quick-chip" onclick="sendQuickPrompt('Flight Surgeon, report my full vital signs.')">"Report vitals"</span>
          <span class="quick-chip" onclick="sendQuickPrompt('Explain current hypoxia and triage warnings.')">"Explain hypoxia"</span>
          <span class="quick-chip" onclick="sendQuickPrompt('Verify suit oxygen pressure and seal integrity.')">"Verify oxygen"</span>
        </div>
        <div class="chat-input-bar">
          <input type="text" id="chat-input" class="chat-input" placeholder="Communicate with Flight Surgeon..." onkeydown="if(event.key==='Enter') sendMessage()">
          <button class="btn btn-bland" onclick="sendMessage()">Transmit</button>
        </div>
      </div>
    </div>

    <div class="bland-panel">
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h3 style="color:#c084fc;">Bland AI Outbound Voice Dispatch</h3>
          <span class="badge badge-bland">VOICE AGENT</span>
        </div>
        <p style="font-size: 12px; color: var(--text-muted); margin: 6px 0 12px;">
          Dispatches an automated high-priority telephone voice agent via Bland AI (https://bland.ai) directly to Astronaut Benjamin or Mission Control with clinical triage directives.
        </p>

        <div style="display: flex; gap: 8px; margin-bottom: 10px;">
          <input type="text" id="bland-phone" class="chat-input" value="+15550192834" style="flex: 1;" placeholder="Phone (E.164: +1...)">
          <button class="btn btn-bland" onclick="dispatchBlandCall()">📞 Dispatch Call</button>
        </div>

        <div style="font-size: 11px; color: var(--text-muted);">Emergency Prompt Preview:</div>
        <div id="bland-task-preview" style="font-size: 11px; color: #cbd5e1; background: #0b1120; padding: 8px; border-radius: 4px; border: 1px solid var(--border); margin-top: 4px; max-height: 80px; overflow-y: auto;">
          Waiting for telemetry update...
        </div>
      </div>

      <div>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 8px;">Dispatch Status & Call Log:</div>
        <div class="bland-call-log" id="bland-log">
          [System] Bland AI Voice Service Standby.
        </div>
      </div>
    </div>
  </div>

  <div class="grid" style="grid-template-columns: 1fr 1fr; margin-bottom: 20px;">
    <div class="panel-box">
      <h3>Immediate Flight Surgeon Directives</h3>
      <ul class="actions-list" id="actions-list">
        <li>Maintain standard biotelemetry polling.</li>
      </ul>
    </div>
    <div class="panel-box">
      <h3>Raw Triage Output (Strict JSON)</h3>
      <pre id="raw-json">{}</pre>
    </div>
  </div>

  <!-- BAND AI Multi-Agent Coordination Infrastructure Section -->
  <div class="band-section">
    <div class="band-header">
      <div>
        <h3>🌐 BAND AI // Multi-Agent Network Coordination</h3>
        <p style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
          Autonomous decentralized multi-agent interaction infrastructure (<a href="https://band.ai" target="_blank" style="color:#22d3ee; text-decoration:none;">https://band.ai</a>) — live peer discovery, room participant management, and specialist clinical escalation.
        </p>
      </div>
      <div>
        <span class="badge badge-band" id="band-conn-pill">INITIALIZING</span>
      </div>
    </div>

    <div class="band-grid">
      <!-- Agent Identity & Platform Connection -->
      <div class="band-box">
        <h4>Agent Identity &amp; Transport</h4>
        <div>
          <div class="band-meta-row">
            <span class="band-meta-label">Identity:</span>
            <span class="band-meta-val" id="band-agent-name">AstroPulse Flight Surgeon</span>
          </div>
          <div class="band-meta-row">
            <span class="band-meta-label">Agent ID:</span>
            <span class="band-meta-val" id="band-agent-id" style="color: #22d3ee; font-size: 10px;">Checking...</span>
          </div>
          <div class="band-meta-row">
            <span class="band-meta-label">Transport Protocol:</span>
            <span class="band-meta-val">Phoenix WebSocket + REST</span>
          </div>
          <div class="band-meta-row">
            <span class="band-meta-label">Mention Routing:</span>
            <span class="band-meta-val" style="color: #34d399;">Active (Judged Turns)</span>
          </div>
          <div class="band-meta-row" style="border-bottom: none;">
            <span class="band-meta-label">Inbound Messages:</span>
            <span class="band-meta-val" id="band-msg-count">0 processed</span>
          </div>
        </div>
      </div>

      <!-- Multi-Agent Collaboration Protocol -->
      <div class="band-box">
        <h4>Multi-Agent Collaboration Engine</h4>
        <div>
          <div class="band-meta-row">
            <span class="band-meta-label">Peer Discovery:</span>
            <span class="band-meta-val">band_lookup_peers()</span>
          </div>
          <div class="band-meta-row">
            <span class="band-meta-label">Room Invitation:</span>
            <span class="band-meta-val">band_add_participant()</span>
          </div>
          <div class="band-meta-row">
            <span class="band-meta-label">Escalation Policy:</span>
            <span class="band-meta-val" style="color: #f59e0b;">WARNING &amp; CRITICAL Triage</span>
          </div>
          <div class="band-meta-row">
            <span class="band-meta-label">Specialist Matching:</span>
            <span class="band-meta-val">Aerospace Medical &amp; Life-Support</span>
          </div>
          <div class="band-meta-row" style="border-bottom: none;">
            <span class="band-meta-label">Telemetry Protection:</span>
            <span class="band-meta-val" style="color: #34d399;">Read-Only (No Stream Adv.)</span>
          </div>
        </div>
      </div>

      <!-- Real-Time Activity & Collaboration Log -->
      <div class="band-box">
        <h4>Recent Multi-Agent Events &amp; Handoffs</h4>
        <div class="band-event-log" id="band-event-log">
          [System] BAND agent integration loaded. Polling connection state...
        </div>
      </div>
    </div>
  </div>

  <div class="controls-bar">
    <span style="font-size: 13px; font-weight: bold; color: var(--text-muted);">SIMULATE ANOMALY SPIKES:</span>
    <button class="btn btn-danger" onclick="injectScenario('hypoxia')">🚨 Simulate Acute Hypoxia Event</button>
    <button class="btn btn-warning" onclick="injectScenario('tachycardia')">⚡ Simulate Cardiac Spike</button>
    <button class="btn btn-warning" onclick="injectScenario('radiation_spike')">☢️ Simulate Radiation Burst</button>
    <button class="btn btn-reset" onclick="injectScenario('reset')">🔄 Restore Nominal Baseline</button>
  </div>

  <script>
    const maxDataPoints = 20;
    const labels = [];
    const hrData = [];
    const spo2Data = [];
    let currentTriage = null;

    const ctx = document.getElementById('telemetryChart').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Heart Rate (BPM)',
            data: hrData,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            yAxisID: 'yHR',
            tension: 0.3
          },
          {
            label: 'SpO2 (%)',
            data: spo2Data,
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            yAxisID: 'ySpO2',
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#1f293d' }, ticks: { color: '#9ca3af' } },
          yHR: {
            type: 'linear',
            position: 'left',
            min: 40,
            max: 180,
            grid: { color: '#1f293d' },
            ticks: { color: '#f59e0b' },
            title: { display: true, text: 'Heart Rate (BPM)', color: '#f59e0b' }
          },
          ySpO2: {
            type: 'linear',
            position: 'right',
            min: 70,
            max: 100,
            grid: { drawOnChartArea: false },
            ticks: { color: '#38bdf8' },
            title: { display: true, text: 'SpO2 (%)', color: '#38bdf8' }
          }
        },
        plugins: {
          legend: { labels: { color: '#cbd5e1' } }
        }
      }
    });

    async function pollLiveTriage() {
      try {
        const res = await fetch('/api/flight-surgeon/live-triage');
        if (!res.ok) return;
        const triage = await res.json();
        currentTriage = triage;
        updateUI(triage);
      } catch (err) {
        console.error("Polling error:", err);
      }
    }

    function updateUI(triage) {
      const v = triage.vital_status;
      document.getElementById('val-hr').innerText = v.heart_rate_bpm;
      document.getElementById('val-spo2').innerText = v.spo2_percent.toFixed(1);
      document.getElementById('val-bp').innerText = v.blood_pressure;
      document.getElementById('val-rad').innerText = v.radiation_msv.toFixed(3);

      // Status pill
      const pill = document.getElementById('triage-status-pill');
      pill.innerText = triage.triage_level;
      pill.className = `triage-badge triage-${triage.triage_level}`;

      // Trend
      const trend = document.getElementById('trend-indicator');
      trend.innerText = `TREND: ${triage.telemetry_trend}`;
      trend.style.color = triage.telemetry_trend === 'DETERIORATING' ? '#ef4444' : (triage.telemetry_trend === 'STABLE' ? '#10b981' : '#f59e0b');

      // Clinical Assessment
      document.getElementById('assessment-text').innerText = triage.clinical_assessment;
      document.getElementById('primary-concern').innerText = triage.primary_concern;

      // Warnings
      const warnContainer = document.getElementById('warnings-container');
      document.getElementById('warn-count').innerText = triage.triage_warnings.length;
      if (triage.triage_warnings.length === 0) {
        warnContainer.innerHTML = '<div style="font-size: 12px; color: var(--text-muted);">No active warnings. Physiological parameters nominal.</div>';
      } else {
        warnContainer.innerHTML = triage.triage_warnings.map(w => 
          `<div class="warning-item ${w.severity}"><strong>[${w.severity}] ${w.parameter.toUpperCase()}:</strong> ${w.message}</div>`
        ).join('');
      }

      // Actions
      const actionsList = document.getElementById('actions-list');
      if (triage.immediate_actions && triage.immediate_actions.length > 0) {
        actionsList.innerHTML = triage.immediate_actions.map(a => `<li>${a}</li>`).join('');
      } else {
        actionsList.innerHTML = '<li>Maintain standard telemetry polling.</li>';
      }

      // Raw JSON
      document.getElementById('raw-json').innerText = JSON.stringify(triage, null, 2);

      // Model badge
      if (triage.source_model) {
        document.getElementById('provider-badge').innerText = `ENGINE: ${triage.source_model}`;
      }

      // Bland AI Task Preview
      const previewText = `[Bland AI Task Prompt] Call Benjamin. Triage: ${triage.triage_level} | Concern: ${triage.primary_concern} | SpO2: ${v.spo2_percent}% | Directives: ${triage.immediate_actions.join(', ')}`;
      document.getElementById('bland-task-preview').innerText = previewText;

      // Update Chart
      const timeLabel = new Date(triage.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      labels.push(timeLabel);
      hrData.push(v.heart_rate_bpm);
      spo2Data.push(v.spo2_percent);

      if (labels.length > maxDataPoints) {
        labels.shift();
        hrData.shift();
        spo2Data.shift();
      }
      chart.update();
    }

    async function injectScenario(scenario) {
      try {
        await fetch('/api/flight-surgeon/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario: scenario, severity: 'severe' })
        });
        setTimeout(pollLiveTriage, 200);
      } catch (err) {
        alert("Failed to inject scenario: " + err);
      }
    }

    // Conversational Interaction
    async function sendMessage() {
      const input = document.getElementById('chat-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      appendMessage(text, 'user');

      try {
        const res = await fetch('/api/flight-surgeon/interact', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, astronaut_id: 'Benjamin' })
        });
        if (res.ok) {
          const data = await res.json();
          appendMessage(data.reply, 'surgeon');
          speakReply(data.reply);
        } else {
          appendMessage("Error communicating with Flight Surgeon.", 'surgeon');
        }
      } catch (err) {
        appendMessage("Comms link failure: " + err, 'surgeon');
      }
    }

    function sendQuickPrompt(promptText) {
      document.getElementById('chat-input').value = promptText;
      sendMessage();
    }

    function appendMessage(text, role) {
      const container = document.getElementById('chat-messages');
      const div = document.createElement('div');
      div.className = `chat-msg ${role}`;
      div.innerHTML = role === 'surgeon' ? `<strong>Flight Surgeon:</strong> ${text}` : `<strong>Benjamin:</strong> ${text}`;
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
    }

    function speakReply(text) {
      const toggle = document.getElementById('voice-synthesis-toggle');
      if (toggle && toggle.checked && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
      }
    }

    // Bland AI Voice Dispatch
    async function dispatchBlandCall() {
      const phone = document.getElementById('bland-phone').value.trim();
      const log = document.getElementById('bland-log');
      log.innerText = `[Dispatching] Contacting Bland AI for phone ${phone}...`;

      try {
        const res = await fetch('/api/flight-surgeon/call', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ phone_number: phone, trigger_reason: "Manual / Emergency Dispatch" })
        });
        const data = await res.json();
        log.innerHTML = `[SUCCESS] Call ID: ${data.call_id}<br>Status: ${data.status.toUpperCase()} (${data.simulated ? 'SIMULATED' : 'LIVE BLAND.AI'})<br>Voice: ${data.voice}<br>${data.message}`;
      } catch (err) {
        log.innerText = `[ERROR] Failed to dispatch Bland AI call: ${err}`;
      }
    }

    // Poll BAND AI status every 3 seconds
    async function pollBandStatus() {
      try {
        const res = await fetch('/api/band/status');
        if (!res.ok) return;
        const b = await res.json();
        const badge = document.getElementById('band-badge');
        const pill = document.getElementById('band-conn-pill');
        const idElem = document.getElementById('band-agent-id');
        const countElem = document.getElementById('band-msg-count');
        const logElem = document.getElementById('band-event-log');

        if (!b.enabled) {
          badge.innerText = 'BAND AI: DISABLED';
          badge.style.color = '#94a3b8';
          badge.style.borderColor = '#475569';
          pill.innerText = 'DISABLED (BAND_ENABLED=false)';
          pill.className = 'badge';
        } else if (!b.configured) {
          badge.innerText = 'BAND AI: UNCONFIGURED';
          badge.style.color = '#f59e0b';
          badge.style.borderColor = '#f59e0b';
          pill.innerText = 'NO CREDENTIALS';
          pill.className = 'badge badge-warning';
        } else if (b.connected) {
          badge.innerText = 'BAND AI: CONNECTED';
          badge.style.color = '#34d399';
          badge.style.borderColor = '#10b981';
          pill.innerText = 'ONLINE / CONNECTED';
          pill.className = 'badge badge-band';
        } else {
          badge.innerText = 'BAND AI: STANDBY';
          badge.style.color = '#38bdf8';
          badge.style.borderColor = '#38bdf8';
          pill.innerText = b.last_error ? 'RECONNECTING' : 'CONNECTING...';
          pill.className = 'badge badge-band';
        }

        if (b.agent_id) {
          idElem.innerText = b.agent_id;
        }

        if (b.adapter) {
          countElem.innerText = `${b.adapter.messages_processed || 0} processed`;

          if (b.adapter.collaboration_events && b.adapter.collaboration_events.length > 0) {
            logElem.innerHTML = b.adapter.collaboration_events.map(ev => 
              `[${ev.timestamp.substring(11, 19)}] Specialist Hand-off: <strong>${ev.specialist_name}</strong> (Room: ${ev.room_id}) - ${ev.triage_level}: ${ev.primary_concern}`
            ).join('<br>');
          } else if (b.connected) {
            logElem.innerText = `[${new Date().toLocaleTimeString()}] Listening for room mentions (@FlightSurgeon) on BAND platform...`;
          } else if (b.last_error) {
            logElem.innerText = `[${new Date().toLocaleTimeString()}] Notice: ${b.last_error}`;
          }
        }
      } catch (err) {
        console.error("BAND status polling error:", err);
      }
    }

    // Initialize and poll every 2.5s
    pollLiveTriage();
    setInterval(pollLiveTriage, 2500);

    // Initialize and poll BAND status every 3s
    pollBandStatus();
    setInterval(pollBandStatus, 3000);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
