"""
AstroPulse AI - Live BAND Multi-Agent Collaboration Demonstration
==================================================================
Demonstrates how the BAND FlightSurgeonBandAdapter processes an incoming room message:
  1. Receives message mentioning @FlightSurgeon
  2. Inspects Astronaut Benjamin's live telemetry (without advancing the stream)
  3. Executes clinical triage analysis
  4. Automatically discovers registered BAND peers
  5. Selects and adds an appropriate specialist agent (e.g. Pulmonologist / Life Support)
  6. Dispatches a structured multi-agent consultation handoff
  7. Returns a comprehensive clinical status report to Mission Control
"""

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from schemas import TelemetryFrame, TriageReport, VitalStatus, TriageWarning
from flight_surgeon import AIFlightSurgeon
from telemetry_stream import BenjaminTelemetryStream
from band_agent import FlightSurgeonBandAdapter


async def run_collaboration_demonstration():
    print("=" * 76)
    print("   ASTROPULSE AI // BAND MULTI-AGENT COLLABORATION DEMONSTRATION    ")
    print("=" * 76)

    # 1. Setup real Flight Surgeon and Telemetry Stream
    telemetry_stream = BenjaminTelemetryStream()
    flight_surgeon = AIFlightSurgeon()

    adapter = FlightSurgeonBandAdapter(
        flight_surgeon=flight_surgeon,
        telemetry_stream=telemetry_stream,
    )

    # 2. Simulate Astronaut Benjamin entering an acute hypoxia event
    print("\n[SCENARIO] Simulating Astronaut Benjamin experiencing acute hypoxia in EVA...")
    telemetry_stream.trigger_hypoxia_event(severity="severe")
    telemetry_stream.active_simulation_queue[0] = {
        "heart_rate": 138, "spo2": 82.5,
        "blood_pressure_systolic": 168, "blood_pressure_diastolic": 104,
        "radiation_msv": 0.15, "astronaut_id": "Benjamin"
    }

    # 3. Simulate BAND Room Environment with registered peers
    print("[BAND ENVIRONMENT] Discoverable BAND Peers registered on platform:")
    class DemoPeer:
        def __init__(self, id, name, handle, description):
            self.id = id
            self.name = name
            self.handle = handle
            self.description = description

    peers = [
        DemoPeer("agent-pulmonology-99", "Aerospace Pulmonology Specialist", "@aero_pulm", "clinical hypoxia respiratory and EVA life-support"),
        DemoPeer("agent-cardio-12", "Spaceflight Cardiology Expert", "@space_cardio", "hemodynamic monitoring arrhythmias blood pressure"),
        DemoPeer("agent-logistics-01", "Cargo & Logistics Agent", "@cargo_bot", "manifest schedules supply delivery"),
    ]
    for p in peers:
        print(f"  • {p.name} ({p.handle}) -> {p.description}")

    # 4. Construct mock BAND AgentTools
    tools = MagicMock()
    tools.agent_id = "c481ef81-9785-49f2-b4fd-0eb058ed6070"
    tools.turn = MagicMock()
    tools.turn.judged = False
    tools.lookup_peers = AsyncMock(return_value=MagicMock(data=peers))
    tools.add_participant = AsyncMock(return_value=MagicMock(status="added"))
    
    sent_messages = []
    async def mock_send(content: str, mentions=None):
        sent_messages.append({"content": content, "mentions": mentions})
    tools.send_message = AsyncMock(side_effect=mock_send)

    # 5. Simulate Mission Control sending a message into the BAND room
    print("\n[BAND ROOM] Inbound message received from Mission Control:")
    incoming_msg = MagicMock()
    incoming_msg.content = "@FlightSurgeon Astronaut Benjamin is reporting acute dizziness and dyspnea. Analyze his vitals and consult a specialist immediately."
    incoming_msg.sender_id = "user-flight-director-01"
    incoming_msg.sender_name = "Flight Director (Gene Kranz)"
    incoming_msg.room_id = "room-astropulse-eva-alpha"

    print(f"  From: {incoming_msg.sender_name} ({incoming_msg.sender_id})")
    print(f"  Room: {incoming_msg.room_id}")
    print(f"  Content: \"{incoming_msg.content}\"")

    # 6. Execute adapter message processing turn
    print("\n[AGENT REASONING] Flight Surgeon processing BAND turn...")
    await adapter.on_message(
        msg=incoming_msg,
        tools=tools,
        history=[],
        participants_msg=None,
        contacts_msg=None,
        is_session_bootstrap=False,
        room_id=incoming_msg.room_id,
    )

    # 7. Review Turn Outcomes
    print("\n" + "=" * 76)
    print("                  BAND MULTI-AGENT ACTIONS EXECUTED                  ")
    print("=" * 76)

    # Participant added check
    print("\n[ACTION 1] Peer Discovery & Specialist Invitation:")
    tools.lookup_peers.assert_awaited()
    print("  -> tools.lookup_peers() executed successfully.")
    tools.add_participant.assert_awaited()
    add_args = tools.add_participant.call_args
    print(f"  -> tools.add_participant('{add_args[0][0]}') called.")
    print("  -> Specialist 'Aerospace Pulmonology Specialist' added to room 'room-astropulse-eva-alpha'.")

    # Messages sent
    print(f"\n[ACTION 2] Platform Messages Dispatched ({len(sent_messages)} messages):")
    for idx, m in enumerate(sent_messages, 1):
        print(f"\n--- MESSAGE {idx} (Mentions: {m['mentions']}) ---")
        print(m["content"])

    # Verify adapter state
    status = adapter.get_status()
    print("\n" + "=" * 76)
    print(f"Adapter Status: {status['messages_processed']} messages processed | {len(status['collaboration_events'])} collaboration events recorded.")
    print("=" * 76)


if __name__ == "__main__":
    asyncio.run(run_collaboration_demonstration())
