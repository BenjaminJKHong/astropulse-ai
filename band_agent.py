"""
AstroPulse AI - BAND Agent Adapter
=====================================
Connects the AI Flight Surgeon to the BAND multi-agent platform (https://band.ai).
Implements a SimpleAdapter that routes incoming BAND room messages through the
existing Flight Surgeon intelligence engine without replacing any existing logic.

Architecture:
  - FlightSurgeonBandAdapter  : SimpleAdapter subclass; handles on_message turns
  - BandAgentService          : Lifecycle wrapper (start/stop) for FastAPI integration

Message flow:
  BAND room mention → on_message() → get latest telemetry → flight surgeon triage
  → generate response → tools.send_message() back to room
  → (if needed) lookup_peers → add_participant → delegate specialist
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from band.core.simple_adapter import SimpleAdapter
from band.core.types import PlatformMessage
from band.core.protocols import AgentToolsProtocol
from band import Agent

if TYPE_CHECKING:
    pass

logger = logging.getLogger("astropulse.band_agent")


# ---------------------------------------------------------------------------
# Collaboration thresholds
# ---------------------------------------------------------------------------

# Only escalate to a specialist when triage level is WARNING or CRITICAL
_ESCALATE_LEVELS = {"WARNING", "CRITICAL"}

# Keyword patterns in user message that signal multi-agent collaboration
_COLLABORATION_KEYWORDS = {
    "assist", "help", "specialist", "collaborate", "consult",
    "second opinion", "expert", "analyse", "analyze"
}


def _wants_collaboration(message_text: str) -> bool:
    """True if the incoming message explicitly requests specialist collaboration."""
    lower = message_text.lower()
    return any(kw in lower for kw in _COLLABORATION_KEYWORDS)


# ---------------------------------------------------------------------------
# SimpleAdapter implementation
# ---------------------------------------------------------------------------

class FlightSurgeonBandAdapter(SimpleAdapter[list[dict[str, Any]]]):
    """
    BAND SimpleAdapter that uses AstroPulse AI Flight Surgeon as its intelligence.

    Receives incoming room turns via on_message(), fetches live telemetry,
    runs the existing triage engine, and sends a structured medical response
    back into the BAND room.  When the message requests specialist collaboration
    and the triage level warrants it, it will:
      1. Look up available peers via tools.lookup_peers()
      2. Add the most relevant peer to the room via tools.add_participant()
      3. Send a concise handoff message mentioning the specialist
    """

    # This adapter handles its own turn reply, so judges_turns = True
    judges_turns = True

    def __init__(
        self,
        flight_surgeon: Any,
        telemetry_stream: Any,
        agent_name: str = "AstroPulse Flight Surgeon",
    ) -> None:
        super().__init__()
        self.flight_surgeon = flight_surgeon
        self.telemetry_stream = telemetry_stream
        self.agent_name = agent_name

        # Metrics / activity log for the status endpoint and dashboard
        self.connected_at: Optional[datetime] = None
        self.last_message_at: Optional[datetime] = None
        self.messages_processed: int = 0
        self.collaboration_events: list[dict[str, Any]] = []
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Called once after the agent successfully connects to BAND."""
        self.connected_at = datetime.utcnow()
        logger.info(
            "[BAND] Flight Surgeon agent started as '%s' — connected to BAND platform.",
            agent_name,
        )

    # ------------------------------------------------------------------
    # Core message handler
    # ------------------------------------------------------------------

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: list[dict[str, Any]],
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Process an incoming BAND room message mentioning the Flight Surgeon.

        Steps:
          1. Extract message text and sender info
          2. Get latest telemetry frame (no stream advance — read-only)
          3. Run Flight Surgeon triage analysis
          4. Check if collaboration was requested
          5. If YES + triage warrants it: discover peers, add relevant specialist
          6. Compose and send response back to BAND room
        """
        self.last_message_at = datetime.utcnow()
        self.messages_processed += 1

        sender_name = msg.sender_name or "Mission Controller"
        message_text = msg.content.strip()

        logger.info(
            "[BAND] Received message in room %s from %s: %s",
            room_id, sender_name, message_text[:120],
        )

        try:
            # ── Step 1: Get telemetry without advancing the stream ──────────
            frame = self.telemetry_stream.get_current_frame(advance=False)

            # ── Step 2: Run triage ─────────────────────────────────────────
            triage = await self.flight_surgeon.analyze_telemetry(frame)

            # ── Step 3: Build primary response ─────────────────────────────
            response_text = self._build_triage_response(
                sender_name=sender_name,
                triage=triage,
                message_text=message_text,
            )

            # ── Step 4: Specialist collaboration ───────────────────────────
            wants_collab = _wants_collaboration(message_text)
            specialist_added = False

            if wants_collab and triage.triage_level in _ESCALATE_LEVELS:
                specialist_added = await self._attempt_specialist_collaboration(
                    tools=tools,
                    room_id=room_id,
                    triage=triage,
                    sender_name=sender_name,
                )

            # ── Step 5: Send primary reply ─────────────────────────────────
            # Mention the original sender so BAND routes the reply correctly
            await tools.send_message(
                content=response_text,
                mentions=[msg.sender_id],
            )

            if specialist_added:
                logger.info(
                    "[BAND] Specialist collaboration initiated in room %s.", room_id
                )

        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("[BAND] Error processing message in room %s", room_id)
            # Attempt a graceful failure reply — if this also fails, the
            # platform's judged-turn mechanism will mark the turn as failed.
            try:
                await tools.send_message(
                    content=(
                        "⚠️ Flight Surgeon encountered an internal error while "
                        "processing your request. Please retry or check the "
                        "AstroPulse system logs."
                    ),
                    mentions=[msg.sender_id],
                )
            except Exception:
                logger.exception("[BAND] Failed to send error reply to room %s", room_id)

    # ------------------------------------------------------------------
    # Response construction
    # ------------------------------------------------------------------

    def _build_triage_response(
        self,
        sender_name: str,
        triage: Any,
        message_text: str,
    ) -> str:
        """Compose a structured aerospace medical response from the triage report."""
        v = triage.vital_status
        level = triage.triage_level

        # Triage level emoji
        level_emoji = {
            "NOMINAL": "🟢",
            "ADVISORY": "🔵",
            "WARNING": "🟡",
            "CRITICAL": "🔴",
        }.get(level, "⚪")

        lines = [
            f"**AstroPulse AI Flight Surgeon — Live Medical Report**",
            f"*Responding to {sender_name} • Astronaut: Benjamin*",
            "",
            f"{level_emoji} **Triage Level: {level}**",
            f"📋 **Primary Concern:** {triage.primary_concern}",
            "",
            "**Current Vitals:**",
            f"  • Heart Rate: {v.heart_rate_bpm} BPM",
            f"  • SpO₂: {v.spo2_percent:.1f}%",
            f"  • Blood Pressure: {v.blood_pressure} mmHg",
            f"  • Radiation: {v.radiation_msv:.3f} mSv",
            f"  • Telemetry Trend: {triage.telemetry_trend}",
            "",
            f"**Clinical Assessment:** {triage.clinical_assessment}",
        ]

        # Add warnings
        if triage.triage_warnings:
            lines.append("")
            lines.append("**Active Warnings:**")
            for w in triage.triage_warnings:
                lines.append(f"  ⚠️ [{w.severity}] {w.parameter.upper()}: {w.message}")

        # Add immediate actions
        if triage.immediate_actions:
            lines.append("")
            lines.append("**Immediate Directives:**")
            for i, action in enumerate(triage.immediate_actions, 1):
                lines.append(f"  {i}. {action}")

        # Secondary concerns
        if triage.secondary_concerns:
            lines.append("")
            lines.append(f"**Secondary Concerns:** {', '.join(triage.secondary_concerns)}")

        lines.append("")
        lines.append(f"*Engine: {triage.source_model or 'aerospace-clinical-expert-v1'} • {datetime.utcnow().strftime('%H:%M:%S UTC')}*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Multi-agent specialist collaboration
    # ------------------------------------------------------------------

    async def _attempt_specialist_collaboration(
        self,
        tools: AgentToolsProtocol,
        room_id: str,
        triage: Any,
        sender_name: str,
    ) -> bool:
        """
        Discover and add a relevant specialist agent to the current room.

        Returns True if a specialist was successfully added.
        """
        try:
            logger.info("[BAND] Attempting specialist peer lookup for room %s", room_id)
            peers_response = await tools.lookup_peers(page=1, page_size=20)

            # peers_response.data is a list of Peer objects
            peers = getattr(peers_response, "data", [])
            if not peers:
                logger.info("[BAND] No peers available for collaboration.")
                return False

            # Filter out the Flight Surgeon itself
            my_id = tools.agent_id
            available = [p for p in peers if getattr(p, "id", None) != my_id]

            if not available:
                logger.info("[BAND] No external peers available.")
                return False

            # Score peers for relevance to the current medical situation
            specialist = _select_specialist(available, triage)
            if specialist is None:
                specialist = available[0]  # fallback: take first available

            specialist_id = getattr(specialist, "id", None)
            specialist_name = getattr(specialist, "name", "Specialist Agent")
            specialist_handle = getattr(specialist, "handle", specialist_id)

            if not specialist_id:
                return False

            logger.info(
                "[BAND] Adding specialist '%s' (%s) to room %s",
                specialist_name, specialist_id, room_id,
            )

            add_result = await tools.add_participant(specialist_id, role="member")
            add_status = getattr(add_result, "status", None)
            logger.info("[BAND] add_participant result: %s", add_status)

            # Build a concise task briefing for the specialist
            task_msg = self._build_specialist_handoff(
                specialist_name=specialist_name,
                triage=triage,
                sender_name=sender_name,
            )

            # Mention the specialist so they receive the message
            await tools.send_message(
                content=task_msg,
                mentions=[specialist_id],
            )

            # Record collaboration event for dashboard/status
            self.collaboration_events.append({
                "timestamp": datetime.utcnow().isoformat(),
                "room_id": room_id,
                "specialist_name": specialist_name,
                "specialist_id": specialist_id,
                "triage_level": triage.triage_level,
                "primary_concern": triage.primary_concern,
            })
            # Keep only last 20 events
            if len(self.collaboration_events) > 20:
                self.collaboration_events = self.collaboration_events[-20:]

            return True

        except Exception as exc:
            logger.warning(
                "[BAND] Specialist collaboration failed: %s", exc, exc_info=True
            )
            return False

    def _build_specialist_handoff(
        self,
        specialist_name: str,
        triage: Any,
        sender_name: str,
    ) -> str:
        """Build a concise task briefing for the specialist agent."""
        v = triage.vital_status
        actions_str = "; ".join(triage.immediate_actions[:3]) or "Monitor closely"

        return (
            f"**🤝 Multi-Agent Consultation Request from AstroPulse Flight Surgeon**\n\n"
            f"Specialist **{specialist_name}**, your expertise is requested regarding "
            f"Astronaut Benjamin's current medical situation.\n\n"
            f"**Triage Status:** {triage.triage_level}\n"
            f"**Primary Concern:** {triage.primary_concern}\n"
            f"**Vitals:** HR {v.heart_rate_bpm} BPM | SpO₂ {v.spo2_percent:.1f}% | "
            f"BP {v.blood_pressure} | Rad {v.radiation_msv:.3f} mSv\n"
            f"**Clinical Assessment:** {triage.clinical_assessment}\n"
            f"**Current Directives:** {actions_str}\n\n"
            f"Please provide your specialist assessment. The requesting controller "
            f"is {sender_name}. Flight Surgeon will consolidate your response."
        )

    # ------------------------------------------------------------------
    # Status reporting (for FastAPI status endpoint)
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Returns current adapter activity state for the status endpoint."""
        return {
            "messages_processed": self.messages_processed,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "collaboration_events": self.collaboration_events[-5:],  # last 5
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# Peer selection heuristics
# ---------------------------------------------------------------------------

_MEDICAL_KEYWORDS = {
    "medical", "health", "physician", "doctor", "nurse", "clinical",
    "surgeon", "cardiology", "pulmonology", "emergency", "trauma",
    "radiation", "oncology", "aerospace", "flight"
}

_ENGINEERING_KEYWORDS = {
    "engineer", "systems", "life support", "pressure", "suit", "EVA",
    "oxygen", "hardware", "maintenance", "diagnostics"
}


def _score_peer(peer: Any, triage: Any) -> int:
    """Score a peer's relevance to the current triage situation (higher = better)."""
    name = (getattr(peer, "name", "") or "").lower()
    description = (getattr(peer, "description", "") or "").lower()
    text = f"{name} {description}"

    score = 0
    level = triage.triage_level

    # Medical/clinical agents are always relevant
    for kw in _MEDICAL_KEYWORDS:
        if kw in text:
            score += 3

    # Engineering agents especially useful for life-support events
    if level in ("CRITICAL", "WARNING"):
        for kw in _ENGINEERING_KEYWORDS:
            if kw in text:
                score += 2

    return score


def _select_specialist(peers: list[Any], triage: Any) -> Optional[Any]:
    """Pick the most relevant peer, or None if none are clearly suitable."""
    if not peers:
        return None
    scored = [(p, _score_peer(p, triage)) for p in peers]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_peer, best_score = scored[0]
    # Only select if there's at least a marginal match
    return best_peer if best_score > 0 else None


# ---------------------------------------------------------------------------
# BandAgentService — FastAPI lifecycle wrapper
# ---------------------------------------------------------------------------

class BandAgentService:
    """
    Wraps the BAND Agent for FastAPI startup/shutdown lifecycle.

    Usage in main.py:
        band_service = BandAgentService(flight_surgeon, telemetry_stream)
        # In lifespan startup:  await band_service.start()
        # In lifespan shutdown: await band_service.stop()
    """

    def __init__(
        self,
        flight_surgeon: Any,
        telemetry_stream: Any,
        agent_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id or os.getenv("BAND_AGENT_ID", "")
        self.api_key = api_key or os.getenv("BAND_AGENT_API_KEY", "")
        self.enabled = os.getenv("BAND_ENABLED", "false").lower() == "true"
        self.flight_surgeon = flight_surgeon
        self.telemetry_stream = telemetry_stream

        self._agent: Optional[Agent] = None
        self._adapter: Optional[FlightSurgeonBandAdapter] = None
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._connected = False
        self._last_error: Optional[str] = None
        self._started_at: Optional[datetime] = None

    def is_configured(self) -> bool:
        """True only when both agent_id and api_key are non-empty."""
        return bool(
            self.agent_id and len(self.agent_id.strip()) > 5
            and self.api_key and len(self.api_key.strip()) > 5
        )

    def is_connected(self) -> bool:
        return self._connected

    async def start(self) -> None:
        """
        Start the BAND agent in a background asyncio Task.

        Uses Agent.create() + agent.start() so Uvicorn retains its own event
        loop and shutdown signals.  If credentials are missing or BAND_ENABLED
        is false, this is a silent no-op — the rest of AstroPulse keeps working.
        """
        if not self.enabled:
            logger.info("[BAND] Integration disabled (BAND_ENABLED != true). Skipping.")
            return

        if not self.is_configured():
            logger.warning(
                "[BAND] Credentials missing (BAND_AGENT_ID / BAND_AGENT_API_KEY). "
                "BAND integration inactive."
            )
            return

        try:
            self._adapter = FlightSurgeonBandAdapter(
                flight_surgeon=self.flight_surgeon,
                telemetry_stream=self.telemetry_stream,
            )

            self._agent = Agent.create(
                adapter=self._adapter,
                agent_id=self.agent_id,
                api_key=self.api_key,
            )

            # Start in background so Uvicorn loop is not blocked
            self._task = asyncio.create_task(
                self._run_agent_loop(), name="band-agent"
            )
            logger.info("[BAND] Agent task started (agent_id=%s).", self.agent_id)

        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("[BAND] Failed to start agent: %s", exc)

    async def _run_agent_loop(self) -> None:
        """Inner loop: connects to BAND with automatic reconnection on transient errors."""
        backoff = 5.0
        max_backoff = 120.0

        while True:
            try:
                if self._agent is None:
                    break
                logger.info("[BAND] Connecting to BAND platform...")
                self._connected = True
                self._started_at = datetime.utcnow()
                if self._adapter:
                    self._adapter.connected_at = datetime.utcnow()

                await self._agent.run()

                # If run() returns normally (clean shutdown), break the loop
                self._connected = False
                logger.info("[BAND] Agent run() returned cleanly.")
                break

            except asyncio.CancelledError:
                self._connected = False
                logger.info("[BAND] Agent task cancelled.")
                break

            except Exception as exc:
                self._connected = False
                self._last_error = str(exc)
                logger.warning(
                    "[BAND] Connection error: %s. Reconnecting in %.0fs...",
                    exc, backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2, max_backoff)

    async def stop(self) -> None:
        """Gracefully stop the BAND agent and cancel the background task."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._agent:
            try:
                await self._agent.stop()
            except Exception as exc:
                logger.warning("[BAND] Error during agent.stop(): %s", exc)

        self._connected = False
        self._agent = None
        self._task = None
        logger.info("[BAND] Agent service stopped.")

    def get_status(self) -> dict[str, Any]:
        """Returns BAND connection status for the API status endpoint."""
        adapter_status = self._adapter.get_status() if self._adapter else {}
        return {
            "enabled": self.enabled,
            "configured": self.is_configured(),
            "connected": self._connected,
            "agent_id": self.agent_id if self.is_configured() else None,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "last_error": self._last_error,
            "adapter": adapter_status,
        }

