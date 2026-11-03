"""
BAND AI Integration Service for AstroPulse AI Flight Surgeon
=============================================================
Provides high-level coordination and status inspection for the BAND multi-agent
infrastructure (https://band.ai).

Separated cleanly from bland_service.py (which handles telephone voice calls).
BAND handles agent-to-agent collaboration, peer discovery, and room chat routing.
"""

from typing import Any, Dict, Optional
from band_agent import BandAgentService, FlightSurgeonBandAdapter

__all__ = ["BandAgentService", "FlightSurgeonBandAdapter", "BandServiceManager"]


class BandServiceManager:
    """
    Coordinator for BAND multi-agent operations.
    Exposes helper utilities and bridges the Flight Surgeon engine to BAND.
    """

    def __init__(self, service: BandAgentService):
        self.service = service

    @property
    def is_operational(self) -> bool:
        return self.service.is_configured() and self.service.is_connected()

    def get_summary(self) -> Dict[str, Any]:
        """Provides a sanitized summary for API and dashboard presentation."""
        status = self.service.get_status()
        return {
            "platform": "BAND Multi-Agent Coordination Infrastructure",
            "url": "https://band.ai",
            "enabled": status.get("enabled", False),
            "configured": status.get("configured", False),
            "connected": status.get("connected", False),
            "agent_id": status.get("agent_id"),
            "messages_processed": status.get("adapter", {}).get("messages_processed", 0),
            "recent_collaborations": status.get("adapter", {}).get("collaboration_events", []),
            "last_error": status.get("last_error"),
        }

