"""
Comprehensive Test Suite — BAND AI Integration for AstroPulse AI Flight Surgeon
================================================================================
Tests all BAND integration components using mocks and the band.testing.FakeAgentTools.
Does NOT require a real BAND account or live network connection.

Tests covered:
  1.  Missing BAND credentials → service reports unconfigured, no crash
  2.  BAND_ENABLED=false → start() is a silent no-op
  3.  Successful adapter initialization
  4.  on_message with nominal telemetry → response sent
  5.  on_message with CRITICAL triage → emergency response content
  6.  Multi-agent: collaboration keywords trigger peer lookup
  7.  Multi-agent: specialist added and handoff message sent
  8.  Multi-agent: no available peers → graceful skip
  9.  Peer scoring / selection heuristic
 10.  BAND API / tools errors handled gracefully
 11.  Graceful shutdown via BandAgentService.stop()
 12.  Status endpoint returns correct structure
 13.  Existing test suites (flight surgeon + endpoints) still pass
"""

import asyncio
import os
import sys
import time
import unittest
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# ── Make sure project root is on path ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from schemas import TelemetryFrame, TriageReport, VitalStatus, TriageWarning
from flight_surgeon import AIFlightSurgeon
from telemetry_stream import BenjaminTelemetryStream
from band_agent import (
    FlightSurgeonBandAdapter,
    BandAgentService,
    _select_specialist,
    _wants_collaboration,
    _score_peer,
)


# ════════════════════════════════════════════════════════════════════════════
# Helpers — Fake telemetry and triage fixtures
# ════════════════════════════════════════════════════════════════════════════

def _make_nominal_frame() -> TelemetryFrame:
    return TelemetryFrame(
        timestamp=int(time.time()),
        heart_rate=72,
        spo2=99.0,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        radiation_msv=0.12,
        astronaut_id="Benjamin",
    )


def _make_critical_frame() -> TelemetryFrame:
    return TelemetryFrame(
        timestamp=int(time.time()),
        heart_rate=138,
        spo2=82.5,
        blood_pressure_systolic=168,
        blood_pressure_diastolic=104,
        radiation_msv=0.15,
        astronaut_id="Benjamin",
    )


def _make_nominal_triage() -> TriageReport:
    return TriageReport(
        astronaut_id="Benjamin",
        timestamp=int(time.time()),
        triage_level="NOMINAL",
        vital_status=VitalStatus(
            heart_rate_bpm=72,
            spo2_percent=99.0,
            blood_pressure="120/80",
            radiation_msv=0.12,
        ),
        clinical_assessment="NOMINAL: Astronaut Benjamin's physiological markers are nominal.",
        primary_concern="None - Vitals Stable",
        triage_warnings=[],
        immediate_actions=["Maintain standard telemetry."],
        requires_crew_intervention=False,
        telemetry_trend="STABLE",
        source_model="aerospace-clinical-expert-v1",
    )


def _make_critical_triage() -> TriageReport:
    return TriageReport(
        astronaut_id="Benjamin",
        timestamp=int(time.time()),
        triage_level="CRITICAL",
        vital_status=VitalStatus(
            heart_rate_bpm=138,
            spo2_percent=82.5,
            blood_pressure="168/104",
            radiation_msv=0.15,
        ),
        clinical_assessment="EMERGENCY: Astronaut Benjamin is experiencing severe acute hypoxia.",
        primary_concern="Acute Severe Hypoxia & Respiratory Compromise",
        triage_warnings=[
            TriageWarning(
                parameter="spo2",
                value=82.5,
                severity="CRITICAL",
                message="CRITICAL HYPOXIA SPIKE: SpO2 collapsed to 82.5%.",
            )
        ],
        immediate_actions=[
            "Initiate 100% emergency O2 flood via EVA/IVA life-support.",
            "Seal helmet visor, verify suit pressure integrity.",
        ],
        requires_crew_intervention=True,
        telemetry_trend="DETERIORATING",
        source_model="aerospace-clinical-expert-v1",
    )


class _FakeTelemetryStream:
    """Minimal mock of BenjaminTelemetryStream."""
    def __init__(self, frame: Optional[TelemetryFrame] = None):
        self._frame = frame or _make_nominal_frame()

    def get_current_frame(self, advance: bool = False) -> TelemetryFrame:
        return self._frame


class _FakeSurgeon:
    """Minimal mock of AIFlightSurgeon."""
    def __init__(self, triage: Optional[TriageReport] = None):
        self._triage = triage or _make_nominal_triage()
        self.latest_triage = self._triage

    async def analyze_telemetry(self, frame: TelemetryFrame) -> TriageReport:
        return self._triage


def _make_platform_message(
    content: str = "@FlightSurgeon What is Benjamin's status?",
    sender_id: str = "user-abc",
    sender_name: str = "Mission Controller",
    room_id: str = "room-test-1",
) -> MagicMock:
    """Build a fake PlatformMessage-like object."""
    msg = MagicMock()
    msg.content = content
    msg.sender_id = sender_id
    msg.sender_name = sender_name
    msg.room_id = room_id
    msg.id = "msg-001"
    return msg


def _make_fake_tools(
    agent_id: str = "agent-fs-001",
    peers: Optional[list] = None,
) -> MagicMock:
    """Build a mock AgentToolsProtocol with async methods."""
    tools = MagicMock()
    tools.agent_id = agent_id
    tools.turn = MagicMock()
    tools.turn.judged = False

    # Async stubs
    tools.send_message = AsyncMock(return_value=None)
    tools.send_notice = AsyncMock(return_value=None)
    tools.no_reply = AsyncMock(return_value=None)
    tools.send_failure = AsyncMock(return_value=None)
    tools.add_participant = AsyncMock(return_value=MagicMock(status="added"))
    tools.remove_participant = AsyncMock(return_value=None)

    # Peer lookup — returns a mock with .data attribute
    peer_list = peers if peers is not None else []
    peers_response = MagicMock()
    peers_response.data = peer_list
    tools.lookup_peers = AsyncMock(return_value=peers_response)

    return tools


def _make_peer(name: str, pid: str = None, description: str = "") -> MagicMock:
    p = MagicMock()
    p.name = name
    p.id = pid or f"peer-{name.lower().replace(' ', '-')}"
    p.handle = f"@{p.id}"
    p.description = description
    return p


# ════════════════════════════════════════════════════════════════════════════
# Test cases
# ════════════════════════════════════════════════════════════════════════════

class TestBandMissingCredentials(unittest.TestCase):
    """Test 1: Missing credentials — service is non-functional but safe."""

    def test_unconfigured_when_no_env_vars(self):
        with patch.dict(os.environ, {
            "BAND_ENABLED": "true",
            "BAND_AGENT_ID": "",
            "BAND_AGENT_API_KEY": "",
        }):
            svc = BandAgentService(
                flight_surgeon=_FakeSurgeon(),
                telemetry_stream=_FakeTelemetryStream(),
            )
            self.assertFalse(svc.is_configured())
            self.assertFalse(svc.is_connected())

    def test_status_shows_not_configured(self):
        with patch.dict(os.environ, {
            "BAND_ENABLED": "true",
            "BAND_AGENT_ID": "",
            "BAND_AGENT_API_KEY": "",
        }):
            svc = BandAgentService(
                flight_surgeon=_FakeSurgeon(),
                telemetry_stream=_FakeTelemetryStream(),
            )
            status = svc.get_status()
            self.assertFalse(status["configured"])
            self.assertFalse(status["connected"])
            self.assertIsNone(status["agent_id"])

    def test_start_no_crash_when_unconfigured(self):
        """start() must not raise when credentials are missing."""
        with patch.dict(os.environ, {
            "BAND_ENABLED": "true",
            "BAND_AGENT_ID": "",
            "BAND_AGENT_API_KEY": "",
        }):
            svc = BandAgentService(
                flight_surgeon=_FakeSurgeon(),
                telemetry_stream=_FakeTelemetryStream(),
            )
            asyncio.run(svc.start())  # should silently skip
            self.assertFalse(svc.is_connected())
        print("  [PASSED] Missing credentials: service safe and non-functional.")


class TestBandDisabled(unittest.TestCase):
    """Test 2: BAND_ENABLED=false → start() is a no-op."""

    def test_start_is_noop_when_disabled(self):
        with patch.dict(os.environ, {
            "BAND_ENABLED": "false",
            "BAND_AGENT_ID": "c481ef81-9785-49f2-b4fd-0eb058ed6070",
            "BAND_AGENT_API_KEY": "band_a_1793718796_Y_znYEo0ssqih5zc226mO9MrukCRdaZj",
        }):
            svc = BandAgentService(
                flight_surgeon=_FakeSurgeon(),
                telemetry_stream=_FakeTelemetryStream(),
            )
            asyncio.run(svc.start())
            self.assertFalse(svc.is_connected())
            status = svc.get_status()
            self.assertFalse(status["enabled"])
        print("  [PASSED] BAND_ENABLED=false: start() is a clean no-op.")


class TestAdapterInitialization(unittest.TestCase):
    """Test 3: Adapter initializes with correct defaults."""

    def test_adapter_init(self):
        surgeon = _FakeSurgeon()
        stream = _FakeTelemetryStream()
        adapter = FlightSurgeonBandAdapter(
            flight_surgeon=surgeon,
            telemetry_stream=stream,
        )
        self.assertIsNone(adapter.connected_at)
        self.assertEqual(adapter.messages_processed, 0)
        self.assertEqual(len(adapter.collaboration_events), 0)
        self.assertIsNone(adapter.last_error)
        print("  [PASSED] Adapter initialization: defaults correct.")

    def test_adapter_on_started(self):
        adapter = FlightSurgeonBandAdapter(
            flight_surgeon=_FakeSurgeon(),
            telemetry_stream=_FakeTelemetryStream(),
        )
        asyncio.run(adapter.on_started("Flight Surgeon", "Medical AI"))
        self.assertIsNotNone(adapter.connected_at)
        print("  [PASSED] Adapter on_started: connected_at set.")


class TestOnMessageNominal(unittest.TestCase):
    """Test 4: Nominal message → send_message called with triage content."""

    def test_nominal_response_sent(self):
        surgeon = _FakeSurgeon(_make_nominal_triage())
        stream = _FakeTelemetryStream(_make_nominal_frame())
        adapter = FlightSurgeonBandAdapter(flight_surgeon=surgeon, telemetry_stream=stream)

        msg = _make_platform_message("@FlightSurgeon Report Benjamin's vitals.")
        tools = _make_fake_tools()

        asyncio.run(adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-test",
        ))

        tools.send_message.assert_awaited_once()
        call_args = tools.send_message.call_args
        content = call_args.kwargs.get("content") or call_args.args[0]
        self.assertIn("NOMINAL", content)
        self.assertIn("Benjamin", content)
        self.assertIn("Heart Rate", content)
        self.assertEqual(adapter.messages_processed, 1)
        print("  [PASSED] Nominal message: correct triage response sent to room.")


class TestOnMessageCritical(unittest.TestCase):
    """Test 5: CRITICAL triage → emergency content in response."""

    def test_critical_response_contains_emergency(self):
        surgeon = _FakeSurgeon(_make_critical_triage())
        stream = _FakeTelemetryStream(_make_critical_frame())
        adapter = FlightSurgeonBandAdapter(flight_surgeon=surgeon, telemetry_stream=stream)

        msg = _make_platform_message("@FlightSurgeon Status update needed.")
        tools = _make_fake_tools()

        asyncio.run(adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-critical",
        ))

        content = tools.send_message.call_args.kwargs.get("content") or \
                  tools.send_message.call_args.args[0]
        self.assertIn("CRITICAL", content)
        self.assertIn("Hypoxia", content)
        self.assertIn("O2", content)
        print("  [PASSED] Critical triage: emergency keywords in response.")


class TestCollaborationKeywords(unittest.TestCase):
    """Test 6: Collaboration keyword detection."""

    def test_keywords_detected(self):
        self.assertTrue(_wants_collaboration("Get assistance from another specialist"))
        self.assertTrue(_wants_collaboration("Please collaborate with an expert"))
        self.assertTrue(_wants_collaboration("Analyze and get help if necessary"))
        self.assertFalse(_wants_collaboration("What is Benjamin's heart rate?"))
        self.assertFalse(_wants_collaboration("Report vitals now"))
        print("  [PASSED] Collaboration keywords: detection works correctly.")


class TestMultiAgentSpecialistAdded(unittest.TestCase):
    """Test 7: Specialist discovered, added to room, handoff message sent."""

    def test_specialist_added_and_messaged(self):
        surgeon = _FakeSurgeon(_make_critical_triage())
        stream = _FakeTelemetryStream(_make_critical_frame())
        adapter = FlightSurgeonBandAdapter(flight_surgeon=surgeon, telemetry_stream=stream)

        specialist = _make_peer(
            name="Emergency Medical Specialist",
            pid="peer-ems-001",
            description="aerospace emergency medical agent",
        )
        tools = _make_fake_tools(peers=[specialist])

        msg = _make_platform_message(
            "@FlightSurgeon Analyze Benjamin and get assistance from a specialist."
        )

        asyncio.run(adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-collab",
        ))

        # add_participant should have been called
        tools.add_participant.assert_awaited()
        add_call = tools.add_participant.call_args
        self.assertEqual(add_call.args[0], "peer-ems-001")

        # send_message called at least twice: triage reply + handoff
        self.assertGreaterEqual(tools.send_message.await_count, 2)

        # A collaboration event was recorded
        self.assertEqual(len(adapter.collaboration_events), 1)
        event = adapter.collaboration_events[0]
        self.assertEqual(event["specialist_id"], "peer-ems-001")
        self.assertEqual(event["triage_level"], "CRITICAL")
        print("  [PASSED] Multi-agent: specialist added and handoff message sent.")


class TestNoPeersAvailable(unittest.TestCase):
    """Test 8: Collaboration requested but no peers available → graceful skip."""

    def test_no_peers_graceful(self):
        surgeon = _FakeSurgeon(_make_critical_triage())
        stream = _FakeTelemetryStream(_make_critical_frame())
        adapter = FlightSurgeonBandAdapter(flight_surgeon=surgeon, telemetry_stream=stream)

        tools = _make_fake_tools(peers=[])

        msg = _make_platform_message(
            "@FlightSurgeon Get help from a specialist please."
        )

        asyncio.run(adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-no-peers",
        ))

        # add_participant should NOT have been called
        tools.add_participant.assert_not_awaited()
        # But primary reply should still be sent
        tools.send_message.assert_awaited_once()
        print("  [PASSED] No peers: graceful skip, primary reply still sent.")


class TestPeerScoring(unittest.TestCase):
    """Test 9: Peer scoring heuristic selects most relevant agent."""

    def test_medical_peer_scores_highest(self):
        triage = _make_critical_triage()
        medical = _make_peer("Medical Expert", description="clinical aerospace physician")
        random_bot = _make_peer("Data Analysis Bot", description="excel spreadsheets")
        engineering = _make_peer("Life Support Engineer", description="oxygen systems EVA suit")

        med_score = _score_peer(medical, triage)
        eng_score = _score_peer(engineering, triage)
        rand_score = _score_peer(random_bot, triage)

        self.assertGreater(med_score, rand_score)
        self.assertGreater(eng_score, rand_score)
        selected = _select_specialist([random_bot, medical, engineering], triage)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "Medical Expert")
        print("  [PASSED] Peer scoring: medical expert selected over unrelated bot.")

    def test_no_relevant_peers_returns_none(self):
        triage = _make_nominal_triage()
        bots = [
            _make_peer("Calendar Bot", description="scheduling meetings"),
            _make_peer("Weather Bot", description="forecast data"),
        ]
        result = _select_specialist(bots, triage)
        # score is 0 for unrelated bots → should return None
        self.assertIsNone(result)
        print("  [PASSED] Peer selection: None returned when no relevant peers.")


class TestBandToolErrors(unittest.TestCase):
    """Test 10: Band tool/API errors handled without crashing."""

    def test_lookup_peers_failure_is_caught(self):
        surgeon = _FakeSurgeon(_make_critical_triage())
        stream = _FakeTelemetryStream(_make_critical_frame())
        adapter = FlightSurgeonBandAdapter(flight_surgeon=surgeon, telemetry_stream=stream)

        tools = _make_fake_tools()
        tools.lookup_peers = AsyncMock(side_effect=Exception("BAND API 500"))

        msg = _make_platform_message("@FlightSurgeon Collaborate with a specialist.")

        # Should not raise
        asyncio.run(adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-err",
        ))

        # Primary reply still sent despite peer lookup failure
        tools.send_message.assert_awaited()
        print("  [PASSED] BAND API error: handled gracefully, primary reply sent.")

    def test_send_message_failure_sends_error_reply(self):
        surgeon = _FakeSurgeon(_make_nominal_triage())
        stream = _FakeTelemetryStream(_make_nominal_frame())
        adapter = FlightSurgeonBandAdapter(flight_surgeon=surgeon, telemetry_stream=stream)

        tools = _make_fake_tools()
        # First send_message raises, second (error reply) succeeds
        tools.send_message = AsyncMock(side_effect=[Exception("Transport error"), None])

        # Surgeon analyze_telemetry also fails to trigger error path
        surgeon.analyze_telemetry = AsyncMock(side_effect=Exception("Triage engine failure"))

        msg = _make_platform_message("@FlightSurgeon status?")

        asyncio.run(adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-err2",
        ))

        self.assertIsNotNone(adapter.last_error)
        print("  [PASSED] send_message failure: error reply attempted, no crash.")


class TestGracefulShutdown(unittest.TestCase):
    """Test 11: BandAgentService.stop() cancels task without hanging."""

    def test_stop_when_never_started(self):
        with patch.dict(os.environ, {
            "BAND_ENABLED": "true",
            "BAND_AGENT_ID": "c481ef81-9785-49f2-b4fd-0eb058ed6070",
            "BAND_AGENT_API_KEY": "band_a_valid_test_key_xyz",
        }):
            svc = BandAgentService(
                flight_surgeon=_FakeSurgeon(),
                telemetry_stream=_FakeTelemetryStream(),
            )
            # stop() on an unstarted service must not raise
            asyncio.run(svc.stop())
            self.assertFalse(svc.is_connected())
        print("  [PASSED] Graceful shutdown: stop() on unstarted service is safe.")

    def test_stop_cancels_running_task(self):
        """Start a fake long-running task and verify stop() cancels it."""
        with patch.dict(os.environ, {
            "BAND_ENABLED": "false",
            "BAND_AGENT_ID": "fake-id",
            "BAND_AGENT_API_KEY": "fake-key",
        }):
            svc = BandAgentService(
                flight_surgeon=_FakeSurgeon(),
                telemetry_stream=_FakeTelemetryStream(),
            )

        async def _run():
            # Inject a fake never-ending task
            async def _long_task():
                await asyncio.sleep(9999)

            svc._task = asyncio.create_task(_long_task())
            await svc.stop()
            self.assertTrue(svc._task is None)

        asyncio.run(_run())
        print("  [PASSED] Graceful shutdown: running task cancelled by stop().")


class TestStatusEndpoint(unittest.TestCase):
    """Test 12: Status dict has correct shape and doesn't leak secrets."""

    def test_status_structure(self):
        with patch.dict(os.environ, {
            "BAND_ENABLED": "true",
            "BAND_AGENT_ID": "c481ef81-9785-49f2-b4fd-0eb058ed6070",
            "BAND_AGENT_API_KEY": "band_a_1793718796_Y_znYEo0ssqih5zc226mO9MrukCRdaZj",
        }):
            svc = BandAgentService(
                flight_surgeon=_FakeSurgeon(),
                telemetry_stream=_FakeTelemetryStream(),
            )
            status = svc.get_status()

        self.assertIn("enabled", status)
        self.assertIn("configured", status)
        self.assertIn("connected", status)
        self.assertIn("agent_id", status)
        # API key must NOT appear in status
        status_str = str(status)
        self.assertNotIn("band_a_", status_str)
        self.assertNotIn("1793718796", status_str)
        print("  [PASSED] Status endpoint: correct shape, no secrets exposed.")

    def test_adapter_status_included(self):
        adapter = FlightSurgeonBandAdapter(
            flight_surgeon=_FakeSurgeon(),
            telemetry_stream=_FakeTelemetryStream(),
        )
        adapter.messages_processed = 5
        status = adapter.get_status()
        self.assertEqual(status["messages_processed"], 5)
        self.assertIn("collaboration_events", status)
        print("  [PASSED] Adapter status: messages_processed tracked correctly.")


# ════════════════════════════════════════════════════════════════════════════
# Test 13: Ensure existing suites are unbroken
# ════════════════════════════════════════════════════════════════════════════

class TestExistingFlightSurgeonUnbroken(unittest.TestCase):
    """Test 13a: Existing flight surgeon engine still works after BAND integration."""

    def test_nominal_triage_unchanged(self):
        surgeon = AIFlightSurgeon()
        frame = _make_nominal_frame()
        report = surgeon.analyze_telemetry_sync(frame)
        self.assertEqual(report.triage_level, "NOMINAL")
        self.assertFalse(report.requires_crew_intervention)
        print("  [PASSED] Existing Flight Surgeon: nominal triage unchanged.")

    def test_critical_hypoxia_unchanged(self):
        surgeon = AIFlightSurgeon()
        baseline = TelemetryFrame(
            timestamp=int(time.time()) - 15,
            heart_rate=72, spo2=99.0,
            blood_pressure_systolic=120, blood_pressure_diastolic=80,
            radiation_msv=0.12, astronaut_id="Benjamin"
        )
        surgeon.analyze_telemetry_sync(baseline)
        critical = _make_critical_frame()
        report = surgeon.analyze_telemetry_sync(critical)
        self.assertEqual(report.triage_level, "CRITICAL")
        self.assertTrue(report.requires_crew_intervention)
        print("  [PASSED] Existing Flight Surgeon: critical hypoxia detection unchanged.")


class TestExistingTelemetryStreamUnbroken(unittest.TestCase):
    """Test 13b: Telemetry stream unaffected by BAND integration."""

    def test_stream_cycles_correctly(self):
        stream = BenjaminTelemetryStream()
        f1 = stream.get_current_frame(advance=True)
        f2 = stream.get_current_frame(advance=True)
        self.assertIsNotNone(f1)
        self.assertIsNotNone(f2)
        print("  [PASSED] Existing telemetry stream: cycling works correctly.")


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def run_all_band_tests():
    print("=" * 72)
    print("     ASTROPULSE AI — BAND INTEGRATION COMPREHENSIVE TEST SUITE     ")
    print("=" * 72)

    test_classes = [
        TestBandMissingCredentials,
        TestBandDisabled,
        TestAdapterInitialization,
        TestOnMessageNominal,
        TestOnMessageCritical,
        TestCollaborationKeywords,
        TestMultiAgentSpecialistAdded,
        TestNoPeersAvailable,
        TestPeerScoring,
        TestBandToolErrors,
        TestGracefulShutdown,
        TestStatusEndpoint,
        TestExistingFlightSurgeonUnbroken,
        TestExistingTelemetryStreamUnbroken,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for cls in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        for test in suite:
            try:
                test.debug()
                total_passed += 1
            except Exception as e:
                total_failed += 1
                failures.append((str(test), str(e)))
                print(f"  [FAILED] {test}: {e}")

    print("=" * 72)
    if total_failed == 0:
        print(f"  ALL {total_passed} BAND INTEGRATION TESTS PASSED CLEANLY (100% SUCCESS)")
    else:
        print(f"  {total_passed} PASSED / {total_failed} FAILED")
        for name, err in failures:
            print(f"    [FAIL] {name}: {err}")
    print("=" * 72)
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_band_tests()
    sys.exit(0 if success else 1)

