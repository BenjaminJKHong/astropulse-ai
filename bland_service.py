"""
Bland AI Voice & Conversational Interaction Service for AstroPulse AI Flight Surgeon.
Enables real-time voice call dispatching via Bland AI (https://bland.ai)
and interactive astronaut-surgeon conversational dialogue.
"""

import os
import time
import uuid
from typing import Optional, Dict, Any
import httpx
from dotenv import load_dotenv

from schemas import TriageReport, BlandCallResponse, SurgeonInteractionResponse
from prompts import FLIGHT_SURGEON_SYSTEM_PROMPT

load_dotenv()

class BlandAIInteractionService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BLAND_API_KEY")
        self.base_url = "https://api.bland.ai/v1"
        self.default_voice = os.getenv("BLAND_VOICE", "maya")
        self.default_phone = os.getenv("BLAND_PHONE_NUMBER", "+15550192834")

    def is_configured(self) -> bool:
        """Checks if live Bland AI API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate_call_task(self, triage: TriageReport, task_override: Optional[str] = None) -> str:
        """Constructs an urgent aerospace voice agent task for Bland AI."""
        if task_override:
            return task_override

        v = triage.vital_status
        actions_str = "; ".join(triage.immediate_actions) if triage.immediate_actions else "Maintain monitoring"
        
        task = (
            f"You are the NASA AstroPulse AI Flight Surgeon conducting an urgent emergency voice call with Astronaut Benjamin and Mission Control. "
            f"Current mission triage status is {triage.triage_level}. "
            f"Live vitals: Heart Rate is {v.heart_rate_bpm} BPM, Oxygen Saturation SpO2 is {v.spo2_percent}%, Blood Pressure is {v.blood_pressure}, Radiation is {v.radiation_msv} mSv. "
            f"Primary clinical diagnosis: {triage.primary_concern}. "
            f"Immediately instruct Astronaut Benjamin to execute the following directives: {actions_str}. "
            f"Speak with steady, calm, authoritative aerospace medical urgency. "
            f"Ask Benjamin: 'Benjamin, this is Flight Surgeon. Can you hear me? Acknowledge your current consciousness and verify your suit pressure gauge now.'"
        )
        return task

    async def initiate_call(
        self,
        phone_number: Optional[str],
        triage: TriageReport,
        task_override: Optional[str] = None
    ) -> BlandCallResponse:
        """
        Dispatches an outbound phone call via Bland AI API.
        If BLAND_API_KEY is not set or network is unreachable, returns a simulated dispatch response.
        """
        target_phone = phone_number or self.default_phone
        task_prompt = self.generate_call_task(triage, task_override)
        now = int(time.time())

        # If live Bland AI API key is configured, make actual call
        if self.is_configured():
            headers = {
                "authorization": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "phone_number": target_phone,
                "task": task_prompt,
                "model": "enhanced",
                "voice": self.default_voice,
                "reduce_latency": True,
                "wait_for_greeting": False,
                "temperature": 0.3,
                "max_duration": 5
            }
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(f"{self.base_url}/calls", headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    call_id = data.get("call_id", str(uuid.uuid4()))
                    return BlandCallResponse(
                        call_id=call_id,
                        status="queued",
                        phone_number=target_phone,
                        task=task_prompt,
                        simulated=False,
                        voice=self.default_voice,
                        created_at=now,
                        message=f"Outbound Bland AI call dispatched successfully to {target_phone}."
                    )
            except Exception as e:
                print(f"[Bland AI] Live call failed: {e}. Falling back to simulated dispatch.")

        # Simulated response (offline/test mode)
        sim_call_id = f"bland_sim_{uuid.uuid4().hex[:10]}"
        return BlandCallResponse(
            call_id=sim_call_id,
            status="simulated_dispatch",
            phone_number=target_phone,
            task=task_prompt,
            simulated=True,
            voice=self.default_voice,
            created_at=now,
            message=f"Bland AI Emergency Voice Agent dispatched (Simulated Mode). Voice: '{self.default_voice}'."
        )

    async def interact(
        self,
        message: str,
        triage: TriageReport,
        flight_surgeon: Any
    ) -> SurgeonInteractionResponse:
        """
        Handles interactive conversational dialogue between the astronaut/user and the AI Flight Surgeon.
        """
        v = triage.vital_status
        cleaned = message.strip()
        lower_msg = cleaned.lower()

        # If LLM API is available on flight surgeon, query it with conversational medical context
        if flight_surgeon and flight_surgeon.api_key and flight_surgeon.provider in ("groq", "nebius", "openai"):
            try:
                system_instruction = (
                    f"{FLIGHT_SURGEON_SYSTEM_PROMPT}\n\n"
                    f"CONVERSATIONAL INTERACTION MODE:\n"
                    f"You are speaking directly to Astronaut Benjamin over the biotelemetry voice loop.\n"
                    f"Current Vitals: HR={v.heart_rate_bpm} BPM, SpO2={v.spo2_percent}%, BP={v.blood_pressure}, Rad={v.radiation_msv} mSv.\n"
                    f"Triage Level: {triage.triage_level}. Primary Concern: {triage.primary_concern}.\n"
                    f"Respond in 1 to 3 direct, calm, professional aerospace medical sentences."
                )
                
                url = f"{flight_surgeon.base_url.rstrip('/')}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {flight_surgeon.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": flight_surgeon.model,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": cleaned}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 150
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    reply = data["choices"][0]["message"]["content"].strip()
                    return SurgeonInteractionResponse(
                        reply=reply,
                        triage_level=triage.triage_level,
                        current_spo2=v.spo2_percent,
                        current_hr=v.heart_rate_bpm,
                        clinical_concern=triage.primary_concern,
                        recommended_actions=triage.immediate_actions
                    )
            except Exception as e:
                print(f"[AI Flight Surgeon] Conversational LLM call failed: {e}. Using deterministic response.")

        # Deterministic Clinical Response Engine
        if "dizzy" in lower_msg or "breath" in lower_msg or "hypoxia" in lower_msg or "air" in lower_msg or "faint" in lower_msg:
            if v.spo2_percent < 92.0:
                reply = (
                    f"Benjamin, sit down immediately. Your SpO2 has dropped to {v.spo2_percent}% and heart rate is {v.heart_rate_bpm} BPM. "
                    f"You are entering acute hypoxia. Switch your suit life-support to emergency 100% O2 now and do not unseal your visor."
                )
            else:
                reply = (
                    f"Benjamin, your current SpO2 is nominal at {v.spo2_percent}%, but heart rate is {v.heart_rate_bpm} BPM. "
                    f"Take slow, measured diaphragmatic breaths. I am actively monitoring your partial O2 pressure."
                )
        elif "status" in lower_msg or "vitals" in lower_msg or "report" in lower_msg or "how am i" in lower_msg:
            reply = (
                f"Flight Surgeon status report for Astronaut Benjamin: Triage level is {triage.triage_level}. "
                f"Heart rate: {v.heart_rate_bpm} BPM, SpO2: {v.spo2_percent}%, Blood Pressure: {v.blood_pressure}. "
                f"Assessment: {triage.clinical_assessment}"
            )
        elif "radiation" in lower_msg or "storm" in lower_msg:
            if v.radiation_msv > 1.0:
                reply = (
                    f"Warning, Benjamin. Telemetry registers radiation at {v.radiation_msv} mSv. "
                    f"Abort all exterior activities and move into the AstroPulse storm shelter immediately."
                )
            else:
                reply = f"Cosmic radiation levels are currently normal at {v.radiation_msv} mSv. Shield integrity is nominal."
        else:
            reply = (
                f"Flight Surgeon reading you loud and clear, Benjamin. Your biometrics show HR {v.heart_rate_bpm} BPM and SpO2 {v.spo2_percent}%. "
                f"Triage status is {triage.triage_level}. Maintain mission protocol and report any sensory changes."
            )

        return SurgeonInteractionResponse(
            reply=reply,
            triage_level=triage.triage_level,
            current_spo2=v.spo2_percent,
            current_hr=v.heart_rate_bpm,
            clinical_concern=triage.primary_concern,
            recommended_actions=triage.immediate_actions
        )

