import asyncio
import os
import json
import logging
from typing import Annotated
from dotenv import load_dotenv

from livekit import agents, rtc, api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    function_tool,
)
from livekit.plugins import groq, sarvam, silero

# Load env variables
load_dotenv()

# Configure logging
logger = logging.getLogger("doctor-feedback-agent")
logger.setLevel(logging.INFO)

class DoctorFeedbackAgent(Agent):
    """Doctor Feedback Collection Agent"""
    
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly medical assistant bot for Dr. Gupta's clinic. "
                "Your task is to call a patient after their visit to collect feedback. "
                "You must ask these 3 questions one by one. Do not ask them all at once: "
                "1. On a scale of 1-5, how would you rate Dr. Gupta's communication? "
                "2. Was your wait time too long, reasonable, or short? "
                "3. Do you have any final comments or suggestions? "
                "Wait for the user to answer each question before moving to the next. "
                "Once you have all 3 answers, you MUST call the 'save_feedback' tool to save the data. "
                "After saving, thank the patient and say goodbye."
            )
        )
        self.current_room_name = None

    def set_room_name(self, room_name: str):
        self.current_room_name = room_name

    @function_tool 
    async def save_feedback(
        self, 
        rating: Annotated[int, "Rating for communication (1-5)"],
        wait_time: Annotated[str, "The patient's perception of wait time"],
        comments: Annotated[str, "Final comments from the patient"]
    ) -> str:
        """Saves the collected patient feedback to the dashboard."""
        
        feedback_data = {
            "doctor": "Dr. Gupta",
            "rating": rating,
            "wait_time": wait_time,
            "comments": comments
        }
        
        print(f"✅ DASHBOARD UPDATE: {json.dumps(feedback_data, indent=2)}")
        
        try:
            with open("feedback_dashboard.json", "a") as f:
                f.write(json.dumps(feedback_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to file: {e}")

        return "Feedback saved successfully. You may now end the call."


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")

    agent = DoctorFeedbackAgent()
    agent.set_room_name(ctx.room.name)

    session = AgentSession(
        vad=silero.VAD.load(),
        
        stt=sarvam.STT(language="en-IN"),
        
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        
        # --- FIX: Added target_language_code ---
        tts=sarvam.TTS(
            speaker="anushka", 
            model="bulbul:v2",
            target_language_code="en-IN"  # <--- REQUIRED ARGUMENT
        ),
    )

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        print(f"Participant {participant.identity} disconnected.")

    await session.start(agent=agent, room=ctx.room)

    await session.generate_reply(
        instructions="Hi, am I speaking with the patient? I'm calling from Dr. Gupta's office for a quick feedback."
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))