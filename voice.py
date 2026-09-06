"""
voice.py — speech-to-text for the app's microphone input.

One job: take the raw audio bytes st.audio_input hands back from the
browser and turn them into a text question, using OpenAI's transcription
endpoint. Kept separate from app.py so the UI stays thin, matching every
other concern in this project (ingestion, retrieval, tools) living in its
own file.
"""

import openai
from dotenv import load_dotenv

from config import PROJECT_ROOT, TRANSCRIBE_MODEL

# Self-sufficient even if this module is used standalone (agent.py also
# loads this same .env, but we shouldn't rely on import order for it).
load_dotenv(PROJECT_ROOT / ".env")


def transcribe_audio(audio_bytes: bytes, filename: str = "question.wav") -> str:
    """Transcribe a recorded clip to text. Returns "" on empty audio."""
    if not audio_bytes:
        return ""
    client = openai.OpenAI()
    response = client.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=(filename, audio_bytes),
    )
    return response.text.strip()
