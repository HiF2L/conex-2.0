"""
Voice note downloader and Whisper transcriber service for Telegram voice messages.
"""
import os
import tempfile
import logging
from typing import Optional
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

async def process_voice_note(bot, file_id: str, llm_client: LLMClient) -> str:
    """
    Downloads a Telegram voice note (.ogg), sends it to OpenAI Whisper API via LLMClient,
    and returns the transcribed string. Ensures temporary file cleanup.
    """
    temp_path = None
    try:
        # Create a temporary file with .ogg extension
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
            temp_path = tmp_file.name

        # Download voice file from Telegram
        file_info = await bot.get_file(file_id)
        await bot.download_file(file_info.file_path, destination=temp_path)

        # Transcribe via Whisper
        transcription = llm_client.transcribe_audio(temp_path)
        logger.info(f"Successfully transcribed voice note file_id={file_id}: {len(transcription)} chars")
        return transcription

    except Exception as e:
        logger.error(f"Error processing Telegram voice note: {e}")
        return f"[Voice Transcription Error: {str(e)}]"
    finally:
        # Guarantee temp file cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup temp audio file {temp_path}: {cleanup_err}")
