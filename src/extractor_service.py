"""
Asynchronous Background Extractor & Memory Compactor Service.
Processes conversation turns out-of-band to update 3-Tier memory without delaying CLI responses.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from src.models import MemoryDiff
from src.llm_client import LLMClient
from src.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

class ExtractorService:
    def __init__(self, memory_engine: MemoryEngine, llm_client: LLMClient):
        self.memory_engine = memory_engine
        self.llm_client = llm_client
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory_extractor")

    def trigger_async_extraction(
        self, 
        user_message: str, 
        agent_response: str, 
        system_prompt: str,
        on_complete_callback: Optional[Callable[[MemoryDiff], None]] = None,
        conversation_history: Optional[list] = None
    ):
        """
        Launch non-blocking background worker task to extract memory diff and update YAML files.
        """
        turn_text = f"USER INPUT:\n{user_message}\n\nAGENT RESPONSE:\n{agent_response}"
        self.executor.submit(self._run_extraction_pipeline, turn_text, system_prompt, on_complete_callback, 0, conversation_history)

    def extract_sync(self, raw_text: str, system_prompt: str, user_id: int = 0, conversation_history: Optional[list] = None) -> MemoryDiff:
        """
        Synchronous extraction pipeline used for /dump stream of consciousness processing.
        """
        import os
        import datetime
        from src.db import sync_tier3_to_postgres, save_scheduled_ping
        current_date = datetime.date.today().isoformat()
        target_uid = user_id or int(os.getenv("ALLOWED_TELEGRAM_ID", "0") or "0")
        diff = self.llm_client.extract_memory_diff(
            system_prompt, raw_text, current_date=current_date, conversation_history=conversation_history
        )
        if diff:
            self.memory_engine.apply_diff(diff, current_date=current_date)
            sync_tier3_to_postgres()
            if diff.scheduled_pings:
                for ping in diff.scheduled_pings:
                    save_scheduled_ping(target_uid, ping.scheduled_at, ping.event_type, ping.context_text)
                    logger.info(f"Saved extracted event ping: {ping.context_text} at {ping.scheduled_at}")
        return diff

    def _run_extraction_pipeline(
        self, 
        text_to_analyze: str, 
        system_prompt: str,
        on_complete_callback: Optional[Callable[[MemoryDiff], None]] = None,
        user_id: int = 0,
        conversation_history: Optional[list] = None
    ):
        """
        Worker logic executed in background thread.
        """
        import os
        import datetime
        from src.db import sync_tier3_to_postgres, save_scheduled_ping
        current_date = datetime.date.today().isoformat()
        target_uid = user_id or int(os.getenv("ALLOWED_TELEGRAM_ID", "0") or "0")
        try:
            diff = self.llm_client.extract_memory_diff(
                system_prompt, text_to_analyze, current_date=current_date, conversation_history=conversation_history
            )
            if diff and (diff.tier1_updates or diff.tier2_updates or diff.tier3_updates or diff.deletions or diff.scheduled_pings):
                self.memory_engine.apply_diff(diff, current_date=current_date)
                sync_tier3_to_postgres()
                if diff.scheduled_pings:
                    for ping in diff.scheduled_pings:
                        save_scheduled_ping(target_uid, ping.scheduled_at, ping.event_type, ping.context_text)
                        logger.info(f"Saved extracted event ping: {ping.context_text} at {ping.scheduled_at}")
                logger.info(f"Background Extractor updated memory: T1={len(diff.tier1_updates)}, T2={len(diff.tier2_updates)}, T3_entities={list(diff.tier3_updates.keys())}")
                if on_complete_callback:
                    on_complete_callback(diff)
        except Exception as e:
            logger.error(f"Background extraction failed safely without corrupting memory: {e}")

    def shutdown(self):
        """Gracefully shutdown background thread pool."""
        self.executor.shutdown(wait=False)
