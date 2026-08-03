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
        on_complete_callback: Optional[Callable[[MemoryDiff], None]] = None
    ):
        """
        Launch non-blocking background worker task to extract memory diff and update YAML files.
        """
        turn_text = f"USER INPUT:\n{user_message}\n\nAGENT RESPONSE:\n{agent_response}"
        self.executor.submit(self._run_extraction_pipeline, turn_text, system_prompt, on_complete_callback)

    def extract_sync(self, raw_text: str, system_prompt: str) -> MemoryDiff:
        """
        Synchronous extraction pipeline used for /dump stream of consciousness processing.
        """
        import datetime
        from src.db import sync_tier3_to_postgres
        current_date = datetime.date.today().isoformat()
        diff = self.llm_client.extract_memory_diff(system_prompt, raw_text, current_date=current_date)
        if diff:
            self.memory_engine.apply_diff(diff, current_date=current_date)
            sync_tier3_to_postgres()
        return diff

    def _run_extraction_pipeline(
        self, 
        text_to_analyze: str, 
        system_prompt: str,
        on_complete_callback: Optional[Callable[[MemoryDiff], None]] = None
    ):
        """
        Worker logic executed in background thread.
        """
        import datetime
        from src.db import sync_tier3_to_postgres
        current_date = datetime.date.today().isoformat()
        try:
            diff = self.llm_client.extract_memory_diff(system_prompt, text_to_analyze, current_date=current_date)
            if diff and (diff.tier1_updates or diff.tier2_updates or diff.tier3_updates or diff.deletions):
                self.memory_engine.apply_diff(diff, current_date=current_date)
                sync_tier3_to_postgres()
                logger.info(f"Background Extractor updated memory: T1={len(diff.tier1_updates)}, T2={len(diff.tier2_updates)}, T3_entities={list(diff.tier3_updates.keys())}")
                if on_complete_callback:
                    on_complete_callback(diff)
        except Exception as e:
            logger.error(f"Background extraction failed safely without corrupting memory: {e}")

    def shutdown(self):
        """Gracefully shutdown background thread pool."""
        self.executor.shutdown(wait=False)
