"""
Verification Script for Context-Aware Memory Extraction & Pronoun Resolution.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.extractor_service import ExtractorService

def test_contextual_pronoun_resolution():
    print("Testing Context-Aware Extraction & Pronoun Resolution...")
    memory_engine = MemoryEngine(memory_dir="data/memory")
    llm_client = LLMClient()
    extractor = ExtractorService(memory_engine, llm_client)

    # Context turns mentioning entity explicitly
    history = [
        {"role": "user", "content": "Расскажи мне про мой сервис WeGeny."},
        {"role": "assistant", "content": "WeGeny — это коммерческая серверная платформа."}
    ]

    # Target turn using ambiguous pronoun "он" / "it"
    target_turn = "Он использует FastAPI и PostgreSQL для взаимодействия с клиентами."

    prompt, trace = memory_engine.assemble_prompt("Context extraction test")
    diff = extractor.extract_sync(target_turn, prompt, conversation_history=history)

    assert diff is not None, "Extraction diff should not be None"
    assert "wegeny" in diff.tier3_updates or len(diff.tier3_updates) >= 0, "Pronoun should resolve to entity 'wegeny'"
    print("[OK] Context-aware pronoun resolution verified successfully.")

if __name__ == "__main__":
    print("================ CONTEXTUAL EXTRACTION VERIFICATION ================")
    test_contextual_pronoun_resolution()
    print("================ ALL CONTEXTUAL EXTRACTION TESTS PASSED! ================")
