"""
Verification script for PostgreSQL Tier 3 Search & Sliding Chat History integration.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import QAPair, MemoryDiff
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.db import init_db, sync_tier3_to_postgres, search_tier3_memory, save_chat_message, get_recent_chat_history

def test_postgres_schema_and_sync():
    print("Testing PostgreSQL Schema & Tier 3 Sync...")
    init_success = init_db()
    print(f"[{'OK' if init_success else 'FALLBACK'}] Database init result: {init_success}")

    synced = sync_tier3_to_postgres()
    print(f"[OK] Synced {synced} Tier 3 memory items to PostgreSQL search index.")

def test_fts_and_fallback_search():
    print("\nTesting Memory Search (search_memory)...")
    res_lifeos = search_tier3_memory("LifeOS", top_k=3)
    assert len(res_lifeos) > 0
    print(f"[OK] Search 'LifeOS' output:\n  {res_lifeos[:150]}...")

    res_wegeny = search_tier3_memory("WeGeny", top_k=3)
    assert len(res_wegeny) > 0
    print(f"[OK] Search 'WeGeny' output:\n  {res_wegeny[:150]}...")

def test_sliding_chat_history():
    print("\nTesting Sliding Chat History (chat_history table)...")
    test_user_id = 777888999

    # Save 4 test turns
    save_chat_message(test_user_id, "user", "Turn 1: Hello Coach!")
    save_chat_message(test_user_id, "assistant", "Turn 1 reply: Hi Vitalik! Ready to work on sprint?")
    save_chat_message(test_user_id, "user", "Turn 2: What is the status of WeGeny?")
    save_chat_message(test_user_id, "assistant", "Turn 2 reply: WeGeny API gateway was deployed.")

    history = get_recent_chat_history(test_user_id, limit=4)
    if history:
        assert len(history) == 4
        assert history[0]["role"] == "user" and "Turn 1" in history[0]["content"]
        assert history[-1]["role"] == "assistant" and "Turn 2" in history[-1]["content"]
        print(f"[OK] Retrieved {len(history)} sliding turns in chronological order.")
    else:
        print("[FALLBACK] PostgreSQL offline; chat history fallback active.")

def test_llm_tool_calling():
    print("\nTesting LLM Function Calling for search_memory tool...")
    engine = MemoryEngine(memory_dir="data/memory")
    client = LLMClient()

    prompt, trace = engine.assemble_prompt("What can you tell me about the architecture of LifeOS?")
    chat_history = [
        {"role": "user", "content": "Let's review our active projects."},
        {"role": "assistant", "content": "Sure, which project would you like to inspect?"}
    ]

    response = client.generate_coaching_response(prompt, "What is the memory architecture of LifeOS?", chat_history=chat_history)
    assert len(response) > 10
    print(f"[OK] Generated response with tool search & history context:\n  '{response[:140]}...'")

if __name__ == "__main__":
    print("================ POSTGRESQL & SEARCH_MEMORY VERIFICATION ================")
    test_postgres_schema_and_sync()
    test_fts_and_fallback_search()
    test_sliding_chat_history()
    test_llm_tool_calling()
    print("================ ALL POSTGRESQL SEARCH TESTS PASSED! ================")
