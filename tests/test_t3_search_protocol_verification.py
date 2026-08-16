"""
Synthetic Test Suite for Mandatory T3 LLM Search Protocol & Broad DB Search Fallback.
STRICTLY USES MOCK SYNTHETIC DATA.
"""
import sys
import os
import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.db import search_tier3_memory

def test_mandatory_t3_search_protocol_directive():
    print("Testing Abstract Mandatory T3 Search Protocol Directive in MemoryEngine...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Search protocol test")

    assert "MANDATORY TIER 3 MEMORY SEARCH PROTOCOL" in prompt, "Protocol directive missing from system prompt"
    assert "Calling search_memory(query) is MANDATORY whenever" in prompt
    assert "personality analysis" in prompt or "holistic synthesis" in prompt
    print("[OK] MANDATORY TIER 3 MEMORY SEARCH PROTOCOL verified in System Prompt.")

def test_search_tier3_memory_fallback():
    print("\nTesting search_tier3_memory Token-Based & Top-K Fallback...")
    # Issue a broad search query
    res = search_tier3_memory("проекты и цели пользователя")
    assert res is not None, "Search result should not be None"
    assert "Tier 3 Memory" in res or "No long-term memory entries found" in res
    print("[OK] search_tier3_memory broad query fallback verified cleanly.")

def test_json_tool_leak_sanitizer():
    print("\nTesting Raw JSON Tool Leak Sanitizer in LLMClient...")
    client = LLMClient()

    raw_leaked_output = (
        '{"query": "пользователь биография личность проекты"}\n'
        'Понял. Подведем итоги по данным памяти T3.'
    )
    cleaned = client._sanitize_tool_leak(raw_leaked_output)

    assert '{"query"' not in cleaned, "Raw JSON query object should be stripped"
    assert "Понял. Подведем итоги по данным памяти T3." in cleaned
    print("[OK] Raw JSON tool call leaks stripped successfully.")

if __name__ == "__main__":
    print("================ T3 SEARCH PROTOCOL & DB FALLBACK VERIFICATION ================")
    test_mandatory_t3_search_protocol_directive()
    test_search_tier3_memory_fallback()
    test_json_tool_leak_sanitizer()
    print("================ ALL PROTOCOL VERIFICATION TESTS PASSED! ================")
