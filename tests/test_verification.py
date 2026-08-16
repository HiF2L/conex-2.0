"""
Verification script for LifeOS Personal Memory & Coaching Agent MVP.
Tests MemoryEngine, ExtractorService, LLMClient, and Pydantic schemas.
"""
import os
import sys
import datetime
from pathlib import Path

# Force UTF-8 stdout encoding if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import QAPair, MemoryDiff, MemoryTrace
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.extractor_service import ExtractorService

def test_models():
    print("Testing Pydantic v2 Models...")
    qa = QAPair(
        id="test_01",
        question="What is the test question?",
        answer="Test answer",
        weight=0.9,
        confidence=1.0,
        origin="unit_test"
    )
    assert qa.id == "test_01"
    assert qa.weight == 0.9

    diff = MemoryDiff(
        tier1_updates=[qa],
        tier2_updates=[],
        tier3_updates={"test_entity": [qa]},
        deletions=[]
    )
    assert len(diff.tier1_updates) == 1
    assert "test_entity" in diff.tier3_updates
    print("[OK] Models validated successfully.")

def test_memory_engine():
    print("\nTesting MemoryEngine...")
    engine = MemoryEngine(memory_dir="data/memory")

    assert len(engine.tier1_items) >= 3, f"Expected >=3 T1 items, got {len(engine.tier1_items)}"
    assert len(engine.tier2_items) >= 3, f"Expected >=3 T2 items, got {len(engine.tier2_items)}"
    assert "lifeos" in engine.tier3_entities, "Expected 'lifeos' entity in T3"

    # Test prompt assembly without T3 mention
    prompt_normal, trace_normal = engine.assemble_prompt("Hello, how are you today?")
    assert trace_normal.t1_count == len(engine.tier1_items)
    assert trace_normal.t2_count == len(engine.tier2_items)
    assert len(trace_normal.t3_entities_loaded) == 0
    assert trace_normal.estimated_tokens > 0
    print(f"[OK] Normal Prompt assembled (~{trace_normal.estimated_tokens} est tokens, no T3).")

    # Test prompt assembly WITH T3 entity mention ("lifeos")
    prompt_t3, trace_t3 = engine.assemble_prompt("Tell me about LifeOS architecture.")
    assert "lifeos" in trace_t3.t3_entities_loaded
    assert trace_t3.t3_entities_loaded["lifeos"] <= 3
    assert "LIFEOS" in prompt_t3
    print(f"[OK] Entity-triggered Prompt assembled (~{trace_t3.estimated_tokens} est tokens, loaded T3: {trace_t3.t3_entities_loaded}).")

    # Test decay
    decayed_count = engine.apply_decay()
    print(f"[OK] Apply decay executed (decayed {decayed_count} items).")

def test_llm_client_and_extractor():
    print("\nTesting LLMClient and ExtractorService...")
    engine = MemoryEngine(memory_dir="data/memory")
    client = LLMClient()
    extractor = ExtractorService(engine, client)

    # Test offline response generation
    prompt, trace = engine.assemble_prompt("What is my sprint goal for LifeOS?")
    response = client.generate_coaching_response(prompt, "What is my sprint goal for LifeOS?")
    assert len(response) > 10
    print(f"[OK] Coach response generated:\n  '{response[:100]}...'")

    # Test synchronous extraction
    raw_dump = "Update sprint goal: Finishing LifeOS MVP memory compaction module and testing token efficiency."
    diff = extractor.extract_sync(raw_dump, prompt)
    assert isinstance(diff, MemoryDiff)
    print(f"[OK] Sync extraction completed: T1={len(diff.tier1_updates)}, T2={len(diff.tier2_updates)}, T3_entities={list(diff.tier3_updates.keys())}")

    extractor.shutdown()

if __name__ == "__main__":
    print("================ LIFEOS MVP VERIFICATION ================")
    test_models()
    test_memory_engine()
    test_llm_client_and_extractor()
    print("================ ALL TESTS PASSED SUCCESSFULLY! ================")
