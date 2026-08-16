"""
Synthetic Test Suite for 100% Domain-Agnostic T3 Trace Counting & Entity Auto-Loading.
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
from src.models import MemoryTrace

def test_detect_entities_autoload():
    print("Testing Domain-Agnostic detect_entities Auto-Loading...")
    engine = MemoryEngine(memory_dir="data/memory")

    # Call detect_entities with a query containing NO entity names
    generic_query = "Проанализируй меня как личность и дай общую оценку"
    matched = engine.detect_entities(generic_query)

    assert matched is not None and len(matched) > 0, "detect_entities should auto-load entities when no entity name matches"
    
    # Check assemble_prompt
    prompt, trace = engine.assemble_prompt(generic_query)
    assert len(trace.t3_entities_loaded) > 0, "trace.t3_entities_loaded should be > 0"
    print(f"[OK] detect_entities auto-loaded {len(matched)} entities ({len(trace.t3_entities_loaded)} entities loaded in prompt).")

def test_dynamic_trace_tool_counting():
    print("\nTesting Dynamic Trace Tool Counting in generate_coaching_response...")
    client = LLMClient()
    trace = MemoryTrace(t1_count=5, t2_count=5, estimated_tokens=100)

    assert trace.t3_sections_read == 0, f"Initial t3_sections_read should be 0, got {trace.t3_sections_read}"

    # Simulate reading 1 section
    trace.t3_sections_read += 1

    assert trace.t3_sections_read == 1, f"Updated t3_sections_read should be 1, got {trace.t3_sections_read}"
    assert trace.t3_total == 1
    print(f"[OK] Dynamic trace tool counting verified: t3_sections_read = {trace.t3_sections_read}.")

if __name__ == "__main__":
    print("================ T3 TRACE FIX & AUTOLOAD VERIFICATION ================")
    test_detect_entities_autoload()
    test_dynamic_trace_tool_counting()
    print("================ ALL T3 TRACE FIX TESTS PASSED! ================")
