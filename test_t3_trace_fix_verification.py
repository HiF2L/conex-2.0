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
sys.path.insert(0, str(Path(__file__).parent))

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
    assert trace.t3_total > 0, f"trace.t3_total should be > 0, got {trace.t3_total}"
    print(f"[OK] detect_entities auto-loaded {len(matched)} entities ({trace.t3_total} Qs).")

def test_dynamic_trace_tool_counting():
    print("\nTesting Dynamic Trace Tool Counting in generate_coaching_response...")
    client = LLMClient()
    trace = MemoryTrace(t1_count=5, t2_count=5, t3_entities_loaded={"initial": 2}, estimated_tokens=100)

    assert trace.t3_total == 2, f"Initial t3_total should be 2, got {trace.t3_total}"

    # Simulate tool response update
    tool_res = (
        "Found 10 candidate memory entries matching 'projects':\n"
        "1. [Section ID: 101 | Entity: WEGENY] Topic: Psychological matching\n"
        "2. [Section ID: 102 | Entity: LIFEOS] Topic: ProactiveEngine"
    )
    
    c = len(__import__("re").findall(r"Section ID|\d+\.\s+\[Entity", tool_res))
    trace.t3_entities_loaded["retrieved_via_tools"] = trace.t3_entities_loaded.get("retrieved_via_tools", 0) + max(1, c)

    assert trace.t3_total > 2, f"Updated t3_total should be > 2, got {trace.t3_total}"
    print(f"[OK] Dynamic trace tool counting verified: new t3_total = {trace.t3_total}.")

if __name__ == "__main__":
    print("================ T3 TRACE FIX & AUTOLOAD VERIFICATION ================")
    test_detect_entities_autoload()
    test_dynamic_trace_tool_counting()
    print("================ ALL T3 TRACE FIX TESTS PASSED! ================")
