"""
Synthetic Test Suite for 100% Index-Based Tier 1 & Tier 2 Question Anchor Memory.
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
from src.db import get_qa_item_db
from src.llm_client import LLMClient

def test_t1_t2_index_prompt_assembly():
    print("Testing T1 & T2 Question Anchor Index Prompt Assembly...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Goal test query")

    assert "TIER 1: CORE PROFILE QUESTION ANCHORS INDEX" in prompt
    assert "TIER 2: DYNAMIC STATE QUESTION ANCHORS INDEX" in prompt
    assert "100% INDEX-BASED MEMORY PROTOCOL" in prompt

    print(f"[OK] Prompt assembled with lightweight T1 ({trace.t1_count} anchors) & T2 ({trace.t2_count} anchors) Question Indexes.")

def test_get_qa_item_db_lookup():
    print("\nTesting get_qa_item_db Item Lookup...")
    engine = MemoryEngine(memory_dir="data/memory")
    
    if engine.tier1_items:
        t1_id = engine.tier1_items[0].id
        res = get_qa_item_db(t1_id)
        assert "Question:" in res and "Answer:" in res
        print(f"[OK] Successfully looked up T1 item '{t1_id}'.")

    if engine.tier2_items:
        t2_id = engine.tier2_items[0].id
        res = get_qa_item_db(t2_id)
        assert "Question:" in res and "Answer:" in res
        print(f"[OK] Successfully looked up T2 item '{t2_id}'.")

if __name__ == "__main__":
    print("================ INDEX-BASED MEMORY VERIFICATION ================")
    test_t1_t2_index_prompt_assembly()
    test_get_qa_item_db_lookup()
    print("================ ALL INDEX-BASED MEMORY TESTS PASSED! ================")
