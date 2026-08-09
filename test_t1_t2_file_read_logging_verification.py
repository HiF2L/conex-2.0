"""
Synthetic Test Suite for Mandatory T1/T2 Anchor Read Execution & Source File Logging.
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
from src.db import get_qa_item_db
from src.models import MemoryTrace

def test_source_file_in_get_qa_item_db():
    print("Testing get_qa_item_db returns Source File label...")
    engine = MemoryEngine(memory_dir="data/memory")

    if engine.tier1_items:
        t1_id = engine.tier1_items[0].id
        res = get_qa_item_db(t1_id)
        assert "Source File: tier1_core.yaml" in res, f"Expected tier1_core.yaml in res, got:\n{res}"
        print(f"[OK] get_qa_item_db T1 returned Source File: tier1_core.yaml")

    if engine.tier2_items:
        t2_id = engine.tier2_items[0].id
        res = get_qa_item_db(t2_id)
        assert "Source File: tier2_state.yaml" in res, f"Expected tier2_state.yaml in res, got:\n{res}"
        print(f"[OK] get_qa_item_db T2 returned Source File: tier2_state.yaml")

def test_strict_directive_7_text():
    print("\nTesting Strict Directive 7 Text in System Prompt...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("test prompt")

    assert "EXPLICITLY FORBIDDEN to guess, invent, or output answer facts" in prompt
    assert "MUST call read_memory_item(item_id) to inspect the factual answer in tier1_core.yaml or tier2_state.yaml" in prompt
    print("[OK] Strict Directive 7 verified successfully.")

if __name__ == "__main__":
    print("================ T1/T2 FILE READ LOGGING VERIFICATION ================")
    test_source_file_in_get_qa_item_db()
    test_strict_directive_7_text()
    print("================ ALL T1/T2 FILE READ LOGGING TESTS PASSED! ================")
