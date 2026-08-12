"""
Synthetic Test Suite for Smart Notifications & Approach Adaptation.
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
from src.db import get_top_focus_tasks_db, create_task_db

def test_directives_and_task_filtering():
    print("Testing Smart Directives & Top-3 Task Filtering...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("test prompt")

    assert "DIRECT-OUTPUT DIRECTIVE" in prompt
    assert "DYNAMIC ADAPTATION DIRECTIVE FOR USER SILENCE" in prompt
    print("[OK] Directives present in System Prompt.")

    top_tasks, total_count = get_top_focus_tasks_db(limit=3)
    assert isinstance(top_tasks, list)
    assert len(top_tasks) <= 3
    print(f"[OK] get_top_focus_tasks_db returned {len(top_tasks)} top focus tasks (Total active in DB: {total_count}).")

if __name__ == "__main__":
    print("================ SMART NOTIFICATIONS VERIFICATION ================")
    test_directives_and_task_filtering()
    print("================ ALL SMART NOTIFICATIONS TESTS PASSED! ================")
