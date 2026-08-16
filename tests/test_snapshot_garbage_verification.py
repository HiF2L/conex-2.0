"""
Verification script for Nightly Memory Snapshots (23:59), Smart T2 Garbage Collection, and forget_memory Tool.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import QAPair
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.scheduler import create_scheduler
from aiogram import Bot

def test_nightly_snapshot():
    print("Testing Nightly Memory History Snapshot...")
    engine = MemoryEngine(memory_dir="data/memory")
    snapshot_path = engine.save_nightly_snapshot()
    
    assert os.path.exists(snapshot_path), f"Snapshot file missing: {snapshot_path}"
    today_str = datetime.date.today().isoformat()
    assert today_str in snapshot_path
    print(f"[OK] Saved and verified snapshot at: {snapshot_path}")

def test_tier2_garbage_cleanup():
    print("\nTesting Smart Tier 2 Garbage Collection...")
    engine = MemoryEngine(memory_dir="data/memory")
    
    # Inject garbage test items
    engine.tier2_items.append(QAPair(
        id="test_completed_chore",
        question="Did user complete the quick test chore?",
        answer="Yes, completed yesterday.",
        weight=0.9,
        valid_from="2026-08-01"
    ))
    engine.tier2_items.append(QAPair(
        id="test_low_weight_item",
        question="What is the temporary noise state?",
        answer="Irrelevant random note.",
        weight=0.05,
        valid_from="2026-08-01"
    ))
    engine._write_yaml_file_safely(engine.tier2_path, engine.tier2_items)

    cleaned_count = engine.cleanup_tier2_garbage()
    assert cleaned_count >= 2, f"Expected at least 2 items cleaned, got {cleaned_count}"
    
    remaining_ids = {qa.id for qa in engine.tier2_items}
    assert "test_completed_chore" not in remaining_ids
    assert "test_low_weight_item" not in remaining_ids
    print(f"[OK] Cleaned {cleaned_count} garbage items from Tier 2 state.")

def test_forget_memory_tool():
    print("\nTesting forget_memory Tool...")
    engine = MemoryEngine(memory_dir="data/memory")

    # Add test items to Tier 2
    test_id = f"test_forget_{int(datetime.datetime.now().timestamp())}"
    engine.tier2_items.append(QAPair(
        id=test_id,
        question="What is the test project secret key?",
        answer="SECRET_FORGET_KEY_123",
        weight=1.0,
        valid_from=datetime.date.today().isoformat()
    ))
    engine._write_yaml_file_safely(engine.tier2_path, engine.tier2_items)

    result_msg = engine.forget_memory(target_tier="tier2", keyword="SECRET_FORGET_KEY_123")
    assert "Successfully removed" in result_msg
    
    remaining_ids = {qa.id for qa in engine.tier2_items}
    assert test_id not in remaining_ids
    print(f"[OK] forget_memory tool executed successfully:\n  {result_msg}")

def test_2359_scheduler_job():
    print("\nTesting 23:59 Scheduler Job Registration...")
    bot = Bot(token="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")
    engine = MemoryEngine(memory_dir="data/memory")
    client = LLMClient()
    
    scheduler = create_scheduler(bot, 123456789, engine, client)
    job_ids = [job.id for job in scheduler.get_jobs()]
    
    assert "nightly_memory_snapshot_job" in job_ids
    assert "morning_briefing_job" in job_ids
    assert "evening_reflection_job" in job_ids
    print(f"[OK] APScheduler correctly registered all 3 daily jobs: {job_ids}")

if __name__ == "__main__":
    print("================ SNAPSHOT & GARBAGE COLLECTION VERIFICATION ================")
    test_nightly_snapshot()
    test_tier2_garbage_cleanup()
    test_forget_memory_tool()
    test_2359_scheduler_job()
    print("================ ALL SNAPSHOT & GARBAGE COLLECTION TESTS PASSED! ================")
