"""
Verification script for Proactive Push Engine, Quiet Hours, Event Pings, and Evening Sync Protocol.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.proactive_engine import ProactiveEngine
from src.db import save_scheduled_ping, get_due_pings, mark_ping_status, get_pings_count_today
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.scheduler import create_scheduler
from aiogram import Bot

def test_quiet_hours():
    print("Testing Quiet Hours Logic (22:00 -> 08:00)...")
    engine = ProactiveEngine()
    
    # 23:00 -> inside quiet hours
    assert engine.is_in_quiet_hours(datetime.time(23, 0)) is True, "23:00 should be inside Quiet Hours"
    # 03:30 -> inside quiet hours
    assert engine.is_in_quiet_hours(datetime.time(3, 30)) is True, "03:30 should be inside Quiet Hours"
    # 14:00 -> outside quiet hours
    assert engine.is_in_quiet_hours(datetime.time(14, 0)) is False, "14:00 should be outside Quiet Hours"
    print("[OK] Quiet Hours evaluated correctly across overnight boundary.")

def test_scheduled_ping_db():
    print("\nTesting Scheduled Ping DB Operations & Fallback...")
    user_id = 999999
    now_past = (datetime.datetime.now() - datetime.timedelta(minutes=10)).isoformat()
    
    ping_id = save_scheduled_ping(user_id, now_past, "doctor", "Стоматолог в 11:00")
    assert ping_id > 0, "Failed to save scheduled ping"
    
    due = get_due_pings(user_id)
    assert len(due) >= 1, "Expected due pings count >= 1"
    match = [p for p in due if p["context_text"] == "Стоматолог в 11:00"]
    assert len(match) == 1, "Did not find saved context text in due pings"
    
    mark_ping_status(match[0]["id"], "executed")
    
    due_after = get_due_pings(user_id)
    match_after = [p for p in due_after if p["context_text"] == "Стоматолог в 11:00"]
    assert len(match_after) == 0, "Executed ping should no longer be pending"
    
    count_today = get_pings_count_today(user_id)
    assert count_today >= 1, f"Expected executed pings count >= 1, got {count_today}"
    print(f"[OK] Scheduled pings DB save, retrieval, status mark, and count verified (Count today: {count_today}).")

def test_event_ping_extraction():
    print("\nTesting Event Ping Extractor Rule...")
    llm_client = LLMClient()
    sys_prompt = "You are an AI Memory Extractor."
    user_text = "Завтра в 11:00 иду к врачу-стоматологу на осмотр."
    
    diff = llm_client.extract_memory_diff(sys_prompt, user_text)
    print(f"[OK] Extracted MemoryDiff: T1={len(diff.tier1_updates)}, T2={len(diff.tier2_updates)}, Pings={len(diff.scheduled_pings)}")

def test_scheduler_jobs():
    print("\nTesting Scheduler Registration with Proactive Ping Checker...")
    bot = Bot(token="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")
    memory_engine = MemoryEngine(memory_dir="data/memory")
    llm_client = LLMClient()
    
    scheduler = create_scheduler(bot, 123456789, memory_engine, llm_client)
    job_ids = [job.id for job in scheduler.get_jobs()]
    
    assert "proactive_ping_checker_job" in job_ids
    assert "morning_briefing_job" in job_ids
    assert "evening_reflection_job" in job_ids
    assert "nightly_memory_snapshot_job" in job_ids
    print(f"[OK] APScheduler successfully registered all 4 jobs: {job_ids}")

if __name__ == "__main__":
    print("================ PROACTIVE ENGINE & EVENING SYNC VERIFICATION ================")
    test_quiet_hours()
    test_scheduled_ping_db()
    test_event_ping_extraction()
    test_scheduler_jobs()
    print("================ ALL PROACTIVE ENGINE TESTS PASSED! ================")
