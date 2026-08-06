"""
Verification script for Proactive Engine Localization, Context Sync & Timezone Fix.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.scheduler import create_scheduler
from aiogram import Bot

def test_timezone_and_scheduler():
    print("Testing APScheduler Timezone Configuration...")
    bot = Bot(token="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")
    memory_engine = MemoryEngine(memory_dir="data/memory")
    llm_client = LLMClient()
    
    scheduler = create_scheduler(bot, 123456789, memory_engine, llm_client)
    assert str(scheduler.timezone) == "Europe/Moscow", f"Expected timezone Europe/Moscow, got {scheduler.timezone}"
    print(f"[OK] APScheduler timezone configured to {scheduler.timezone}")

def test_proactive_awareness_directive():
    print("\nTesting Model Awareness of ProactiveEngine in System Prompt...")
    memory_engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = memory_engine.assemble_prompt("General query")
    
    assert "ProactiveEngine" in prompt, "System prompt should inform model of ProactiveEngine"
    assert "Do NOT tell the user you cannot text them proactively" in prompt
    print("[OK] System Prompt contains proactive engine awareness directive.")

def test_daily_systems_extraction():
    print("\nTesting Daily Systems & Routines Extractor Rule...")
    llm_client = LLMClient()
    sys_prompt = "You are an AI Memory Extractor."
    user_text = "Мои 4 главных правила системности каждый день: 1) Проектный шаг, 2) Уборка 4 зон, 3) 1 час прогулка и мысли, 4) Отклик на вакансию."
    
    diff = llm_client.extract_memory_diff(sys_prompt, user_text)
    assert len(diff.tier2_updates) >= 1, "Expected at least 1 Tier 2 update for daily systems"
    t2_answers = [qa.answer.lower() for qa in diff.tier2_updates]
    print(f"[OK] Extracted {len(diff.tier2_updates)} Tier 2 items for Daily Systems: {t2_answers[0][:80]}...")

if __name__ == "__main__":
    print("================ CRITICAL BUGFIX VERIFICATION ================")
    test_timezone_and_scheduler()
    test_proactive_awareness_directive()
    test_daily_systems_extraction()
    print("================ ALL CRITICAL BUGFIX TESTS PASSED! ================")
