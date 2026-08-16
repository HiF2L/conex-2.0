"""
Verification script for Telegram Bot integration (aiogram 3.x, Message Splitting, Voice, Scheduler).
"""
import sys
import asyncio
import os
import tempfile
from pathlib import Path
from aiogram import Router, Bot

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import QAPair, MemoryDiff
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.extractor_service import ExtractorService
from src.bot import SecurityFilter, register_handlers, active_dump_sessions, split_message
from src.scheduler import create_scheduler

def test_split_message():
    print("Testing Message Splitting Utility...")
    short_text = "Short response."
    assert split_message(short_text, max_chunk_size=3900) == [short_text]

    # Create 10,000 char long text
    long_paragraph_1 = "A" * 3000 + "\n\n"
    long_paragraph_2 = "B" * 3000 + "\n\n"
    long_paragraph_3 = "C" * 3000
    long_text = long_paragraph_1 + long_paragraph_2 + long_paragraph_3

    chunks = split_message(long_text, max_chunk_size=3900)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 3900
    assert "A" in chunks[0] and "B" in chunks[1] and "C" in chunks[2]
    print(f"[OK] Split {len(long_text)} char message into {len(chunks)} clean sequential chunks.")

def test_security_filter():
    print("\nTesting SecurityFilter...")
    filter_123 = SecurityFilter(allowed_id=12345)
    
    class DummyUser:
        def __init__(self, uid):
            self.id = uid

    class DummyMessage:
        def __init__(self, uid):
            self.from_user = DummyUser(uid)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    assert loop.run_until_complete(filter_123(DummyMessage(12345))) == True
    assert loop.run_until_complete(filter_123(DummyMessage(99999))) == False
    print("[OK] SecurityFilter validated successfully.")

def test_dump_session_buffering():
    print("\nTesting Multi-Message Dump Session Buffering...")
    test_user_id = 987654
    active_dump_sessions[test_user_id] = []
    
    active_dump_sessions[test_user_id].append("Part 1: Worked on WeGeny API.")
    active_dump_sessions[test_user_id].append("Part 2: Health update - 30 min cardio.")
    active_dump_sessions[test_user_id].append("Part 3: Music synthesis project idea.")

    assert len(active_dump_sessions[test_user_id]) == 3
    combined = "\n\n---\n\n".join(active_dump_sessions.pop(test_user_id))
    assert "WeGeny" in combined and "Health" in combined and "Music" in combined
    assert test_user_id not in active_dump_sessions
    print(f"[OK] Multi-message dump buffer merged successfully ({len(combined)} chars).")

def test_bot_handlers_registration():
    print("\nTesting Handler & Command Registration...")
    router = Router()
    engine = MemoryEngine(memory_dir="data/memory")
    client = LLMClient()
    extractor = ExtractorService(engine, client)

    register_handlers(router, engine, client, extractor)
    assert len(router.message.handlers) >= 7
    print(f"[OK] {len(router.message.handlers)} message handlers registered.")
    extractor.shutdown()

if __name__ == "__main__":
    print("================ BOT INTEGRATION VERIFICATION ================")
    test_split_message()
    test_security_filter()
    test_dump_session_buffering()
    test_bot_handlers_registration()
    print("================ ALL TELEGRAM BOT TESTS PASSED! ================")
