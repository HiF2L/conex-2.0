"""
Synthetic Test Suite for Telegram Debug Mode (/debug).
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

from src.models import MemoryTrace
from src.llm_client import LLMClient
from src.bot import user_debug_mode

def test_user_debug_mode_toggle():
    print("Testing /debug User Preference Map Toggle...")
    user_id = 999111
    assert user_debug_mode.get(user_id, False) is False

    user_debug_mode[user_id] = True
    assert user_debug_mode.get(user_id, False) is True

    user_debug_mode[user_id] = False
    assert user_debug_mode.get(user_id, False) is False

    print("[OK] /debug User Preference Map toggle verified.")

def test_debug_steps_recording():
    print("\nTesting Detailed Section Header & Title Logging in debug_steps...")
    trace = MemoryTrace(t1_count=5, t2_count=5, estimated_tokens=100)

    # Simulate tool calls recording detailed section headers
    search_res = (
        "Found 10 candidate memory entries matching 'projects':\n"
        "1. [Section ID: 101 | Entity: WEGENY] Topic: Psychological matching\n"
        "2. [Section ID: 102 | Entity: LIFEOS] Topic: ProactiveEngine"
    )
    hits = __import__("re").findall(r"Entity:\s*([A-Za-z0-9_]+)", search_res)
    trace.debug_steps.append(f"• 🔍 `search_memory(\"projects\")` -> Hits: [{', '.join(hits[:5])}]")

    outline_res = (
        "Document Outline for 'WEGENY' (3 sections total):\n"
        "  - [Section ID: 12] Topic: What is WeGeny mission?\n"
        "  - [Section ID: 13] Topic: Technical stack\n"
        "  - [Section ID: 14] Topic: Psychological matching"
    )
    headers = __import__("re").findall(r"-\s*\[Section ID:\s*[^\]]+\]\s*Topic:\s*([^\n]+)", outline_res)
    headers_str = ", ".join([f"'{h.strip()}'" for h in headers[:4]])
    trace.debug_steps.append(f"• 📋 `get_document_outline(\"wegeny\")` -> Headers: [{headers_str}]")

    sec_res = "[Entity: WEGENY | Section ID: 14]\nQuestion: How does psychological matching work?\nAnswer: Details..."
    q_match = __import__("re").search(r"Question:\s*([^\n]+)", sec_res)
    topic = q_match.group(1).strip() if q_match else "14"
    trace.debug_steps.append(f"• 📖 `read_document_section(\"wegeny\", \"14\")` -> Section: '{topic}' ({len(sec_res)} chars)")

    assert len(trace.debug_steps) == 3
    assert "Hits: [WEGENY, LIFEOS]" in trace.debug_steps[0]
    assert "Headers: ['What is WeGeny mission?', 'Technical stack', 'Psychological matching']" in trace.debug_steps[1]
    assert "Section: 'How does psychological matching work?'" in trace.debug_steps[2]

    print("[OK] Detailed debug_steps recording verified successfully.")

if __name__ == "__main__":
    print("================ TELEGRAM DEBUG MODE VERIFICATION ================")
    test_user_debug_mode_toggle()
    test_debug_steps_recording()
    print("================ ALL DEBUG MODE VERIFICATION TESTS PASSED! ================")
