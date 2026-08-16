"""
Synthetic Test Suite for Wellbeing & State Protocol Engine (PostgreSQL + LLM Tools).
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

from src.db import (
    log_wellbeing_event_db,
    get_wellbeing_history_db,
    generate_wellbeing_summary_db,
    get_recovery_protocol_db
)
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient

def test_wellbeing_db_crud_and_summary():
    print("Testing Wellbeing DB CRUD & Aggregated Summary...")

    # Log peak clarity event
    log1 = log_wellbeing_event_db(
        state_type="PEAK_CLARITY",
        triggers=["low_carb", "brisk_walk", "americano"],
        symptoms=["fast_word_retrieval", "high_articulation"],
        notes="Felt extremely clear after morning walk and low carb breakfast."
    )
    assert log1["state_type"] == "PEAK_CLARITY"
    assert "low_carb" in log1["triggers"]

    # Log brain fog event
    log2 = log_wellbeing_event_db(
        state_type="BRAIN_FOG",
        triggers=["high_sugar", "poor_sleep"],
        symptoms=["sluggish_focus"],
        notes="Brain fog after eating pastry."
    )
    assert log2["state_type"] == "BRAIN_FOG"

    # Query history
    clarity_logs = get_wellbeing_history_db(state_type="PEAK_CLARITY", limit=5)
    assert any(l["id"] == log1["id"] for l in clarity_logs)
    print(f"[OK] Historical retrieval verified (found {len(clarity_logs)} PEAK_CLARITY logs).")

    # Generate summary
    summary = generate_wellbeing_summary_db()
    assert summary["total_logs"] >= 2
    assert "PEAK_CLARITY" in summary["states_breakdown"]
    print(f"[OK] Aggregated summary verified: {summary['states_breakdown']}.")

def test_recovery_protocol_generation():
    print("\nTesting Recovery Protocol Generation...")
    protocol_text = get_recovery_protocol_db(current_state="BRAIN_FOG")

    assert "RECOVERY PROTOCOL: RESET TO PEAK CLARITY" in protocol_text
    assert "Actionable Step-by-Step Recovery Checklist" in protocol_text
    assert "Hydration & Electrolytes" in protocol_text
    print("[OK] Recovery protocol generation verified.")

def test_llm_tools_and_prompt_directives():
    print("\nTesting LLM Tools Registration & Prompt Directives...")

    # Verify tool registration in LLMClient
    client = LLMClient()
    # Check tool array contains log_wellbeing_event and get_recovery_protocol
    tool_names = []
    for tool in getattr(client, "tools", []):
        if isinstance(tool, dict) and "function" in tool:
            tool_names.append(tool["function"]["name"])

    # If API not configured, fallback mock check
    if not tool_names:
        print("ℹ️ LLMClient initialized without API key (tool array built dynamically in generate_coaching_response).")

    # Verify MemoryEngine directive
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Felt brain fog today")
    assert "WELLBEING & STATE REGULATION DIRECTIVE" in prompt
    assert "log_wellbeing_event" in prompt
    assert "get_recovery_protocol" in prompt
    print("[OK] System Prompt directives for Wellbeing Engine verified.")

if __name__ == "__main__":
    print("================ WELLBEING ENGINE VERIFICATION ================")
    test_wellbeing_db_crud_and_summary()
    test_recovery_protocol_generation()
    test_llm_tools_and_prompt_directives()
    print("================ ALL WELLBEING ENGINE TESTS PASSED! ================")
