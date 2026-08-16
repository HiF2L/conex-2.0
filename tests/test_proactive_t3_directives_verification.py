"""
Synthetic Test Suite for Proactive Tier 3 Search & Zero-Fluff Directives.
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
from src.llm_client import LLMClient, is_trivial_user_turn

def test_zero_fluff_and_proactive_directives():
    print("Testing Zero-Fluff & Proactive T3 Search Directives in MemoryEngine...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Directive test query")

    assert "MAXIMUM DENSITY & ZERO-FLUFF DIRECTIVE" in prompt, "Zero-fluff directive missing"
    assert "DEFAULT PROACTIVE TIER 3 SEARCH PROTOCOL" in prompt, "Proactive T3 search protocol missing"
    assert "relying solely on T1/T2 is EXPLICITLY FORBIDDEN" in prompt
    assert "Calling search_memory(query) on Step 1 is MANDATORY for all non-trivial turns" in prompt
    print("[OK] System Prompt Directives verified successfully.")

def test_is_trivial_user_turn():
    print("\nTesting Turn Classifier (is_trivial_user_turn)...")
    
    trivial_inputs = [
        "привет", "хай", "спасибо", "ок", "понял",
        "заверши задачу 5", "добавь задачу купить хлеб", "удали проект test"
    ]
    for inp in trivial_inputs:
        assert is_trivial_user_turn(inp) is True, f"Failed to classify trivial turn: '{inp}'"

    non_trivial_inputs = [
        "Расскажи о моих проектах и их архитектуре.",
        "Проанализируй мою личность и жизненные цели.",
        "Как устроен движок символического синтеза в Intelligence Bit?",
        "What is the complete architecture of WeGeny?"
    ]
    for inp in non_trivial_inputs:
        assert is_trivial_user_turn(inp) is False, f"Failed to classify non-trivial turn: '{inp}'"

    print("[OK] Turn Classifier accurately separates trivial task commands from non-trivial research turns.")

if __name__ == "__main__":
    print("================ PROACTIVE T3 & ZERO-FLUFF VERIFICATION ================")
    test_zero_fluff_and_proactive_directives()
    test_is_trivial_user_turn()
    print("================ ALL DIRECTIVE VERIFICATION TESTS PASSED! ================")
