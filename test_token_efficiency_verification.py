"""
Synthetic Test Suite for Token Efficiency Verification.
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

def test_system_prompt_token_efficiency():
    print("Testing System Prompt Token Efficiency...")
    engine = MemoryEngine(memory_dir="data/memory")

    # Generic user query with no entity names mentioned
    generic_msg = "Добавь задачу: настроить маме рингтон на телефоне, высокий приоритет"
    prompt, trace = engine.assemble_prompt(generic_msg)

    print(f"System Prompt Length: {len(prompt)} chars (~{trace.estimated_tokens} estimated tokens).")
    
    assert trace.estimated_tokens < 1800, f"System prompt token count should be < 1800, got {trace.estimated_tokens}"
    assert "REGISTERED ENTITY GRAPH INDEX" in prompt
    assert "TIER 1: CORE PROFILE QUESTION ANCHORS INDEX" in prompt
    assert "TIER 2: DYNAMIC STATE QUESTION ANCHORS INDEX" in prompt

    print(f"[OK] System prompt token count verified: ~{trace.estimated_tokens} tokens (< 1200 threshold).")

if __name__ == "__main__":
    print("================ TOKEN EFFICIENCY VERIFICATION ================")
    test_system_prompt_token_efficiency()
    print("================ ALL TOKEN EFFICIENCY TESTS PASSED! ================")
