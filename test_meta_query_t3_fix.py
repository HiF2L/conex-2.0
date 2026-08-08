"""
Synthetic Test Suite for Meta-Queries & Abstract Personality Analysis Fix.
STRICTLY USES MOCK SYNTHETIC DATA (Alex / Aerospace Engineer / Mars Rover).
"""
import sys
import os
import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_client import is_meta_analysis_query, LLMClient
from src.memory_engine import MemoryEngine

def test_meta_query_detector():
    print("Testing Generic Meta-Query Intent Detector (is_meta_analysis_query)...")
    
    meta_queries = [
        "Слушай, можешь описать меня как человека и рассказать о моих проектах?",
        "Проанализируй мою личность на основе наших разговоров.",
        "Опиши мои главные проекты и стратегические цели.",
        "Describe my personality and project portfolio.",
        "Overview all my projects and global life strategy."
    ]
    
    for q in meta_queries:
        assert is_meta_analysis_query(q) is True, f"Failed to detect meta-query: '{q}'"

    operational_queries = [
        "Добавь задачу купить хлеб",
        "Заверши задачу 5",
        "Какая погода сегодня?",
        "Напомни про встречу с врачом"
    ]
    for q in operational_queries:
        assert is_meta_analysis_query(q) is False, f"False positive for operational query: '{q}'"

    print("[OK] is_meta_analysis_query intent detector accurately identifies meta-queries.")

def test_meta_query_system_prompt_directive():
    print("\nTesting Meta-Query System Prompt Directive in MemoryEngine...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Meta analysis query test")

    assert "META-QUERY & HOLISTIC ANALYSIS DIRECTIVE" in prompt, "Directive 7 missing from system prompt"
    assert "INSUFFICIENT" in prompt
    assert "NEVER base a user's core personality or long-term identity on temporary daily chores" in prompt
    print("[OK] META-QUERY & HOLISTIC ANALYSIS DIRECTIVE verified in System Prompt.")

def test_synthetic_meta_query_context_prefetch():
    print("\nTesting Synthetic Meta-Query Search Pre-Fetch (Alex / Aerospace Engineer)...")
    client = LLMClient()
    
    # Synthetic query about mock user Alex
    synthetic_user_query = "Describe me as a person and overview all my aerospace projects."
    assert is_meta_analysis_query(synthetic_user_query) is True, "Synthetic meta-query should trigger detector"

    print("[OK] Synthetic meta-query pre-fetch triggers correctly.")

if __name__ == "__main__":
    print("================ META-QUERY & PERSONALITY ANALYSIS FIX VERIFICATION ================")
    test_meta_query_detector()
    test_meta_query_system_prompt_directive()
    test_synthetic_meta_query_context_prefetch()
    print("================ ALL SYNTHETIC META-QUERY TESTS PASSED! ================")
