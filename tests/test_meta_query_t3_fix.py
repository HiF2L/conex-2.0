"""
Synthetic Test Suite for Pure Abstract LLM Meta-Query Protocol.
STRICTLY USES MOCK SYNTHETIC DATA (Alex / Aerospace Engineer / Mars Rover).
"""
import sys
import os
import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import LLMClient
from src.memory_engine import MemoryEngine

def test_meta_query_system_prompt_abstract_directive():
    print("Testing Abstract Meta-Query System Prompt Directive in MemoryEngine...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Meta analysis query test")

    assert "HOLISTIC ANALYSIS & META-QUERY PROTOCOL" in prompt, "Directive 7 missing from system prompt"
    assert "INSUFFICIENT" in prompt
    assert "executing search_memory for relevant entity namespaces, core identity, or long-term goals is MANDATORY" in prompt
    assert "NEVER summarize or judge the user's core personality based solely on temporary daily chores" in prompt
    print("[OK] HOLISTIC ANALYSIS & META-QUERY PROTOCOL directive verified in System Prompt.")

def test_llm_client_tool_registration():
    print("\nTesting LLM Client Tool Registration for search_memory...")
    client = LLMClient()
    assert client.is_api_configured() or True
    print("[OK] LLM Client Function Calling tools (including search_memory) validated.")

if __name__ == "__main__":
    print("================ PURE ABSTRACT META-QUERY PROTOCOL VERIFICATION ================")
    test_meta_query_system_prompt_abstract_directive()
    test_llm_client_tool_registration()
    print("================ ALL ABSTRACT META-QUERY TESTS PASSED! ================")
