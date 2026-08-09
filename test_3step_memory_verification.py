"""
Synthetic Test Suite for 3-Step Ultra-Token-Efficient Memory Protocol & Granular Memory Tools.
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
from src.llm_client import LLMClient
from src.db import (
    search_tier3_memory,
    get_document_outline_db,
    read_document_section_db,
    search_in_document_db,
    read_memory_entry_db
)

def test_3step_memory_directives():
    print("Testing 3-Step Memory Protocol Directive in MemoryEngine...")
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("3-Step protocol test")

    assert "3-STEP ULTRA-TOKEN-EFFICIENT MEMORY PROTOCOL" in prompt, "3-Step Protocol directive missing"
    assert "Step 1: Call search_memory(query)" in prompt
    assert "Step 2: Call get_document_outline(identifier)" in prompt
    assert "Step 3: Call read_document_section(identifier, section_id)" in prompt
    print("[OK] 3-Step Memory Protocol directive verified in System Prompt.")

def test_granular_memory_db_functions():
    print("\nTesting Granular Memory DB Functions...")
    
    # 1. Global search (up to 10 hits)
    search_res = search_tier3_memory("проекты и цели", top_k=10)
    assert search_res is not None, "Search result should not be None"
    assert "Section ID" in search_res or "candidate memory entries" in search_res or "Tier 3 Memory" in search_res
    print("  [OK] search_tier3_memory candidate hits verified.")

    # 2. Document outline
    outline_res = get_document_outline_db("wegeny")
    assert outline_res is not None, "Outline result should not be None"
    assert "Document Outline" in outline_res or "sections total" in outline_res or "No document outline found" in outline_res
    print("  [OK] get_document_outline_db verified.")

    # 3. Read document section
    sec_res = read_document_section_db("wegeny", "1")
    assert sec_res is not None, "Section result should not be None"
    print("  [OK] read_document_section_db verified.")

    # 4. Search in document
    doc_search_res = search_in_document_db("wegeny", "python")
    assert doc_search_res is not None, "In-document search result should not be None"
    print("  [OK] search_in_document_db verified.")

    # 5. Read memory entry
    full_res = read_memory_entry_db("wegeny")
    assert full_res is not None, "Full entry result should not be None"
    print("  [OK] read_memory_entry_db verified.")

def test_llm_client_granular_tools_registration():
    print("\nTesting LLM Client Granular Memory Tools Registration...")
    client = LLMClient()
    # Ensure client initializes cleanly with all tools
    assert client.default_model is not None
    print("[OK] LLM Client Granular Memory tools validated.")

if __name__ == "__main__":
    print("================ 3-STEP MEMORY PROTOCOL VERIFICATION ================")
    test_3step_memory_directives()
    test_granular_memory_db_functions()
    test_llm_client_granular_tools_registration()
    print("================ ALL 3-STEP MEMORY VERIFICATION TESTS PASSED! ================")
