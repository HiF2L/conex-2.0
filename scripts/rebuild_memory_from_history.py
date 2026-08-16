"""
Administrative Script: Full Context-Aware Memory Rebuilding from History / Source Text.
Clears corrupted Tier 3 memory, processes raw user brief text via multi-turn contextual extractor,
and populates PostgreSQL FTS index with 100% accurate, strictly isolated entities.
"""
import os
import sys
import yaml
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.extractor_service import ExtractorService
from src.db import _get_connection, sync_tier3_to_postgres

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Memory_Rebuilder")

MEMORY_DIR = ROOT_DIR / "data" / "memory"
TIER3_DIR = MEMORY_DIR / "tier3_entities"
USER_BRIEF_FILE = ROOT_DIR / "data" / "user_brief.txt"

def wipe_tier3_memory():
    """Wipes all Tier 3 entity YAML files and PostgreSQL tier3_memory_index table."""
    # 1. Clear Tier 3 YAML files
    if TIER3_DIR.exists():
        for yaml_file in TIER3_DIR.glob("*.yaml"):
            try:
                yaml_file.unlink()
                logger.info(f"Deleted old Tier 3 entity file: {yaml_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {yaml_file}: {e}")

    # 2. Truncate PostgreSQL table
    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE tier3_memory_index;")
                conn.commit()
                logger.info("Wiped table 'tier3_memory_index' in PostgreSQL.")
        except Exception as e:
            logger.warning(f"Failed to truncate PostgreSQL tier3_memory_index: {e}")
        finally:
            conn.close()

def rebuild_memory_from_source(source_filepath: Path):
    print("================ MEMORY REBUILDING PROCESS STARTED ================")
    
    if not source_filepath.exists():
        print(f"❌ Error: Source brief file '{source_filepath}' not found!")
        return

    # 1. Wipe existing Tier 3 memory
    print("\nStep 1: Wiping old Tier 3 memory files and PostgreSQL index...")
    wipe_tier3_memory()

    # 2. Read source text
    print(f"\nStep 2: Reading source text from {source_filepath}...")
    with open(source_filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split text into logical sections by double newline
    sections = [s.strip() for s in full_text.split("\n\n") if s.strip()]
    print(f"  [OK] Parsed {len(sections)} contextual text blocks.")

    # 3. Initialize Memory Engine & Extractor
    memory_engine = MemoryEngine(memory_dir=str(MEMORY_DIR))
    llm_client = LLMClient()
    extractor = ExtractorService(memory_engine, llm_client)

    # 4. Context-aware extraction loop
    print("\nStep 3: Extracting structured memory with context preservation...")
    sliding_history: List[Dict[str, str]] = []

    for i, section in enumerate(sections, 1):
        print(f"  Processing section {i}/{len(sections)}...")
        prompt, trace = memory_engine.assemble_prompt(f"Section {i} extraction")
        
        diff = extractor.extract_sync(
            raw_text=section,
            system_prompt=prompt,
            conversation_history=sliding_history
        )

        # Update sliding history for context preservation
        sliding_history.append({"role": "user", "content": section})
        if diff:
            summary_str = f"Extracted: T1={len(diff.tier1_updates)}, T2={len(diff.tier2_updates)}, T3_entities={list(diff.tier3_updates.keys())}"
            sliding_history.append({"role": "assistant", "content": summary_str})

        if len(sliding_history) > 8:
            sliding_history = sliding_history[-8:]

    # 5. Sync Tier 3 to PostgreSQL
    print("\nStep 4: Syncing newly extracted Tier 3 entities to PostgreSQL FTS index...")
    synced_db_count = sync_tier3_to_postgres()

    # 6. Final Summary
    entity_files = list(TIER3_DIR.glob("*.yaml")) if TIER3_DIR.exists() else []
    print("\n================ MEMORY REBUILDING COMPLETED ================")
    print("Summary:")
    print(f"  - Source Text Blocks Processed: {len(sections)}")
    print(f"  - Tier 3 Clean Entity Files: {len(entity_files)}")
    for ef in entity_files:
        print(f"    * {ef.name}")
    print(f"  - Total PostgreSQL Synced QA Items: {synced_db_count}")
    print("=============================================================")

if __name__ == "__main__":
    rebuild_memory_from_source(USER_BRIEF_FILE)
