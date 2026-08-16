"""
Administrative Memory Sanitization, Deduplication, and PostgreSQL Re-indexing Script.
Wipes corrupted DB indexes, cleans hallucinated cross-entity links & duplicate QA items from YAML files,
and rebuilds PostgreSQL Tier 3 search index from scratch.
"""
import os
import sys
import re
import yaml
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import QAPair
from src.db import _get_connection, sync_tier3_to_postgres, delete_tier3_memory_by_keyword

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Memory_Sanitizer")

MEMORY_DIR = ROOT_DIR / "data" / "memory"
TIER2_FILE = MEMORY_DIR / "tier2_state.yaml"
TIER3_DIR = MEMORY_DIR / "tier3_entities"

# Known false cross-entity hallucination patterns to remove/clean
FALSE_CROSS_ENTITY_PATTERNS = [
    r"intelligence bit is a component or product within the wegeny ecosystem",
    r"conex may be a conceptual or former name for lifeos",
    r"user input:\s*agent response:",
]

def is_corrupted_or_hallucinated(item: Dict[str, Any]) -> bool:
    """Checks if QA pair contains raw prompt text or false cross-entity claims."""
    q = str(item.get("question", "")).lower()
    a = str(item.get("answer", "")).lower()
    text = f"{q} {a}"

    # Check for raw turn dumps
    if "user input:" in a or "agent response:" in a:
        return True

    # Check for hallucinated cross-entity links
    for pattern in FALSE_CROSS_ENTITY_PATTERNS:
        if re.search(pattern, text):
            return True

    return False

def sanitize_and_deduplicate_qa_list(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitizes raw QA dictionaries: removes corrupted items, and ensures exactly ONE
    canonical QA pair per question (picking highest weight / latest date).
    """
    cleaned_items = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if is_corrupted_or_hallucinated(item):
            logger.info(f"Removing corrupted/hallucinated QA item: {item.get('id')} - {item.get('question')[:40]}...")
            continue
        cleaned_items.append(item)

    # Group by normalized question
    question_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in cleaned_items:
        norm_q = str(item.get("question", "")).strip().lower()
        if norm_q not in question_groups:
            question_groups[norm_q] = []
        question_groups[norm_q].append(item)

    canonical_items = []
    for norm_q, group in question_groups.items():
        if len(group) == 1:
            canonical_items.append(group[0])
        else:
            # Pick canonical item: highest weight, latest valid_from, highest confidence
            best_item = max(
                group,
                key=lambda x: (
                    float(x.get("weight", 1.0)),
                    str(x.get("valid_from", "")),
                    float(x.get("confidence", 1.0))
                )
            )
            canonical_items.append(best_item)

    return canonical_items

def wipe_postgres_indexes():
    """Wipes tier3_memory_index and chat_history from PostgreSQL."""
    conn = _get_connection()
    if not conn:
        print("ℹ️ PostgreSQL connection offline. Skipping DB wipe.")
        return 0

    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE tier3_memory_index;")
            cur.execute("TRUNCATE TABLE chat_history;")
            conn.commit()
            print("  [OK] Wiped tables 'tier3_memory_index' and 'chat_history' in PostgreSQL.")
            return 1
    except Exception as e:
        logger.warning(f"Failed to wipe PostgreSQL tables: {e}")
        return 0
    finally:
        conn.close()

def run_memory_sanitization_and_reindex():
    print("================ MEMORY SANITIZATION & RE-INDEXING ================")
    
    # 1. Wipe PostgreSQL Index & Chat History
    print("\nStep 1: Wiping PostgreSQL Tier 3 Index & Chat History...")
    wipe_postgres_indexes()

    # 2. Sanitize & Deduplicate Tier 2 State YAML
    print("\nStep 2: Sanitizing & Deduplicating Tier 2 State YAML...")
    t2_initial = 0
    t2_final = 0
    if TIER2_FILE.exists():
        with open(TIER2_FILE, "r", encoding="utf-8") as f:
            raw_t2 = yaml.safe_load(f) or []
        t2_initial = len(raw_t2)
        clean_t2 = sanitize_and_deduplicate_qa_list(raw_t2)
        t2_final = len(clean_t2)
        
        with open(TIER2_FILE, "w", encoding="utf-8") as f:
            yaml.dump(clean_t2, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"  [OK] Tier 2 State: Cleaned from {t2_initial} -> {t2_final} canonical QA items.")

    # 3. Sanitize & Deduplicate Tier 3 Entity YAMLs
    print("\nStep 3: Sanitizing & Deduplicating Tier 3 Entity YAMLs...")
    t3_entities_count = 0
    t3_initial_items = 0
    t3_final_items = 0

    if TIER3_DIR.exists():
        for file_path in TIER3_DIR.glob("*.yaml"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_items = yaml.safe_load(f) or []
                if not raw_items:
                    file_path.unlink()
                    continue

                t3_initial_items += len(raw_items)
                clean_items = sanitize_and_deduplicate_qa_list(raw_items)

                if not clean_items:
                    file_path.unlink()
                    logger.info(f"Removed empty entity file: {file_path.name}")
                else:
                    with open(file_path, "w", encoding="utf-8") as f:
                        yaml.dump(clean_items, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
                    t3_final_items += len(clean_items)
                    t3_entities_count += 1
            except Exception as e:
                logger.error(f"Error sanitizing {file_path}: {e}")

        print(f"  [OK] Tier 3 Entities: {t3_entities_count} entity files sanitized ({t3_initial_items} -> {t3_final_items} QA items).")

    # 4. Re-index Tier 3 into PostgreSQL
    print("\nStep 4: Re-indexing Sanitized Tier 3 Memory into PostgreSQL...")
    synced_db_count = sync_tier3_to_postgres()
    print(f"  [OK] Re-indexed {synced_db_count} clean Tier 3 QA items into PostgreSQL FTS index.")

    print("\n================ RE-INDEXING & SANITIZATION COMPLETED ================")
    print(f"Summary:")
    print(f"  - Tier 2 State QA Items: {t2_final} (deduplicated from {t2_initial})")
    print(f"  - Tier 3 Entities Count: {t3_entities_count}")
    print(f"  - Tier 3 Total QA Items: {t3_final_items} (deduplicated from {t3_initial_items})")
    print(f"  - PostgreSQL DB Synced Items: {synced_db_count}")
    print("======================================================================")

if __name__ == "__main__":
    run_memory_sanitization_and_reindex()
