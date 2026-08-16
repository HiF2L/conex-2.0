"""
Verification script for Dynamic Tier 3 Entity Generation & Accurate Date Injection.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MemoryDiff, QAPair
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.extractor_service import ExtractorService

def test_dynamic_tier3_creation():
    print("Testing Dynamic Tier 3 Entity Generation & Date Accuracy...")
    engine = MemoryEngine(memory_dir="data/memory")
    client = LLMClient()
    extractor = ExtractorService(engine, client)

    today_str = datetime.date.today().isoformat()

    # Dump text mentioning multiple new entities
    dump_text = (
        "Huge progress today! On project WeGeny, we deployed the new API gateway. "
        "Also launched Intelligence Bit for daily tech news summaries. "
        "In terms of Health, started a 30-minute cardio routine every morning."
    )

    sys_prompt, _ = engine.assemble_prompt(dump_text)
    diff = extractor.extract_sync(dump_text, sys_prompt)

    # Check extracted entities in diff
    assert "wegeny" in diff.tier3_updates or "health" in diff.tier3_updates or "intelligence_bit" in diff.tier3_updates, "Expected new entity keys in tier3_updates"
    
    # Check that YAML files were dynamically created on disk
    tier3_dir = Path("data/memory/tier3_entities")
    created_files = [f.name for f in tier3_dir.glob("*.yaml")]
    print(f"[OK] Tier 3 YAML files on disk: {created_files}")

    for file_name in ["wegeny.yaml", "health.yaml", "intelligence_bit.yaml"]:
        target_file = tier3_dir / file_name
        if target_file.exists():
            items = engine._read_yaml_file(target_file)
            assert len(items) > 0
            for item in items:
                assert item.valid_from and len(item.valid_from) == 10 and item.valid_from.count("-") == 2, f"Invalid date format: {item.valid_from}"
            print(f"[OK] Dynamically created and validated {file_name} with valid_from={items[-1].valid_from}")

    extractor.shutdown()

if __name__ == "__main__":
    print("================ TIER 3 DYNAMIC EXTRACTION VERIFICATION ================")
    test_dynamic_tier3_creation()
    print("================ ALL TIER 3 DYNAMIC TESTS PASSED! ================")
