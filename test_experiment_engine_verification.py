"""
Synthetic Test Suite for Sprints, A/B Testing & Dynamic Daily Scheduler.
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

from src.db import (
    create_experiment_db,
    get_active_experiments_db,
    advance_experiment_phase_db
)
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient

def test_experiment_db_crud_and_phase_advancement():
    print("Testing Experiment DB CRUD & Phase Advancement...")

    # Create Sprint experiment
    sprint_exp = create_experiment_db(
        title="2-Week Python Coding Sprint",
        type="SPRINT",
        hypothesis_a="Coding 2 hours daily builds sustainable momentum",
        duration_days=14,
        daily_actions=["Code 2 hours", "Log commit"]
    )
    assert sprint_exp["type"] == "SPRINT"
    assert sprint_exp["phase"] == "PHASE_A"
    assert sprint_exp["status"] == "active"
    print(f"[OK] Sprint created: ID #{sprint_exp['id']} '{sprint_exp['title']}'")

    # Create A/B Test experiment
    ab_exp = create_experiment_db(
        title="Dietary Focus Experiment: Keto vs Mediterranean",
        type="AB_TEST",
        hypothesis_a="Low-carb keto eliminates afternoon brain fog",
        hypothesis_b="Mediterranean diet maintains stable energy",
        duration_days=14,
        daily_actions=["1 keto meal", "no refined sugar"]
    )
    assert ab_exp["type"] == "AB_TEST"
    assert ab_exp["phase"] == "PHASE_A"
    print(f"[OK] A/B Test created: ID #{ab_exp['id']} '{ab_exp['title']}'")

    # Fetch active experiments
    active_list = get_active_experiments_db()
    assert any(e["id"] == ab_exp["id"] for e in active_list)
    print(f"[OK] Fetched {len(active_list)} active experiments.")

    # Advance A/B test phase: PHASE_A -> PHASE_B
    advanced_exp = advance_experiment_phase_db(ab_exp["id"])
    assert advanced_exp["phase"] == "PHASE_B"
    assert advanced_exp["status"] == "active"
    print(f"[OK] Advanced A/B test #{ab_exp['id']} to PHASE_B.")

    # Advance again: PHASE_B -> COMPLETED
    completed_exp = advance_experiment_phase_db(ab_exp["id"])
    assert completed_exp["phase"] == "COMPLETED"
    assert completed_exp["status"] == "completed"
    print(f"[OK] Advanced A/B test #{ab_exp['id']} to COMPLETED.")

def test_llm_tools_and_prompt_directives():
    print("\nTesting LLM Tools Registration & Prompt Directives for Experiments...")

    # Verify MemoryEngine directive
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("I want to try keto for 2 weeks")
    assert "EXPERIMENTS & DAILY SCHEDULING DIRECTIVE" in prompt
    assert "create_experiment" in prompt
    assert "MANDATORY ON-DEMAND SCHEDULING TOOL-CHAINING" in prompt
    assert "get_active_rules(domain='productivity')" in prompt
    assert "list_tasks(status='todo')" in prompt
    assert "get_active_experiments()" in prompt
    print("[OK] System Prompt directives for Experiment Engine and On-Demand Daily Scheduling Tool-Chaining verified.")

if __name__ == "__main__":
    print("================ EXPERIMENT ENGINE VERIFICATION ================")
    test_experiment_db_crud_and_phase_advancement()
    test_llm_tools_and_prompt_directives()
    print("================ ALL EXPERIMENT ENGINE TESTS PASSED! ================")
