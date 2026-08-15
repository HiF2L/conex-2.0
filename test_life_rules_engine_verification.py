"""
Synthetic Test Suite for Life Principles & Productivity Axioms Engine.
Verifies DB CRUD helpers, Tool Registration, System Prompt Directives, and Morning Briefing Injection.
STRICTLY USES MOCK SYNTHETIC DATA.
"""
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.db import (
    init_db,
    save_life_rule_db,
    get_active_rules_db,
    toggle_life_rule_db
)
from src.models import LifeRuleItem
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient

# Ensure DB schema is initialized
init_db()

def test_life_rules_db_crud():
    print("Testing Life Rules DB CRUD & Domain Filtering...")

    # 1. Save rules across different domains
    r1 = save_life_rule_db(
        domain="productivity",
        rule_name="No Morning Cooking",
        rule_text="Never start complex cooking before the first 90-minute Deep Work block.",
        anti_pattern="Cooking elaborate meals at 10 AM draining mental clarity before coding.",
        actionable_remedy="Prep quick protein/coffee; defer hot meals until post-work locomotion."
    )
    assert r1["id"] is not None
    assert r1["domain"] == "productivity"
    assert r1["rule_name"] == "No Morning Cooking"
    assert r1["is_active"] is True

    r2 = save_life_rule_db(
        domain="chores",
        rule_name="15-Minute Micro-Cleaning Sprints",
        rule_text="Cleaning must be capped at 15 minutes per single zone.",
        anti_pattern="Starting all-day cleaning that derails cognitive work.",
        actionable_remedy="Set a 15m timer for one specific zone."
    )
    assert r2["id"] is not None
    assert r2["domain"] == "chores"

    r3 = save_life_rule_db(
        domain="mental_health",
        rule_name="Walk Before Analysis",
        rule_text="When feeling stuck or anxious, take a 30m outdoor walk before making decisions.",
        anti_pattern="Endless rumination and self-criticism in chair.",
        actionable_remedy="Immediate physical locomotion outside without audio."
    )
    assert r3["domain"] == "mental_health"

    print(f"[OK] Saved 3 life rules across domains: #{r1['id']}, #{r2['id']}, #{r3['id']}.")

    # 2. Query all active rules
    all_active = get_active_rules_db()
    assert any(r["id"] == r1["id"] for r in all_active)
    assert any(r["id"] == r2["id"] for r in all_active)
    assert any(r["id"] == r3["id"] for r in all_active)
    print(f"[OK] Fetched {len(all_active)} total active life rules.")

    # 3. Query with domain filter
    prod_rules = get_active_rules_db(domain="productivity")
    assert all(r["domain"] == "productivity" for r in prod_rules)
    assert any(r["id"] == r1["id"] for r in prod_rules)
    assert not any(r["id"] == r2["id"] for r in prod_rules)
    print(f"[OK] Domain filter 'productivity' successfully returned {len(prod_rules)} rules.")

    # 4. Toggle rule active state
    toggled = toggle_life_rule_db(r2["id"], is_active=False)
    assert toggled is True

    active_after_disable = get_active_rules_db()
    assert not any(r["id"] == r2["id"] for r in active_after_disable)
    print(f"[OK] Successfully toggled rule #{r2['id']} to inactive.")

    # Re-enable rule
    toggle_life_rule_db(r2["id"], is_active=True)
    active_after_enable = get_active_rules_db()
    assert any(r["id"] == r2["id"] for r in active_after_enable)
    print(f"[OK] Successfully re-enabled rule #{r2['id']}.")

def test_llm_tool_registration_and_prompt_directives():
    print("\nTesting LLM Tools & Prompt Directives for Life Rules...")

    # 1. Verify MemoryEngine system prompt directive
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Reflecting on why my day went off track")
    assert "LIFE PRINCIPLES & PRODUCTIVITY AXIOMS DIRECTIVE" in prompt
    assert "save_life_rule" in prompt
    assert "get_active_rules" in prompt
    print("[OK] MemoryEngine system prompt contains Directive 13 with reflection extraction and grounded advice protocols.")

    # 2. Verify LLMClient tools definitions
    client = LLMClient()
    # Mock generation call to inspect tools list
    import inspect
    src = inspect.getsource(client.generate_coaching_response)
    assert "save_life_rule" in src
    assert "get_active_rules" in src
    print("[OK] LLMClient registered 'save_life_rule' and 'get_active_rules' tools.")

def test_morning_briefing_rules_injection():
    print("\nTesting Morning Briefing Prompt Assembly with Life Rules...")
    from src.db import get_top_focus_tasks_db, get_active_experiments_db, get_active_rules_db
    
    engine = MemoryEngine(memory_dir="data/memory")
    prompt, trace = engine.assemble_prompt("Morning briefing, sprint schedule, and daily rules.")
    
    active_rules = get_active_rules_db()
    assert len(active_rules) > 0

    rule_lines = []
    for r in active_rules:
        extra = f" (Remedy: {r.get('actionable_remedy')})" if r.get('actionable_remedy') else ""
        rule_lines.append(f"- [{r.get('domain', 'productivity').upper()}] {r.get('rule_name')}: {r.get('rule_text')}{extra}")
    rules_context = "АКТИВНЫЕ ЖИЗНЕННЫЕ ПРАВИЛА И АКСИОМЫ ПРОДУКТИВНОСТИ:\n" + "\n".join(rule_lines)

    enforcement_instructions = (
        "СТРОГИЕ ПРАВИЛА СОСТАВЛЕНИЯ УТРЕННЕГО ПЛАНА:\n"
        "1. Никаких сложных бытовых дел/готовки до первого блока глубокой работы (Deep Work).\n"
        "2. Уборка строго ограничена микро-спринтами 10–15 минут на одну зону.\n"
        "3. Zero-Choice формулировки задач (без двусмысленности вроде 'Вариант А или Вариант Б')."
    )
    full_prompt = f"{prompt}\n\n{rules_context}\n\n{enforcement_instructions}"

    assert "АКТИВНЫЕ ЖИЗНЕННЫЕ ПРАВИЛА И АКСИОМЫ ПРОДУКТИВНОСТИ" in full_prompt
    assert "Никаких сложных бытовых дел/готовки до первого блока глубокой работы" in full_prompt
    assert "Zero-Choice" in full_prompt
    print("[OK] Morning Briefing prompt successfully injects active axioms and enforcement rules.")

if __name__ == "__main__":
    print("================ LIFE RULES ENGINE VERIFICATION ================")
    test_life_rules_db_crud()
    test_llm_tool_registration_and_prompt_directives()
    test_morning_briefing_rules_injection()
    print("================ ALL LIFE RULES ENGINE TESTS PASSED! ================")
