"""
Verification Script for Tool Leak Fix, Exact Task ID Resolution, and Evening Sync Limits.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_client import LLMClient
from src.db import (
    create_task_db,
    get_active_tasks_db,
    complete_task_db,
    update_task_db,
    delete_task_db
)

def test_sanitize_tool_leak():
    print("Testing Tool Call Output Leak Sanitizer...")
    client = LLMClient()
    
    leaked_text = (
        "Отлично, задача выполнена!\n"
        "to=complete_task {\"identifier\": \"5\"}\n"
        "to=functions.forgetmemory (json): {\"keyword\": \"tired\"}\n"
        "Чем еще могу помочь?"
    )
    cleaned = client._sanitize_tool_leak(leaked_text)
    
    assert "to=complete_task" not in cleaned, "Function leak to=complete_task should be removed"
    assert "to=functions.forgetmemory" not in cleaned, "Function leak to=functions should be removed"
    assert "Отлично, задача выполнена!" in cleaned
    assert "Чем еще могу помочь?" in cleaned
    print("[OK] _sanitize_tool_leak successfully stripped raw function execution syntax.")

def test_task_id_exact_matching():
    print("\nTesting Exact Task ID Resolution (Preventing Substring Matches)...")
    
    # Create two tasks: one with ID 5 (conceptually), another with "5" in title
    t1 = create_task_db("Deploy API Gateway", project_name="Fix_Test")
    t2 = create_task_db("Design 5 UI Mockups", project_name="Fix_Test")
    
    t1_id = t1["id"]
    t2_id = t2["id"]

    # Completing t1 by exact string representation of ID
    completed = complete_task_db(str(t1_id))
    assert completed is True, f"Failed to complete task ID {t1_id}"

    # Verify t2 (which has '5' in title) is STILL TODO and not accidentally completed
    t2_active = get_active_tasks_db(project_name="Fix_Test", status="todo")
    t2_active_ids = [t["id"] for t in t2_active]
    
    assert t2_id in t2_active_ids, f"Task {t2_id} ('Design 5 UI Mockups') should NOT be completed when ID {t1_id} was requested!"
    print(f"[OK] Task ID {t1_id} completed cleanly without affecting task {t2_id} ('Design 5 UI Mockups').")

def test_update_task_db():
    print("\nTesting update_task_db Functionality...")
    t = create_task_db("Initial Task Title", project_name="Fix_Test", priority=3)
    t_id = t["id"]

    success = update_task_db(str(t_id), title="Updated Task Title", priority=1, status="in_progress")
    assert success is True, "update_task_db should return True"

    active = get_active_tasks_db(project_name="Fix_Test", status="all")
    updated_task = next(item for item in active if item["id"] == t_id)
    assert updated_task["title"] == "Updated Task Title"
    assert updated_task["priority"] == 1
    assert updated_task["status"] == "in_progress"
    print("[OK] update_task_db updated title, priority, and status successfully.")

def test_evening_sync_today_filter():
    print("\nTesting Evening Sync Today Only & Limit Filter...")
    tasks = get_active_tasks_db(today_only=True, limit=10)
    assert len(tasks) <= 10, f"Expected at most 10 tasks, got {len(tasks)}"
    print(f"[OK] get_active_tasks_db(today_only=True, limit=10) returned {len(tasks)} items.")

if __name__ == "__main__":
    print("================ CRITICAL BUGFIX VERIFICATION ================")
    test_sanitize_tool_leak()
    test_task_id_exact_matching()
    test_update_task_db()
    test_evening_sync_today_filter()
    print("================ ALL BUGFIX TESTS PASSED! ================")
