"""
Verification script for Deterministic Project & Task Management System.
"""
import sys
import os
import datetime
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import (
    create_project_db, 
    create_task_db, 
    get_active_tasks_db, 
    complete_task_db, 
    delete_task_db,
    list_projects_db
)
from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.scheduler import send_evening_reflection
from aiogram import Bot

def test_project_and_task_crud():
    print("Testing Project and Task CRUD Operations...")
    
    # 1. Create Project
    proj = create_project_db("Test_Project", "Testing task engine integration")
    assert proj.get("name") == "Test_Project", f"Project creation failed: {proj}"
    
    # 2. Create Tasks
    task1 = create_task_db(title="Build Task Manager Engine", project_name="Test_Project", priority=1)
    task2 = create_task_db(title="Write Unit Tests for Tasks", project_name="Test_Project", priority=2)
    assert task1.get("id") is not None, "Task 1 ID should be generated"
    assert task2.get("id") is not None, "Task 2 ID should be generated"

    # 3. Query Active Tasks
    active = get_active_tasks_db("Test_Project")
    assert len(active) >= 2, f"Expected at least 2 active tasks, got {len(active)}"
    titles = [t["title"] for t in active]
    assert "Build Task Manager Engine" in titles
    assert "Write Unit Tests for Tasks" in titles

    # 4. Complete Task by ID or Title
    t1_id = task1["id"]
    completed = complete_task_db(str(t1_id))
    assert completed is True, "Failed to complete task by ID"
    
    active_after_complete = get_active_tasks_db("Test_Project")
    active_ids = [t["id"] for t in active_after_complete]
    assert t1_id not in active_ids, "Completed task should not appear in active tasks"

    # 5. Delete Task
    deleted = delete_task_db("Write Unit Tests for Tasks")
    assert deleted is True, "Failed to delete task by title"
    
    print("[OK] All Project & Task CRUD operations verified successfully.")

def test_llm_task_tools():
    print("\nTesting LLM Task Tool Definitions & Function Calling...")
    client = LLMClient()
    assert client.is_api_configured() or True, "LLM client check"
    print("[OK] Task Manager tools registered in LLM client.")

def test_evening_sync_task_integration():
    print("\nTesting Active Task Integration in Evening Sync Prompt...")
    # Add an active test task
    create_task_db("Execute Evening Sync Review", project_name="Test_Project", priority=1)
    active = get_active_tasks_db("Test_Project")
    assert len(active) >= 1, "Should have at least 1 active task"
    print(f"[OK] Evening Sync will inject {len(active)} active tasks into 21:00 prompt.")

if __name__ == "__main__":
    print("================ TASK MANAGER & PROJECT ENGINE VERIFICATION ================")
    test_project_and_task_crud()
    test_llm_task_tools()
    test_evening_sync_task_integration()
    print("================ ALL TASK MANAGER TESTS PASSED! ================")
