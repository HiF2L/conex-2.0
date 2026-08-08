"""
PostgreSQL database layer for LifeOS Personal Memory Agent.
Provides Hybrid Vector (pgvector + ProxyAPI Embeddings) + Full-Text Search (FTS) for Tier 3 memory (search_memory tool) 
and sliding window conversation history.
Supports automatic local YAML fallback if PostgreSQL connection is unavailable.
"""
import os
import yaml
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://wegeny_admin:wegeny_password@localhost:5432/lifeos_db").strip()

def _get_connection():
    """Attempt to establish a psycopg 3 or psycopg2 connection to PostgreSQL."""
    try:
        import psycopg
        return psycopg.connect(DATABASE_URL, connect_timeout=3)
    except ImportError:
        try:
            import psycopg2
            return psycopg2.connect(DATABASE_URL, connect_timeout=3)
        except Exception as e:
            logger.debug(f"PostgreSQL psycopg2 connection failed: {e}")
            return None
    except Exception as e:
        logger.debug(f"PostgreSQL connection offline: {e}")
        return None

def init_db() -> bool:
    """
    Initialize PostgreSQL schema, pgvector extension, and GIN/vector indexes if available.
    """
    conn = _get_connection()
    if not conn:
        logger.info("PostgreSQL offline. Running with local YAML & memory fallback.")
        return False

    try:
        with conn.cursor() as cur:
            # 1. Try initializing pgvector extension
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                has_pgvector = True
            except Exception as ext_err:
                logger.info(f"pgvector extension not installed in PostgreSQL instance: {ext_err}. Using text/FTS mode.")
                has_pgvector = False
                conn.rollback()

            # 2. Chat history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history (user_id, id DESC);
            """)

            # 3. Scheduled Pings table for Proactive Engine
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_pings (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    scheduled_at TIMESTAMP NOT NULL,
                    event_type VARCHAR(50) NOT NULL DEFAULT 'event_followup',
                    context_text TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_pings_status ON scheduled_pings (status, scheduled_at);
            """)

            # 4. Projects and Tasks tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    priority INT DEFAULT 2,
                    status VARCHAR(20) DEFAULT 'todo',
                    due_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status, project_id);
            """)

            # 3. Tier 3 memory index table with FTS tsvector and optional vector column
            if has_pgvector:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tier3_memory_index (
                        id VARCHAR(100) PRIMARY KEY,
                        entity_name VARCHAR(100) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        weight FLOAT DEFAULT 1.0,
                        valid_from VARCHAR(20),
                        search_vector tsvector,
                        embedding vector(1536)
                    );
                    CREATE INDEX IF NOT EXISTS idx_tier3_search ON tier3_memory_index USING GIN (search_vector);
                """)
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tier3_memory_index (
                        id VARCHAR(100) PRIMARY KEY,
                        entity_name VARCHAR(100) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        weight FLOAT DEFAULT 1.0,
                        valid_from VARCHAR(20),
                        search_vector tsvector,
                        embedding_json TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_tier3_search ON tier3_memory_index USING GIN (search_vector);
                """)

            conn.commit()
            logger.info("PostgreSQL database schema & indexes initialized successfully.")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL schema: {e}")
        return False
    finally:
        conn.close()

# In-memory fallback storage for scheduled pings when DB is offline
_in_memory_scheduled_pings: List[Dict[str, Any]] = []

def save_scheduled_ping(user_id: int, scheduled_at_iso: str, event_type: str, context_text: str) -> int:
    """Saves a scheduled event ping for proactive follow-up."""
    if not user_id or not scheduled_at_iso or not context_text:
        return 0

    ping_data = {
        "id": len(_in_memory_scheduled_pings) + 1,
        "user_id": user_id,
        "scheduled_at": scheduled_at_iso,
        "event_type": event_type or "event_followup",
        "context_text": context_text,
        "status": "pending",
        "created_at": datetime.datetime.now().isoformat()
    }
    _in_memory_scheduled_pings.append(ping_data)

    conn = _get_connection()
    if not conn:
        return ping_data["id"]

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scheduled_pings (user_id, scheduled_at, event_type, context_text, status)
                VALUES (%s, %s, %s, %s, 'pending')
                RETURNING id;
            """, (user_id, scheduled_at_iso, event_type or "event_followup", context_text))
            inserted_id = cur.fetchone()[0]
            conn.commit()
            return inserted_id
    except Exception as e:
        logger.warning(f"Failed to save scheduled ping to PostgreSQL: {e}")
        return ping_data["id"]
    finally:
        conn.close()

def get_due_pings(target_user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve pending pings scheduled at or before NOW()."""
    now = datetime.datetime.now()
    now_iso = now.isoformat()

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if target_user_id:
                    cur.execute("""
                        SELECT id, user_id, scheduled_at, event_type, context_text, status
                        FROM scheduled_pings
                        WHERE status = 'pending' AND scheduled_at <= %s AND user_id = %s
                        ORDER BY scheduled_at ASC;
                    """, (now, target_user_id))
                else:
                    cur.execute("""
                        SELECT id, user_id, scheduled_at, event_type, context_text, status
                        FROM scheduled_pings
                        WHERE status = 'pending' AND scheduled_at <= %s
                        ORDER BY scheduled_at ASC;
                    """, (now,))
                rows = cur.fetchall()
                if rows:
                    return [
                        {
                            "id": r[0],
                            "user_id": r[1],
                            "scheduled_at": str(r[2]),
                            "event_type": r[3],
                            "context_text": r[4],
                            "status": r[5]
                        }
                        for r in rows
                    ]
        except Exception as e:
            logger.warning(f"Failed to fetch due pings from PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory fallback
    due = []
    for p in _in_memory_scheduled_pings:
        if p["status"] == "pending" and p["scheduled_at"] <= now_iso:
            if not target_user_id or p["user_id"] == target_user_id:
                due.append(p)
    return due

def mark_ping_status(ping_id: int, status: str = "executed") -> None:
    """Mark ping status as executed or cancelled."""
    for p in _in_memory_scheduled_pings:
        if p["id"] == ping_id:
            p["status"] = status

    conn = _get_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE scheduled_pings SET status = %s WHERE id = %s;
            """, (status, ping_id))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to update ping status in PostgreSQL: {e}")
    finally:
        conn.close()

def get_pings_count_today(user_id: int) -> int:
    """Get count of pings executed today for user_id."""
    today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_iso = today_start.isoformat()

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM scheduled_pings
                    WHERE user_id = %s AND status = 'executed' AND scheduled_at >= %s;
                """, (user_id, today_start))
                return cur.fetchone()[0]
        except Exception as e:
            logger.warning(f"Failed to get pings count today from PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory count
    return sum(
        1 for p in _in_memory_scheduled_pings
        if p["user_id"] == user_id and p["status"] == "executed" and p["scheduled_at"] >= today_start_iso
    )

def sync_tier3_to_postgres(memory_dir: str = "data/memory") -> int:
    """
    Scans data/memory/tier3_entities/*.yaml and populates/upserts tier3_memory_index in PostgreSQL with ProxyAPI embeddings.
    """
    from src.llm_client import LLMClient
    llm_client = LLMClient()

    conn = _get_connection()
    if not conn:
        return 0

    tier3_path = Path(memory_dir) / "tier3_entities"
    if not tier3_path.exists():
        return 0

    synced_count = 0
    try:
        with conn.cursor() as cur:
            # Check if embedding column is vector type or text type
            cur.execute("""
                SELECT data_type FROM information_schema.columns 
                WHERE table_name = 'tier3_memory_index' AND column_name = 'embedding';
            """)
            has_vector_col = bool(cur.fetchone())

            for file_path in tier3_path.glob("*.yaml"):
                entity_name = file_path.stem.lower()
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_data = yaml.safe_load(f)
                    if not raw_data or not isinstance(raw_data, list):
                        continue

                    for item in raw_data:
                        qa_id = str(item.get("id", f"{entity_name}_{synced_count}"))
                        question = str(item.get("question", ""))
                        answer = str(item.get("answer", ""))
                        weight = float(item.get("weight", 1.0))
                        valid_from = item.get("valid_from")

                        # Generate embedding via ProxyAPI
                        embed_text = f"{entity_name}: {question} - {answer}"
                        vec = llm_client.generate_embedding(embed_text)

                        if has_vector_col:
                            vec_str = str(vec) if vec else None
                            cur.execute("""
                                INSERT INTO tier3_memory_index (id, entity_name, question, answer, weight, valid_from, search_vector, embedding)
                                VALUES (%s, %s, %s, %s, %s, %s,
                                    to_tsvector('russian', COALESCE(%s, '') || ' ' || COALESCE(%s, '') || ' ' || COALESCE(%s, '')) ||
                                    to_tsvector('english', COALESCE(%s, '') || ' ' || COALESCE(%s, '') || ' ' || COALESCE(%s, '')),
                                    %s::vector
                                )
                                ON CONFLICT (id) DO UPDATE SET
                                    entity_name = EXCLUDED.entity_name,
                                    question = EXCLUDED.question,
                                    answer = EXCLUDED.answer,
                                    weight = EXCLUDED.weight,
                                    valid_from = EXCLUDED.valid_from,
                                    search_vector = EXCLUDED.search_vector,
                                    embedding = EXCLUDED.embedding;
                            """, (
                                qa_id, entity_name, question, answer, weight, valid_from,
                                question, answer, entity_name,
                                question, answer, entity_name,
                                vec_str
                            ))
                        else:
                            vec_json = json.dumps(vec) if vec else None
                            cur.execute("""
                                INSERT INTO tier3_memory_index (id, entity_name, question, answer, weight, valid_from, search_vector, embedding_json)
                                VALUES (%s, %s, %s, %s, %s, %s,
                                    to_tsvector('russian', COALESCE(%s, '') || ' ' || COALESCE(%s, '') || ' ' || COALESCE(%s, '')) ||
                                    to_tsvector('english', COALESCE(%s, '') || ' ' || COALESCE(%s, '') || ' ' || COALESCE(%s, '')),
                                    %s
                                )
                                ON CONFLICT (id) DO UPDATE SET
                                    entity_name = EXCLUDED.entity_name,
                                    question = EXCLUDED.question,
                                    answer = EXCLUDED.answer,
                                    weight = EXCLUDED.weight,
                                    valid_from = EXCLUDED.valid_from,
                                    search_vector = EXCLUDED.search_vector,
                                    embedding_json = EXCLUDED.embedding_json;
                            """, (
                                qa_id, entity_name, question, answer, weight, valid_from,
                                question, answer, entity_name,
                                question, answer, entity_name,
                                vec_json
                            ))
                        synced_count += 1
                except Exception as file_err:
                    logger.warning(f"Failed to sync Tier 3 YAML {file_path}: {file_err}")

            conn.commit()
            logger.info(f"Synced {synced_count} Tier 3 memory items to PostgreSQL search index with ProxyAPI embeddings.")
            return synced_count
    except Exception as e:
        logger.error(f"Error during Tier 3 PostgreSQL sync: {e}")
        return 0
    finally:
        conn.close()

def search_tier3_memory(query: str, top_k: int = 4) -> str:
    """
    Executes PostgreSQL Hybrid Vector + Full-Text Search query to retrieve relevant Tier 3 memory items.
    Falls back to local YAML search if PostgreSQL is offline or yields no rows.
    """
    from src.llm_client import LLMClient
    clean_query = query.strip()
    if not clean_query:
        return "No search query provided."

    conn = _get_connection()
    results = []

    if conn:
        try:
            with conn.cursor() as cur:
                # Generate query vector via ProxyAPI
                llm_client = LLMClient()
                q_vec = llm_client.generate_embedding(clean_query)

                # Check if pgvector column exists
                cur.execute("""
                    SELECT data_type FROM information_schema.columns 
                    WHERE table_name = 'tier3_memory_index' AND column_name = 'embedding';
                """)
                has_vector_col = bool(cur.fetchone())

                like_pattern = f"%{clean_query}%"

                if has_vector_col and q_vec:
                    q_vec_str = str(q_vec)
                    cur.execute("""
                        SELECT entity_name, question, answer, weight, valid_from,
                               (ts_rank(search_vector, websearch_to_tsquery('russian', %s)) +
                                ts_rank(search_vector, websearch_to_tsquery('english', %s)) +
                                COALESCE(1.0 - (embedding <=> %s::vector), 0.0)) AS rank
                        FROM tier3_memory_index
                        ORDER BY rank DESC, weight DESC
                        LIMIT %s;
                    """, (clean_query, clean_query, q_vec_str, top_k))
                else:
                    cur.execute("""
                        SELECT entity_name, question, answer, weight, valid_from,
                               (ts_rank(search_vector, websearch_to_tsquery('russian', %s)) +
                                ts_rank(search_vector, websearch_to_tsquery('english', %s))) AS rank
                        FROM tier3_memory_index
                        WHERE search_vector @@ websearch_to_tsquery('russian', %s)
                           OR search_vector @@ websearch_to_tsquery('english', %s)
                           OR question ILIKE %s OR answer ILIKE %s OR entity_name ILIKE %s
                        ORDER BY rank DESC, weight DESC
                        LIMIT %s;
                    """, (clean_query, clean_query, clean_query, clean_query, like_pattern, like_pattern, like_pattern, top_k))
                
                rows = cur.fetchall()
                for r in rows:
                    results.append({
                        "entity": r[0],
                        "question": r[1],
                        "answer": r[2],
                        "weight": r[3],
                        "valid_from": r[4]
                    })
        except Exception as e:
            logger.warning(f"PostgreSQL hybrid search error: {e}. Switching to YAML search fallback.")
        finally:
            conn.close()

    # Fallback to local YAML search if DB returned no results
    if not results:
        results = _fallback_yaml_search(clean_query, top_k=top_k)

    if not results:
        return f"No long-term memory entries found matching '{clean_query}'."

    output_lines = [f"Found {len(results)} relevant entries in Tier 3 Memory for '{clean_query}':\n"]
    for idx, item in enumerate(results, 1):
        output_lines.append(
            f"{idx}. [Entity: {item['entity'].upper()}] Question: {item['question']}\n"
            f"   Answer: {item['answer']} (weight: {item.get('weight', 1.0)})"
        )

    return "\n\n".join(output_lines)

def _fallback_yaml_search(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Local YAML search fallback when PostgreSQL is offline or FTS finds no matches."""
    tier3_dir = Path("data/memory/tier3_entities")
    if not tier3_dir.exists():
        return []

    q_lower = query.lower()
    matches = []

    for file_path in tier3_dir.glob("*.yaml"):
        entity_name = file_path.stem.lower()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
            if not raw_data or not isinstance(raw_data, list):
                continue

            for item in raw_data:
                q = str(item.get("question", ""))
                a = str(item.get("answer", ""))
                
                # Check entity name or text match
                if entity_name in q_lower or q_lower in entity_name or q_lower in q.lower() or q_lower in a.lower():
                    matches.append({
                        "entity": entity_name,
                        "question": q,
                        "answer": a,
                        "weight": float(item.get("weight", 1.0)),
                        "valid_from": item.get("valid_from")
                    })
        except Exception:
            pass

    # Sort matches by weight descending
    matches = sorted(matches, key=lambda x: x.get("weight", 1.0), reverse=True)
    return matches[:top_k]

# In-memory sliding chat history fallback (user_id -> List[Dict[str, str]])
_in_memory_chat_history: Dict[int, List[Dict[str, str]]] = {}

def save_chat_message(user_id: int, role: str, content: str) -> None:
    """Save user or assistant chat turn into chat_history table with in-memory fallback."""
    if not user_id or not content:
        return

    # Always maintain in-memory sliding history
    if user_id not in _in_memory_chat_history:
        _in_memory_chat_history[user_id] = []
    _in_memory_chat_history[user_id].append({"role": role, "content": content})
    if len(_in_memory_chat_history[user_id]) > 20:
        _in_memory_chat_history[user_id] = _in_memory_chat_history[user_id][-20:]

    conn = _get_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_history (user_id, role, content)
                VALUES (%s, %s, %s);
            """, (user_id, role, content))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to save chat message to PostgreSQL: {e}")
    finally:
        conn.close()

def get_recent_chat_history(user_id: int, limit: int = 8) -> List[Dict[str, str]]:
    """
    Retrieve the last `limit` chat turns for user_id in chronological order.
    Uses PostgreSQL if available, otherwise falls back to in-memory history.
    """
    if not user_id:
        return []

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT role, content FROM (
                        SELECT role, content, id FROM chat_history
                        WHERE user_id = %s
                        ORDER BY id DESC
                        LIMIT %s
                    ) sub ORDER BY id ASC;
                """, (user_id, limit))
                rows = cur.fetchall()
                if rows:
                    return [{"role": r[0], "content": r[1]} for r in rows]
        except Exception as e:
            logger.warning(f"Failed to retrieve chat history from PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory fallback
    mem_hist = _in_memory_chat_history.get(user_id, [])
    return mem_hist[-limit:] if mem_hist else []

def delete_tier3_memory_by_keyword(keyword: str) -> int:
    """Delete Tier 3 memory items matching keyword from PostgreSQL table."""
    clean_kw = keyword.strip()
    if not clean_kw:
        return 0

    conn = _get_connection()
    if not conn:
        return 0

    deleted_count = 0
    try:
        with conn.cursor() as cur:
            like_pattern = f"%{clean_kw}%"
            cur.execute("""
                DELETE FROM tier3_memory_index
                WHERE entity_name ILIKE %s OR question ILIKE %s OR answer ILIKE %s OR id ILIKE %s;
            """, (like_pattern, like_pattern, like_pattern, like_pattern))
            deleted_count = cur.rowcount
            conn.commit()
            db_deleted = delete_tier3_memory_by_keyword(clean_kw)
            logger.info(f"PostgreSQL Tier 3 deletion for '{clean_kw}' removed {db_deleted} DB rows.")
            return db_deleted
    except Exception as e:
        logger.warning(f"Failed to delete Tier 3 memory from PostgreSQL: {e}")
        return 0
    finally:
        conn.close()

# In-memory fallback storage for projects and tasks
_in_memory_projects: List[Dict[str, Any]] = []
_in_memory_tasks: List[Dict[str, Any]] = []

def create_project_db(name: str, description: str = "") -> Dict[str, Any]:
    """Create a new project in PostgreSQL with in-memory fallback."""
    clean_name = name.strip()
    if not clean_name:
        return {"error": "Project name cannot be empty"}

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO projects (name, description, status)
                    VALUES (%s, %s, 'active')
                    ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description, status = 'active'
                    RETURNING id, name, description, status, created_at;
                """, (clean_name, description))
                row = cur.fetchone()
                conn.commit()
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "status": row[3],
                    "created_at": str(row[4])
                }
        except Exception as e:
            logger.warning(f"Failed to create project in PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory fallback
    for p in _in_memory_projects:
        if p["name"].lower() == clean_name.lower():
            p["description"] = description
            p["status"] = "active"
            return p

    proj = {
        "id": len(_in_memory_projects) + 1,
        "name": clean_name,
        "description": description,
        "status": "active",
        "created_at": datetime.datetime.now().isoformat()
    }
    _in_memory_projects.append(proj)
    return proj

def list_projects_db() -> List[Dict[str, Any]]:
    """List all projects."""
    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, description, status, created_at FROM projects ORDER BY id ASC;
                """)
                rows = cur.fetchall()
                if rows:
                    return [
                        {"id": r[0], "name": r[1], "description": r[2], "status": r[3], "created_at": str(r[4])}
                        for r in rows
                    ]
        except Exception as e:
            logger.warning(f"Failed to list projects from PostgreSQL: {e}")
        finally:
            conn.close()
    return _in_memory_projects

def create_task_db(title: str, project_name: Optional[str] = None, priority: int = 2, due_date: Optional[str] = None, description: str = "") -> Dict[str, Any]:
    """Create a new task under an optional project."""
    clean_title = title.strip()
    if not clean_title:
        return {"error": "Task title cannot be empty"}

    project_id = None
    if project_name:
        proj = create_project_db(project_name)
        project_id = proj.get("id")

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tasks (project_id, title, description, priority, status, due_date)
                    VALUES (%s, %s, %s, %s, 'todo', %s)
                    RETURNING id, project_id, title, description, priority, status, due_date, created_at;
                """, (project_id, clean_title, description, priority, due_date if due_date else None))
                row = cur.fetchone()
                conn.commit()
                return {
                    "id": row[0],
                    "project_id": row[1],
                    "project_name": project_name,
                    "title": row[2],
                    "description": row[3],
                    "priority": row[4],
                    "status": row[5],
                    "due_date": str(row[6]) if row[6] else None,
                    "created_at": str(row[7])
                }
        except Exception as e:
            logger.warning(f"Failed to create task in PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory fallback
    task = {
        "id": len(_in_memory_tasks) + 1,
        "project_id": project_id,
        "project_name": project_name,
        "title": clean_title,
        "description": description,
        "priority": priority,
        "status": "todo",
        "due_date": due_date,
        "created_at": datetime.datetime.now().isoformat()
    }
    _in_memory_tasks.append(task)
    return task

def get_active_tasks_db(project_name: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve tasks optionally filtered by project_name and status ('todo', 'in_progress', 'done', 'all')."""
    clean_status = (status or "").lower().strip()
    if clean_status == "all":
        status_filter = ("todo", "in_progress", "done", "cancelled")
    elif clean_status:
        status_filter = (clean_status,)
    else:
        status_filter = ("todo", "in_progress")

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if project_name:
                    cur.execute("""
                        SELECT t.id, t.title, t.description, t.priority, t.status, t.due_date, p.name
                        FROM tasks t
                        LEFT JOIN projects p ON t.project_id = p.id
                        WHERE t.status = ANY(%s) AND LOWER(p.name) = LOWER(%s)
                        ORDER BY t.priority ASC, t.id ASC;
                    """, (list(status_filter), project_name.strip()))
                else:
                    cur.execute("""
                        SELECT t.id, t.title, t.description, t.priority, t.status, t.due_date, p.name
                        FROM tasks t
                        LEFT JOIN projects p ON t.project_id = p.id
                        WHERE t.status = ANY(%s)
                        ORDER BY t.priority ASC, t.id ASC;
                    """, (list(status_filter),))
                rows = cur.fetchall()
                if rows is not None:
                    return [
                        {
                            "id": r[0],
                            "title": r[1],
                            "description": r[2],
                            "priority": r[3],
                            "status": r[4],
                            "due_date": str(r[5]) if r[5] else None,
                            "project_name": r[6]
                        }
                        for r in rows
                    ]
        except Exception as e:
            logger.warning(f"Failed to fetch active tasks from PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory fallback
    res = []
    for t in _in_memory_tasks:
        if t["status"] in status_filter:
            if not project_name or (t.get("project_name") and t["project_name"].lower() == project_name.lower()):
                res.append(t)
    return res

def complete_task_db(identifier: str) -> bool:
    """Complete a task by ID (integer) or title match."""
    clean_id = str(identifier).strip()
    if not clean_id:
        return False

    is_int = clean_id.isdigit()
    task_id = int(clean_id) if is_int else None

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if task_id is not None:
                    cur.execute("""
                        UPDATE tasks SET status = 'done', completed_at = CURRENT_TIMESTAMP WHERE id = %s;
                    """, (task_id,))
                else:
                    pattern = f"%{clean_id}%"
                    cur.execute("""
                        UPDATE tasks SET status = 'done', completed_at = CURRENT_TIMESTAMP WHERE title ILIKE %s;
                    """, (pattern,))
                updated = cur.rowcount > 0
                conn.commit()
                if updated:
                    return True
        except Exception as e:
            logger.warning(f"Failed to complete task in PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory fallback
    for t in _in_memory_tasks:
        if (task_id is not None and t["id"] == task_id) or (clean_id.lower() in t["title"].lower()):
            t["status"] = "done"
            t["completed_at"] = datetime.datetime.now().isoformat()
            return True
    return False

def delete_task_db(identifier: str) -> bool:
    """Delete a task by ID or title match."""
    clean_id = str(identifier).strip()
    if not clean_id:
        return False

    is_int = clean_id.isdigit()
    task_id = int(clean_id) if is_int else None

    conn = _get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                if task_id is not None:
                    cur.execute("""
                        DELETE FROM tasks WHERE id = %s;
                    """, (task_id,))
                else:
                    pattern = f"%{clean_id}%"
                    cur.execute("""
                        DELETE FROM tasks WHERE title ILIKE %s;
                    """, (pattern,))
                deleted = cur.rowcount > 0
                conn.commit()
                if deleted:
                    return True
        except Exception as e:
            logger.warning(f"Failed to delete task in PostgreSQL: {e}")
        finally:
            conn.close()

    # In-memory fallback
    for t in list(_in_memory_tasks):
        if (task_id is not None and t["id"] == task_id) or (clean_id.lower() in t["title"].lower()):
            _in_memory_tasks.remove(t)
            return True
    return False
