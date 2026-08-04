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
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://wegeny_admin:wegeny_password@localhost:5432/lifeos_db").strip()

def _get_connection():
    """Attempt to establish a psycopg 3 connection to PostgreSQL."""
    try:
        import psycopg
        conn = psycopg.connect(DATABASE_URL, connect_timeout=3)
        return conn
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

def save_chat_message(user_id: int, role: str, content: str) -> None:
    """Save user or assistant chat turn into chat_history table."""
    if not user_id or not content:
        return

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
    Returns list of {"role": "user"|"assistant", "content": "..."}.
    """
    if not user_id:
        return []

    conn = _get_connection()
    if not conn:
        return []

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
            return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception as e:
        logger.warning(f"Failed to retrieve chat history from PostgreSQL: {e}")
        return []
    finally:
        conn.close()

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
            logger.info(f"Deleted {deleted_count} Tier 3 memory entries from PostgreSQL matching '{clean_kw}'.")
            return deleted_count
    except Exception as e:
        logger.warning(f"Failed to delete Tier 3 memory from PostgreSQL: {e}")
        return 0
    finally:
        conn.close()
