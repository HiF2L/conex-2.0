"""
PostgreSQL Initialization & Memory Seed Script for LifeOS.
Creates database, tables, GIN search indexes, pgvector extension, and populates Tier 3 memory items from YAML.
"""
import os
import sys
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DB_Setup")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://wegeny_admin:wegeny_password@localhost:5432/lifeos_db").strip()

def setup_database():
    print("================ LIFEOS POSTGRESQL SETUP ================")
    print(f"Target Database URL: {DATABASE_URL}")

    # Import psycopg 3 or psycopg2
    try:
        import psycopg
        use_v3 = True
        print("[OK] Loaded psycopg v3.")
    except ImportError:
        try:
            import psycopg2
            from psycopg2 import sql
            use_v3 = False
            print("[OK] Loaded psycopg2.")
        except ImportError:
            print("[ERROR] Neither 'psycopg' nor 'psycopg2' is installed in your Python environment.")
            print("Please run: pip install psycopg[binary]")
            sys.exit(1)

    if use_v3:
        from psycopg import sql

    # 1. Connect to base PostgreSQL server to ensure lifeos_db exists
    db_name = "lifeos_db"
    if "/" in DATABASE_URL:
        base_url = DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
        target_db = DATABASE_URL.rsplit("/", 1)[1]
        if target_db:
            db_name = target_db.split("?")[0]
    else:
        base_url = "postgresql://wegeny_admin:wegeny_password@localhost:5432/postgres"

    print(f"\n1. Checking PostgreSQL Server Connection & Database '{db_name}'...")
    try:
        if use_v3:
            conn = psycopg.connect(base_url, autocommit=True, connect_timeout=5)
        else:
            conn = psycopg2.connect(base_url, connect_timeout=5)
            conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cur.fetchone()
            if not exists:
                print(f"  └ Database '{db_name}' does not exist. Creating database...")
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                print(f"  [OK] Database '{db_name}' created successfully.")
            else:
                print(f"  [OK] Database '{db_name}' already exists.")
        conn.close()
    except Exception as e:
        print(f"ℹ️ Base database check notice ({e}). Attempting direct connection to '{db_name}'...")

    # 2. Connect to target database and initialize schema
    print(f"\n2. Connecting to '{db_name}' and initializing tables & indexes...")
    try:
        if use_v3:
            conn = psycopg.connect(DATABASE_URL, connect_timeout=5)
        else:
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
    except Exception as e:
        print(f"[ERROR] Failed to connect to PostgreSQL at {DATABASE_URL}: {e}")
        print("\nPlease verify:")
        print(" 1. Is PostgreSQL server running on localhost:5432?")
        print(" 2. Are credentials in .env correct? (DATABASE_URL=postgresql://user:pass@localhost:5432/lifeos_db)")
        sys.exit(1)

    has_pgvector = False
    try:
        with conn.cursor() as cur:
            # Try initializing pgvector extension
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                has_pgvector = True
                print("  [OK] Extension 'pgvector' enabled successfully.")
            except Exception as ext_err:
                conn.rollback()
                print(f"  ℹ️ pgvector extension not installed in PostgreSQL instance. Proceeding with FTS text mode.")

            # Create chat_history table
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

            # Create projects and tasks tables
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

            # Create tier3_memory_index table
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
            print("  [OK] Tables 'chat_history' and 'tier3_memory_index' initialized with GIN search indexes.")

    except Exception as e:
        print(f"[ERROR] Error setting up database schema: {e}")
        conn.close()
        sys.exit(1)

    # 3. Seed / Sync Tier 3 Memory YAML files into PostgreSQL
    print("\n3. Seeding Tier 3 Memory files from data/memory/tier3_entities/...")
    from src.db import sync_tier3_to_postgres
    synced_count = sync_tier3_to_postgres()
    print(f"  [OK] Successfully indexed {synced_count} Tier 3 memory QA items into PostgreSQL!")

    print("\n================ SETUP COMPLETED SUCCESSFULLY! ================")
    print("Your PostgreSQL database is fully configured and ready for LifeOS.")
    conn.close()

if __name__ == "__main__":
    setup_database()
