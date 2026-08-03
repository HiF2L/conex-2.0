# LifeOS Personal Memory & Coaching Agent MVP

A production-ready, highly token-efficient Personal AI Agent ("Senior Friend & Coach") featuring a **3-Tier Question-Anchored Memory Architecture** stored in human-readable local YAML files.

---

## 🌟 Key Architecture Features

1. **3-Tier Question-Anchored Memory System (`data/memory/`)**:
   - **Tier 1 (Core Profile)**: Static QA about user personality, values, communication style (*Always loaded*).
   - **Tier 2 (Dynamic State)**: Rolling QA about current energy, sprint goals, blockers with exponential weight decay (`W_new = W_old * 0.95`, *Always loaded*).
   - **Tier 3 (Entity Graph)**: Dynamically generated QA pairs for projects/topics (e.g. `lifeos.yaml`). Loaded **only** when an entity is explicitly mentioned in prompt, selecting the **Top-3 highest-weighted QA pairs**.

2. **Pydantic v2 Models (`src/models.py`)**:
   - Strict validation of all memory schemas (`QAPair`, `MemoryDiff`, `MemoryTrace`).

3. **Asynchronous Background Extractor (`src/extractor_service.py`)**:
   - Out-of-band non-blocking extraction thread that parses chat turns using a fast LLM model into structured `MemoryDiff` JSON.
   - Safe atomic file operations (never corrupts YAML on disk).

4. **Rich Terminal CLI Interface (`src/main.py`)**:
   - Interactive prompt with Markdown-formatted coach output.
   - Non-intrusive memory trace debug footer: `[Memory Trace: T1: 3 Qs | T2: 3 Qs | T3: LifeOS (3 Qs) | Est. Tokens: ~850]`.
   - Special commands: `/memory`, `/dump`, `/decay`, `/help`, `/exit`.

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API credentials:
```bash
cp .env.example .env
```
In `.env`:
```env
OPENAI_API_KEY=your_api_token
OPENAI_BASE_URL=https://api.provod.ai/v1
DEFAULT_MODEL=openai/gpt-5.6-luna
FAST_MODEL=deepseek/deepseek-v4-flash
```
*(Note: If `OPENAI_API_KEY` is omitted, the CLI operates smoothly using an intelligent local fallback simulator).*

### 3. Run the Interactive CLI
```bash
python -m src.main
# or
python src/main.py
```

---

## 🛠️ CLI Commands & Usage

- **Normal Conversation**:
  Type your message. If you mention an entity (e.g., `"How can I improve LifeOS memory architecture?"`), the engine dynamically injects Top-3 Tier 3 QA pairs into context.
- **`/memory`**: Displays Tier 1, Tier 2, and Tier 3 memory state in formatted Rich tables.
- **`/dump`**: Opens "Stream of Consciousness" mode to paste voice transcripts or unorganized notes for immediate memory extraction.
- **`/decay`**: Applies weight decay (`W_new = W_old * 0.95`) to Tier 2 items.
- **`/exit`**: Gracefully quits the app.

---

## 📁 Project Structure

```
.
├── .env                     # Local environment variables
├── .env.example             # Template for API credentials & models
├── data/
│   └── memory/              # Human-editable local YAML memory files
│       ├── tier1_core.yaml
│       ├── tier2_state.yaml
│       └── tier3_entities/
│           └── lifeos.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models.py            # Pydantic v2 schemas
│   ├── llm_client.py        # OpenAI SDK wrapper with fallback
│   ├── memory_engine.py     # Prompt assembly & 3-tier memory engine
│   ├── extractor_service.py # Async background compactor/extractor
│   └── main.py              # Interactive Rich CLI app
└── README.md
```
