# 🧠 Conex 2.0 / LifeOS: Autonomous AI Coach & Memory Operating System

Conex 2.0 (LifeOS) is a personal AI Agent ("Senior Friend & Coach") designed for high-density cognitive leverage, proactive accountability, and lifelong memory continuity.

It features a **3-Tier Index-Based Memory Architecture** (PostgreSQL + Local YAML), **Automated Proactive Coaching Loops**, **Sprints & Experiments Engine**, **Personal Life Principles & Productivity Axioms Engine**, and full **Telegram Bot Integration** with voice-message transcription.

---

## 🌟 Key Architecture & Capabilities

```mermaid
graph TD
    User([Telegram User / Voice / Text]) <--> Bot[Aiogram 3 Telegram Bot]
    Bot <--> LLM[LLM Function Calling Engine]
    LLM <--> MemEngine[3-Tier Memory Engine]
    LLM <--> DB[(PostgreSQL + Full-Text / pgvector)]
    LLM <--> Proactive[Proactive Push Engine]
    
    subgraph Storage Layer
        T1[Tier 1: Core Long-Term Profile]
        T2[Tier 2: Dynamic Rolling State]
        T3[Tier 3: Isolated Entity Knowledge Graph]
        Rules[(Life Rules & Axioms Table)]
        Tasks[(Tasks & Projects Table)]
        Sprints[(Sprints & A/B Experiments)]
        Wellbeing[(Wellbeing & Energy Logs)]
    end

    MemEngine --> T1
    MemEngine --> T2
    MemEngine --> T3
    DB --> Rules
    DB --> Tasks
    DB --> Sprints
    DB --> Wellbeing
```

### 1. 🧠 3-Tier Index-Based Memory System
- **Tier 1 (Core Profile)**: Question-anchored long-term traits, cognitive style, core values, and strategic vision (`data/memory/tier1_core.yaml`).
- **Tier 2 (Dynamic Rolling State)**: Current energy state, active habits, and recent context with exponential weight decay (`data/memory/tier2_state.yaml`).
- **Tier 3 (Entity Knowledge Graph & Search)**: Strict namespace-isolated entity YAMLs (`data/memory/tier3_entities/*.yaml`) synchronized to PostgreSQL for hybrid full-text and vector search.
- **3-Step Memory Protocol**: Zero-hallucination factual lookup using `search_memory` → `get_document_outline` → `read_document_section`.

### 2. 📜 Personal Life Principles & Productivity Axioms Engine
- Stores extracted personal rules across 5 domains: `productivity`, `nutrition`, `mental_health`, `chores`, `career`.
- **Reflection Extraction**: Dynamically identifies systemic insights during reflections and saves them via `save_life_rule`.
- **Grounded Advice**: When encountering confusion or breakdowns, queries active rules to ground coaching in user-established axioms rather than generic clichés.
- **Morning Plan Enforcement**: Automatically enforces axioms (e.g. No complex cooking before Deep Work, 10–15m micro-cleaning sprints, Zero-choice task formulations).

### 3. 🔬 Sprints & Experiments Engine
- Manages structured 14-day habits, lifestyle sprints, and A/B hypotheses.
- Tracks phase progression: `ACTIVE` → `CHECKIN` → `ADAPT` → `COMPLETED`.
- Daily tracking integrated into morning briefings and evening reflections.

### 4. ⏰ Proactive Push Engine & Automated Cycles
- **🌅 Morning Briefing (09:00)**: Synthesizes active sprint status, top focus tasks, active axioms, and atomic deep-work blocks.
- **🌆 Evening Sync (21:00)**: Daily check-in evaluating systemic rules, completed tasks, energy blockers, and selecting 1 atomic micro-step for tomorrow.
- **⚡ Proactive Follow-up Pings**: Periodic background checker (every 15 min) for scheduled follow-ups and accountability reminders with quiet hours enforcement (`22:00–08:00`).

### 5. 📱 Telegram Bot & Voice Dump Transcription
- **Aiogram 3 Integration**: Real-time interaction with single-user authentication security (`ALLOWED_TELEGRAM_ID`).
- **Speech-to-Text**: Automatic voice message transcription using dedicated OpenAI Whisper / ProxyAPI endpoints.
- **Mobile-Optimized Telegram Formatting**: Automatic conversion of Markdown tables to bullet lists, elimination of raw hashtag headers, clean bold/italic typography without artificial clock timestamps.

### 6. 🛡️ Resilient LLM Client
- Multi-step OpenAI function-calling loop with `tool_choice="auto"`.
- **Flattened Message Context Fallback**: Auto-translates tool responses to standard dialog context for proxy endpoints (Google Gemini, DeepSeek, Claude, GPT).
- **Intelligent Offline Fallback**: Dynamically aggregates active database context even during network/provider outages.

---

## 📁 Repository Structure

```
.
├── .env.example             # Environment configuration template
├── .gitignore               # Excludes personal runtime YAMLs & caches
├── setup_db.py              # PostgreSQL database initialization & migration
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
├── data/
│   └── memory/              # Memory storage directory
│       ├── .gitkeep
│       ├── tier1_core.yaml.example
│       ├── tier2_state.yaml.example
│       ├── history/
│       │   └── .gitkeep
│       └── tier3_entities/
│           ├── .gitkeep
│           └── sample_entity.yaml.example
└── src/
    ├── __init__.py
    ├── bot.py               # Aiogram 3 Telegram bot & voice handlers
    ├── db.py                # PostgreSQL persistence layer & CRUD helpers
    ├── extractor_service.py # Async background memory extraction
    ├── llm_client.py        # LLM client, tool execution & Telegram formatter
    ├── main.py              # Interactive Rich CLI terminal interface
    ├── memory_engine.py     # 3-Tier prompt assembly & coaching directives
    ├── models.py            # Pydantic v2 schemas
    ├── proactive_engine.py  # Push notification & scheduled ping scheduler
    └── scheduler.py         # AsyncIO cron jobs (Morning 09:00 & Evening 21:00)
```

---

## 🚀 Getting Started

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/HiF2L/conex-2.0.git
cd conex-2.0
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your credentials in `.env`:
```env
# LLM Provider API Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.provod.ai/v1
DEFAULT_MODEL=openai/gpt-5.6-luna
FAST_MODEL=deepseek/deepseek-v4-flash

# Speech-to-Text Configuration (ProxyAPI / Whisper)
STT_API_KEY=your_proxyapi_key_here
STT_BASE_URL=https://api.proxyapi.ru/openai/v1
STT_MODEL=gpt-4o-mini-transcribe

# PostgreSQL Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/lifeos_db

# Telegram Bot Credentials (Aiogram 3)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
ALLOWED_TELEGRAM_ID=123456789

# Proactive Engine Settings
PROACTIVE_MAX_PINGS_PER_DAY=3
PROACTIVE_QUIET_START=22:00
PROACTIVE_QUIET_END=08:00
EVENING_SYNC_TIME=21:00
```

### 3. Initialize Memory Templates & Database
Create your local memory files from examples:
```bash
cp data/memory/tier1_core.yaml.example data/memory/tier1_core.yaml
cp data/memory/tier2_state.yaml.example data/memory/tier2_state.yaml
cp data/memory/tier3_entities/sample_entity.yaml.example data/memory/tier3_entities/lifeos.yaml
```

Run database schema setup and sync Tier 3 memory into PostgreSQL:
```bash
python setup_db.py
```

---

## 🏃 Running Conex 2.0

### Run Telegram Bot
```bash
python -m src.bot
# or
python src/bot.py
```

### Run Interactive Terminal CLI
```bash
python -m src.main
# or
python src/main.py
```

### Run Full Test Suite
```bash
python -m pytest
```

---

## 🛠️ Telegram Bot & CLI Commands

| Command | Description |
|---|---|
| `/start` | Welcome message and system status check |
| `/tasks` | List top active tasks with priority tags |
| `/experiments` | View active habits, sprints, and A/B tests |
| `/rules` | View active Life Principles & Productivity Axioms |
| `/memory` | Inspect Tier 1, Tier 2, and Tier 3 memory anchors |
| `/decay` | Manually apply exponential weight decay to Tier 2 items |
| `/debug` | Toggle memory trace and tool-calling execution inspect footer |
| `/dump` | Paste raw text or voice transcript for immediate memory extraction |

---

## 📄 License
MIT License. Created by [Vitalik (Hitori)](https://github.com/HiF2L).
