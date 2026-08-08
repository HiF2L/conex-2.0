"""
LLM Client Wrapper utilizing standard OpenAI client with dotenv configuration,
Whisper audio transcription, and intelligent fallback support.
"""
import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from openai import OpenAI

from src.models import MemoryDiff, QAPair

# Load environment configuration
load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        # Main Provider Configuration (provod.ai)
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        self.default_model = os.getenv("DEFAULT_MODEL", "openai/gpt-5.6-luna").strip()
        self.fast_model = os.getenv("FAST_MODEL", "deepseek/deepseek-v4-flash").strip()

        # Dedicated STT Configuration (ProxyAPI)
        self.stt_base_url = os.getenv("STT_BASE_URL", "https://api.proxyapi.ru/openai/v1").strip()
        self.stt_api_key = os.getenv("STT_API_KEY", "").strip() or self.api_key
        self.stt_model = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe").strip()

        # Initialize Main Chat SDK client (provod.ai)
        self.client = OpenAI(
            api_key=self.api_key if self.api_key else "offline_key",
            base_url=self.base_url
        )

        # Initialize Dedicated STT SDK client (ProxyAPI)
        self.stt_client = OpenAI(
            api_key=self.stt_api_key if self.stt_api_key else "offline_key",
            base_url=self.stt_base_url
        )

    def is_api_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_api_key_here")

    def generate_coaching_response(
        self, 
        system_prompt: str, 
        user_input: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        memory_engine: Optional[Any] = None
    ) -> str:
        """
        Generate coaching agent response using DEFAULT_MODEL.
        Supports sliding conversation history and OpenAI Function Calling for search_memory and forget_memory tools.
        """
        from src.db import search_tier3_memory

        search_tool = {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Search deep long-term memory (Tier 3) about specific projects (e.g. WeGeny, Intelligence Bit), health, music, YouTube, or roadmap details from PostgreSQL when context is required.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query or entity name to look up in long-term memory."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        forget_tool = {
            "type": "function",
            "function": {
                "name": "forget_memory",
                "description": "Explicitly delete or remove outdated or resolved QA pairs from Tier 2 state or Tier 3 entity memory when requested by the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_tier": {
                            "type": "string",
                            "enum": ["tier2", "tier3", "all"],
                            "description": "Memory tier to clean ('tier2', 'tier3', or 'all')."
                        },
                        "keyword": {
                            "type": "string",
                            "description": "Keyword, topic, entity name, or item ID to match and delete."
                        }
                    },
                    "required": ["target_tier", "keyword"]
                }
            }
        }

        create_project_tool = {
            "type": "function",
            "function": {
                "name": "create_project",
                "description": "Create a new project category in PostgreSQL Task Manager.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Unique name of the project (e.g. WeGeny, Health, LifeOS)"},
                        "description": {"type": "string", "description": "Optional description of project goals"}
                    },
                    "required": ["name"]
                }
            }
        }

        create_task_tool = {
            "type": "function",
            "function": {
                "name": "create_task",
                "description": "Create a new task in PostgreSQL Task Manager.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title or action item"},
                        "project_name": {"type": "string", "description": "Optional project category (e.g. WeGeny, LifeOS)"},
                        "priority": {"type": "integer", "description": "Priority level (1: High, 2: Medium, 3: Low)", "default": 2},
                        "due_date": {"type": "string", "description": "Optional due date ISO string (YYYY-MM-DD)"},
                        "description": {"type": "string", "description": "Optional task details"}
                    },
                    "required": ["title"]
                }
            }
        }

        complete_task_tool = {
            "type": "function",
            "function": {
                "name": "complete_task",
                "description": "Mark a task as completed ('done') in PostgreSQL Task Manager by task ID or title match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Task ID integer (e.g. '3') or task title substring"}
                    },
                    "required": ["identifier"]
                }
            }
        }

        list_tasks_tool = {
            "type": "function",
            "function": {
                "name": "list_tasks",
                "description": "List tasks from PostgreSQL Task Manager.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string", "description": "Optional project name filter"},
                        "status": {"type": "string", "description": "Optional status filter ('todo', 'in_progress', 'done', 'all')"}
                    }
                }
            }
        }

        delete_task_tool = {
            "type": "function",
            "function": {
                "name": "delete_task",
                "description": "Delete a task from PostgreSQL Task Manager by ID or title match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Task ID integer or title substring"}
                    },
                    "required": ["identifier"]
                }
            }
        }

        update_task_tool = {
            "type": "function",
            "function": {
                "name": "update_task",
                "description": "Update details (title, status, priority, due_date) of an existing task in PostgreSQL Task Manager by task ID or title match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Task ID integer (e.g. '5') or title substring"},
                        "title": {"type": "string", "description": "Optional new task title"},
                        "status": {"type": "string", "description": "Optional new status ('todo', 'in_progress', 'done', 'cancelled')"},
                        "priority": {"type": "integer", "description": "Optional new priority (1: High, 2: Medium, 3: Low)"},
                        "due_date": {"type": "string", "description": "Optional new due date (YYYY-MM-DD)"}
                    },
                    "required": ["identifier"]
                }
            }
        }

        tools = [search_tool, forget_tool, create_project_tool, create_task_tool, complete_task_tool, list_tasks_tool, delete_task_tool, update_task_tool]

        if self.is_api_configured():
            try:
                # Construct messages payload: system prompt + sliding chat history + current user message
                messages = [{"role": "system", "content": system_prompt}]
                if chat_history:
                    for turn in chat_history:
                        if isinstance(turn, dict) and "role" in turn and "content" in turn:
                            messages.append({"role": turn["role"], "content": turn["content"]})
                messages.append({"role": "user", "content": user_input})

                # Initial completion call with tools enabled
                response = self.client.chat.completions.create(
                    model=self.default_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7
                )
                
                choice = response.choices[0]

                # Check if model requested tool call
                if choice.message.tool_calls:
                    messages.append(choice.message) # append assistant tool request
                    
                    for tool_call in choice.message.tool_calls:
                        fn_name = tool_call.function.name
                        if fn_name == "search_memory":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                query = args.get("query", user_input)
                            except Exception:
                                query = user_input
                            
                            logger.info(f"LLM triggered tool search_memory(query='{query}')")
                            search_results = search_tier3_memory(query)
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": search_results
                            })

                        elif fn_name == "forget_memory" and memory_engine:
                            try:
                                args = json.loads(tool_call.function.arguments)
                                target_tier = args.get("target_tier", "tier2")
                                keyword = args.get("keyword", "")
                            except Exception:
                                target_tier, keyword = "tier2", user_input
                            
                            logger.info(f"LLM triggered tool forget_memory(target_tier='{target_tier}', keyword='{keyword}')")
                            forget_results = memory_engine.forget_memory(target_tier, keyword)

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": forget_results
                            })

                        elif fn_name == "create_project":
                            from src.db import create_project_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                res = create_project_db(args.get("name", ""), args.get("description", ""))
                            except Exception as pe:
                                res = {"error": str(pe)}
                            logger.info(f"LLM triggered tool create_project: {res}")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(res, ensure_ascii=False)})

                        elif fn_name == "create_task":
                            from src.db import create_task_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                res = create_task_db(
                                    title=args.get("title", ""),
                                    project_name=args.get("project_name"),
                                    priority=int(args.get("priority", 2)),
                                    due_date=args.get("due_date"),
                                    description=args.get("description", "")
                                )
                            except Exception as te:
                                res = {"error": str(te)}
                            logger.info(f"LLM triggered tool create_task: {res}")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(res, ensure_ascii=False)})

                        elif fn_name == "complete_task":
                            from src.db import complete_task_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                success = complete_task_db(args.get("identifier", ""))
                            except Exception as ce:
                                success = False
                            logger.info(f"LLM triggered tool complete_task: success={success}")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Task completion result: {success}"})

                        elif fn_name == "list_tasks":
                            from src.db import get_active_tasks_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                tasks = get_active_tasks_db(args.get("project_name"), args.get("status"))
                            except Exception as le:
                                tasks = []
                            logger.info(f"LLM triggered tool list_tasks: found {len(tasks)} tasks")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tasks, ensure_ascii=False)})

                        elif fn_name == "delete_task":
                            from src.db import delete_task_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                success = delete_task_db(args.get("identifier", ""))
                            except Exception as de:
                                success = False
                            logger.info(f"LLM triggered tool delete_task: success={success}")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Task deletion result: {success}"})

                        elif fn_name == "update_task":
                            from src.db import update_task_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                success = update_task_db(
                                    identifier=args.get("identifier", ""),
                                    title=args.get("title"),
                                    status=args.get("status"),
                                    priority=args.get("priority"),
                                    due_date=args.get("due_date")
                                )
                            except Exception as ue:
                                success = False
                            logger.info(f"LLM triggered tool update_task: success={success}")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Task update result: {success}"})

                    # Second call to generate final answer using tool execution results
                    final_response = self.client.chat.completions.create(
                        model=self.default_model,
                        messages=messages,
                        temperature=0.7
                    )
                    if final_response.choices and final_response.choices[0].message.content:
                        return self._sanitize_tool_leak(final_response.choices[0].message.content.strip())

                elif choice.message.content:
                    return self._sanitize_tool_leak(choice.message.content.strip())

            except Exception as e:
                logger.warning(f"API call failed: {e}. Falling back to offline simulator.")

        # Offline / Fallback Response
        return self._sanitize_tool_leak(self._generate_offline_coaching_response(system_prompt, user_input))

    def _sanitize_tool_leak(self, text: str) -> str:
        """Removes leaked raw tool call syntax (e.g. to=func_name ..., to=functions.xyz, {"query": ...}) from text responses."""
        if not text:
            return ""
        # Strip patterns like `to=function_name ...` or `to=functions.xyz ...`
        cleaned = re.sub(r"to=\w+(\.\w+)?\s*(\(json\))?:?\s*\{.*?\}", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"to=\w+(\.\w+)?\s+[^\n]+", "", cleaned)
        cleaned = re.sub(r"to=functions\.\w+[^\n]*", "", cleaned)
        # Strip standalone raw JSON tool call leaks (e.g. {"query": "..."})
        cleaned = re.sub(r'^\s*\{\s*"(query|target_tier|keyword|identifier|title|project_name)"\s*:.*?\n?', "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def extract_memory_diff(
        self, 
        system_prompt: str, 
        text_to_analyze: str, 
        current_date: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> MemoryDiff:
        """
        Extract memory updates as a structured MemoryDiff using FAST_MODEL.
        Injects CURRENT_DATE, CONVERSATION CONTEXT (last 3-4 turns), and enforces
        dynamic Tier 3 entity categorization and pronoun resolution.
        """
        import datetime
        today_str = current_date or datetime.date.today().isoformat()

        context_str = ""
        if conversation_history:
            recent_turns = conversation_history[-4:]
            context_lines = [f"{turn.get('role', 'user').upper()}: {turn.get('content', '')}" for turn in recent_turns]
            context_str = "\n\n=== CONVERSATION CONTEXT (LAST 4 TURNS) ===\n" + "\n".join(context_lines)

        if self.is_api_configured():
            extraction_prompt = (
                f"{system_prompt}\n\n"
                f"CURRENT_DATE: {today_str}"
                f"{context_str}\n\n"
                "CRITICAL EXTRACTION RULES:\n"
                f"1. DATE ACCURACY: All new or updated QA items MUST use valid_from = '{today_str}'. DO NOT use past years or hallucinated dates.\n"
                "2. TIER 3 DYNAMIC ENTITIES: Identify any specific projects, topics, or domains mentioned in the text "
                "(e.g., 'wegeny', 'intelligence_bit', 'health', 'music', 'youtube', 'cleaning', 'lifeos', etc.). "
                "You MUST create separate entry keys in 'tier3_updates' for each entity (e.g., 'tier3_updates': {'wegeny': [...], 'health': [...]}). "
                "Do NOT lump project-specific facts into Tier 1 or Tier 2.\n"
                "3. Tier 1 is ONLY for static core identity/values/communication preferences.\n"
                "4. Tier 2 is ONLY for current energy level, overall sprint goal, and immediate blockers.\n"
                "5. DEDUPLICATION & REPLACEMENT: If new user information updates, replaces, or contradicts an existing Tier 2 item, place the ID of the old Tier 2 item in 'deletions' to prevent duplicates.\n"
                "6. PROMOTION TO TIER 3: If a task or idea matures into a specific project feature, create an entry in 'tier3_updates[entity]' and list the old Tier 2 item ID in 'deletions'.\n"
                "7. EVENT PING EXTRACTION: Identify any specific time-sensitive events mentioned by the user (e.g. 'Завтра в 11:00 иду к врачу', 'Meeting at 3pm'). Populate 'scheduled_pings': [{'scheduled_at': ISO_STR (2-3 hours after event), 'event_type': str, 'context_text': str}].\n"
                "8. DAILY SYSTEMS & ROUTINES: Identify any recurring daily system rules defined by the user (e.g. project step, cleaning 4 zones, 1hr walk/think, job applications). Persist them to 'tier2_updates' with question 'What are the user's active daily system rules/routines?' and weight 1.0.\n"
                "9. PRONOUN RESOLUTION & CONTEXT BINDING: Use the preceding conversation turns to accurately resolve pronouns ('he', 'it', 'this project') and bind facts strictly to the correct entity name. Never link Entity A to Entity B unless explicitly declared in text.\n\n"
                "Return a strict JSON object matching MemoryDiff schema:\n"
                "{\n"
                '  "tier1_updates": [{"id": str, "question": str, "answer": str, "weight": float, "confidence": float, "origin": str, "valid_from": str, "valid_until": str|null}],\n'
                '  "tier2_updates": [...],\n'
                '  "tier3_updates": {"entity_name": [...]},\n'
                '  "deletions": [id_str, ...]\n'
                "}\n"
                "Return ONLY valid raw JSON."
            )

            # Models to attempt for extraction (FAST_MODEL then DEFAULT_MODEL)
            models_to_try = [self.fast_model]
            if self.default_model not in models_to_try:
                models_to_try.append(self.default_model)

            for model_name in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": extraction_prompt},
                            {"role": "user", "content": text_to_analyze}
                        ],
                        temperature=0.2
                    )
                    raw_json = response.choices[0].message.content.strip()
                    data = self._clean_and_parse_json(raw_json)
                    return MemoryDiff.model_validate(data)
                except Exception as e:
                    logger.warning(f"API extraction with model '{model_name}' failed: {e}.")

        # Offline / Rule-based Fallback Extractor
        return self._rule_based_memory_extractor(text_to_analyze, current_date=today_str)

    def _clean_and_parse_json(self, raw_text: str) -> dict:
        """Strip markdown codeblocks and parse JSON dictionary."""
        cleaned = raw_text.strip()
        if "```" in cleaned:
            lines = cleaned.splitlines()
            code_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    code_lines.append(line)
            if code_lines:
                cleaned = "\n".join(code_lines).strip()

        # Find first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx+1]

        return json.loads(cleaned)

    def generate_embedding(self, text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
        """
        Generate text vector embedding via ProxyAPI endpoint (https://api.proxyapi.ru/openai/v1).
        """
        stt_api_ready = bool(self.stt_api_key and self.stt_api_key != "your_proxyapi_key_here")
        if stt_api_ready:
            try:
                response = self.stt_client.embeddings.create(
                    model=model,
                    input=text
                )
                if response.data and len(response.data) > 0:
                    return response.data[0].embedding
            except Exception as primary_e:
                logger.warning(f"ProxyAPI embedding generation failed: {primary_e}. Trying main provider fallback...")
                if self.is_api_configured():
                    try:
                        response = self.client.embeddings.create(
                            model=model,
                            input=text
                        )
                        if response.data and len(response.data) > 0:
                            return response.data[0].embedding
                    except Exception as fallback_e:
                        logger.warning(f"Main provider embedding fallback failed: {fallback_e}.")
        return None

    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribe audio file using ProxyAPI STT endpoint (https://api.proxyapi.ru/openai/v1) with gpt-4o-mini-transcribe.
        """
        stt_api_ready = bool(self.stt_api_key and self.stt_api_key != "your_proxyapi_key_here")
        if stt_api_ready:
            try:
                with open(file_path, "rb") as audio_file:
                    transcript = self.stt_client.audio.transcriptions.create(
                        model=self.stt_model,
                        file=audio_file
                    )
                    if hasattr(transcript, "text"):
                        return transcript.text.strip()
                    elif isinstance(transcript, dict) and "text" in transcript:
                        return transcript["text"].strip()
            except Exception as primary_e:
                logger.warning(f"ProxyAPI STT model '{self.stt_model}' failed: {primary_e}. Trying main provider fallback...")
                if self.is_api_configured():
                    try:
                        with open(file_path, "rb") as audio_file:
                            transcript = self.client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            )
                            if hasattr(transcript, "text"):
                                return transcript.text.strip()
                            elif isinstance(transcript, dict) and "text" in transcript:
                                return transcript["text"].strip()
                    except Exception as fallback_e:
                        logger.warning(f"Main provider STT fallback failed: {fallback_e}.")

        return f"[Simulated Voice Dump Transcribed]: High energy state today. Working on WeGeny backend and Health routine improvements."

    def _generate_offline_coaching_response(self, system_prompt: str, user_input: str) -> str:
        """
        Simulate thoughtful Senior Friend & Coach response offline.
        """
        lower_input = user_input.lower()
        if "lifeos" in lower_input:
            return (
                "Regarding **LifeOS**, your 3-tier memory setup (Core Profile + Rolling State + Dynamic Entity Graph) "
                "is extremely token efficient. Since Tier 3 entity QA pairs are loaded dynamically, we keep context tight. "
                "How are you feeling about the Telegram bot deployment so far?"
            )
        elif "energy" in lower_input or "tired" in lower_input or "sprint" in lower_input:
            return (
                "I notice you're reflecting on your current sprint and energy. "
                "As an architectural thinker who hates overengineering, let's keep your focus sharp on the essential core. "
                "What single milestone will yield 80% of the value today?"
            )
        else:
            return (
                f"Got it, Vitalik. I've analyzed: '{user_input}'. "
                "Based on your core values (direct feedback, clean architecture, no fluff), "
                "let's ensure this aligns with your current sprint goals while keeping system overhead low. "
                "What's the next step you'd like to execute?"
            )

    def _rule_based_memory_extractor(self, text: str, current_date: Optional[str] = None) -> MemoryDiff:
        """
        Rule-based memory diff generator when offline with dynamic entity detection.
        """
        import datetime
        import re
        today_str = current_date or datetime.date.today().isoformat()
        lower_text = text.lower()
        
        t2_updates = []
        t3_updates = {}

        # 1. State / Energy / Daily Systems in Tier 2
        if any(k in lower_text for k in ["energy", "dump", "voice", "sprint", "систем", "правил", "рутин", "каждый день"]):
            t2_updates.append(QAPair(
                id=f"state_energy_{int(datetime.datetime.now().timestamp())}",
                question="What are the user's latest reported energy state, active daily system rules, or sprint focus?",
                answer=f"User mentioned context: '{text[:120]}'",
                weight=1.0,
                confidence=0.9,
                origin="voice_dump_extractor",
                valid_from=today_str
            ))

        # 2. Dynamic Entity Extraction for Tier 3
        known_entities = ["wegeny", "health", "intelligence_bit", "music", "youtube", "cleaning", "lifeos"]
        
        for entity in known_entities:
            if entity in lower_text or entity.replace("_", " ") in lower_text:
                t3_updates[entity] = [
                    QAPair(
                        id=f"{entity}_update_{int(datetime.datetime.now().timestamp())}",
                        question=f"What is the recent update or status regarding {entity.upper()}?",
                        answer=text[:160],
                        weight=0.95,
                        confidence=0.9,
                        origin="voice_dump_extractor",
                        valid_from=today_str
                    )
                ]

        # 3. Check for any generic capitalization or project keywords if none matched
        if not t3_updates:
            # Look for project keywords like "project X" or words ending in "-bot" / "-app"
            matches = re.findall(r'\b([a-z0-9_]{3,15})_(?:project|app|bot|service)\b', lower_text)
            for match in matches:
                clean_ent = match.strip().lower()
                t3_updates[clean_ent] = [
                    QAPair(
                        id=f"{clean_ent}_update_{int(datetime.datetime.now().timestamp())}",
                        question=f"What is the recent update regarding project {clean_ent}?",
                        answer=text[:160],
                        weight=0.9,
                        confidence=0.85,
                        origin="voice_dump_extractor",
                        valid_from=today_str
                    )
                ]

        return MemoryDiff(
            tier1_updates=[],
            tier2_updates=t2_updates,
            tier3_updates=t3_updates,
            deletions=[]
        )
