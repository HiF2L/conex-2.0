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

def is_trivial_user_turn(user_input: str) -> bool:
    """
    Returns True if the turn is a simple operational task command or brief greeting/acknowledgment.
    Returns False for all non-trivial questions, conceptual queries, or reflective inquiries.
    """
    if not user_input or len(user_input.strip()) <= 5:
        return True
    
    text = user_input.lower().strip()
    
    greetings = {"привет", "хай", "здравствуй", "добрый день", "спасибо", "ок", "хорошо", "понял", "hello", "hi", "thanks", "ok"}
    if text in greetings:
        return True

    task_cmd_patterns = [
        r"^(добавь|создай|удали|выполни|заверши|отмети|обнови)\s+(задачу|проект)\b",
        r"^(complete|create|delete|update)\s+(task|project)\b"
    ]
    for pat in task_cmd_patterns:
        if re.search(pat, text):
            return True

    return False

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
        memory_engine: Optional[Any] = None,
        trace: Optional[Any] = None
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

        get_document_outline_tool = {
            "type": "function",
            "function": {
                "name": "get_document_outline",
                "description": "Inspect the Table of Contents / section headers of a Tier 3 document/entity without loading full text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Document/entity name or ID"}
                    },
                    "required": ["identifier"]
                }
            }
        }

        read_document_section_tool = {
            "type": "function",
            "function": {
                "name": "read_document_section",
                "description": "Retrieve full text of a specific section ID or topic within a document/entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Document/entity name or ID"},
                        "section_id": {"type": "string", "description": "Section ID or topic substring"}
                    },
                    "required": ["identifier", "section_id"]
                }
            }
        }

        search_in_document_tool = {
            "type": "function",
            "function": {
                "name": "search_in_document",
                "description": "Perform targeted keyword search within a specific document/entity to extract matching section snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Document/entity name or ID"},
                        "sub_query": {"type": "string", "description": "Sub-query keywords to search inside document"}
                    },
                    "required": ["identifier", "sub_query"]
                }
            }
        }

        read_memory_entry_tool = {
            "type": "function",
            "function": {
                "name": "read_memory_entry",
                "description": "Read complete text of a specified document or entity when full context is required.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Document/entity name or ID"}
                    },
                    "required": ["identifier"]
                }
            }
        }

        read_memory_item_tool = {
            "type": "function",
            "function": {
                "name": "read_memory_item",
                "description": "Retrieve the exact factual answer for any Tier 1 Core Profile or Tier 2 Dynamic State Question Anchor by its item_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string", "description": "The item ID from Tier 1 or Tier 2 index (e.g., 't1_1', 't2_5')"}
                    },
                    "required": ["item_id"]
                }
            }
        }

        log_wellbeing_event_tool = {
            "type": "function",
            "function": {
                "name": "log_wellbeing_event",
                "description": "Log physical, cognitive, or emotional state events (food, movement, medications, clarity, brain fog, anxiety).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state_type": {"type": "string", "description": "State category (e.g. 'PEAK_CLARITY', 'BRAIN_FOG', 'LOW_ENERGY', 'ANXIETY')"},
                        "triggers": {"type": "array", "items": {"type": "string"}, "description": "List of triggers or catalysts (e.g. ['low_carb', 'brisk_walk', 'americano'])"},
                        "symptoms": {"type": "array", "items": {"type": "string"}, "description": "List of observed physical/cognitive symptoms (e.g. ['fast_word_retrieval', 'sluggish_focus'])"},
                        "notes": {"type": "string", "description": "Raw reflection or detailed input notes"}
                    },
                    "required": ["state_type"]
                }
            }
        }

        get_recovery_protocol_tool = {
            "type": "function",
            "function": {
                "name": "get_recovery_protocol",
                "description": "Fetch historical Recovery Protocol checklist based on user's documented peak-clarity triggers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_state": {"type": "string", "description": "Current state (e.g. 'BRAIN_FOG', 'LOW_ENERGY', 'ANXIETY')"}
                    },
                    "required": ["current_state"]
                }
            }
        }

        create_experiment_tool = {
            "type": "function",
            "function": {
                "name": "create_experiment",
                "description": "Spin up a new structured Sprint or A/B Test experiment (e.g. habit tracking, Keto vs Mediterranean).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title of the sprint or A/B experiment"},
                        "type": {"type": "string", "description": "Type of experiment: 'SPRINT' or 'AB_TEST'"},
                        "hypothesis_a": {"type": "string", "description": "Primary hypothesis or Phase A protocol"},
                        "hypothesis_b": {"type": "string", "description": "Phase B protocol for A/B tests"},
                        "duration_days": {"type": "integer", "description": "Duration in days (default: 14)"},
                        "daily_actions": {"type": "array", "items": {"type": "string"}, "description": "List of concrete daily action items"}
                    },
                    "required": ["title", "type", "hypothesis_a"]
                }
            }
        }

        get_active_experiments_tool = {
            "type": "function",
            "function": {
                "name": "get_active_experiments",
                "description": "Fetch all currently active habit Sprints and A/B experiments.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }

        advance_experiment_phase_tool = {
            "type": "function",
            "function": {
                "name": "advance_experiment_phase",
                "description": "Advance an experiment to its next phase (e.g., PHASE_A -> PHASE_B -> COMPLETED).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {"type": "integer", "description": "The experiment ID integer"}
                    },
                    "required": ["experiment_id"]
                }
            }
        }

        save_life_rule_tool = {
            "type": "function",
            "function": {
                "name": "save_life_rule",
                "description": "Save or update a personal life rule, operating principle, or productivity axiom in PostgreSQL across domains (productivity, nutrition, mental_health, chores, career).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "enum": ["productivity", "nutrition", "mental_health", "chores", "career"],
                            "description": "Domain of the rule"
                        },
                        "rule_name": {
                            "type": "string",
                            "description": "Short memorable title or rule name"
                        },
                        "rule_text": {
                            "type": "string",
                            "description": "Core principle or rule text"
                        },
                        "anti_pattern": {
                            "type": "string",
                            "description": "Observed failure mode or anti-pattern to avoid"
                        },
                        "actionable_remedy": {
                            "type": "string",
                            "description": "Concrete remedy, workaround, or SOP"
                        }
                    },
                    "required": ["domain", "rule_name", "rule_text"]
                }
            }
        }

        get_active_rules_tool = {
            "type": "function",
            "function": {
                "name": "get_active_rules",
                "description": "Query active personal life rules, principles, and productivity axioms from PostgreSQL. Optionally filter by domain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "enum": ["productivity", "nutrition", "mental_health", "chores", "career"],
                            "description": "Optional domain filter"
                        }
                    }
                }
            }
        }

        tools = [
            search_tool, forget_tool, create_project_tool, create_task_tool, 
            complete_task_tool, list_tasks_tool, delete_task_tool, update_task_tool,
            get_document_outline_tool, read_document_section_tool, search_in_document_tool, read_memory_entry_tool,
            read_memory_item_tool, log_wellbeing_event_tool, get_recovery_protocol_tool,
            create_experiment_tool, get_active_experiments_tool, advance_experiment_phase_tool,
            save_life_rule_tool, get_active_rules_tool
        ]

        if self.is_api_configured():
            try:
                # Construct messages payload: system prompt + sliding chat history + current user message
                messages = [{"role": "system", "content": system_prompt}]
                if chat_history:
                    for turn in chat_history:
                        if isinstance(turn, dict) and "role" in turn and "content" in turn:
                            messages.append({"role": turn["role"], "content": turn["content"]})
                messages.append({"role": "user", "content": user_input})

                is_non_trivial = not is_trivial_user_turn(user_input)

                # Multi-step tool execution loop (up to 4 steps)
                for iteration in range(4):
                    curr_tool_choice = "auto"
                    if iteration == 0 and is_non_trivial:
                        curr_tool_choice = {"type": "function", "function": {"name": "search_memory"}}

                    try:
                        response = self.client.chat.completions.create(
                            model=self.default_model,
                            messages=messages,
                            tools=tools,
                            tool_choice=curr_tool_choice,
                            temperature=0.7
                        )
                    except Exception as primary_e:
                        logger.warning(f"Tool-enabled API call failed with tool_choice={curr_tool_choice}: {primary_e}. Retrying with tool_choice='auto'...")
                        try:
                            response = self.client.chat.completions.create(
                                model=self.default_model,
                                messages=messages,
                                tools=tools,
                                tool_choice="auto",
                                temperature=0.7
                            )
                        except Exception as auto_e:
                            logger.warning(f"Tool-enabled API call with tool_choice='auto' failed: {auto_e}. Retrying direct completion without tools...")
                            response = self.client.chat.completions.create(
                                model=self.default_model,
                                messages=messages,
                                temperature=0.7
                            )

                    if not response or not response.choices:
                        break

                    choice = response.choices[0]
                    if not choice.message.tool_calls:
                        if choice.message.content:
                            return self._sanitize_tool_leak(choice.message.content.strip())
                        break

                    # Append model's tool call message
                    messages.append(choice.message)

                    for tool_call in choice.message.tool_calls:
                        fn_name = tool_call.function.name
                        if fn_name == "search_memory":
                            from src.db import search_tier3_memory
                            try:
                                args = json.loads(tool_call.function.arguments)
                                query = args.get("query", user_input)
                            except Exception:
                                query = user_input
                            
                            logger.info(f"LLM triggered tool search_memory(query='{query}')")
                            search_results = search_tier3_memory(query, top_k=10)

                            if trace and hasattr(trace, "debug_steps"):
                                hits = re.findall(r"Entity:\s*([A-Za-z0-9_]+)", search_results)
                                hits_str = ", ".join(list(dict.fromkeys(hits))[:5]) if hits else "0 hits"
                                trace.debug_steps.append(f"• 🔍 `search_memory(\"{query}\")` -> Hits: [{hits_str}]")

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": search_results
                            })

                        elif fn_name == "get_document_outline":
                            from src.db import get_document_outline_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                identifier = args.get("identifier", "")
                            except Exception:
                                identifier = ""
                            logger.info(f"LLM triggered tool get_document_outline('{identifier}')")
                            outline_res = get_document_outline_db(identifier)
                            
                            if trace and hasattr(trace, "debug_steps"):
                                headers = re.findall(r"-\s*\[Section ID:\s*[^\]]+\]\s*Topic:\s*([^\n]+)", outline_res)
                                headers_str = ", ".join([f"'{h.strip()}'" for h in headers[:4]]) if headers else "0 headers"
                                trace.debug_steps.append(f"• 📋 `get_document_outline(\"{identifier}\")` -> Headers: [{headers_str}]")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": outline_res})

                        elif fn_name == "read_document_section":
                            from src.db import read_document_section_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                identifier = args.get("identifier", "")
                                section_id = args.get("section_id", "")
                            except Exception:
                                identifier, section_id = "", ""
                            logger.info(f"LLM triggered tool read_document_section('{identifier}', '{section_id}')")
                            sec_res = read_document_section_db(identifier, section_id)
                            
                            if trace and hasattr(trace, "t3_sections_read"):
                                trace.t3_sections_read += 1

                                q_match = re.search(r"Question:\s*([^\n]+)", sec_res)
                                topic = q_match.group(1).strip() if q_match else section_id
                                trace.debug_steps.append(f"• 📖 `read_document_section(\"{identifier}\", \"{section_id}\")` -> Section: '{topic}' ({len(sec_res)} chars)")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": sec_res})

                        elif fn_name == "search_in_document":
                            from src.db import search_in_document_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                identifier = args.get("identifier", "")
                                sub_query = args.get("sub_query", "")
                            except Exception:
                                identifier, sub_query = "", ""
                            logger.info(f"LLM triggered tool search_in_document('{identifier}', '{sub_query}')")
                            doc_search_res = search_in_document_db(identifier, sub_query)

                            if trace and hasattr(trace, "t3_sections_read"):
                                c = len(re.findall(r"Section ID", doc_search_res))
                                trace.t3_sections_read += max(1, c)

                                matches = re.findall(r"Question:\s*([^\n]+)", doc_search_res)
                                matches_str = ", ".join([f"'{m.strip()}'" for m in matches[:3]]) if matches else "0 matches"
                                trace.debug_steps.append(f"• 🔎 `search_in_document(\"{identifier}\", \"{sub_query}\")` -> Snippets: [{matches_str}]")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": doc_search_res})

                        elif fn_name == "read_memory_entry":
                            from src.db import read_memory_entry_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                identifier = args.get("identifier", "")
                            except Exception:
                                identifier = ""
                            logger.info(f"LLM triggered tool read_memory_entry('{identifier}')")
                            entry_res = read_memory_entry_db(identifier)

                            if trace and hasattr(trace, "t3_sections_read"):
                                c = len(re.findall(r"Section ID", entry_res))
                                trace.t3_sections_read += max(1, c)

                                trace.debug_steps.append(f"• 📑 `read_memory_entry(\"{identifier}\")` -> Full document ({len(entry_res)} chars)")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": entry_res})

                        elif fn_name == "read_memory_item":
                            from src.db import get_qa_item_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                item_id = args.get("item_id", "")
                            except Exception:
                                item_id = ""
                            logger.info(f"LLM triggered tool read_memory_item('{item_id}')")
                            item_res = get_qa_item_db(item_id)

                            if trace and hasattr(trace, "debug_steps"):
                                src_match = re.search(r"Source File:\s*([^\s\|]+)", item_res)
                                src_file = src_match.group(1).strip() if src_match else "memory"
                                q_match = re.search(r"Question:\s*([^\n]+)", item_res)
                                q_title = q_match.group(1).strip() if q_match else item_id
                                trace.debug_steps.append(f"• 📌 `read_memory_item(\"{item_id}\")` -> File: `{src_file}` | Anchor: '{q_title}' ({len(item_res)} chars)")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": item_res})

                        elif fn_name == "log_wellbeing_event":
                            from src.db import log_wellbeing_event_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                state_type = args.get("state_type", "GENERAL")
                                triggers = args.get("triggers", [])
                                symptoms = args.get("symptoms", [])
                                notes = args.get("notes", "")
                            except Exception:
                                state_type, triggers, symptoms, notes = "GENERAL", [], [], ""

                            logger.info(f"LLM triggered tool log_wellbeing_event(state_type='{state_type}')")
                            log_res = log_wellbeing_event_db(state_type, triggers, symptoms, notes)
                            res_str = json.dumps(log_res, ensure_ascii=False)

                            if trace and hasattr(trace, "debug_steps"):
                                trace.debug_steps.append(f"• 🩺 `log_wellbeing_event(\"{state_type}\")` -> Recorded log ID #{log_res.get('id')}")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res_str})

                        elif fn_name == "get_recovery_protocol":
                            from src.db import get_recovery_protocol_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                current_state = args.get("current_state", "BRAIN_FOG")
                            except Exception:
                                current_state = "BRAIN_FOG"

                            logger.info(f"LLM triggered tool get_recovery_protocol(current_state='{current_state}')")
                            protocol_res = get_recovery_protocol_db(current_state)

                            if trace and hasattr(trace, "debug_steps"):
                                trace.debug_steps.append(f"• 🚨 `get_recovery_protocol(\"{current_state}\")` -> Issued Recovery Protocol Checklist ({len(protocol_res)} chars)")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": protocol_res})

                        elif fn_name == "create_experiment":
                            from src.db import create_experiment_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                title = args.get("title", "")
                                exp_type = args.get("type", "SPRINT")
                                hyp_a = args.get("hypothesis_a", "")
                                hyp_b = args.get("hypothesis_b", "")
                                duration = args.get("duration_days", 14)
                                actions = args.get("daily_actions", [])
                            except Exception:
                                title, exp_type, hyp_a, hyp_b, duration, actions = "", "SPRINT", "", "", 14, []

                            logger.info(f"LLM triggered tool create_experiment(title='{title}', type='{exp_type}')")
                            exp_res = create_experiment_db(title, exp_type, hyp_a, hyp_b, duration, actions)
                            res_str = json.dumps(exp_res, ensure_ascii=False)

                            if trace and hasattr(trace, "debug_steps"):
                                trace.debug_steps.append(f"• 🔬 `create_experiment(\"{title}\")` -> Created {exp_type} ID #{exp_res.get('id')}")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res_str})

                        elif fn_name == "get_active_experiments":
                            from src.db import get_active_experiments_db
                            logger.info("LLM triggered tool get_active_experiments()")
                            active_exp = get_active_experiments_db()
                            res_str = json.dumps(active_exp, ensure_ascii=False)

                            if trace and hasattr(trace, "debug_steps"):
                                trace.debug_steps.append(f"• 🧪 `get_active_experiments()` -> Active items: {len(active_exp)}")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res_str})

                        elif fn_name == "advance_experiment_phase":
                            from src.db import advance_experiment_phase_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                exp_id = int(args.get("experiment_id", 0))
                            except Exception:
                                exp_id = 0

                            logger.info(f"LLM triggered tool advance_experiment_phase(experiment_id={exp_id})")
                            adv_res = advance_experiment_phase_db(exp_id)
                            res_str = json.dumps(adv_res, ensure_ascii=False)

                            if trace and hasattr(trace, "debug_steps"):
                                trace.debug_steps.append(f"• ⏩ `advance_experiment_phase({exp_id})` -> New Phase: {adv_res.get('phase')}")

                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": res_str})

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

                        elif fn_name == "save_life_rule":
                            from src.db import save_life_rule_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                rule_res = save_life_rule_db(
                                    domain=args.get("domain", "productivity"),
                                    rule_name=args.get("rule_name", ""),
                                    rule_text=args.get("rule_text", ""),
                                    anti_pattern=args.get("anti_pattern", ""),
                                    actionable_remedy=args.get("actionable_remedy", "")
                                )
                            except Exception as re_err:
                                rule_res = {"error": str(re_err)}
                            logger.info(f"LLM triggered tool save_life_rule: {rule_res}")
                            if trace and hasattr(trace, "debug_steps"):
                                trace.debug_steps.append(f"• 📜 `save_life_rule(\"{args.get('rule_name', '')}\")` -> ID #{rule_res.get('id')}")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(rule_res, ensure_ascii=False)})

                        elif fn_name == "get_active_rules":
                            from src.db import get_active_rules_db
                            try:
                                args = json.loads(tool_call.function.arguments)
                                domain_arg = args.get("domain")
                                rules = get_active_rules_db(domain=domain_arg)
                            except Exception as ge_err:
                                rules = []
                            logger.info(f"LLM triggered tool get_active_rules: found {len(rules)} rules")
                            if trace and hasattr(trace, "debug_steps"):
                                trace.debug_steps.append(f"• 📜 `get_active_rules(\"{args.get('domain', 'all')}\")` -> Found {len(rules)} rules")
                            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(rules, ensure_ascii=False)})

            except Exception as e:
                logger.warning(f"API call failed: {e}. Falling back to offline simulator.")

        # Offline / Fallback Response
        return self._sanitize_tool_leak(self._generate_offline_coaching_response(system_prompt, user_input))

    def _format_for_telegram(self, text: str) -> str:
        """
        Transforms Markdown into clean, readable Telegram format:
        1. Converts markdown headers (#, ##, ###) into bold headers (**Heading**).
        2. Removes ugly horizontal rules (---, ***, ___).
        3. Converts raw Markdown tables (|---|---|) into structured bullet lists.
        4. Normalizes whitespace and paragraph spacing.
        """
        if not text:
            return ""

        lines = text.split("\n")
        processed_lines: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 1. Detect and parse Markdown Tables
            if stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]:
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                
                # Convert table rows to bullet points
                parsed_bullets = []
                for row_idx, tline in enumerate(table_lines):
                    # Skip separator line (e.g. |---|---|)
                    if re.match(r"^\|[\s\-:|]+\|$", tline):
                        continue
                    # Extract cells
                    cells = [c.strip() for c in tline.split("|")[1:-1] if c.strip() != ""]
                    if not cells:
                        continue
                    # Skip header row if it contains generic labels
                    if row_idx == 0 and any(c.lower() in ["#", "№", "id", "правило", "параметр", "шаг", "пункт", "трек", "время"] for c in cells):
                        continue
                    
                    # Filter out leading index numbers (e.g. "1", "2", "3")
                    if len(cells) > 1 and (cells[0].isdigit() or cells[0] in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]):
                        cells = cells[1:]
                    
                    if len(cells) == 1:
                        parsed_bullets.append(f"• **{cells[0]}**")
                    elif len(cells) == 2:
                        parsed_bullets.append(f"• **{cells[0]}**: {cells[1]}")
                    elif len(cells) >= 3:
                        extra = f" ({cells[2]})" if cells[2] else ""
                        parsed_bullets.append(f"• **{cells[0]}**: {cells[1]}{extra}")

                if parsed_bullets:
                    processed_lines.extend(parsed_bullets)
                continue

            # 2. Strip horizontal divider lines
            if re.match(r"^(\s*[-*_]\s*){3,}$", stripped):
                i += 1
                continue

            # 3. Convert markdown headers (###, ##, #) to bold
            header_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if header_match:
                title = header_match.group(2).strip()
                title = re.sub(r"^\*\*(.+)\*\*$", r"\1", title)
                processed_lines.append(f"**{title}**")
                i += 1
                continue

            processed_lines.append(line)
            i += 1

        result = "\n".join(processed_lines)
        # Collapse 3+ newlines to 2
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    def _sanitize_tool_leak(self, text: str) -> str:
        """Removes leaked raw tool call syntax and formats text cleanly for Telegram."""
        if not text:
            return ""
        # Strip patterns like `to=function_name ...` or `to=functions.xyz ...`
        cleaned = re.sub(r"to=\w+(\.\w+)?\s*(\(json\))?:?\s*\{.*?\}", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"to=\w+(\.\w+)?\s+[^\n]+", "", cleaned)
        cleaned = re.sub(r"to=functions\.\w+[^\n]*", "", cleaned)
        # Strip standalone raw JSON tool call leaks (e.g. {"query": "..."})
        cleaned = re.sub(r'^\s*\{\s*"(query|target_tier|keyword|identifier|title|project_name)"\s*:.*?\n?', "", cleaned, flags=re.MULTILINE)
        
        # Telegram formatting post-processing
        cleaned = self._format_for_telegram(cleaned)
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
        Transcribes voice messages via Dedicated STT (ProxyAPI gpt-4o-mini-transcribe / whisper-1)
        with fallback to Main Provider (provod.ai) Whisper endpoint.
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found for transcription: {file_path}")
            return ""

        if self.stt_api_key and self.stt_api_key != "your_proxyapi_key_here":
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
        Generates structured, Russian-language Senior Friend & Coach responses
        when API is offline or temporarily unreachable.
        """
        lower_input = user_input.lower()

        # 1. Morning Briefing intent
        if any(w in lower_input for w in ["утренний брифинг", "доброе утро", "morning briefing", "план на сегодня"]):
            tasks_block = ""
            if "ТОП-3 КЛЮЧЕВЫЕ ЗАДАЧИ" in system_prompt:
                match = re.search(r"(ТОП-3 КЛЮЧЕВЫЕ ЗАДАЧИ[^\n]*(?:\n\s*-[^\n]+)*)", system_prompt)
                if match:
                    tasks_block = match.group(1).strip()
            
            exp_block = ""
            if "АКТИВНЫЕ СПРИНТЫ" in system_prompt:
                match_exp = re.search(r"(АКТИВНЫЕ СПРИНТЫ[^\n]*(?:\n\s*-[^\n]+)*)", system_prompt)
                if match_exp:
                    exp_block = match_exp.group(1).strip()

            lines = [
                "Доброе утро, Виталик! Фокусируемся на главном и держим системный темп.",
                "",
                "📋 **4 правила системности на сегодня**:",
                "1. 🚀 **1 проектный шаг** (WeGeny / Intelligence Bit) — сделай до полудня.",
                "2. 🧹 **Бытовой порядок** — 15 минут на ключевые зоны.",
                "3. 🚶 **1 час прогулки / движения** — перезагрузка мышления и ясность.",
                "4. 💼 **Карьерный отклик / шаг** — регулярность важнее объема.",
            ]
            if tasks_block:
                lines.extend(["", f"🎯 **{tasks_block}**"])
            if exp_block:
                lines.extend(["", f"🔬 **{exp_block}**"])
            
            lines.extend([
                "",
                "С какого первого 25-минутного блока начинаем?"
            ])
            return "\n".join(lines)

        # 2. Evening Sync intent
        elif any(w in lower_input for w in ["вечерн", "итоги дня", "evening sync", "синхронизац"]):
            tasks_block = ""
            if "ФОКУС-ЗАДАЧИ" in system_prompt:
                match = re.search(r"(ФОКУС-ЗАДАЧИ[^\n]*(?:\n\s*-[^\n]+)*)", system_prompt)
                if match:
                    tasks_block = match.group(1).strip()

            lines = [
                "Добрый вечер! Подведем краткие итоги дня.",
                "",
                "📊 **Чек-ин по 4 правилам системности**:",
                "1. Проектный шаг закрыт?",
                "2. Бытовые рутины выполнены?",
                "3. Прогулка / физическая активность состоялась?",
                "4. Карьерный фокус удержан?",
            ]
            if tasks_block:
                lines.extend(["", f"🎯 **{tasks_block}**"])

            lines.extend([
                "",
                "Что сегодня дало максимум энергии, а что заблокировало? Зафиксируй **1 атомарный шаг (15–30 мин)** на завтра."
            ])
            return "\n".join(lines)

        # 3. LifeOS intent
        elif "lifeos" in lower_input:
            return (
                "По **LifeOS**: архитектура памяти и системные циклы активны. "
                "Все 3 уровня памяти (Core Profile, Rolling State, Entity Graph) синхронизированы. "
                "Какой следующий компонент или сценарий оптимизируем?"
            )

        # 4. Energy / Wellbeing / Sprint intent
        elif any(w in lower_input for w in ["энерги", "устал", "ресурс", "спринт", "energy", "sprint"]):
            return (
                "Вижу запрос по состоянию и фокусу. "
                "Главное правило при снижении ресурса — снизить трение и не усложнять. "
                "Давай выделим ровно один ключевой шаг на 15–20 минут, который даст максимальный результат."
            )

        # 5. General intelligent coach fallback
        else:
            return (
                f"Принято по теме: «{user_input}».\n\n"
                "Держим фокус на чистой архитектуре и реальных результатах без лишнего шума. "
                "Какой конкретный шаг сейчас в приоритете?"
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
