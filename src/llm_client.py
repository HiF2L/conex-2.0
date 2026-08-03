"""
LLM Client Wrapper utilizing standard OpenAI client with dotenv configuration,
Whisper audio transcription, and intelligent fallback support.
"""
import os
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
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate coaching agent response using DEFAULT_MODEL.
        Supports sliding conversation history and OpenAI Function Calling for search_memory tool.
        """
        from src.db import search_tier3_memory

        tool_definition = {
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
                    tools=[tool_definition],
                    tool_choice="auto",
                    temperature=0.7
                )
                
                choice = response.choices[0]

                # Check if model requested tool call (search_memory)
                if choice.message.tool_calls:
                    messages.append(choice.message) # append assistant tool request
                    
                    for tool_call in choice.message.tool_calls:
                        if tool_call.function.name == "search_memory":
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

                    # Second call to generate final answer using search results
                    final_response = self.client.chat.completions.create(
                        model=self.default_model,
                        messages=messages,
                        temperature=0.7
                    )
                    if final_response.choices and final_response.choices[0].message.content:
                        return final_response.choices[0].message.content.strip()

                elif choice.message.content:
                    return choice.message.content.strip()

            except Exception as e:
                logger.warning(f"API call failed: {e}. Falling back to offline simulator.")

        # Offline / Fallback Response
        return self._generate_offline_coaching_response(system_prompt, user_input)

    def extract_memory_diff(self, system_prompt: str, text_to_analyze: str, current_date: Optional[str] = None) -> MemoryDiff:
        """
        Extract memory updates as a structured MemoryDiff using FAST_MODEL.
        Injects CURRENT_DATE and enforces dynamic Tier 3 entity categorization.
        Robustly handles API provider constraints (e.g. provod.ai 400 errors).
        """
        import datetime
        today_str = current_date or datetime.date.today().isoformat()

        if self.is_api_configured():
            extraction_prompt = (
                f"{system_prompt}\n\n"
                f"CURRENT_DATE: {today_str}\n\n"
                "CRITICAL EXTRACTION RULES:\n"
                f"1. DATE ACCURACY: All new or updated QA items MUST use valid_from = '{today_str}'. DO NOT use past years or hallucinated dates.\n"
                "2. TIER 3 DYNAMIC ENTITIES: Identify any specific projects, topics, or domains mentioned in the text "
                "(e.g., 'wegeny', 'intelligence_bit', 'health', 'music', 'youtube', 'cleaning', 'lifeos', etc.). "
                "You MUST create separate entry keys in 'tier3_updates' for each entity (e.g., 'tier3_updates': {'wegeny': [...], 'health': [...]}). "
                "Do NOT lump project-specific facts into Tier 1 or Tier 2.\n"
                "3. Tier 1 is ONLY for static core identity/values/communication preferences.\n"
                "4. Tier 2 is ONLY for current energy level, overall sprint goal, and immediate blockers.\n\n"
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

        # 1. State / Energy in Tier 2
        if "energy" in lower_text or "dump" in lower_text or "voice" in lower_text or "sprint" in lower_text:
            t2_updates.append(QAPair(
                id=f"state_energy_{int(datetime.datetime.now().timestamp())}",
                question="What is the user's latest reported energy state or dump focus?",
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
