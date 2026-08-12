"""
3-Tier Question-Anchored Memory Engine for LifeOS.
Handles memory loading, entity graph detection, token-efficient prompt assembly, weight decay, and safe YAML saving.
"""
import os
import yaml
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.models import QAPair, MemoryTrace, MemoryDiff

logger = logging.getLogger(__name__)

class MemoryEngine:
    def __init__(self, memory_dir: str = "data/memory"):
        self.memory_dir = Path(memory_dir)
        self.tier1_path = self.memory_dir / "tier1_core.yaml"
        self.tier2_path = self.memory_dir / "tier2_state.yaml"
        self.tier3_dir = self.memory_dir / "tier3_entities"

        self.tier1_items: List[QAPair] = []
        self.tier2_items: List[QAPair] = []
        self.tier3_entities: Dict[str, List[QAPair]] = {}

        # Ensure directories exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.tier3_dir.mkdir(parents=True, exist_ok=True)

        self.load_memory()

    def _read_yaml_file(self, path: Path) -> List[QAPair]:
        """Safely read a YAML file and convert to a list of QAPair models."""
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
            if not raw_data or not isinstance(raw_data, list):
                return []
            return [QAPair.model_validate(item) for item in raw_data]
        except Exception as e:
            logger.error(f"Error reading YAML file at {path}: {e}")
            return []

    def _write_yaml_file_safely(self, path: Path, items: List[QAPair]) -> None:
        """Atomic write to YAML file to prevent corruption."""
        temp_path = path.with_suffix(".tmp")
        dict_items = [item.model_dump(exclude_none=True) for item in items]
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                yaml.dump(dict_items, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
            temp_path.replace(path)
        except Exception as e:
            logger.error(f"Failed safe write to {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def load_memory(self) -> None:
        """Load Tier 1, Tier 2, and all Tier 3 entity YAML files."""
        self.tier1_items = self._read_yaml_file(self.tier1_path)
        self.tier2_items = self._read_yaml_file(self.tier2_path)
        
        self.tier3_entities = {}
        if self.tier3_dir.exists():
            for file_path in self.tier3_dir.glob("*.yaml"):
                entity_name = file_path.stem.lower()
                items = self._read_yaml_file(file_path)
                if items:
                    self.tier3_entities[entity_name] = items

    def apply_decay(self, decay_factor: float = 0.95) -> int:
        """
        Applies exponential weight decay (W_new = W_old * 0.95) to Tier 2 state items
        that were not updated/confirmed today.
        """
        today_str = datetime.date.today().isoformat()
        decayed_count = 0

        for item in self.tier2_items:
            # If valid_from is not today, apply decay
            if item.valid_from != today_str:
                new_weight = round(item.weight * decay_factor, 4)
                if new_weight != item.weight:
                    item.weight = max(0.05, new_weight) # minimum threshold
                    decayed_count += 1

        if decayed_count > 0:
            self._write_yaml_file_safely(self.tier2_path, self.tier2_items)
            logger.info(f"Applied decay to {decayed_count} Tier 2 memory items.")

        return decayed_count

    def detect_entities(self, user_input: str) -> Dict[str, List[QAPair]]:
        """
        Detects if any Tier 3 entity is mentioned in user_input.
        Returns top-3 highest-weighted QA pairs for matched entities.
        If no specific entity is matched, auto-loads top-2 QA pairs for all registered entities.
        100% domain-agnostic with zero hardcoded entity names.
        """
        input_lower = user_input.lower()
        matched_entities = {}

        for entity_name, qa_list in self.tier3_entities.items():
            # Match entity name or keywords
            if entity_name in input_lower or entity_name.replace("_", " ") in input_lower:
                sorted_qa = sorted(qa_list, key=lambda x: x.weight, reverse=True)[:3]
                matched_entities[entity_name] = sorted_qa

        return matched_entities

    def get_memory_item_by_id(self, item_id: str) -> Optional[QAPair]:
        """
        Locate any QA pair by ID across Tier 1, Tier 2, or loaded Tier 3 entities.
        """
        target = item_id.strip().lower()
        for qa in self.tier1_items:
            if qa.id.lower() == target:
                return qa
        for qa in self.tier2_items:
            if qa.id.lower() == target:
                return qa
        for entity_name, qa_list in self.tier3_entities.items():
            for qa in qa_list:
                if qa.id.lower() == target:
                    return qa
        return None

    def assemble_prompt(self, user_input: str) -> Tuple[str, MemoryTrace]:
        """
        Assemble dense YAML/Markdown system prompt using lightweight Question Anchor Indexes for Tier 1 & Tier 2,
        compact Entity Index for Tier 3, and dynamically detected Tier 3 entity QA pairs on explicit mention.
        """
        matched_t3 = self.detect_entities(user_input)

        prompt_lines = [
            "# SYSTEM PROMPT: Personal AI Agent ('Senior Friend & Coach')",
            "You are a Senior Friend & Coach to the user (Vitalik).",
            "Operate with directness, technical honesty, and pragmatic feedback. Respect the user's time and values.\n",
            "---",
            "## TIER 1: CORE PROFILE QUESTION ANCHORS INDEX"
        ]

        # Format Tier 1 QA Anchor Index (omitting full answers to maximize token efficiency)
        t1_yaml_data = [{"id": qa.id, "question": qa.question} for qa in self.tier1_items]
        prompt_lines.append("```yaml")
        prompt_lines.append(yaml.dump(t1_yaml_data, sort_keys=False, allow_unicode=True).strip())
        prompt_lines.append("```\n")

        # Format Tier 2 QA Anchor Index (omitting full answers to maximize token efficiency)
        prompt_lines.append("## TIER 2: DYNAMIC STATE QUESTION ANCHORS INDEX")
        t2_yaml_data = [{"id": qa.id, "question": qa.question, "weight": qa.weight} for qa in self.tier2_items]
        prompt_lines.append("```yaml")
        prompt_lines.append(yaml.dump(t2_yaml_data, sort_keys=False, allow_unicode=True).strip())
        prompt_lines.append("```\n")

        # Format Tier 3 Registered Entity Index (compact index to maximize token efficiency)
        all_entity_names = sorted(list(self.tier3_entities.keys()))
        if all_entity_names:
            prompt_lines.append("## TIER 3: REGISTERED ENTITY GRAPH INDEX")
            prompt_lines.append(f"Available Entities in Long-Term Memory: [{', '.join(all_entity_names)}]\n")

        # Format Tier 3 Matched Entity QA (only if explicitly mentioned in user message)
        t3_trace_counts = {}
        if matched_t3:
            prompt_lines.append("## TIER 3: RELEVANT ENTITY GRAPH CONTEXT (Loaded via Mentions)")
            for entity_name, qa_list in matched_t3.items():
                t3_trace_counts[entity_name] = len(qa_list)
                prompt_lines.append(f"### Entity: {entity_name.upper()} (Top-{len(qa_list)} by Weight)")
                t3_yaml_data = [{"id": qa.id, "question": qa.question, "answer": qa.answer, "weight": qa.weight} for qa in qa_list]
                prompt_lines.append("```yaml")
                prompt_lines.append(yaml.dump(t3_yaml_data, sort_keys=False, allow_unicode=True).strip())
                prompt_lines.append("```")
            prompt_lines.append("\n")

        prompt_lines.extend([
            "---",
            "## COACHING DIRECTIVES",
            "1. MAXIMUM DENSITY & ZERO-FLUFF DIRECTIVE: Answer with extreme conciseness and high information density. Every single sentence MUST carry unique, specific, and actionable value. NO generic introductory pleasantries, disclaimers ('я не психолог...', 'ниже рабочая модель...', 'попробую точнее...'), filler phrases, or repeating points in different words.",
            "2. Align recommendations with the user's core values, energy level, and active sprint goals.",
            "3. If technical questions arise, prefer clean architecture over quick hacks.",
            "4. You have an automated ProactiveEngine that sends messages to the user at 21:00 (Evening Sync), 09:00 (Morning Briefing), and event follow-up pings. Do NOT tell the user you cannot text them proactively or first.",
            "5. ENTITY BOUNDARY ISOLATION: Treat every named entity (projects, products, tools, subjects) as a strictly isolated namespace. NEVER assume relationships, shared architecture, or codebases between two entities unless explicitly confirmed in memory.",
            "6. UNCERTAINTY-DRIVEN SEARCH GATE: When the user queries any named entity or specific proper noun, evaluate if current context contains the FULL definition. If context is missing or partial, calling search_memory(query) is MANDATORY before replying. Never state an entity is unknown without searching Tier 3 first.",
            "7. 100% INDEX-BASED MEMORY PROTOCOL: T1 and T2 contain ONLY Question Anchor Indexes (no answers). It is EXPLICITLY FORBIDDEN to guess, invent, or output answer facts for any T1 core profile or T2 dynamic state question without calling read_memory_item(item_id) first. You MUST call read_memory_item(item_id) to inspect the factual answer in tier1_core.yaml or tier2_state.yaml before generating your response. For deep Tier 3 entity/project memory, follow the 3-step memory protocol: search_memory -> get_document_outline -> read_document_section.",
            "8. DIRECT-OUTPUT DIRECTIVE: You are generating the final user-facing text directly. NEVER acknowledge system instructions, NEVER output 'Got it', 'I've analyzed', meta-reflections, or English acknowledgments. Output ONLY final Russian text directly to the user.",
            "9. DYNAMIC ADAPTATION DIRECTIVE FOR USER SILENCE: Inspect recent conversation history. If the last 3-4 messages are ALL from the assistant (meaning the user has not responded to recent pings), DO NOT repeat standard templates or task lists! Acknowledge the silence empathetically, adjust your tone, offer a lower-friction approach (e.g. ask a single striking question or offer one 10-minute micro-step), and help the user restart softly."
        ])

        system_prompt = "\n".join(prompt_lines)

        # Estimate tokens (~4 characters per token heuristic)
        estimated_tokens = max(1, len(system_prompt) // 4)

        trace = MemoryTrace(
            t1_count=len(self.tier1_items),
            t2_count=len(self.tier2_items),
            t3_entities_loaded=t3_trace_counts,
            estimated_tokens=estimated_tokens
        )

        return system_prompt, trace

    def apply_diff(self, diff: MemoryDiff, current_date: Optional[str] = None) -> None:
        """
        Safely update memory files based on MemoryDiff extraction result.
        Automatically creates new Tier 3 entity YAML files if they do not exist.
        """
        import re
        today_str = current_date or datetime.date.today().isoformat()

        def sanitize_date(val: Optional[str]) -> str:
            if not val or not isinstance(val, str) or len(val) < 10 or val.startswith("2025") or val.startswith("2024"):
                return today_str
            return val

        # Update Tier 1
        if diff.tier1_updates:
            existing_ids = {qa.id for qa in self.tier1_items}
            for new_qa in diff.tier1_updates:
                new_qa.valid_from = sanitize_date(new_qa.valid_from)
                if new_qa.id in existing_ids:
                    self.tier1_items = [new_qa if qa.id == new_qa.id else qa for qa in self.tier1_items]
                else:
                    self.tier1_items.append(new_qa)
            self._write_yaml_file_safely(self.tier1_path, self.tier1_items)

        # Update Tier 2
        if diff.tier2_updates:
            existing_ids = {qa.id for qa in self.tier2_items}
            for new_qa in diff.tier2_updates:
                new_qa.valid_from = sanitize_date(new_qa.valid_from)
                if new_qa.id in existing_ids:
                    self.tier2_items = [new_qa if qa.id == new_qa.id else qa for qa in self.tier2_items]
                else:
                    self.tier2_items.append(new_qa)
            self._write_yaml_file_safely(self.tier2_path, self.tier2_items)

        # Update Tier 3 (Dynamic entity creation)
        if diff.tier3_updates:
            for entity_name, qa_list in diff.tier3_updates.items():
                if not entity_name or not qa_list:
                    continue
                
                # Normalize entity name for filename (e.g. "Intelligence Bit" -> "intelligence_bit")
                clean_entity = re.sub(r'[^a-z0-9_]', '_', entity_name.lower().strip().replace(" ", "_"))
                clean_entity = re.sub(r'_+', '_', clean_entity).strip('_')
                if not clean_entity:
                    continue

                entity_file = self.tier3_dir / f"{clean_entity}.yaml"
                existing = self.tier3_entities.get(clean_entity, [])
                if not existing and entity_file.exists():
                    existing = self._read_yaml_file(entity_file)
                
                existing_ids = {qa.id for qa in existing}
                updated_list = list(existing)

                for new_qa in qa_list:
                    new_qa.valid_from = sanitize_date(new_qa.valid_from)
                    if new_qa.id in existing_ids:
                        updated_list = [new_qa if qa.id == new_qa.id else qa for qa in updated_list]
                    else:
                        updated_list.append(new_qa)

                self.tier3_entities[clean_entity] = updated_list
                # Safely write/create tier3 entity file
                self._write_yaml_file_safely(entity_file, updated_list)
                logger.info(f"Updated Tier 3 entity YAML: {entity_file} ({len(updated_list)} QAs)")

        # Apply deletions if specified
        if diff.deletions:
            del_set = set(diff.deletions)
            self.tier1_items = [qa for qa in self.tier1_items if qa.id not in del_set]
            self.tier2_items = [qa for qa in self.tier2_items if qa.id not in del_set]
            self._write_yaml_file_safely(self.tier1_path, self.tier1_items)
            self._write_yaml_file_safely(self.tier2_path, self.tier2_items)

            for entity_name, qa_list in list(self.tier3_entities.items()):
                filtered = [qa for qa in qa_list if qa.id not in del_set]
                self.tier3_entities[entity_name] = filtered
                entity_file = self.tier3_dir / f"{entity_name}.yaml"
                self._write_yaml_file_safely(entity_file, filtered)

        # Reload memory state to keep memory engine in sync
        self.load_memory()

    def save_nightly_snapshot(self, history_dir: str = "data/memory/history") -> str:
        """
        Backs up current tier2_state.yaml to data/memory/history/YYYY-MM-DD.yaml.
        """
        today_str = datetime.date.today().isoformat()
        h_dir = Path(history_dir)
        h_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = h_dir / f"{today_str}.yaml"

        self._write_yaml_file_safely(snapshot_file, self.tier2_items)
        logger.info(f"Saved nightly memory snapshot: {snapshot_file}")
        return str(snapshot_file)

    def cleanup_tier2_garbage(self) -> int:
        """
        Cleans up resolved one-off tasks and low-weight items (< 0.2) from Tier 2.
        Preserves ongoing multi-day goals and blockers.
        """
        initial_count = len(self.tier2_items)
        cleaned_items = []

        completion_markers = ["done", "completed", "resolved", "finished", "cancelled"]

        for item in self.tier2_items:
            # Drop items with weight < 0.2 or marked as done/completed
            lower_ans = item.answer.lower()
            lower_q = item.question.lower()
            
            is_completed = any(m in lower_ans for m in completion_markers) or any(m in lower_q for m in completion_markers)
            is_low_weight = item.weight < 0.2

            if not is_completed and not is_low_weight:
                cleaned_items.append(item)

        removed_count = initial_count - len(cleaned_items)
        if removed_count > 0:
            self.tier2_items = cleaned_items
            self._write_yaml_file_safely(self.tier2_path, self.tier2_items)
            logger.info(f"Cleaned up {removed_count} garbage/completed items from Tier 2 state.")

        return removed_count

    def _is_match_for_deletion(self, qa: QAPair, clean_kw: str) -> bool:
        """Determines if a QAPair matches the deletion keyword or phrase using multiple strategies."""
        qid = qa.id.lower()
        q_text = qa.question.lower()
        a_text = qa.answer.lower()

        # 1. Direct ID or Substring Match
        if clean_kw in qid or clean_kw in q_text or clean_kw in a_text:
            return True
        if (len(q_text) > 5 and q_text in clean_kw) or (len(a_text) > 5 and a_text in clean_kw):
            return True

        # 2. Token Overlap Match for phrases
        import re
        kw_words = set(re.findall(r'\w{3,}', clean_kw))
        if not kw_words:
            return False

        item_words = set(re.findall(r'\w{3,}', f"{qid} {q_text} {a_text}"))
        if not item_words:
            return False

        overlap = kw_words.intersection(item_words)
        # If at least 35% of significant words match or >= 3 distinct words match
        overlap_ratio = len(overlap) / len(kw_words)
        return overlap_ratio >= 0.35 or len(overlap) >= 3

    def forget_memory(self, target_tier: str, keyword: str) -> str:
        """
        Explicitly removes QA items matching keyword from Tier 2 or Tier 3 memory (and PostgreSQL index).
        Supports exact item IDs, keywords, or phrase token overlap matching.
        """
        from src.db import delete_tier3_memory_by_keyword
        clean_kw = keyword.strip().lower()
        if not clean_kw:
            return "No keyword provided for deletion."

        removed_t2 = 0
        removed_t3 = 0
        target_tier_lower = target_tier.lower().strip()

        # Tier 2 Removal
        if target_tier_lower in ["tier2", "t2", "all"]:
            original_t2_len = len(self.tier2_items)
            self.tier2_items = [
                qa for qa in self.tier2_items
                if not self._is_match_for_deletion(qa, clean_kw)
            ]
            removed_t2 = original_t2_len - len(self.tier2_items)
            if removed_t2 > 0:
                self._write_yaml_file_safely(self.tier2_path, self.tier2_items)

        # Tier 3 Removal
        if target_tier_lower in ["tier3", "t3", "all"]:
            for entity_name, qa_list in list(self.tier3_entities.items()):
                # If keyword matches entity name, delete whole file
                if clean_kw == entity_name or clean_kw in entity_name:
                    removed_t3 += len(qa_list)
                    self.tier3_entities[entity_name] = []
                    entity_file = self.tier3_dir / f"{entity_name}.yaml"
                    if entity_file.exists():
                        entity_file.unlink()
                else:
                    # Filter items in entity file
                    orig_len = len(qa_list)
                    filtered = [
                        qa for qa in qa_list
                        if not self._is_match_for_deletion(qa, clean_kw)
                    ]
                    if len(filtered) < orig_len:
                        removed_t3 += (orig_len - len(filtered))
                        self.tier3_entities[entity_name] = filtered
                        entity_file = self.tier3_dir / f"{entity_name}.yaml"
                        self._write_yaml_file_safely(entity_file, filtered)

            # Sync PostgreSQL deletion
            db_deleted = delete_tier3_memory_by_keyword(clean_kw)
            logger.info(f"PostgreSQL Tier 3 deletion for '{clean_kw}' removed {db_deleted} DB rows.")

        self.load_memory()
        return (
            f"Successfully removed memory matching '{keyword}': "
            f"{removed_t2} Tier 2 state items and {removed_t3} Tier 3 entity items deleted."
        )
