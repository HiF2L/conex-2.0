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
        Returns top-3 highest-weighted QA pairs for each matched entity.
        """
        input_lower = user_input.lower()
        matched_entities = {}

        for entity_name, qa_list in self.tier3_entities.items():
            # Match entity name or keywords
            if entity_name in input_lower or entity_name.replace("_", " ") in input_lower:
                # Sort by weight descending, pick top 3
                sorted_qa = sorted(qa_list, key=lambda x: x.weight, reverse=True)[:3]
                matched_entities[entity_name] = sorted_qa

        return matched_entities

    def assemble_prompt(self, user_input: str) -> Tuple[str, MemoryTrace]:
        """
        Assemble dense YAML/Markdown system prompt using loaded Tier 1, Tier 2,
        and dynamically detected Tier 3 entity QA pairs.
        """
        matched_t3 = self.detect_entities(user_input)

        prompt_lines = [
            "# SYSTEM PROMPT: Personal AI Agent ('Senior Friend & Coach')",
            "You are a Senior Friend & Coach to the user (Vitalik).",
            "Operate with directness, technical honesty, and pragmatic feedback. Respect the user's time and values.\n",
            "---",
            "## TIER 1: CORE PROFILE (Always Loaded)"
        ]

        # Format Tier 1 QA
        t1_yaml_data = [{"question": qa.question, "answer": qa.answer} for qa in self.tier1_items]
        prompt_lines.append("```yaml")
        prompt_lines.append(yaml.dump(t1_yaml_data, sort_keys=False, allow_unicode=True).strip())
        prompt_lines.append("```\n")

        # Format Tier 2 QA
        prompt_lines.append("## TIER 2: DYNAMIC STATE & SPRINT GOALS (Always Loaded)")
        t2_yaml_data = [{"question": qa.question, "answer": qa.answer, "weight": qa.weight} for qa in self.tier2_items]
        prompt_lines.append("```yaml")
        prompt_lines.append(yaml.dump(t2_yaml_data, sort_keys=False, allow_unicode=True).strip())
        prompt_lines.append("```\n")

        # Format Tier 3 Matched Entity QA
        t3_trace_counts = {}
        if matched_t3:
            prompt_lines.append("## TIER 3: RELEVANT ENTITY GRAPH CONTEXT (Loaded via Mentions)")
            for entity_name, qa_list in matched_t3.items():
                t3_trace_counts[entity_name] = len(qa_list)
                prompt_lines.append(f"### Entity: {entity_name.upper()} (Top-{len(qa_list)} by Weight)")
                t3_yaml_data = [{"question": qa.question, "answer": qa.answer, "weight": qa.weight} for qa in qa_list]
                prompt_lines.append("```yaml")
                prompt_lines.append(yaml.dump(t3_yaml_data, sort_keys=False, allow_unicode=True).strip())
                prompt_lines.append("```")
            prompt_lines.append("\n")

        prompt_lines.extend([
            "---",
            "## COACHING DIRECTIVES",
            "1. Answer concisely and directly without fluff or unnecessary introductory pleasantries.",
            "2. Align recommendations with the user's core values, energy level, and active sprint goals.",
            "3. If technical questions arise, prefer clean architecture over quick hacks."
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
