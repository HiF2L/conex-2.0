"""
Pydantic v2 data models for LifeOS memory system and LLM interactions.
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """
    Core Question-Anchored Memory Item.
    """
    id: str = Field(..., description="Unique identifier for the QA pair")
    question: str = Field(..., description="The anchor question defining this memory item")
    answer: str = Field(..., description="The factual or contextual answer")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance / importance weight (decayable)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score in validity")
    origin: str = Field(default="user_input", description="Source of memory (e.g. initial_seed, conversation, voice_dump)")
    valid_from: Optional[str] = Field(default=None, description="ISO Date string when item became valid")
    valid_until: Optional[str] = Field(default=None, description="ISO Date string when item expired/superseded")


class ScheduledPingItem(BaseModel):
    """
    Scheduled follow-up event ping model.
    """
    id: Optional[str] = Field(default=None, description="Optional identifier for the scheduled ping")
    scheduled_at: str = Field(..., description="ISO Timestamp (e.g. 2026-08-04T14:30:00) when ping should be triggered")
    event_type: str = Field(default="event_followup", description="Event category (e.g. event_followup, meeting, doctor)")
    context_text: str = Field(..., description="Details/context about the time-sensitive event")


class MemoryDiff(BaseModel):
    """
    Structured extraction outcome representing changes to be committed to disk.
    """
    tier1_updates: List[QAPair] = Field(default_factory=list, description="New or updated Tier 1 Core QA items")
    tier2_updates: List[QAPair] = Field(default_factory=list, description="New or updated Tier 2 Dynamic State QA items")
    tier3_updates: Dict[str, List[QAPair]] = Field(
        default_factory=dict, 
        description="Map of entity_name (e.g., 'wegeny', 'intelligence_bit', 'health', 'music', 'youtube', 'cleaning', 'lifeos') -> list of new/updated Tier 3 QA items"
    )
    deletions: List[str] = Field(default_factory=list, description="List of QA IDs to remove or invalidate")
    scheduled_pings: List[ScheduledPingItem] = Field(default_factory=list, description="Extracted time-sensitive event pings")


class MemoryTrace(BaseModel):
    """
    Metadata trace tracking prompt composition and token count.
    """
    t1_count: int = 0
    t2_count: int = 0
    t3_entities_loaded: Dict[str, int] = Field(default_factory=dict)
    t3_sections_read: int = 0
    estimated_tokens: int = 0
    debug_steps: List[str] = Field(default_factory=list, description="Step-by-step tool execution logs for /debug mode")

    @property
    def t3_total(self) -> int:
        return self.t3_sections_read

    def format_trace_str(self) -> str:
        t3_parts = [f"{entity.capitalize()} ({count} Qs)" for entity, count in self.t3_entities_loaded.items()]
        t3_str = ", ".join(t3_parts) if t3_parts else "None"
        return f"[Memory Trace: T1: {self.t1_count} Qs | T2: {self.t2_count} Qs | T3: {t3_str} | Est. Tokens: ~{self.estimated_tokens}]"


class ChatTurn(BaseModel):
    """
    A single turn of conversation.
    """
    user_message: str
    agent_response: str
    timestamp: Optional[str] = None


class ProjectItem(BaseModel):
    """
    Project category model.
    """
    id: Optional[int] = None
    name: str = Field(..., description="Unique project name")
    description: Optional[str] = Field(default="", description="Project description or mission")
    status: str = Field(default="active", description="Project status (active, archived)")
    created_at: Optional[str] = None


class TaskItem(BaseModel):
    """
    Task model for PostgreSQL Task Manager.
    """
    id: Optional[int] = None
    project_id: Optional[int] = None
    project_name: Optional[str] = None
    title: str = Field(..., description="Task title or action item")
    description: Optional[str] = Field(default="", description="Task description")
    priority: int = Field(default=2, ge=1, le=3, description="Priority (1=High, 2=Medium, 3=Low)")
    status: str = Field(default="todo", description="Status (todo, in_progress, done, cancelled)")
    due_date: Optional[str] = Field(default=None, description="Due date ISO string (YYYY-MM-DD)")
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class WellbeingLogItem(BaseModel):
    """
    Physical, cognitive, or emotional wellbeing event log.
    """
    id: Optional[int] = None
    user_id: Optional[int] = None
    state_type: str = Field(..., description="State category (e.g. PEAK_CLARITY, BRAIN_FOG, LOW_ENERGY, ANXIETY)")
    triggers: List[str] = Field(default_factory=list, description="Triggers or catalysts")
    symptoms: List[str] = Field(default_factory=list, description="Observed symptoms")
    notes: Optional[str] = Field(default="", description="Reflections and observations")
    created_at: Optional[str] = None


class ExperimentItem(BaseModel):
    """
    Sprint or A/B test habit experiment model.
    """
    id: Optional[int] = None
    user_id: Optional[int] = None
    title: str = Field(..., description="Experiment title")
    type: str = Field(default="SPRINT", description="Type: SPRINT or AB_TEST")
    phase: str = Field(default="PHASE_A", description="Phase: PHASE_A, PHASE_B, COMPLETED")
    duration_days: int = Field(default=14, description="Duration in days")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    hypothesis_a: str = Field(..., description="Primary hypothesis / Protocol A")
    hypothesis_b: Optional[str] = Field(default="", description="Protocol B for A/B tests")
    daily_actions: List[str] = Field(default_factory=list, description="Daily actionable steps")
    status: str = Field(default="active", description="Status: active, completed, cancelled")
    created_at: Optional[str] = None

