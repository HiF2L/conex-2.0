"""
Proactive Push Engine for LifeOS Personal AI Agent.
Manages event-driven pings, Quiet Hours (22:00 - 08:00), daily ping limits, and empathetic follow-up delivery.
"""
import os
import logging
import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from aiogram import Bot

from src.db import get_due_pings, mark_ping_status, get_pings_count_today
from src.llm_client import LLMClient

load_dotenv()

logger = logging.getLogger(__name__)


class ProactiveEngine:
    def __init__(self):
        self.max_pings_per_day = int(os.getenv("PROACTIVE_MAX_PINGS_PER_DAY", "3"))
        self.quiet_start_str = os.getenv("PROACTIVE_QUIET_START", "22:00").strip()
        self.quiet_end_str = os.getenv("PROACTIVE_QUIET_END", "08:00").strip()

        # Parse Quiet Start / End times
        try:
            sh, sm = map(int, self.quiet_start_str.split(":"))
            self.quiet_start = datetime.time(hour=sh, minute=sm)
        except Exception:
            self.quiet_start = datetime.time(hour=22, minute=0)

        try:
            eh, em = map(int, self.quiet_end_str.split(":"))
            self.quiet_end = datetime.time(hour=eh, minute=em)
        except Exception:
            self.quiet_end = datetime.time(hour=8, minute=0)

    def is_in_quiet_hours(self, current_time: Optional[datetime.time] = None) -> bool:
        """
        Check if current_time falls inside Quiet Hours (e.g. 22:00 to 08:00).
        """
        now_time = current_time or datetime.datetime.now().time()
        
        if self.quiet_start > self.quiet_end:
            # Overnight Quiet Hours (e.g. 22:00 -> 08:00)
            return now_time >= self.quiet_start or now_time < self.quiet_end
        else:
            # Daytime Quiet Hours (e.g. 13:00 -> 15:00)
            return self.quiet_start <= now_time < self.quiet_end

    async def check_and_execute_pings(self, bot: Bot, target_user_id: int, llm_client: LLMClient) -> int:
        """
        Checks for due event pings, enforces Quiet Hours & daily limits, and sends follow-ups.
        Returns count of executed pings.
        """
        from src.scheduler import send_safe_bot_message

        if not target_user_id:
            return 0

        # 1. Enforce Quiet Hours
        if self.is_in_quiet_hours():
            logger.info("ProactiveEngine: Currently inside Quiet Hours. Suppressing event pings.")
            return 0

        # 2. Enforce Daily Max Pings Limit
        pings_today = get_pings_count_today(target_user_id)
        if pings_today >= self.max_pings_per_day:
            logger.info(f"ProactiveEngine: Daily limit reached ({pings_today}/{self.max_pings_per_day} pings). Suppressing non-critical pings.")
            return 0

        # 3. Retrieve Due Pings
        due_pings = get_due_pings(target_user_id)
        if not due_pings:
            return 0

        executed_count = 0
        for ping in due_pings:
            # Check remaining quota
            if (pings_today + executed_count) >= self.max_pings_per_day:
                logger.info("ProactiveEngine: Reached daily max pings limit during loop.")
                break

            ping_id = ping["id"]
            context_text = ping["context_text"]
            event_type = ping.get("event_type", "event_followup")

            prompt = (
                f"Ты — дружелюбный и внимательный Senior Friend & Coach пользователя. "
                f"Сформируй короткое (1–2 предложения) сообщение на русском языке в Telegram, поинтересовавшись, как прошло мероприятие: '{context_text}'. "
                "Пиши тепло, естественным языком, прямо и без воды. Отвечай строго на русском языке."
            )
            
            try:
                ping_message = llm_client.generate_coaching_response(
                    system_prompt=prompt,
                    user_input=f"Check in on event: {context_text}"
                )
                
                formatted_text = f"🔔 **Event Follow-Up**\n\n{ping_message}"
                await send_safe_bot_message(bot, target_user_id, formatted_text, parse_mode="Markdown")
                
                mark_ping_status(ping_id, "executed")
                executed_count += 1
                logger.info(f"Executed proactive ping #{ping_id} for user {target_user_id}: '{context_text}'")
            except Exception as e:
                logger.error(f"Failed to execute proactive ping #{ping_id}: {e}")

        return executed_count
