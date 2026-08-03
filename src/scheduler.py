"""
Daily proactive scheduler service using APScheduler.
Pushes Morning Briefings (09:00) and Evening Reflections (21:00) to the allowed Telegram ID.
"""
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest

from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

async def send_safe_bot_message(bot: Bot, target_user_id: int, text: str, parse_mode: str = "Markdown"):
    """Send message to target user, splitting into chunks if text > 3900 chars."""
    from src.bot import split_message
    chunks = split_message(text, max_chunk_size=3900)
    for chunk in chunks:
        try:
            await bot.send_message(chat_id=target_user_id, text=chunk, parse_mode=parse_mode)
        except TelegramBadRequest:
            await bot.send_message(chat_id=target_user_id, text=chunk, parse_mode=None)
        except Exception as e:
            logger.error(f"Failed to send scheduled message chunk: {e}")

async def send_morning_briefing(bot: Bot, target_user_id: int, memory_engine: MemoryEngine, llm_client: LLMClient):
    """09:00 Proactive Morning Briefing."""
    if not target_user_id:
        return
    
    # 1. Apply daily decay at start of day
    memory_engine.apply_decay()
    
    # 2. Assemble morning context
    prompt, trace = memory_engine.assemble_prompt("Morning briefing and today's sprint focus plan.")
    briefing_query = "Generate a concise, high-impact 3-bullet Morning Briefing for today focusing on my active sprint goals."
    response = llm_client.generate_coaching_response(prompt, briefing_query)
    
    full_message = f"🌅 **Morning Briefing (09:00)**\n\n{response}\n\n_🧠 [Memory Trace: T1: {trace.t1_count} Qs | T2: {trace.t2_count} Qs | T3: {trace.t3_total} Qs | ~{trace.estimated_tokens} tokens]_"

    await send_safe_bot_message(bot, target_user_id, full_message, parse_mode="Markdown")

async def send_evening_reflection(bot: Bot, target_user_id: int, memory_engine: MemoryEngine, llm_client: LLMClient):
    """21:00 Proactive Evening Reflection."""
    if not target_user_id:
        return
    
    reflection_text = (
        "🌆 **Evening Reflection (21:00)**\n\n"
        "How did your sprint progress today? Send a quick text or voice dump with your wins, blockers, or mindset updates to update your 3-Tier memory!"
    )

    await send_safe_bot_message(bot, target_user_id, reflection_text, parse_mode="Markdown")

def create_scheduler(bot: Bot, target_user_id: int, memory_engine: MemoryEngine, llm_client: LLMClient) -> AsyncIOScheduler:
    """
    Initialize and return APScheduler instance configured for daily 09:00 and 21:00 triggers.
    """
    scheduler = AsyncIOScheduler()
    
    # 09:00 Morning Briefing
    scheduler.add_job(
        send_morning_briefing,
        trigger="cron",
        hour=9,
        minute=0,
        args=[bot, target_user_id, memory_engine, llm_client],
        id="morning_briefing_job",
        replace_existing=True
    )
    
    # 21:00 Evening Reflection
    scheduler.add_job(
        send_evening_reflection,
        trigger="cron",
        hour=21,
        minute=0,
        args=[bot, target_user_id, memory_engine, llm_client],
        id="evening_reflection_job",
        replace_existing=True
    )
    
    return scheduler
