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

    # 2. Fetch sliding chat history (last 8 turns) to detect user silence & adapt tone
    from src.db import get_recent_chat_history, get_top_focus_tasks_db
    chat_history = get_recent_chat_history(target_user_id, limit=8)

    # 3. Fetch top-3 focus tasks
    top_tasks, total_tasks = get_top_focus_tasks_db(limit=3)
    if top_tasks:
        items = [f"- [P{t.get('priority', 2)}] {t['title']}" for t in top_tasks]
        remaining = max(0, total_tasks - len(top_tasks))
        queue_str = f"\n  (и еще {remaining} задач в очереди — отправь /tasks)" if remaining > 0 else ""
        task_context = "ТОП-3 КЛЮЧЕВЫЕ ЗАДАЧИ ИЗ БАЗЫ:\n" + "\n".join(items) + queue_str
    else:
        task_context = "АКТИВНЫЕ ЗАДАЧИ: Все текущие задачи выполнены!"

    # 4. Assemble morning prompt & trace
    prompt, trace = memory_engine.assemble_prompt("Morning briefing & sprint focus.")
    full_prompt = f"{prompt}\n\n{task_context}"

    user_trigger = "Доброе утро! Проведи утренний брифинг на сегодня."
    response = llm_client.generate_coaching_response(
        full_prompt, 
        user_trigger, 
        chat_history=chat_history, 
        memory_engine=memory_engine, 
        trace=trace
    )
    
    full_message = f"🌅 **Morning Briefing (09:00)**\n\n{response}\n\n_🧠 [Memory Trace: T1: {trace.t1_count} Qs | T2: {trace.t2_count} Qs | T3: {trace.t3_sections_read} Secs | ~{trace.estimated_tokens} tokens]_"

    await send_safe_bot_message(bot, target_user_id, full_message, parse_mode="Markdown")

async def send_evening_reflection(bot: Bot, target_user_id: int, memory_engine: MemoryEngine, llm_client: LLMClient):
    """21:00 Consolidated Evening Sync Protocol."""
    if not target_user_id:
        return
    
    # 1. Fetch sliding chat history (last 8 turns) to detect user silence & adapt tone
    from src.db import get_recent_chat_history, get_top_focus_tasks_db
    chat_history = get_recent_chat_history(target_user_id, limit=8)

    # 2. Fetch top-3 focus tasks for today
    top_tasks, total_tasks = get_top_focus_tasks_db(limit=3)
    if top_tasks:
        items = [f"- [P{t.get('priority', 2)}] {t['title']}" for t in top_tasks]
        remaining = max(0, total_tasks - len(top_tasks))
        queue_str = f"\n  (и еще {remaining} задач в очереди — отправь /tasks)" if remaining > 0 else ""
        tasks_block = "ФОКУС-ЗАДАЧИ ДНЯ ИЗ БАЗЫ:\n" + "\n".join(items) + queue_str
    else:
        tasks_block = "ФОКУС-ЗАДАЧИ ДНЯ: Нет незавершенных задач!"

    # 3. Assemble evening prompt & trace
    prompt, trace = memory_engine.assemble_prompt("Evening Sync task review and atomic planning for tomorrow.")
    full_prompt = f"{prompt}\n\n{tasks_block}"

    user_trigger = "Добрый вечер! Проведи вечернюю синхронизацию и подведи итоги дня."
    
    try:
        response = llm_client.generate_coaching_response(
            full_prompt, 
            user_trigger, 
            chat_history=chat_history, 
            memory_engine=memory_engine, 
            trace=trace
        )
    except Exception as e:
        logger.warning(f"LLM call for Evening Sync failed: {e}. Using fallback format.")
        response = (
            "Подведем итоги сегодняшнего дня!\n\n"
            f"1. **Проверка 4 правил системности** (Проектный шаг, Уборка 4 зон, 1ч Прогулка/Мысли, Отклик/Работа) — что удалось закрыть?\n"
            f"2. **Фокус-задачи**:\n{tasks_block}\n\n"
            "3. Что сегодня получилось идеально, а что заблокировало?\n"
            "4. Выбери **1 атомарный микро-шаг (15–30 мин)** на завтра!"
        )
    
    full_message = (
        f"🌆 **Evening Sync (21:00)**\n\n"
        f"{response}\n\n"
        f"_🧠 [Memory Trace: T1: {trace.t1_count} Qs | T2: {trace.t2_count} Qs | T3: {trace.t3_sections_read} Secs | ~{trace.estimated_tokens} tokens]_"
    )

    await send_safe_bot_message(bot, target_user_id, full_message, parse_mode="Markdown")

async def run_proactive_ping_checker(bot: Bot, target_user_id: int, memory_engine: MemoryEngine, llm_client: LLMClient):
    """Periodic Proactive Push Engine checker running every 15 minutes."""
    from src.proactive_engine import ProactiveEngine
    try:
        engine = ProactiveEngine()
        executed_count = await engine.check_and_execute_pings(bot, target_user_id, llm_client)
        if executed_count > 0:
            logger.info(f"Proactive ping checker executed {executed_count} pings.")
    except Exception as e:
        logger.error(f"Proactive ping checker failed: {e}")

async def run_nightly_snapshot_and_cleanup(bot: Bot, target_user_id: int, memory_engine: MemoryEngine):
    """23:59 Nightly Memory History Snapshot & Smart T2 Garbage Collection."""
    try:
        snapshot_path = memory_engine.save_nightly_snapshot()
        cleaned_count = memory_engine.cleanup_tier2_garbage()
        logger.info(f"Nightly snapshot completed ({snapshot_path}), cleaned {cleaned_count} items from Tier 2.")
    except Exception as e:
        logger.error(f"Nightly snapshot job failed: {e}")

def create_scheduler(bot: Bot, target_user_id: int, memory_engine: MemoryEngine, llm_client: LLMClient) -> AsyncIOScheduler:
    """
    Initialize and return APScheduler instance with Europe/Moscow timezone for daily 09:00, Evening Sync, 23:59, and 15-min proactive triggers.
    """
    import os
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")

    evening_sync_str = os.getenv("EVENING_SYNC_TIME", "21:00").strip()
    try:
        eh, em = map(int, evening_sync_str.split(":"))
    except Exception:
        eh, em = 21, 0
    
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
    
    # Evening Sync (EVENING_SYNC_TIME, default 21:00)
    scheduler.add_job(
        send_evening_reflection,
        trigger="cron",
        hour=eh,
        minute=em,
        args=[bot, target_user_id, memory_engine, llm_client],
        id="evening_reflection_job",
        replace_existing=True
    )

    # 23:59 Nightly Memory Snapshot & Garbage Collection
    scheduler.add_job(
        run_nightly_snapshot_and_cleanup,
        trigger="cron",
        hour=23,
        minute=59,
        args=[bot, target_user_id, memory_engine],
        id="nightly_memory_snapshot_job",
        replace_existing=True
    )

    # Every 15 minutes Proactive Ping Checker
    scheduler.add_job(
        run_proactive_ping_checker,
        trigger="cron",
        minute="*/15",
        args=[bot, target_user_id, memory_engine, llm_client],
        id="proactive_ping_checker_job",
        replace_existing=True
    )
    
    return scheduler
