"""
Asynchronous Telegram Bot for LifeOS Personal Memory & Coaching Agent using aiogram 3.x.
Normal voice and text messages utilize sliding chat history (last 6-8 turns) and search_memory tool calling via PostgreSQL.
Multi-message dump mode (/dump ... /done) collects texts/voices into a memory dump buffer.
"""
import os
import asyncio
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, BaseFilter
from aiogram.types import Message, BotCommand
from aiogram.exceptions import TelegramBadRequest

from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.extractor_service import ExtractorService
from src.voice import process_voice_note
from src.scheduler import create_scheduler
from src.db import init_db, sync_tier3_to_postgres, save_chat_message, get_recent_chat_history

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LifeOS_Bot")

load_dotenv()

# Load Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
raw_allowed_id = os.getenv("ALLOWED_TELEGRAM_ID", "").strip()
ALLOWED_TELEGRAM_ID = int(raw_allowed_id) if raw_allowed_id.isdigit() else None

# State dictionary for multi-message dump sessions (user_id -> List[str])
active_dump_sessions: Dict[int, List[str]] = {}

class SecurityFilter(BaseFilter):
    """
    Security filter enforcing single-user access matching ALLOWED_TELEGRAM_ID.
    """
    def __init__(self, allowed_id: Optional[int]):
        self.allowed_id = allowed_id

    async def __call__(self, message: Message) -> bool:
        if not self.allowed_id:
            # If no ID configured, log warning and allow
            return True
        is_allowed = message.from_user is not None and message.from_user.id == self.allowed_id
        if not is_allowed:
            logger.warning(f"Unauthorized access attempt from User ID: {message.from_user.id if message.from_user else 'Unknown'}")
        return is_allowed

def split_message(text: str, max_chunk_size: int = 3900) -> List[str]:
    """
    Splits long messages into clean chunked messages <= max_chunk_size,
    preserving paragraph and line breaks where possible.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current = text

    while len(current) > max_chunk_size:
        # Try double newline break
        split_pos = current.rfind("\n\n", 0, max_chunk_size)
        if split_pos == -1 or split_pos < max_chunk_size // 3:
            # Try single newline break
            split_pos = current.rfind("\n", 0, max_chunk_size)
        if split_pos == -1 or split_pos < max_chunk_size // 3:
            # Try space break
            split_pos = current.rfind(" ", 0, max_chunk_size)
        if split_pos == -1:
            # Hard slice fallback
            split_pos = max_chunk_size

        chunk = current[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        current = current[split_pos:].strip()

    if current:
        chunks.append(current)

    return chunks

async def send_safe_reply(message: Message, text: str, parse_mode: str = "Markdown"):
    """
    Safely send message to Telegram, splitting long text into multiple sequential messages
    and using fallback exception handling if Markdown formatting fails.
    """
    chunks = split_message(text, max_chunk_size=3900)
    
    for chunk in chunks:
        try:
            await message.answer(chunk, parse_mode=parse_mode)
        except TelegramBadRequest as e:
            logger.warning(f"Telegram formatting error ('{e}'). Falling back to plain text (parse_mode=None).")
            await message.answer(chunk, parse_mode=None)
        except Exception as e:
            logger.error(f"Error sending message chunk: {e}")
            try:
                await message.answer(chunk, parse_mode=None)
            except Exception:
                pass

def register_handlers(router: Router, memory_engine: MemoryEngine, llm_client: LLMClient, extractor_service: ExtractorService):
    
    # 1. Command /start & /help
    @router.message(Command("start", "help"))
    async def cmd_help(message: Message):
        help_text = (
            "🧠 **LifeOS Personal Memory & Coaching Agent**\n\n"
            "Senior Friend & Coach powered by 3-Tier Question-Anchored Memory & PostgreSQL Hybrid Search.\n\n"
            "📋 **Available Commands:**\n"
            "• `/memory` - Display current Tier 1 (Core), Tier 2 (State), and Tier 3 (Entities) memory\n"
            "• `/dump` - Start a **Multi-Message Dump Session** (or `/dump <text>` for immediate inline dump)\n"
            "• `/done` or `/stop` - Commit and process an active dump session\n"
            "• `/decay` - Manually apply exponential weight decay (`W_new = W_old * 0.95`) to Tier 2 items\n"
            "• `/help` - Show this list of all available commands\n"
            "• `/start` - Show welcome message and guide\n"
            "• `/exit` or `/quit` - Display bot background service status and scheduled tasks\n\n"
            "💡 *Tip: Your coach maintains sliding conversation history (last 8 turns) and automatically searches deep long-term memory when needed!*"
        )
        await send_safe_reply(message, help_text)

    # 2. Command /memory
    @router.message(Command("memory"))
    async def cmd_memory(message: Message):
        lines = ["🧠 **Current 3-Tier Memory State**\n"]

        # Tier 1
        lines.append("📌 **Tier 1: Core Profile (Always Loaded)**")
        for qa in memory_engine.tier1_items:
            lines.append(f"• *{qa.question}*\n  └ `{qa.answer}` (weight: {qa.weight:.2f})")
        lines.append("")

        # Tier 2
        lines.append("⚡ **Tier 2: Dynamic State (Rolling Decay)**")
        for qa in memory_engine.tier2_items:
            lines.append(f"• *{qa.question}*\n  └ `{qa.answer}` (weight: {qa.weight:.2f})")
        lines.append("")

        # Tier 3
        if memory_engine.tier3_entities:
            lines.append("🌐 **Tier 3: Entity Graph (Loaded on Mention & Tool Search)**")
            for entity, qas in memory_engine.tier3_entities.items():
                lines.append(f"🔹 *Entity: {entity.upper()}*")
                for qa in qas:
                    lines.append(f"  └ *{qa.question}*: `{qa.answer}` (weight: {qa.weight:.2f})")
        else:
            lines.append("🌐 **Tier 3:** No entities registered yet.")

        full_text = "\n".join(lines)
        await send_safe_reply(message, full_text)

    # 3. Command /dump [text]
    @router.message(Command("dump"))
    async def cmd_dump(message: Message):
        user_id = message.from_user.id if message.from_user else 0
        args = message.text.split(maxsplit=1)
        inline_dump = args[1].strip() if len(args) > 1 else ""

        # Inline one-shot dump
        if inline_dump:
            status_msg = await message.answer("🔄 *Extracting memory updates from dump...*", parse_mode="Markdown")
            sys_prompt, _ = memory_engine.assemble_prompt(inline_dump)
            diff = extractor_service.extract_sync(inline_dump, sys_prompt)

            t1_c = len(diff.tier1_updates) if diff else 0
            t2_c = len(diff.tier2_updates) if diff else 0
            t3_keys = list(diff.tier3_updates.keys()) if diff else []

            summary_text = (
                f"📝 **Одноразовый текстовый поток обработан!**\n\n"
                f"**Обновления памяти:**\n"
                f"• Tier 1: `{t1_c}` новых QA\n"
                f"• Tier 2: `{t2_c}` новых QA\n"
                f"• Tier 3: `{len(t3_keys)}` сущностей (`{', '.join(t3_keys) if t3_keys else 'нет'}`)"
            )
            await send_safe_reply(message, summary_text)
            return

        # Multi-message session toggle
        if user_id in active_dump_sessions:
            current_count = len(active_dump_sessions[user_id])
            msg = (
                f"🎙️ **Сессия выгрузки уже активна!**\n\n"
                f"В вашем буфере уже `{current_count}` сообщений.\n"
                "Продолжайте отправлять тексты или голосовые сообщения.\n\n"
                "Когда закончите, отправьте `/done` для обработки и обновления памяти!"
            )
            await send_safe_reply(message, msg)
        else:
            active_dump_sessions[user_id] = []
            msg = (
                "🎙️ **Режим выгрузки памяти (/dump) активирован!**\n\n"
                "Все ваши последующие тексты и голосовые сообщения будут аккумулироваться в выгрузку.\n\n"
                "Когда закончите, отправьте команду `/done` для сохранения изменений в память!"
            )
            await send_safe_reply(message, msg)

    # 4. Command /done & /stop (Commit Multi-message Dump Session)
    @router.message(Command("done", "stop"))
    async def cmd_done(message: Message):
        user_id = message.from_user.id if message.from_user else 0

        if user_id not in active_dump_sessions:
            await send_safe_reply(message, "ℹ️ *У вас нет активной сессии выгрузки.* Чтобы начать, отправьте `/dump`!")
            return

        buffered_items = active_dump_sessions.pop(user_id)

        if not buffered_items:
            await send_safe_reply(message, "⚠️ *Буфер выгрузки был пуст.* Сессия выгрузки закрыта.")
            return

        combined_dump = "\n\n---\n\n".join(buffered_items)
        status_msg = await message.answer(f"🔄 *Обработка накопленного потока ({len(buffered_items)} сообщений)...*", parse_mode="Markdown")

        sys_prompt, _ = memory_engine.assemble_prompt(combined_dump)
        diff = extractor_service.extract_sync(combined_dump, sys_prompt)

        t1_c = len(diff.tier1_updates) if diff else 0
        t2_c = len(diff.tier2_updates) if diff else 0
        t3_keys = list(diff.tier3_updates.keys()) if diff else []

        summary_text = (
            f"✅ **Поток сознания успешно обработан!**\n\n"
            f"**Сообщений в выгрузке:** `{len(buffered_items)}`\n\n"
            f"**Экстрагированные обновления памяти:**\n"
            f"• Tier 1: `{t1_c}` новых QA\n"
            f"• Tier 2: `{t2_c}` новых QA\n"
            f"• Tier 3: `{len(t3_keys)}` сущностей (`{', '.join(t3_keys) if t3_keys else 'нет'}`)"
        )
        await send_safe_reply(message, summary_text)

    # 5. Command /decay
    @router.message(Command("decay"))
    async def cmd_decay(message: Message):
        count = memory_engine.apply_decay()
        reply = f"✓ **Exponential Weight Decay Applied!**\nUpdated weights for `{count}` Tier 2 state items (`W_new = W_old * 0.95`)."
        await send_safe_reply(message, reply)

    # 6. Command /exit & /quit
    @router.message(Command("exit", "quit"))
    async def cmd_exit(message: Message):
        user_id = message.from_user.id if message.from_user else 0
        in_session = user_id in active_dump_sessions
        session_info = f" (Активна сессия `/dump`: {len(active_dump_sessions.get(user_id, []))} сообщений)" if in_session else ""
        
        status_text = (
            "🤖 **LifeOS Background Service Status**\n\n"
            "• **Bot Engine**: Running continuously in Telegram\n"
            "• **Memory Tiers**: Tier 1 (`" + str(len(memory_engine.tier1_items)) + "`), Tier 2 (`" + str(len(memory_engine.tier2_items)) + "`), Tier 3 Entities (`" + str(len(memory_engine.tier3_entities)) + "`)\n"
            "• **Dump Session**: " + ("Active" + session_info if in_session else "Inactive") + "\n"
            "• **Scheduler**: Active (Morning Briefing @ 09:00, Evening Reflection @ 21:00)\n\n"
            "The bot will remain active for proactive coaching and memory management."
        )
        await send_safe_reply(message, status_text)

    # 7. Voice Note Handler
    @router.message(F.voice)
    async def handle_voice_message(message: Message):
        user_id = message.from_user.id if message.from_user else 0
        in_dump_session = user_id in active_dump_sessions

        # If in /dump session mode, accumulate voice note transcription into buffer!
        if in_dump_session:
            status_msg = await message.answer("🎙️ *Расшифровка голосового сообщения для выгрузки...*", parse_mode="Markdown")
            transcription = await process_voice_note(message.bot, message.voice.file_id, llm_client)

            if transcription and "Voice Transcription Error" not in transcription:
                active_dump_sessions[user_id].append(f"[Voice Note Transcription]: {transcription}")
                count = len(active_dump_sessions[user_id])
                await status_msg.edit_text(f"📥 *Голосовое сообщение #{count} добавлено в буфер выгрузки!* (Отправьте еще или `/done`)", parse_mode="Markdown")
            else:
                await status_msg.edit_text(f"❌ Ошибка расшифровки: {transcription}")
            return

        # Normal mode: Transcribe voice note and process as a STANDARD COACHING QUERY
        status_msg = await message.answer("🎙️ *Расшифровка голосового сообщения...*", parse_mode="Markdown")
        transcription = await process_voice_note(message.bot, message.voice.file_id, llm_client)

        if not transcription or "Voice Transcription Error" in transcription:
            await status_msg.edit_text(f"❌ Не удалось распознать голос: {transcription}")
            return

        # 1. Fetch sliding chat history from PostgreSQL (last 8 turns)
        chat_history = get_recent_chat_history(user_id, limit=8)

        # 2. Assemble prompt with 3-Tier memory
        system_prompt, trace = memory_engine.assemble_prompt(transcription)

        # 3. Generate coaching response for transcribed text with tool search & history
        response = llm_client.generate_coaching_response(system_prompt, transcription, chat_history=chat_history, memory_engine=memory_engine)

        # 4. Save dialogue turns to PostgreSQL
        save_chat_message(user_id, "user", f"[Voice Note]: {transcription}")
        save_chat_message(user_id, "assistant", response)

        # 5. Append status trace footer
        trace_str = f"🧠 [Memory Trace: T1: {trace.t1_count} Qs | T2: {trace.t2_count} Qs | T3: {trace.t3_total} Qs | ~{trace.estimated_tokens} tokens]"
        full_reply = f"🎙️ _[Распознано]: \"{transcription}\"_\n\n{response}\n\n_{trace_str}_"

        # 6. Delete temporary status and send full coaching reply
        try:
            await status_msg.delete()
        except Exception:
            pass

        await send_safe_reply(message, full_reply)

        # 7. Non-blocking background extraction turn
        extractor_service.trigger_async_extraction(transcription, response, system_prompt)

    # 8. General Text Message Handler
    @router.message(F.text)
    async def handle_text_message(message: Message):
        user_text = message.text.strip()
        if not user_text:
            return

        user_id = message.from_user.id if message.from_user else 0

        # If user is in multi-message dump session, accumulate text into dump buffer!
        if user_id in active_dump_sessions:
            active_dump_sessions[user_id].append(user_text)
            count = len(active_dump_sessions[user_id])
            await message.answer(f"📥 *Сообщение #{count} добавлено в буфер выгрузки!* (Отправьте еще или `/done`)", parse_mode="Markdown")
            return

        # Normal mode: Standard conversation with Senior Friend & Coach
        # 1. Fetch sliding chat history from PostgreSQL (last 8 turns)
        chat_history = get_recent_chat_history(user_id, limit=8)

        # 2. Assemble prompt with 3-Tier memory
        system_prompt, trace = memory_engine.assemble_prompt(user_text)

        # 3. Generate coaching response with tool search & history
        response = llm_client.generate_coaching_response(system_prompt, user_text, chat_history=chat_history, memory_engine=memory_engine)

        # 4. Save dialogue turns to PostgreSQL
        save_chat_message(user_id, "user", user_text)
        save_chat_message(user_id, "assistant", response)

        # 5. Append status trace footer
        trace_str = f"🧠 [Memory Trace: T1: {trace.t1_count} Qs | T2: {trace.t2_count} Qs | T3: {trace.t3_total} Qs | ~{trace.estimated_tokens} tokens]"
        full_reply = f"{response}\n\n_{trace_str}_"

        # 6. Send safe message to user
        await send_safe_reply(message, full_reply)

        # 7. Launch non-blocking async background memory extraction
        extractor_service.trigger_async_extraction(user_text, response, system_prompt)

async def set_bot_commands(bot: Bot):
    """Register menu commands with Telegram."""
    commands = [
        BotCommand(command="start", description="Welcome message & Quick Guide"),
        BotCommand(command="help", description="List all available commands"),
        BotCommand(command="memory", description="View 3-Tier Memory State"),
        BotCommand(command="dump", description="Start multi-message dump session"),
        BotCommand(command="done", description="Commit and finish dump session"),
        BotCommand(command="decay", description="Apply weight decay to Tier 2 items"),
        BotCommand(command="exit", description="Check bot service & scheduler status"),
    ]
    try:
        await bot.set_my_commands(commands)
        logger.info("Registered Telegram menu commands successfully.")
    except Exception as e:
        logger.warning(f"Failed to set Telegram bot commands menu: {e}")

async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not configured in .env! Please set TELEGRAM_BOT_TOKEN.")
        return

    logger.info("Initializing LifeOS Telegram Bot...")
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    router = Router()

    # Apply single-user security filter
    router.message.filter(SecurityFilter(ALLOWED_TELEGRAM_ID))

    # Initialize PostgreSQL schema & sync Tier 3 memory
    init_db()
    sync_tier3_to_postgres()

    # Initialize Engine & Services
    memory_engine = MemoryEngine()
    llm_client = LLMClient()
    extractor_service = ExtractorService(memory_engine, llm_client)

    # Daily decay check
    memory_engine.apply_decay()

    # Register handlers and menu commands
    register_handlers(router, memory_engine, llm_client, extractor_service)
    dp.include_router(router)
    await set_bot_commands(bot)

    # Initialize Scheduler
    scheduler = create_scheduler(bot, ALLOWED_TELEGRAM_ID, memory_engine, llm_client)
    scheduler.start()
    logger.info("APScheduler started (Morning Briefing @ 09:00, Evening Reflection @ 21:00).")

    logger.info("Bot polling starting...")
    try:
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown()
        extractor_service.shutdown()
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
