"""
Synthetic Test Suite for Telegram Formatting Normalization & Markdown Sanitization.
STRICTLY USES MOCK SYNTHETIC DATA.
"""
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_client import LLMClient

def test_telegram_markdown_table_conversion():
    print("Testing Markdown Table Conversion to Clean Telegram Bullets...")
    client = LLMClient()

    raw_text = (
        "### 4 правила системности (ежедневный каркас)\n\n"
        "| # | Правило | Формат |\n"
        "|---|---|---|\n"
        "| 1 | 20/10 спринты | 20 мин глубокий фокус → 10 мин ходьба/растяжка |\n"
        "| 2 | Walk & Think | 1 час на улице, без гаджетов, свободное мышление |\n"
        "| 3 | Уборка 4 зон | По зоне в день или все сразу — час общий лимит |\n"
        "| 4 | Sprint-трекинг | Закрыть все пункты активного спринта до отбоя |\n\n"
        "---\n\n"
        "### План на сегодня\n\n"
        "- 🔲 Проект: зафиксировать выбор\n"
        "- 🔲 Отклик на вакансию"
    )

    formatted = client._format_for_telegram(raw_text)

    # Assert no table pipes or divider lines remain
    assert "|---|" not in formatted
    assert "---" not in formatted
    assert "###" not in formatted

    # Assert bold headings and clean bullets
    assert "**4 правила системности (ежедневный каркас)**" in formatted
    assert "• **20/10 спринты**: 20 мин глубокий фокус → 10 мин ходьба/растяжка" in formatted
    assert "• **Walk & Think**: 1 час на улице, без гаджетов, свободное мышление" in formatted
    assert "**План на сегодня**" in formatted
    print("[OK] Markdown tables, dividers, and hashtag headings converted cleanly for Telegram.")

def test_sanitize_tool_leak_with_formatting():
    print("\nTesting _sanitize_tool_leak with full Telegram formatting...")
    client = LLMClient()

    raw_with_leaks = (
        'to=functions.search_memory {"query": "sprint"}\n'
        "### Утренний план\n\n"
        "---\n\n"
        "• **Фокус**: Deep Work блок 90 минут\n"
        "• **Прогулка**: 1 час без гаджетов\n"
    )

    cleaned = client._sanitize_tool_leak(raw_with_leaks)
    assert "to=functions" not in cleaned
    assert "###" not in cleaned
    assert "---" not in cleaned
    assert "**Утренний план**" in cleaned
    assert "• **Фокус**: Deep Work блок 90 минут" in cleaned
    print("[OK] _sanitize_tool_leak stripped tool syntax and formatted message for Telegram.")

if __name__ == "__main__":
    print("================ TELEGRAM FORMATTING VERIFICATION ================")
    test_telegram_markdown_table_conversion()
    test_sanitize_tool_leak_with_formatting()
    print("================ ALL TELEGRAM FORMATTING TESTS PASSED! ================")
