"""
LifeOS Personal Memory & Coaching Agent - Interactive Rich CLI Application.
"""
import sys
import logging

# Reconfigure standard output encoding for Windows compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text
from rich.style import Style

from src.memory_engine import MemoryEngine
from src.llm_client import LLMClient
from src.extractor_service import ExtractorService
from src.models import MemoryTrace

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")

console = Console()

def display_welcome_banner():
    banner_text = Text()
    banner_text.append("🧠 LifeOS Personal Memory & Coaching Agent\n", style="bold magenta")
    banner_text.append("Senior Friend & Coach • 3-Tier Question-Anchored Memory Architecture", style="dim cyan")
    console.print(Panel(banner_text, border_style="cyan", expand=False))
    console.print("[dim]Type your message below. Commands: [bold]/memory[/bold], [bold]/dump[/bold], [bold]/decay[/bold], [bold]/help[/bold], [bold]/exit[/bold][/dim]\n")

def display_memory_tables(engine: MemoryEngine):
    console.print("\n[bold magenta]════════════════════ CURRENT MEMORY STATE ════════════════════[/bold magenta]\n")
    
    # Tier 1 Table
    t1_table = Table(title="Tier 1: Core Profile (Always Loaded)", title_style="bold cyan", border_style="cyan")
    t1_table.add_column("ID", style="dim", width=16)
    t1_table.add_column("Anchor Question", style="bold white")
    t1_table.add_column("Answer / Value", style="green")
    t1_table.add_column("Weight", justify="right", style="yellow")
    
    for qa in engine.tier1_items:
        t1_table.add_row(qa.id, qa.question, qa.answer, f"{qa.weight:.2f}")
    console.print(t1_table)
    console.print()

    # Tier 2 Table
    t2_table = Table(title="Tier 2: Dynamic State (Rolling Decay)", title_style="bold yellow", border_style="yellow")
    t2_table.add_column("ID", style="dim", width=16)
    t2_table.add_column("Anchor Question", style="bold white")
    t2_table.add_column("Answer / State", style="green")
    t2_table.add_column("Weight", justify="right", style="yellow")
    t2_table.add_column("Valid From", style="dim cyan", width=12)

    for qa in engine.tier2_items:
        t2_table.add_row(qa.id, qa.question, qa.answer, f"{qa.weight:.2f}", qa.valid_from or "N/A")
    console.print(t2_table)
    console.print()

    # Tier 3 Tables
    if engine.tier3_entities:
        console.print("[bold blue]Tier 3: Entity Graph (Loaded on Mention)[/bold blue]")
        for entity, qa_list in engine.tier3_entities.items():
            t3_table = Table(title=f"Entity: {entity.upper()}", title_style="bold blue", border_style="blue")
            t3_table.add_column("ID", style="dim", width=18)
            t3_table.add_column("Question", style="bold white")
            t3_table.add_column("Answer", style="green")
            t3_table.add_column("Weight", justify="right", style="yellow")
            
            for qa in qa_list:
                t3_table.add_row(qa.id, qa.question, qa.answer, f"{qa.weight:.2f}")
            console.print(t3_table)
            console.print()
    else:
        console.print("[dim]No Tier 3 entities currently registered.[/dim]\n")

def handle_dump_command(engine: MemoryEngine, extractor: ExtractorService):
    console.print(Panel(
        "[bold cyan]🎙️ Stream of Consciousness Processing Mode (/dump)[/bold cyan]\n"
        "Paste raw voice transcripts, unorganized thoughts, or sprint reflection text below.\n"
        "The extractor will parse insights and update your 3-Tier memory.",
        border_style="magenta"
    ))
    raw_dump = Prompt.ask("[bold magenta]Paste dump text[/bold magenta]")
    if not raw_dump.strip():
        console.print("[yellow]Empty dump cancelled.[/yellow]\n")
        return

    console.print("[dim cyan]Processing dump through fast memory compactor...[/dim cyan]")
    sys_prompt, _ = engine.assemble_prompt(raw_dump)
    diff = extractor.extract_sync(raw_dump, sys_prompt)

    if diff:
        t1_count = len(diff.tier1_updates)
        t2_count = len(diff.tier2_updates)
        t3_entities = list(diff.tier3_updates.keys())
        console.print(f"[bold green]✓ Memory updated successfully![/bold green] (Added/Updated: T1={t1_count}, T2={t2_count}, T3 Entities={t3_entities})\n")
    else:
        console.print("[yellow]No significant memory updates extracted from dump.[/yellow]\n")

def display_help():
    help_text = (
        "[bold cyan]Available Commands:[/bold cyan]\n"
        "• [bold]/memory[/bold] - View current Tier 1, Tier 2, and Tier 3 memory items in formatted tables\n"
        "• [bold]/dump[/bold]   - Enter Stream of Consciousness processing mode for voice/text dumps\n"
        "• [bold]/decay[/bold]  - Manually apply exponential weight decay (W = W * 0.95) to Tier 2 state items\n"
        "• [bold]/help[/bold]   - Show this help message\n"
        "• [bold]/exit[/bold]   - Gracefully quit LifeOS CLI\n"
    )
    console.print(Panel(help_text, border_style="dim cyan"))

def main():
    display_welcome_banner()

    # Initialize Engine & Services
    memory_engine = MemoryEngine()
    llm_client = LLMClient()
    extractor_service = ExtractorService(memory_engine, llm_client)

    # Initial daily decay check
    memory_engine.apply_decay()

    if not llm_client.is_api_configured():
        console.print(
            "[dim yellow]Notice: OPENAI_API_KEY not set in .env. Running with local response & extractor fallback.[/dim yellow]\n"
        )

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]Vitalik[/bold cyan]").strip()
            if not user_input:
                continue

            # Command Handling
            cmd = user_input.lower()
            if cmd in ["/exit", "exit", "quit", "/quit"]:
                console.print("[dim magenta]Shutting down LifeOS CLI. Stay focused![/dim magenta]")
                extractor_service.shutdown()
                break
            elif cmd == "/memory":
                display_memory_tables(memory_engine)
                continue
            elif cmd == "/dump":
                handle_dump_command(memory_engine, extractor_service)
                continue
            elif cmd == "/decay":
                count = memory_engine.apply_decay()
                console.print(f"[bold green]✓ Weight decay applied to {count} Tier 2 items.[/bold green]\n")
                continue
            elif cmd == "/help":
                display_help()
                continue

            # 1. Assemble prompt with 3-Tier memory
            system_prompt, trace = memory_engine.assemble_prompt(user_input)

            # 2. Generate response from Senior Friend & Coach
            response = llm_client.generate_coaching_response(system_prompt, user_input)

            # 3. Render Agent Response
            console.print()
            console.print(Panel(Markdown(response), title="[bold magenta]Senior Friend & Coach[/bold magenta]", border_style="magenta"))

            # 4. Display Non-intrusive Debug Panel Trace Footer
            console.print(f"[dim grey50]{trace.format_trace_str()}[/dim grey50]\n")

            # 5. Non-blocking Async Extraction in Background Thread
            extractor_service.trigger_async_extraction(user_input, response, system_prompt)

        except KeyboardInterrupt:
            console.print("\n[dim magenta]Session interrupted. Exiting...[/dim magenta]")
            extractor_service.shutdown()
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}\n")

if __name__ == "__main__":
    main()
