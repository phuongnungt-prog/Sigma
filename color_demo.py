#!/usr/bin/env python3
"""
🎨 Color Demo - Preview tất cả màu prompts
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

# Header
console.print("\n" * 2)
console.print("╔═══════════════════════════════════════════════════════════╗", style="bright_cyan")
console.print("║  🎨 ULTRA AI v16.0 - COLOR PROMPTS DEMO 🎨               ║", style="bright_cyan")
console.print("╚═══════════════════════════════════════════════════════════╝", style="bright_cyan")
console.print()

# Bet Settings (CYAN)
console.print(Panel(
    "[bold bright_cyan]💰 BET SETTINGS (CYAN)[/bold bright_cyan]\n\n"
    "[bold bright_cyan]💰 Số BUILD đặt mỗi ván:[/bold bright_cyan] _\n"
    "[bold bright_cyan]📈 Hệ số nhân sau khi thua (ổn định = 2):[/bold bright_cyan] _\n"
    "[bold bright_cyan]🛡️  Chống soi (số ván đặt trước khi nghỉ 1 ván):[/bold bright_cyan] _",
    title="[bold bright_cyan]CYAN - Bet Settings[/bold bright_cyan]",
    border_style="cyan",
    box=box.ROUNDED
))
console.print()

# Pause (MAGENTA)
console.print(Panel(
    "[bold bright_magenta]⏸️  PAUSE SETTINGS (MAGENTA)[/bold bright_magenta]\n\n"
    "[bold bright_magenta]⏸️  Nếu thua thì nghỉ bao nhiêu ván trước khi cược lại:[/bold bright_magenta] [dim yellow](ví dụ: 2)[/dim yellow] _",
    title="[bold bright_magenta]MAGENTA - Pause[/bold bright_magenta]",
    border_style="magenta",
    box=box.ROUNDED
))
console.print()

# Take Profit (GREEN)
console.print(Panel(
    "[bold bright_green]💵 TAKE PROFIT (GREEN)[/bold bright_green]\n\n"
    "[bold bright_green]💵 Chốt lời khi đạt bao nhiêu BUILD:[/bold bright_green] [dim yellow](ví dụ: 100)[/dim yellow] _",
    title="[bold bright_green]GREEN - Take Profit ✅[/bold bright_green]",
    border_style="green",
    box=box.ROUNDED
))
console.print()

# Stop Loss (RED)
console.print(Panel(
    "[bold bright_red]🛑 STOP LOSS (RED)[/bold bright_red]\n\n"
    "[bold bright_red]🛑 Cắt lỗ khi lỗ bao nhiêu BUILD:[/bold bright_red] [dim yellow](ví dụ: 100)[/dim yellow] _",
    title="[bold bright_red]RED - Stop Loss 🚨[/bold bright_red]",
    border_style="red",
    box=box.ROUNDED
))
console.print()

# Ready (YELLOW)
console.print(Panel(
    "[bold bright_yellow]💯 READY PROMPT (YELLOW)[/bold bright_yellow]\n\n"
    "[bold bright_yellow]💯bạn đã sẵn sàng hãy nhấn enter để bắt đầu💯:[/bold bright_yellow] _",
    title="[bold bright_yellow]YELLOW - Ready ⚡[/bold bright_yellow]",
    border_style="yellow",
    box=box.ROUNDED
))
console.print()

# Summary
console.print("╔═══════════════════════════════════════════════════════════╗", style="bright_cyan")
console.print("║  ✨ TẤT CẢ PROMPTS ĐỀU CÓ MÀU ĐẸP! ✨                    ║", style="bright_cyan")
console.print("╚═══════════════════════════════════════════════════════════╝", style="bright_cyan")
console.print()

console.print("[bold bright_green]✅ CYAN[/bold bright_green] = Bet settings (trung tính)")
console.print("[bold bright_magenta]⏸️  MAGENTA[/bold bright_magenta] = Pause (đặc biệt)")
console.print("[bold bright_green]💵 GREEN[/bold bright_green] = Take profit (tích cực!)")
console.print("[bold bright_red]🛑 RED[/bold bright_red] = Stop loss (cảnh báo!)")
console.print("[bold bright_yellow]💯 YELLOW[/bold bright_yellow] = Ready (hành động!)")
console.print()
console.print("[bold bright_cyan]🎨 Professional UI với màu sắc đẹp![/bold bright_cyan]")
console.print()
