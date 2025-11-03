"""
?? ULTIMATE AI v17.0 - UI SYSTEM
Optimized, Clean, Beautiful UI Display
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from typing import Dict, Any, List, Optional

console = Console()


def display_ultimate_decision(
    room_id: int,
    ultimate_result: Dict[str, Any],
    room_features: Dict[str, float],
    room_stats: Dict[int, Dict],
    confidence_level: str
) -> None:
    """
    ?? HI?N TH? QUY?T ??NH ULTIMATE AI
    Clean, professional, essential info only
    """
    
    # Main decision panel
    decision = Text()
    decision.append("?? PH?NG ???C CH?N: ", style="bold bright_white")
    decision.append(f"#{room_id}", style="bold bright_yellow blink")
    decision.append(f"\n\n", style="")
    
    # Confidence bar
    conf = ultimate_result['confidence']
    conf_bar = "?" * int(conf * 20)
    decision.append("?? ?? TIN C?Y: ", style="bright_cyan")
    decision.append(f"{conf:.1%}", style="bold bright_green")
    decision.append(f" [{conf_bar}]", style="bright_green")
    decision.append(f"\n    ? {confidence_level}", style="dim bright_magenta")
    decision.append(f"\n\n", style="")
    
    # Prediction score
    pred = ultimate_result['prediction']
    decision.append("?? D? ?O?N: ", style="bright_cyan")
    decision.append(f"{pred:.1%} ", style="bold bright_green")
    decision.append("SAFE ?" if pred >= 0.75 else "MODERATE ??" if pred >= 0.60 else "RISKY ?", 
                   style="bright_green" if pred >= 0.75 else "bright_yellow" if pred >= 0.60 else "bright_red")
    decision.append(f"\n\n", style="")
    
    # Room stats
    stats = room_stats.get(room_id, {})
    wins = stats.get("survives", 0)
    losses = stats.get("kills", 0)
    total = wins + losses
    win_rate = (wins / total * 100) if total > 0 else 0
    
    decision.append("?? L?CH S?: ", style="bright_cyan")
    decision.append(f"{wins}W/{losses}L", style="bold bright_white")
    if total > 0:
        decision.append(f" ({win_rate:.0f}% Win)", 
                       style="bright_green" if win_rate >= 60 else "bright_yellow" if win_rate >= 40 else "bright_red")
    decision.append(f"\n\n", style="")
    
    # Recommendation
    decision.append("?? KHUY?N NGH?: ", style="bright_cyan")
    decision.append(ultimate_result['recommendation'], style="bold bright_yellow")
    
    console.print(Panel(
        decision,
        title="[bold bright_yellow blink]?? ULTIMATE AI DECISION ??[/bold bright_yellow blink]",
        border_style="bright_yellow",
        box=box.HEAVY
    ))


def display_ultimate_analysis(
    top_rooms: List[tuple],
    ultimate_predictions: Dict[int, Dict],
    room_stats: Dict[int, Dict]
) -> None:
    """
    ?? PH?N T?CH T? ULTIMATE ENGINE
    Compact table showing top 3 rooms
    """
    
    table = Table(
        title="[bold bright_cyan]?? ULTIMATE ENGINE ANALYSIS[/bold bright_cyan]",
        box=box.ROUNDED,
        border_style="bright_cyan",
        show_header=True,
        header_style="bold bright_magenta"
    )
    
    table.add_column("?? Room", style="bright_white", justify="center")
    table.add_column("?? Prediction", style="bright_cyan", justify="center")
    table.add_column("?? Confidence", style="bright_green", justify="center")
    table.add_column("?? History", style="bright_yellow", justify="center")
    table.add_column("?? Level", style="bright_magenta", justify="center")
    
    for rid, base_score in top_rooms[:3]:
        if rid not in ultimate_predictions:
            continue
            
        up = ultimate_predictions[rid]
        stats = room_stats.get(rid, {})
        wins = stats.get("survives", 0)
        losses = stats.get("kills", 0)
        total = wins + losses
        
        # Emojis based on values
        pred_emoji = "?" if up['score'] >= 0.75 else "??" if up['score'] >= 0.60 else "?"
        conf_emoji = "??" if up['confidence'] >= 0.80 else "??" if up['confidence'] >= 0.60 else "??"
        
        table.add_row(
            f"#{rid}",
            f"{up['score']:.1%} {pred_emoji}",
            f"{up['confidence']:.1%} {conf_emoji}",
            f"{wins}W/{losses}L" if total > 0 else "N/A",
            up['confidence_level']
        )
    
    console.print(table)
    console.print("")


def display_round_summary(
    round_num: int,
    prediction: int,
    killed: int,
    result: str,
    profit: float,
    cumulative: float
) -> None:
    """
    ?? K?T QU? V?N - COMPACT
    """
    
    is_win = "TH?NG" in result.upper() or "WIN" in result.upper()
    
    summary = Text()
    summary.append(f"?? V?N #{round_num} ", style="bold bright_white")
    summary.append("? ", style="dim")
    summary.append(f"Ch?n: #{prediction} ", style="bright_cyan")
    summary.append("? ", style="dim")
    summary.append(f"Kill: #{killed} ", style="bright_red")
    summary.append("? ", style="dim")
    
    if is_win:
        summary.append(f"? {result}", style="bold bright_green")
    else:
        summary.append(f"? {result}", style="bold bright_red")
    
    summary.append(f"\n?? L?i/L?: ", style="bright_white")
    if profit >= 0:
        summary.append(f"+{profit:.1f} BUILD", style="bold bright_green")
    else:
        summary.append(f"{profit:.1f} BUILD", style="bold bright_red")
    
    summary.append(f" | T?ng: ", style="dim")
    if cumulative >= 0:
        summary.append(f"+{cumulative:.1f} BUILD", style="bold bright_green")
    else:
        summary.append(f"{cumulative:.1f} BUILD", style="bold bright_red")
    
    border = "bright_green" if is_win else "bright_red"
    console.print(Panel(
        summary,
        border_style=border,
        box=box.ROUNDED
    ))


def display_learning_progress(
    total_rounds: int,
    accuracy: float,
    recent_wins: int,
    recent_total: int
) -> None:
    """
    ?? TI?N TR?NH H?C - MINIMAL
    """
    
    progress = Text()
    progress.append("?? LEARNING: ", style="bright_cyan")
    progress.append(f"{total_rounds} rounds", style="bold bright_white")
    progress.append(" ? ", style="dim")
    progress.append(f"{accuracy:.1%} accuracy", style="bold bright_green")
    
    if recent_total >= 10:
        recent_acc = recent_wins / recent_total
        progress.append(" ? ", style="dim")
        progress.append(f"Recent: {recent_acc:.1%}", 
                       style="bright_green" if recent_acc >= 0.60 else "bright_yellow")
    
    console.print(progress)


def display_algorithm_breakdown(ultimate_result: Dict[str, Any]) -> None:
    """
    ?? PH?N T?CH C?C ALGORITHMS - ADVANCED
    """
    
    algo_table = Table(
        title="[bold bright_magenta]?? ULTIMATE ENGINE - 6 ALGORITHMS[/bold bright_magenta]",
        box=box.DOUBLE,
        border_style="bright_magenta",
        show_header=True,
        header_style="bold bright_cyan"
    )
    
    algo_table.add_column("?? Algorithm", style="bright_yellow", justify="left")
    algo_table.add_column("?? Score", style="bright_cyan", justify="center")
    algo_table.add_column("?? Impact", style="bright_green", justify="center")
    
    # Extract algorithm results
    algorithms = [
        ("? Bayesian Inference", ultimate_result.get('bayesian', 0), "?????"),
        ("? Kalman Filter", ultimate_result.get('kalman', 0), "????"),
        ("? Monte Carlo", ultimate_result.get('monte_carlo', 0), "?????"),
        ("? Game Theory", ultimate_result.get('game_theory', 0), "???"),
        ("? Statistical Test", ultimate_result.get('stat_test', 0), "????"),
        ("? Ensemble Fusion", ultimate_result.get('ensemble', 0), "?????"),
    ]
    
    for algo_name, score, impact in algorithms:
        score_str = f"{score:.1%}" if score > 0 else "N/A"
        algo_table.add_row(algo_name, score_str, impact)
    
    console.print(algo_table)
    console.print("")


def display_compact_status(
    round_idx: int,
    balance: float,
    cumulative: float,
    win_streak: int,
    lose_streak: int
) -> None:
    """
    ?? STATUS - SUPER COMPACT ONE-LINER
    """
    
    status = Text()
    status.append(f"?? V?n #{round_idx}", style="bold bright_cyan")
    status.append(" | ", style="dim")
    status.append(f"?? {balance:.1f} BUILD", style="bright_white")
    status.append(" | ", style="dim")
    
    if cumulative >= 0:
        status.append(f"?? +{cumulative:.1f}", style="bold bright_green")
    else:
        status.append(f"?? {cumulative:.1f}", style="bold bright_red")
    
    status.append(" | ", style="dim")
    
    if win_streak > 0:
        status.append(f"?? {win_streak} win streak", style="bright_green")
    elif lose_streak > 0:
        status.append(f"?? {lose_streak} lose streak", style="bright_red")
    else:
        status.append("? No streak", style="dim")
    
    console.print(status)
    console.print("")
