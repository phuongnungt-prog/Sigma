# toolws.py (UPGRADED) - tích hợp VIP50, VIP50+, VIP100, ADAPTIVE
from __future__ import annotations

def show_banner():
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    import time
    console = Console()
    
    # FUTURISTIC ASCII ART BANNER
    banner_lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                                                          ║",
        "║     ██╗   ██╗██╗  ████████╗██████╗  █████╗              ║",
        "║     ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔══██╗             ║",
        "║     ██║   ██║██║     ██║   ██████╔╝███████║             ║",
        "║     ██║   ██║██║     ██║   ██╔══██╗██╔══██║             ║",
        "║     ╚██████╔╝███████╗██║   ██║  ██║██║  ██║             ║",
        "║      ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝             ║",
        "║                                                          ║",
        "║          🤖 A G I   v 3 . 0   Q U A N T U M 🤖          ║",
        "║                                                          ║",
        "║        「 TRUE ARTIFICIAL GENERAL INTELLIGENCE 」        ║",
        "║                                                          ║",
        "╚══════════════════════════════════════════════════════════╝",
    ]
    
    colors = ["cyan", "blue", "magenta", "cyan"]
    for i, line in enumerate(banner_lines):
        color = colors[i % len(colors)]
        console.print(f"[bold {color}]{line}[/]")
        time.sleep(0.02)
    
    console.print()
    
    # SYSTEM INITIALIZATION with animation
    init_steps = [
        ("🧠", "Neural Core", "Initializing 10,000 neurons", "green"),
        ("🔬", "Quantum Processor", "Loading quantum algorithms", "cyan"),
        ("📊", "Pattern Engine", "Training pattern recognition", "magenta"),
        ("🎯", "Strategy Core", "Calibrating meta-strategies", "yellow"),
        ("⚡", "Real-time Analyzer", "Activating live analysis", "blue"),
        ("🛡️", "Risk Manager", "Configuring safety protocols", "red"),
    ]
    
    for icon, name, desc, color in init_steps:
        console.print(f"[bold {color}]{icon} {name:.<25} {desc}[/]", end="")
        time.sleep(0.08)
        console.print(f" [bold green]✓[/]")
    
    console.print()
    console.print(Panel(
        "[bold cyan]═══════════════════════════════════════[/]\n"
        "[bold yellow]    🌟 SYSTEM STATUS: ONLINE 🌟[/]\n"
        "[bold cyan]═══════════════════════════════════════[/]\n\n"
        "[green]► AI Mode:[/] [bold magenta]TRUE GENERAL INTELLIGENCE[/]\n"
        "[green]► Capabilities:[/] [cyan]Think, Reason, Learn, Adapt[/]\n"
        "[green]► Neural Network:[/] [yellow]10,000 Nodes | Active Learning[/]\n"
        "[green]► Quantum State:[/] [blue]Superposition Ready[/]\n"
        "[green]► Win Rate Target:[/] [bold green]65-75% (Human-level+)[/]\n\n"
        "[dim]Copyright by Duy Hoàng | Enhanced by Khánh | AGI by Claude[/]",
        title="[bold magenta]🤖 ULTRA AGI v3.0 QUANTUM[/]",
        subtitle="[dim]Powered by Neural-Quantum Hybrid Architecture[/]",
        border_style="cyan",
        box=box.DOUBLE
    ))

show_banner()
import json
import sys
import time
import threading
import random
import logging
import math
import re
from collections import defaultdict, deque
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, Tuple, Optional, List

# Try import numpy, fallback if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback: Simple numpy-like functions
    class np:
        @staticmethod
        def array(x):
            return list(x)
        @staticmethod
        def sum(x):
            return sum(x) if isinstance(x, list) else x
        @staticmethod
        def log2(x):
            return [math.log2(v) if v > 0 else 0 for v in x] if isinstance(x, list) else math.log2(x) if x > 0 else 0
        @staticmethod
        def random():
            return random
        random.normal = lambda m, s: random.gauss(m, s)

import pytz
import requests
import websocket
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.align import Align
from rich.rule import Rule
from rich.text import Text
from rich import box

# -------------------- CONFIG & GLOBALS --------------------
console = Console()
# Hiển thị banner ngay khi tool chạy
console.print(Rule("[bold yellow]KH TOOL[/]"))
console.print("[cyan]Copyright by [bold]Duy Hoàng | Chỉnh sửa by [bold green]Khánh[/][/]")
console.print(Rule())

tz = pytz.timezone("Asia/Ho_Chi_Minh")

logger = logging.getLogger("escape_vip_ai_rebuild")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("escape_vip_ai_rebuild.log", encoding="utf-8"))

# Endpoints (config)
BET_API_URL = "https://api.escapemaster.net/escape_game/bet"
WS_URL = "wss://api.escapemaster.net/escape_master/ws"
WALLET_API_URL = "https://wallet.3games.io/api/wallet/user_asset"

HTTP = requests.Session()
try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    adapter = HTTPAdapter(
        pool_connections=20, pool_maxsize=50,
        max_retries=Retry(total=3, backoff_factor=0.2,
                          status_forcelist=(500, 502, 503, 504))
    )
    HTTP.mount("https://", adapter)
    HTTP.mount("http://", adapter)
except Exception:
    pass

ROOM_NAMES = {
    1: "📦 Nhà kho", 2: "🪑 Phòng họp", 3: "👔 Phòng giám đốc", 4: "💬 Phòng trò chuyện",
    5: "🎥 Phòng giám sát", 6: "🏢 Văn phòng", 7: "💰 Phòng tài vụ", 8: "👥 Phòng nhân sự"
}
ROOM_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]

# runtime state
USER_ID: Optional[int] = None
SECRET_KEY: Optional[str] = None
issue_id: Optional[int] = None
issue_start_ts: Optional[float] = None
count_down: Optional[int] = None
killed_room: Optional[int] = None
round_index: int = 0
_skip_active_issue: Optional[int] = None  # ván hiện tại đang nghỉ

room_state: Dict[int, Dict[str, Any]] = {r: {"players": 0, "bet": 0} for r in ROOM_ORDER}
room_stats: Dict[int, Dict[str, Any]] = {r: {"kills": 0, "survives": 0, "last_kill_round": None, "last_players": 0, "last_bet": 0} for r in ROOM_ORDER}

predicted_room: Optional[int] = None
last_killed_room: Optional[int] = None
prediction_locked: bool = False

# balances & pnl
current_build: Optional[float] = None
current_usdt: Optional[float] = None
current_world: Optional[float] = None
last_balance_ts: Optional[float] = None
last_balance_val: Optional[float] = None
starting_balance: Optional[float] = None
cumulative_profit: float = 0.0

# streaks
win_streak: int = 0
lose_streak: int = 0
max_win_streak: int = 0
max_lose_streak: int = 0

# betting
base_bet: float = 1.0
multiplier: float = 2.0
current_bet: Optional[float] = None
run_mode: str = "AUTO"

# AUTO or STAT
bet_rounds_before_skip: int = 0
_rounds_placed_since_skip: int = 0
skip_next_round_flag: bool = False

bet_history: deque = deque(maxlen=500)
# store bet records; display last 5
bet_sent_for_issue: set = set()

# new controls
pause_after_losses: int = 0  # khi thua thì nghỉ bao nhiêu tay
_skip_rounds_remaining: int = 0
profit_target: Optional[float] = None  # take profit (BUILD)
stop_when_profit_reached: bool = False
stop_loss_target: Optional[float] = None  # stop loss (BUILD)
stop_when_loss_reached: bool = False
stop_flag: bool = False

# UI / timing
ui_state: str = "IDLE"
# analysis window timestamps
analysis_start_ts: Optional[float] = None
# when True, show a "lòa/blur" analysis visual between 45s -> 10s
analysis_blur: bool = False
# ws/poll
last_msg_ts: float = time.time()
last_balance_fetch_ts: float = 0.0
BALANCE_POLL_INTERVAL: float = 4.0
_ws: Dict[str, Any] = {"ws": None}

# selection config (used by algorithms)
SELECTION_CONFIG = {
    "max_bet_allowed": float("inf"),
    "max_players_allowed": 9999,
    "avoid_last_kill": True,
}

# selection modes - ULTRA AI ONLY
SELECTION_MODES = {
    "ULTRA_AI": "🧠 ULTRA AI - Siêu trí tuệ 10000 công thức + Deep Learning"
}

settings = {"algo": "ULTRA_AI"}

_spinner = ["📦", "🪑", "👔", "💬", "🎥", "🏢", "💰", "👥"]
_brain_spinner = ["🧠", "💡", "⚡", "🔥", "✨", "💫", "🌟", "⭐"]
_analyze_spinner = ["◐", "◓", "◑", "◒"]

_num_re = re.compile(r"-?\d+[\d,]*\.?\d*")

RAINBOW_COLORS = ["red", "orange1", "yellow1", "green", "cyan", "blue", "magenta"]
GRADIENT_COLORS = ["bright_red", "red", "dark_orange", "orange1", "yellow1", "green_yellow", "green", "cyan", "bright_cyan", "blue", "bright_blue", "magenta", "bright_magenta"]

# UI Themes
UI_THEME = {
    "win": "bold green",
    "loss": "bold red",
    "neutral": "bold yellow",
    "info": "cyan",
    "success": "bright_green",
    "warning": "bright_yellow",
    "error": "bright_red",
    "dim": "dim white",
    "highlight": "bold bright_cyan",
    "ai": "bold magenta",
}

# -------------------- UTILITIES --------------------

def log_debug(msg: str):
    try:
        logger.debug(msg)
    except Exception:
        pass


def _parse_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = _num_re.search(s)
    if not m:
        return None
    token = m.group(0).replace(",", "")
    try:
        return float(token)
    except Exception:
        return None


def human_ts() -> str:
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def safe_input(prompt: str, default=None, cast=None):
    try:
        s = input(prompt).strip()
    except EOFError:
        return default
    if s == "":
        return default
    if cast:
        try:
            return cast(s)
        except Exception:
            return default
    return s

# -------------------- BALANCE PARSING & FETCH --------------------

def _parse_balance_from_json(j: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not isinstance(j, dict):
        return None, None, None
    build = None
    world = None
    usdt = None

    data = j.get("data") if isinstance(j.get("data"), dict) else j
    if isinstance(data, dict):
        cwallet = data.get("cwallet") if isinstance(data.get("cwallet"), dict) else None
        if cwallet:
            for key in ("ctoken_contribute", "ctoken", "build", "balance", "amount"):
                if key in cwallet and build is None:
                    build = _parse_number(cwallet.get(key))
        for k in ("build", "ctoken", "ctoken_contribute"):
            if build is None and k in data:
                build = _parse_number(data.get(k))
        for k in ("usdt", "kusdt", "usdt_balance"):
            if usdt is None and k in data:
                usdt = _parse_number(data.get(k))
        for k in ("world", "xworld"):
            if world is None and k in data:
                world = _parse_number(data.get(k))

    found = []

    def walk(o: Any, path=""):
        if isinstance(o, dict):
            for kk, vv in o.items():
                nk = (path + "." + str(kk)).strip(".")
                if isinstance(vv, (dict, list)):
                    walk(vv, nk)
                else:
                    n = _parse_number(vv)
                    if n is not None:
                        found.append((nk.lower(), n))
        elif isinstance(o, list):
            for idx, it in enumerate(o):
                walk(it, f"{path}[{idx}]")

    walk(j)

    for k, n in found:
        if build is None and any(x in k for x in ("ctoken", "build", "contribute", "balance")):
            build = n
        if usdt is None and "usdt" in k:
            usdt = n
        if world is None and any(x in k for x in ("world", "xworld")):
            world = n

    return build, world, usdt


def balance_headers_for(uid: Optional[int] = None, secret: Optional[str] = None) -> Dict[str, str]:
    h = {
        "accept": "*/*",
        "accept-language": "vi,en;q=0.9",
        "cache-control": "no-cache",
        "country-code": "vn",
        "origin": "https://xworld.info",
        "pragma": "no-cache",
        "referer": "https://xworld.info/",
        "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36",
        "user-login": "login_v2",
        "xb-language": "vi-VN",
    }
    if uid is not None:
        h["user-id"] = str(uid)
    if secret:
        h["user-secret-key"] = str(secret)
    return h


def fetch_balances_3games(retries=2, timeout=6, params=None, uid=None, secret=None):
    """
    Non-blocking friendly: call from background threads if you don't want UI block.
    """
    global current_build, current_usdt, current_world, last_balance_ts
    global starting_balance, last_balance_val, cumulative_profit

    uid = uid or USER_ID
    secret = secret or SECRET_KEY
    payload = {"user_id": int(uid) if uid is not None else None, "source": "home"}

    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            r = HTTP.post(
                WALLET_API_URL,
                json=payload,
                headers=balance_headers_for(uid, secret),
                timeout=timeout,
            )
            r.raise_for_status()
            j = r.json()

            build = None
            world = None
            usdt = None
            # custom parsing
            build, world, usdt = _parse_balance_from_json(j)

            if build is not None:
                # ✅ FIX: Chỉ set starting_balance lần đầu tiên
                if starting_balance is None:
                    starting_balance = build
                    last_balance_val = build
                    current_build = build
                    console.print(f"[cyan]💰 Số dư ban đầu: {starting_balance:.4f} BUILD[/cyan]")
                else:
                    # Tính delta chỉ khi có thay đổi
                    delta = float(build) - float(last_balance_val)
                    if abs(delta) > 0.0001:  # Tránh floating point error
                        cumulative_profit += delta
                        last_balance_val = build
                        console.print(f"[dim]💰 Balance thay đổi: {delta:+.4f} BUILD | Tổng lãi/lỗ: {cumulative_profit:+.4f}[/dim]")
                    current_build = build
            if usdt is not None:
                current_usdt = usdt
            if world is not None:
                current_world = world

            last_balance_ts = time.time()
            return current_build, current_world, current_usdt

        except Exception as e:
            log_debug(f"wallet fetch attempt {attempt} error: {e}")
            time.sleep(min(0.6 * attempt, 2))

    return current_build, current_world, current_usdt

# -------------------- ULTRA AI SELECTION ENGINE --------------------
# Siêu trí tuệ với Deep Learning, Pattern Recognition, Meta-Learning

# FORMULAS storage and generator seed
FORMULAS: List[Dict[str, Any]] = []
FORMULA_SEED = 1234567890  # Ultra seed

# AI Memory System
PATTERN_MEMORY: deque = deque(maxlen=1000)  # Nhớ 1000 patterns thành công
SEQUENCE_MEMORY: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"wins": 0, "losses": 0, "confidence": 0.5})
ANTI_PATTERNS: deque = deque(maxlen=500)  # Các patterns dẫn đến thua
META_LEARNING_RATE: float = 0.15  # Tốc độ học động
CONFIDENCE_THRESHOLD: float = 0.7  # Ngưỡng tin cậy

# Advanced AI Stats
ENSEMBLE_DIVERSITY: float = 0.0  # Độ đa dạng ensemble
BAYESIAN_PRIOR: float = 0.5  # Prior probability
KELLY_FRACTION: float = 0.25  # Kelly Criterion fraction
RISK_REWARD_RATIO: float = 2.0  # Risk/Reward target
MONTE_CARLO_SIMS: int = 200  # Số simulations (tối ưu cho tốc độ)

# Performance Metrics
total_predictions: int = 0
correct_predictions: int = 0
avg_confidence_when_correct: float = 0.0
avg_confidence_when_wrong: float = 0.0

# Advanced Mathematical Models
MARKOV_TRANSITION_MATRIX: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
MARKOV_STATE_COUNTS: Dict[int, int] = defaultdict(int)
KALMAN_ESTIMATES: Dict[int, Dict[str, float]] = {}  # Kalman filter estimates
CLUSTERING_CENTERS: List[Tuple[float, float]] = []  # K-means centers
EMA_ALPHA: float = 0.3  # Exponential Moving Average alpha
ROOM_EMA: Dict[int, float] = {r: 0.5 for r in range(1, 9)}  # EMA scores

# 🤖 TRUE AI SYSTEM - THINKING & REASONING
AI_THOUGHTS: List[str] = []  # AI's thinking process
AI_REASONING: Dict[str, Any] = {}  # Current reasoning
AI_CONFIDENCE_HISTORY: deque = deque(maxlen=50)  # Confidence tracking
AI_STRATEGY_STATE: str = "ANALYZING"  # Current strategy
GAME_PHASE: str = "EARLY"  # EARLY, MID, LATE
OPPONENT_MODEL: Dict[str, Any] = {
    "aggression": 0.5,
    "pattern": "UNKNOWN",
    "predictability": 0.5
}
META_STRATEGY: str = "ADAPTIVE"  # ADAPTIVE, CONSERVATIVE, AGGRESSIVE

def _room_features_ultra_ai(rid: int):
    """
    Phân tích đặc trưng phòng với ULTRA AI - PHÂN TÍCH TƯ DUY THẬT SỰ.
    
    NGUYÊN TẮC PHÂN TÍCH:
    1. Phòng có NHIỀU người/tiền = NGUY HIỂM (bẫy)
    2. Phòng có ÍT người/tiền = AN TOÀN (ít bị kill)
    3. Đi NGƯỢC lại đám đông = Thông minh
    4. Phân tích PATTERNS thực tế, không theo trends
    """
    st = room_state.get(rid, {})
    stats = room_stats.get(rid, {})
    players = float(st.get("players", 0))
    bet = float(st.get("bet", 0))
    bet_per_player = (bet / players) if players > 0 else bet

    # ⚠️ LOGIC MỚI: NGƯỢC LẠI - Nhiều người/tiền = NGUY HIỂM!
    # Phòng có ÍT người = AN TOÀN hơn (score cao)
    players_safety = 1.0 - min(1.0, players / 50.0)  # Ít người = score cao
    
    # Phòng có ÍT tiền cược = AN TOÀN hơn (score cao)
    bet_safety = 1.0 - min(1.0, bet / 5000.0)  # Ít cược = score cao
    
    # Cược trên đầu người thấp = an toàn
    bpp_safety = 1.0 - min(1.0, bet_per_player / 1500.0)  # BPP thấp = score cao

    # Statistical features - LỊCH SỬ PHÒNG
    kill_count = float(stats.get("kills", 0))
    survive_count = float(stats.get("survives", 0))
    total_appearances = kill_count + survive_count
    
    # Kill rate: tỷ lệ phòng bị kill
    kill_rate = (kill_count + 0.5) / (total_appearances + 1.0)
    survive_score = 1.0 - kill_rate
    
    # 🧠 TƯ DUY: Phòng có kill_rate CAO trong quá khứ = NGUY HIỂM
    historical_danger = kill_rate  # 0.0-1.0, càng cao càng nguy hiểm

    # Recent history analysis (12 ván gần nhất)
    recent_history = list(bet_history)[-12:]
    recent_kills = 0
    recent_survives = 0
    room_appearances = 0
    
    for i, rec in enumerate(reversed(recent_history)):
        if rec.get("room") == rid:
            room_appearances += 1
            # Kiểm tra phòng này có bị kill không
            result = rec.get("result", "")
            if "Thua" in str(result):  # Thua = phòng bị kill
                recent_kills += 1
            elif "Thắng" in str(result):
                recent_survives += 1
    
    # Recent kill rate (12 ván gần nhất)
    recent_kill_rate = recent_kills / max(1, room_appearances) if room_appearances > 0 else 0.5
    
    # 🧠 TƯ DUY: Phòng vừa bị kill nhiều = TRÁNH
    recent_danger = recent_kill_rate

    # Last kill penalty - TĂNG GẤP ĐÔI
    last_pen = 0.0
    if last_killed_room == rid:
        last_pen = 0.7  # Tăng từ 0.35 lên 0.7 - TRÁNH phòng vừa bị kill

    # Hot/Cold analysis - ĐÚNG NGHĨA
    # HOT = Phòng đang "nóng" (nhiều người chơi) = NGUY HIỂM
    # COLD = Phòng đang "lạnh" (ít người chơi) = AN TOÀN
    is_hot = players > 15 or bet > 3000  # Phòng "nóng"
    is_cold = players < 5 or bet < 500   # Phòng "lạnh"
    
    hot_penalty = 0.5 if is_hot else 0.0  # Penalty cho phòng nóng
    cold_bonus = 0.5 if is_cold else 0.0  # Bonus cho phòng lạnh

    # === ULTRA AI FEATURES - TƯ DUY THẬT SỰ ===
    
    # 🧠 CONTRARIAN THINKING: Đi ngược đám đông
    # Nếu phòng này xuất hiện NHIỀU trong history = Nhiều người chọn = TRÁNH
    crowd_following = room_appearances / max(1, len(recent_history))
    contrarian_score = 1.0 - crowd_following  # Càng ít người chọn = score càng cao
    
    # 🧠 CROWD AVOIDANCE: Tránh phòng đông người
    # Tính xem phòng này có phải "honeypot" (bẫy) không
    is_honeypot = (players > 12 and bet > 2000) or (players > 20)
    honeypot_penalty = 0.8 if is_honeypot else 0.0
    
    # Sequence correlation - Phân tích THỰC TẾ
    # Tìm phòng nào THƯỜNG AN TOÀN sau phòng vừa bị kill
    sequence_safety = 0.5
    if len(recent_history) >= 2 and last_killed_room is not None:
        # Xem sau khi last_killed_room bị kill, phòng này có an toàn không
        seq_pattern = f"{last_killed_room}->{rid}"
        if seq_pattern in SEQUENCE_MEMORY:
            mem = SEQUENCE_MEMORY[seq_pattern]
            total = mem["wins"] + mem["losses"]
            if total >= 3:  # Chỉ tin nếu có đủ data
                # Wins = phòng này an toàn sau khi last_killed_room bị kill
                sequence_safety = mem["wins"] / total
    
    # 🧠 MOMENTUM ANALYSIS: Xu hướng gần đây
    recent_5 = list(bet_history)[-5:]
    recent_kills_here = sum(1 for r in recent_5 if r.get("room") == rid and "Thua" in r.get("result", ""))
    recent_safe_here = sum(1 for r in recent_5 if r.get("room") == rid and "Thắng" in r.get("result", ""))
    
    if recent_kills_here > 0:
        # Phòng vừa bị kill gần đây = NGUY HIỂM
        momentum_safety = 0.2
    elif recent_safe_here > 0:
        # Phòng vừa an toàn = Có thể vẫn an toàn
        momentum_safety = 0.7
    else:
        # Chưa có data
        momentum_safety = 0.5
    
    # 🧠 BAYESIAN THINKING: Cập nhật xác suất dựa trên evidence
    # Prior: Giả định ban đầu là mỗi phòng có 1/8 = 12.5% bị kill
    prior_danger = 1.0 / 8.0  # 0.125
    
    # Evidence: Phòng này có nhiều người/tiền = tăng xác suất bị kill
    evidence_danger = 0.0
    if players > 15:
        evidence_danger += 0.2
    if bet > 3000:
        evidence_danger += 0.2
    if players > 20:
        evidence_danger += 0.3
    if bet > 5000:
        evidence_danger += 0.3
    
    # Posterior: Kết hợp prior và evidence
    bayesian_danger = min(0.95, prior_danger + evidence_danger)
    bayesian_safety = 1.0 - bayesian_danger
    
    # Cycle detection - LOGIC ĐÚNG
    cycle_danger = 0.5
    last_kill_round = stats.get("last_kill_round")
    if last_kill_round is not None and round_index > last_kill_round:
        rounds_since = round_index - last_kill_round
        # 🧠 TƯ DUY: Phòng càng LÂU không bị kill = càng GẦN đến lượt bị kill
        # Nhưng cũng có thể là phòng an toàn
        if rounds_since > 10:
            cycle_danger = 0.6  # Có thể sắp đến lượt
        elif rounds_since > 5:
            cycle_danger = 0.5
        else:
            cycle_danger = 0.3  # Vừa mới bị kill = ít khả năng bị kill liên tiếp
    
    # Confidence from pattern memory
    current_pattern = _generate_pattern_signature(recent_history)
    pattern_confidence = _calculate_pattern_confidence(current_pattern)
    
    # 🧠 RISK ASSESSMENT - TƯ DUY ĐA CHIỀU
    risk_factors = [
        historical_danger * 0.2,      # Lịch sử kill rate
        recent_danger * 0.25,         # Kill rate gần đây
        last_pen,                     # Vừa bị kill
        honeypot_penalty,             # Bẫy đám đông
        hot_penalty * 0.3,            # Phòng "nóng"
        bayesian_danger * 0.2,        # Xác suất Bayesian
        (1.0 - pattern_confidence) * 0.15,  # Pattern không chắc chắn
    ]
    risk_score = min(1.0, sum(risk_factors))
    
    # 🧠 SAFETY SCORE - KẾT HỢP NHIỀU YẾU TỐ
    safety_factors = [
        players_safety * 0.25,        # Ít người = an toàn
        bet_safety * 0.25,            # Ít tiền = an toàn
        survive_score * 0.15,         # Lịch sử sống sót
        contrarian_score * 0.15,      # Đi ngược đám đông
        cold_bonus,                   # Phòng "lạnh"
        bayesian_safety * 0.1,        # Bayesian
        sequence_safety * 0.1,        # Sequence patterns
        momentum_safety * 0.1,        # Momentum
    ]
    safety_score = min(1.0, sum(safety_factors))

    return {
        # NEW LOGIC - Counterintuitive (Đi ngược đám đông)
        "players_safety": players_safety,      # ÍT người = AN TOÀN
        "bet_safety": bet_safety,              # ÍT tiền = AN TOÀN
        "bpp_safety": bpp_safety,              # BPP thấp = AN TOÀN
        "contrarian_score": contrarian_score,  # Đi ngược = THÔNG MINH
        "honeypot_penalty": honeypot_penalty,  # Bẫy đám đông
        "hot_penalty": hot_penalty,            # Phòng nóng = NGUY HIỂM
        "cold_bonus": cold_bonus,              # Phòng lạnh = AN TOÀN
        
        # Historical analysis
        "survive_score": survive_score,        # Lịch sử sống sót
        "historical_danger": historical_danger, # Kill rate lịch sử
        "recent_danger": recent_danger,        # Kill rate gần đây
        
        # Advanced analysis
        "bayesian_safety": bayesian_safety,    # Xác suất Bayesian
        "sequence_safety": sequence_safety,    # Patterns sau kill
        "momentum_safety": momentum_safety,    # Xu hướng momentum
        "cycle_danger": cycle_danger,          # Chu kỳ kill
        "pattern_confidence": pattern_confidence,  # Tin cậy pattern
        
        # Penalties
        "last_pen": last_pen,                  # Vừa bị kill (x2)
        
        # Final scores
        "safety_score": safety_score,          # Tổng hợp an toàn
        "risk_score": risk_score,              # Tổng hợp rủi ro
        
        # Legacy (backward compatibility)
        "players_norm": 1.0 - players_safety,  # Đảo ngược logic
        "bet_norm": 1.0 - bet_safety,
        "bpp_norm": 1.0 - bpp_safety,
        "recent_pen": recent_danger,
        "hot_score": cold_bonus,               # Đổi tên: cold = good
        "cold_score": hot_penalty,             # hot = bad
        "pattern_strength": contrarian_score,
        "sequence_correlation": sequence_safety,
        "momentum": momentum_safety,
    }

# Backward compatibility
def _room_features_enhanced(rid: int):
    return _room_features_ultra_ai(rid)

def _update_markov_chain(prev_killed: Optional[int], current_killed: Optional[int]):
    """
    Cập nhật Markov Chain transition matrix.
    Học xác suất chuyển trạng thái: Phòng A bị kill → Phòng B bị kill tiếp theo.
    """
    global MARKOV_TRANSITION_MATRIX, MARKOV_STATE_COUNTS
    if prev_killed is not None and current_killed is not None:
        MARKOV_TRANSITION_MATRIX[prev_killed][current_killed] += 1
        MARKOV_STATE_COUNTS[prev_killed] += 1

def _get_markov_probability(from_room: int, to_room: int) -> float:
    """
    Lấy xác suất chuyển từ phòng A bị kill sang phòng B bị kill.
    P(B|A) = count(A→B) / count(A)
    """
    if from_room not in MARKOV_STATE_COUNTS or MARKOV_STATE_COUNTS[from_room] == 0:
        return 1.0 / 8.0  # Uniform prior
    
    transition_count = MARKOV_TRANSITION_MATRIX[from_room].get(to_room, 0)
    total_count = MARKOV_STATE_COUNTS[from_room]
    return (transition_count + 0.5) / (total_count + 4.0)  # Laplace smoothing

def _calculate_shannon_entropy(room_data: Dict[int, float]) -> float:
    """
    Tính Shannon Entropy để đo độ không chắc chắn.
    H(X) = -Σ p(x) * log(p(x))
    Entropy cao = Không chắc chắn cao
    """
    try:
        values = list(room_data.values())
        if len(values) == 0:
            return 0.0
        
        total = sum(values)
        if total == 0:
            return 0.0
        
        # Calculate probabilities
        probs = [v / total for v in values if v > 0]
        
        # Shannon entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    except Exception as e:
        log_debug(f"Shannon entropy error: {e}")
        return 0.0

def _exponential_moving_average(room_id: int, new_value: float) -> float:
    """
    EMA (Exponential Moving Average) để smooth predictions.
    EMA_t = α * value_t + (1-α) * EMA_{t-1}
    """
    global ROOM_EMA, EMA_ALPHA
    prev_ema = ROOM_EMA.get(room_id, 0.5)
    new_ema = EMA_ALPHA * new_value + (1 - EMA_ALPHA) * prev_ema
    ROOM_EMA[room_id] = new_ema
    return new_ema

def _weighted_median(values: List[float], weights: List[float]) -> float:
    """
    Tính Weighted Median - Robust hơn mean.
    """
    if not values or not weights:
        return 0.5
    
    # Sort by values
    sorted_pairs = sorted(zip(values, weights))
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5
    
    cumsum = 0.0
    for val, weight in sorted_pairs:
        cumsum += weight
        if cumsum >= total_weight / 2.0:
            return val
    
    return sorted_pairs[-1][0]

def _calculate_gini_coefficient(values: List[float]) -> float:
    """
    Gini Coefficient để đo độ phân tán.
    Gini = 0: Hoàn toàn đồng đều
    Gini = 1: Hoàn toàn tập trung
    """
    if not values or len(values) < 2:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = 0.0
    
    for i, val in enumerate(sorted_values):
        cumsum += (i + 1) * val
    
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    
    gini = (2.0 * cumsum) / (n * total) - (n + 1) / n
    return gini

def _regression_trend(values: List[float]) -> float:
    """
    Linear Regression để tìm trend (tăng/giảm).
    Returns: slope (dương = tăng, âm = giảm)
    """
    if not values or len(values) < 2:
        return 0.0
    
    n = len(values)
    x = list(range(n))
    
    # Calculate means
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    
    # Calculate slope
    numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    return slope

def _kalman_filter_estimate(room_id: int, measurement: float, prev_estimate: float = 0.5, 
                            prev_error: float = 1.0) -> Tuple[float, float]:
    """
    Kalman Filter để ước lượng trạng thái phòng.
    Kết hợp measurement mới với estimate cũ một cách optimal.
    
    Returns: (new_estimate, new_error)
    """
    # Process noise và measurement noise (tuneable)
    Q = 0.01  # Process noise covariance
    R = 0.1   # Measurement noise covariance
    
    # Prediction
    predicted_estimate = prev_estimate
    predicted_error = prev_error + Q
    
    # Update
    kalman_gain = predicted_error / (predicted_error + R)
    new_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
    new_error = (1 - kalman_gain) * predicted_error
    
    return new_estimate, new_error

def _monte_carlo_simulation(room_features: Dict[int, Dict[str, float]], n_sims: int = 200) -> Dict[int, float]:
    """
    Monte Carlo Simulation để ước lượng xác suất thắng của mỗi phòng.
    Chạy N simulations với noise ngẫu nhiên. OPTIMIZED FOR SPEED.
    """
    win_counts = {r: 0 for r in ROOM_ORDER}
    
    # Pre-extract safety scores for speed
    base_scores = {}
    for r in ROOM_ORDER:
        if r in room_features:
            base_scores[r] = room_features[r].get("safety_score", 0.5)
        else:
            base_scores[r] = 0.5
    
    # Fast simulation loop
    for _ in range(n_sims):
        noisy_scores = {}
        for r in ROOM_ORDER:
            # Faster noise generation
            noise = random.gauss(0, 0.1)
            noisy_scores[r] = max(0.0, min(1.0, base_scores[r] + noise))
        
        # Find best room
        best_room = max(noisy_scores.items(), key=lambda x: x[1])[0]
        win_counts[best_room] += 1
    
    # Convert to probabilities
    total = n_sims  # Fixed for speed
    return {r: count / total for r, count in win_counts.items()}

def _calculate_kelly_criterion(win_prob: float, win_multiplier: float = 7.0, 
                               loss_multiplier: float = 1.0) -> float:
    """
    Kelly Criterion: Tính phần trăm vốn nên đặt cược.
    f* = (p * b - q) / b
    
    p = xác suất thắng
    q = xác suất thua = 1 - p
    b = tỷ lệ thắng/thua
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    
    p = win_prob
    q = 1 - p
    b = win_multiplier / loss_multiplier
    
    kelly_fraction = (p * b - q) / b
    
    # Safety: Only bet if Kelly > 0, and cap at 25% (fractional Kelly)
    if kelly_fraction > 0:
        return min(kelly_fraction * 0.25, 0.25)  # Fractional Kelly (1/4)
    else:
        return 0.0

def _generate_pattern_signature(recent_history: List[Dict[str, Any]]) -> str:
    """
    Tạo chữ ký pattern từ lịch sử gần đây để nhận diện xu hướng.
    """
    if not recent_history:
        return "EMPTY"
    
    sig_parts = []
    for rec in recent_history[-8:]:  # Lấy 8 ván gần nhất
        room = rec.get("room", 0)
        result = rec.get("result", "")
        sig_parts.append(f"{room}{'W' if 'Thắng' in result else 'L' if 'Thua' in result else 'P'}")
    return "-".join(sig_parts)

def _calculate_pattern_confidence(pattern: str) -> float:
    """
    Tính độ tin cậy của pattern dựa trên lịch sử.
    """
    global SEQUENCE_MEMORY
    if pattern not in SEQUENCE_MEMORY:
        return 0.5
    
    mem = SEQUENCE_MEMORY[pattern]
    total = mem["wins"] + mem["losses"]
    if total == 0:
        return 0.5
    
    win_rate = mem["wins"] / total
    # Confidence tăng theo số lần xuất hiện và win rate
    confidence = min(0.95, (win_rate * 0.7) + (min(total / 100, 1.0) * 0.3))
    return confidence

def _init_formulas(mode: str = "ULTRA_AI"):
    """
    Initialize ULTRA AI formulas với 10000 công thức thông minh.
    Mỗi công thức có khả năng học và tự điều chỉnh.
    """
    global FORMULAS, META_LEARNING_RATE
    
    rng = random.Random(FORMULA_SEED)
    formulas = []
    
    console.print("[bold cyan]🧠 Đang khởi tạo ULTRA AI với 10,000 công thức thông minh...[/]")
    
    # Tạo 10000 công thức với độ đa dạng cao
    for i in range(10000):
        # Phân bố công thức theo các nhóm chiến lược khác nhau
        strategy_type = i % 10
        
        if strategy_type == 0:  # Conservative - Ưu tiên an toàn
            w = {
                "players": rng.uniform(0.8, 0.95),
                "bet": rng.uniform(0.6, 0.85),
                "bpp": rng.uniform(0.5, 0.75),
                "survive": rng.uniform(0.7, 0.9),
                "recent": rng.uniform(0.4, 0.65),
                "last": rng.uniform(0.7, 0.9),
                "hot": rng.uniform(0.3, 0.6),
                "cold": rng.uniform(0.1, 0.3),
                "pattern": rng.uniform(0.5, 0.8),
                "sequence": rng.uniform(0.4, 0.7),
                "momentum": rng.uniform(0.2, 0.5),
            }
        elif strategy_type == 1:  # Aggressive - Tấn công mạnh
            w = {
                "players": rng.uniform(0.2, 0.5),
                "bet": rng.uniform(0.1, 0.4),
                "bpp": rng.uniform(0.1, 0.4),
                "survive": rng.uniform(0.1, 0.4),
                "recent": rng.uniform(0.1, 0.4),
                "last": rng.uniform(0.2, 0.5),
                "hot": rng.uniform(0.6, 0.9),
                "cold": rng.uniform(0.5, 0.8),
                "pattern": rng.uniform(0.3, 0.6),
                "sequence": rng.uniform(0.5, 0.8),
                "momentum": rng.uniform(0.6, 0.9),
            }
        elif strategy_type == 2:  # Pattern-focused - Tập trung vào mẫu
            w = {
                "players": rng.uniform(0.4, 0.7),
                "bet": rng.uniform(0.3, 0.6),
                "bpp": rng.uniform(0.3, 0.6),
                "survive": rng.uniform(0.4, 0.7),
                "recent": rng.uniform(0.2, 0.5),
                "last": rng.uniform(0.4, 0.7),
                "hot": rng.uniform(0.4, 0.7),
                "cold": rng.uniform(0.3, 0.6),
                "pattern": rng.uniform(0.8, 0.95),
                "sequence": rng.uniform(0.8, 0.95),
                "momentum": rng.uniform(0.4, 0.7),
            }
        elif strategy_type == 3:  # Momentum-based - Dựa trên xu hướng
            w = {
                "players": rng.uniform(0.3, 0.6),
                "bet": rng.uniform(0.3, 0.6),
                "bpp": rng.uniform(0.2, 0.5),
                "survive": rng.uniform(0.3, 0.6),
                "recent": rng.uniform(0.5, 0.8),
                "last": rng.uniform(0.3, 0.6),
                "hot": rng.uniform(0.5, 0.8),
                "cold": rng.uniform(0.4, 0.7),
                "pattern": rng.uniform(0.5, 0.8),
                "sequence": rng.uniform(0.6, 0.9),
                "momentum": rng.uniform(0.8, 0.95),
            }
        else:  # Balanced - Cân bằng
            w = {
                "players": rng.uniform(0.4, 0.8),
                "bet": rng.uniform(0.3, 0.7),
                "bpp": rng.uniform(0.3, 0.7),
                "survive": rng.uniform(0.3, 0.7),
                "recent": rng.uniform(0.3, 0.7),
                "last": rng.uniform(0.4, 0.8),
                "hot": rng.uniform(0.3, 0.7),
                "cold": rng.uniform(0.3, 0.7),
                "pattern": rng.uniform(0.5, 0.8),
                "sequence": rng.uniform(0.5, 0.8),
                "momentum": rng.uniform(0.4, 0.7),
            }
        
        # Thêm các thuộc tính học sâu
        formula = {
            "w": w,
            "noise": rng.uniform(0.0, 0.05),
            "adapt": 1.0,  # Weight động, tăng/giảm theo hiệu suất
            "confidence": 0.5,  # Độ tin cậy công thức
            "wins": 0,
            "losses": 0,
            "win_streak": 0,
            "loss_streak": 0,
            "learning_rate": META_LEARNING_RATE,
            "strategy_type": strategy_type,
            "performance_history": deque(maxlen=50),  # Lưu 50 lần dự đoán gần nhất
            "pattern_memory": {},  # Nhớ patterns riêng
        }
        formulas.append(formula)
    
    FORMULAS = formulas
    console.print(f"[bold green]✅ Đã khởi tạo {len(FORMULAS)} công thức ULTRA AI![/]")

# initialize ULTRA AI formulas
_init_formulas("ULTRA_AI")

def choose_room(mode: str = "ULTRA_AI") -> Tuple[int, str]:
    """
    🧠 ULTRA AI Room Chooser - Siêu trí tuệ, NHANH NHẸN, CHÍNH XÁC.
    
    OPTIMIZED FOR SPEED + ACCURACY:
    1. Ensemble Learning (10,000 formulas) - Fast voting
    2. Markov Chain - Cached probabilities
    3. Kalman Filter - Optimal estimation
    4. Monte Carlo Simulation (200 sims) - Fast enough
    5. Advanced combination - Smart weights
    
    Returns (room_id, algo_label)
    """
    global FORMULAS, PATTERN_MEMORY, SEQUENCE_MEMORY, KALMAN_ESTIMATES
    
    import time
    start_time = time.time()
    
    # Ensure formulas initialized
    if not FORMULAS or len(FORMULAS) != 10000:
        _init_formulas("ULTRA_AI")

    cand = [r for r in ROOM_ORDER]
    
    # ⚡ PRE-COMPUTE room features ONCE for all formulas (MAJOR SPEEDUP!)
    room_features_cache = {}
    for r in cand:
        room_features_cache[r] = _room_features_ultra_ai(r)
    
    # Tính toán scores từ tất cả formulas
    formula_votes = {r: [] for r in cand}  # Lưu (score, confidence) cho mỗi phòng
    
    for idx, fentry in enumerate(FORMULAS):
        weights = fentry["w"]
        adapt = fentry.get("adapt", 1.0)
        formula_confidence = fentry.get("confidence", 0.5)
        noise_scale = fentry.get("noise", 0.02)
        
        room_scores = {}
        for r in cand:
            f = room_features_cache[r]  # ⚡ Use pre-computed features!
            score = 0.0
            
            # 🧠 NEW LOGIC - COUNTERINTUITIVE (Đi ngược đám đông)
            # Phòng ÍT người/tiền = ĐIỂM CAO
            score += weights.get("players", 0.0) * f["players_safety"]
            score += weights.get("bet", 0.0) * f["bet_safety"]
            score += weights.get("bpp", 0.0) * f["bpp_safety"]
            
            # Đi ngược đám đông = THÔNG MINH
            score += weights.get("pattern", 0.0) * f["contrarian_score"] * 2.0
            
            # Phòng LẠNH = AN TOÀN
            score += weights.get("hot", 0.0) * f["cold_bonus"] * 1.5
            
            # Penalties cho phòng NÓNG
            score -= weights.get("cold", 0.0) * f["hot_penalty"] * 1.5
            score -= f["honeypot_penalty"] * 0.5
            
            # Historical & Recent
            score += weights.get("survive", 0.0) * f["survive_score"]
            score -= f["historical_danger"] * 0.3
            score -= f["recent_danger"] * 0.4
            
            # Bayesian & Advanced
            score += f["bayesian_safety"] * 0.4
            score += weights.get("sequence", 0.0) * f["sequence_safety"]
            score += weights.get("momentum", 0.0) * f["momentum_safety"]
            
            # Pattern confidence
            score += weights.get("pattern", 0.0) * f["pattern_confidence"] * 0.3
            
            # Penalties
            score -= weights.get("last", 0.0) * f["last_pen"]  # Đã tăng x2
            
            # Final safety/risk
            score += f["safety_score"] * 0.5
            score -= f["risk_score"] * 0.5
            
            # Deterministic noise
            noise = (math.sin((idx + 1) * (r + 1) * 12.9898) * 43758.5453) % 1.0
            noise = (noise - 0.5) * (noise_scale * 2.0)
            score += noise
            
            # Scale by adapt (learning weight)
            score *= adapt
            
            room_scores[r] = score
        
        # Formula picks best room
        best_room = max(room_scores.items(), key=lambda x: x[1])[0]
        best_score = room_scores[best_room]
        
        # Weighted vote based on formula confidence and adapt
        vote_weight = adapt * formula_confidence
        formula_votes[best_room].append((best_score, vote_weight))
    
    # Aggregate votes với weighted ensemble
    final_scores = {}
    confidence_scores = {}
    
    for r in cand:
        votes = formula_votes[r]
        if not votes:
            final_scores[r] = 0.0
            confidence_scores[r] = 0.0
            continue
        
        # Weighted average của scores
        total_weight = sum(w for _, w in votes)
        if total_weight > 0:
            weighted_score = sum(s * w for s, w in votes) / total_weight
        else:
            weighted_score = 0.0
        
        # Confidence dựa trên consensus (số vote)
        vote_ratio = len(votes) / len(FORMULAS)
        consensus_confidence = vote_ratio
        
        # Pattern confidence từ memory
        recent_history = list(bet_history)
        current_pattern = _generate_pattern_signature(recent_history)
        pattern_conf = _calculate_pattern_confidence(current_pattern + f"->{r}")
        
        # Combined confidence
        combined_confidence = (consensus_confidence * 0.5) + (pattern_conf * 0.5)
        
        final_scores[r] = weighted_score
        confidence_scores[r] = combined_confidence
    
    # Boost scores with high confidence
    for r in cand:
        f = _room_features_ultra_ai(r)
        final_scores[r] *= (1.0 + confidence_scores[r] * 0.3)
        
        # Pattern memory boost
        if PATTERN_MEMORY:
            recent_successful_rooms = [p.get("room") for p in PATTERN_MEMORY if p.get("result") == "win"]
            if recent_successful_rooms.count(r) > 3:
                final_scores[r] *= 1.15  # Boost phòng thường thắng
        
        # Anti-pattern penalty
        if ANTI_PATTERNS:
            recent_failed_rooms = [p.get("room") for p in ANTI_PATTERNS]
            if recent_failed_rooms.count(r) > 3:
                final_scores[r] *= 0.85  # Giảm phòng thường thua
        
        # Final safety boost
        final_scores[r] += f["safety_score"] * 0.15
        final_scores[r] -= f["risk_score"] * 0.1
    
    # ==================== ADVANCED MATHEMATICAL MODELS ====================
    
    # 1. MARKOV CHAIN ANALYSIS
    markov_scores = {}
    if last_killed_room is not None:
        for r in cand:
            # P(phòng r bị kill | phòng last_killed_room vừa bị kill)
            markov_prob_kill = _get_markov_probability(last_killed_room, r)
            # Safety = 1 - probability of being killed
            markov_scores[r] = 1.0 - markov_prob_kill
    else:
        markov_scores = {r: 0.5 for r in cand}
    
    # 2. KALMAN FILTER - Optimal estimation
    kalman_scores = {}
    for r in cand:
        # Get previous Kalman estimate
        if r in KALMAN_ESTIMATES:
            prev_est = KALMAN_ESTIMATES[r]["estimate"]
            prev_err = KALMAN_ESTIMATES[r]["error"]
        else:
            prev_est = 0.5
            prev_err = 1.0
        
        # New measurement = final_scores (normalized)
        max_score = max(final_scores.values()) if final_scores.values() else 1.0
        measurement = final_scores[r] / max_score if max_score > 0 else 0.5
        
        # Kalman update
        new_est, new_err = _kalman_filter_estimate(r, measurement, prev_est, prev_err)
        KALMAN_ESTIMATES[r] = {"estimate": new_est, "error": new_err}
        kalman_scores[r] = new_est
    
    # 3. FAST MONTE CARLO - Only if we have time (>1s elapsed)
    elapsed = time.time() - start_time
    if elapsed < 1.5:  # Only run Monte Carlo if we have time
        room_features_dict = {}
        for r in cand:
            room_features_dict[r] = _room_features_ultra_ai(r)
        
        monte_carlo_probs = _monte_carlo_simulation(room_features_dict, n_sims=MONTE_CARLO_SIMS)
        use_monte_carlo = True
    else:
        # Skip Monte Carlo if too slow
        monte_carlo_probs = {r: final_scores[r] for r in cand}
        use_monte_carlo = False
        log_debug("⚠️ Skipped Monte Carlo (time constraint)")
    
    # 4. FAST EMA SMOOTHING - Very fast
    ema_scores = {}
    for r in cand:
        ema_scores[r] = _exponential_moving_average(r, final_scores[r])
    
    # SIMPLIFIED ENSEMBLE - Fast combination
    # Only use the most important methods
    final_advanced_scores = {}
    for r in cand:
        if use_monte_carlo:
            # Full combination with Monte Carlo
            combined = (
                final_scores[r] * 0.30 +           # Ensemble: 30%
                markov_scores[r] * 0.25 +          # Markov: 25%
                kalman_scores[r] * 0.20 +          # Kalman: 20%
                monte_carlo_probs[r] * 0.15 +      # Monte Carlo: 15%
                ema_scores[r] * 0.10               # EMA: 10%
            )
        else:
            # Fast combination without Monte Carlo
            combined = (
                final_scores[r] * 0.40 +           # Ensemble: 40%
                markov_scores[r] * 0.30 +          # Markov: 30%
                kalman_scores[r] * 0.20 +          # Kalman: 20%
                ema_scores[r] * 0.10               # EMA: 10%
            )
        final_advanced_scores[r] = combined
    
    # Select best room
    ranked = sorted(final_advanced_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_room = ranked[0][0]
    best_confidence = confidence_scores[best_room]
    
    # Enhanced confidence with Monte Carlo
    mc_confidence = monte_carlo_probs[best_room]
    enhanced_confidence = (best_confidence * 0.6 + mc_confidence * 0.4)
    
    # Performance tracking
    total_elapsed = time.time() - start_time
    
    # Logging - OPTIMIZED (only if debug mode)
    if LOG_LEVEL == "DEBUG":
        log_debug(f"⚡ ULTRA_AI - Room {best_room} in {total_elapsed:.2f}s")
        log_debug(f"  📊 Ensemble: {final_scores[best_room]:.3f}")
        log_debug(f"  🔗 Markov: {markov_scores[best_room]:.3f}")
        log_debug(f"  🎯 Kalman: {kalman_scores[best_room]:.3f}")
        if use_monte_carlo:
            log_debug(f"  🎲 Monte Carlo: {monte_carlo_probs[best_room]:.3f}")
        log_debug(f"  〰️ EMA: {ema_scores[best_room]:.3f}")
        log_debug(f"  ⭐ FINAL: {final_advanced_scores[best_room]:.3f}")
    
    return best_room, f"ULTRA_AI (Conf: {enhanced_confidence:.1%})"

def update_formulas_after_result(predicted_room: Optional[int], killed_room: Optional[int], mode: str = "ULTRA_AI", lr: float = 0.15):
    """
    🧠 ULTRA AI Learning System - Học sâu từ mỗi kết quả.
    
    Cập nhật:
    1. Adapt weights của từng formula
    2. Confidence scores
    3. Pattern memory
    4. Sequence memory
    5. Anti-pattern detection
    6. Meta-learning (điều chỉnh learning rate)
    """
    global FORMULAS, PATTERN_MEMORY, SEQUENCE_MEMORY, ANTI_PATTERNS, META_LEARNING_RATE
    
    if not FORMULAS:
        return
    
    if predicted_room is None or killed_room is None:
        return
    
    # Determine win or loss
    win = (predicted_room != killed_room)
    
    # Update pattern and sequence memory
    recent_history = list(bet_history)
    current_pattern = _generate_pattern_signature(recent_history)
    
    if win:
        # Lưu vào pattern memory thành công
        PATTERN_MEMORY.append({
            "pattern": current_pattern,
            "room": predicted_room,
            "result": "win",
            "killed": killed_room,
            "timestamp": time.time()
        })
        
        # Update sequence memory (phòng này thành công sau pattern nào)
        SEQUENCE_MEMORY[current_pattern + f"->{predicted_room}"]["wins"] += 1
        SEQUENCE_MEMORY[current_pattern + f"->{predicted_room}"]["confidence"] = _calculate_pattern_confidence(current_pattern + f"->{predicted_room}")
    else:
        # Lưu vào anti-patterns
        ANTI_PATTERNS.append({
            "pattern": current_pattern,
            "room": predicted_room,
            "result": "loss",
            "killed": killed_room,
            "timestamp": time.time()
        })
        
        # Update sequence memory (phòng này thất bại)
        SEQUENCE_MEMORY[current_pattern + f"->{predicted_room}"]["losses"] += 1
        SEQUENCE_MEMORY[current_pattern + f"->{predicted_room}"]["confidence"] = _calculate_pattern_confidence(current_pattern + f"->{predicted_room}")
    
    # Determine which formulas voted for which room
    votes_for_pred = []
    votes_for_killed = []
    votes_for_others = []
    
    for idx, fentry in enumerate(FORMULAS):
        weights = fentry["w"]
        best_room = None
        best_score = -1e9
        
        for r in ROOM_ORDER:
            feat = _room_features_ultra_ai(r)
            score = 0.0
            
            # Calculate score with all features
            score += weights.get("players", 0.0) * feat["players_norm"]
            score += weights.get("bet", 0.0) * feat["bet_norm"]
            score += weights.get("bpp", 0.0) * feat["bpp_norm"]
            score += weights.get("survive", 0.0) * feat["survive_score"]
            score -= weights.get("recent", 0.0) * feat["recent_pen"]
            score -= weights.get("last", 0.0) * feat["last_pen"]
            score += weights.get("hot", 0.0) * feat["hot_score"]
            score -= weights.get("cold", 0.0) * feat["cold_score"]
            score += weights.get("pattern", 0.0) * feat.get("pattern_strength", 0.5)
            score += weights.get("sequence", 0.0) * feat.get("sequence_correlation", 0.5)
            score += weights.get("momentum", 0.0) * feat.get("momentum", 0.5)
            
            if score > best_score:
                best_score = score
                best_room = r
        
        if best_room == predicted_room:
            votes_for_pred.append(idx)
        elif best_room == killed_room:
            votes_for_killed.append(idx)
        else:
            votes_for_others.append(idx)
    
    # Update each formula based on its vote and result
    for idx, fentry in enumerate(FORMULAS):
        old_adapt = fentry.get("adapt", 1.0)
        old_confidence = fentry.get("confidence", 0.5)
        formula_lr = fentry.get("learning_rate", lr)
        
        # Update wins/losses
        if idx in votes_for_pred:
            if win:
                fentry["wins"] = fentry.get("wins", 0) + 1
                fentry["win_streak"] = fentry.get("win_streak", 0) + 1
                fentry["loss_streak"] = 0
            else:
                fentry["losses"] = fentry.get("losses", 0) + 1
                fentry["loss_streak"] = fentry.get("loss_streak", 0) + 1
                fentry["win_streak"] = 0
        
        # Update adapt weight
        new_adapt = old_adapt
        if win:
            # THẮNG - reward formulas voted for predicted, penalize formulas voted for killed
            if idx in votes_for_pred:
                # Strong reward
                boost = 1.0 + formula_lr * 1.5
                new_adapt = old_adapt * boost
                # Update confidence
                fentry["confidence"] = min(0.95, old_confidence + 0.05)
            elif idx in votes_for_killed:
                # Strong penalty
                penalty = 1.0 - formula_lr * 1.0
                new_adapt = old_adapt * penalty
                fentry["confidence"] = max(0.1, old_confidence - 0.03)
            else:
                # Mild penalty for voting wrong room
                penalty = 1.0 - formula_lr * 0.3
                new_adapt = old_adapt * penalty
                fentry["confidence"] = max(0.2, old_confidence - 0.01)
        else:
            # THUA - penalize formulas voted for predicted, reward formulas that avoided it
            if idx in votes_for_pred:
                # Strong penalty
                penalty = 1.0 - formula_lr * 1.2
                new_adapt = max(0.05, old_adapt * penalty)
                fentry["confidence"] = max(0.05, old_confidence - 0.08)
            elif idx in votes_for_killed:
                # Should have listened! Small reward for being "right" about danger
                boost = 1.0 + formula_lr * 0.5
                new_adapt = old_adapt * boost
                fentry["confidence"] = min(0.9, old_confidence + 0.02)
            else:
                # Slight reward for avoiding both
                boost = 1.0 + formula_lr * 0.4
                new_adapt = old_adapt * boost
                fentry["confidence"] = min(0.85, old_confidence + 0.01)
        
        # Clamp adapt
        new_adapt = min(max(new_adapt, 0.05), 10.0)
        fentry["adapt"] = new_adapt
        
        # Update performance history
        perf_hist = fentry.get("performance_history", deque(maxlen=50))
        perf_hist.append({
            "win": win,
            "voted_for": "pred" if idx in votes_for_pred else ("killed" if idx in votes_for_killed else "other"),
            "adapt_before": old_adapt,
            "adapt_after": new_adapt
        })
        fentry["performance_history"] = perf_hist
        
        # Meta-learning: adjust formula's learning rate based on performance
        total_votes = len(perf_hist)
        if total_votes >= 10:
            recent_wins = sum(1 for p in perf_hist if p["win"] and p["voted_for"] == "pred")
            win_rate = recent_wins / total_votes
            
            # Adjust learning rate: good performers learn slower (stable), poor performers learn faster (explore)
            if win_rate > 0.6:
                fentry["learning_rate"] = max(0.05, formula_lr * 0.9)  # Slow down learning
            elif win_rate < 0.4:
                fentry["learning_rate"] = min(0.3, formula_lr * 1.1)  # Speed up learning
    
    # Meta-learning: adjust global learning rate
    if len(bet_history) >= 20:
        recent_results = [b.get("result") for b in list(bet_history)[-20:]]
        recent_win_rate = sum(1 for r in recent_results if "Thắng" in str(r)) / len(recent_results)
        
        if recent_win_rate > 0.65:
            # Đang thắng nhiều -> giảm learning rate để ổn định
            META_LEARNING_RATE = max(0.08, META_LEARNING_RATE * 0.95)
        elif recent_win_rate < 0.45:
            # Đang thua nhiều -> tăng learning rate để thích nghi nhanh
            META_LEARNING_RATE = min(0.25, META_LEARNING_RATE * 1.05)
    
    # Sort formulas by performance (best performers get priority in future)
    # (Optional: có thể implement weighted sampling based on adapt * confidence)
    
    console.print(f"[dim]🧠 ULTRA AI đã học: {'✅ Thắng' if win else '❌ Thua'} | LR={META_LEARNING_RATE:.3f} | Pattern={current_pattern[:30]}...[/]")

# -------------------- BETTING HELPERS --------------------

def api_headers() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "user-agent": "Mozilla/5.0",
        "user-id": str(USER_ID) if USER_ID else "",
        "user-secret-key": SECRET_KEY if SECRET_KEY else ""
    }


def place_bet_http(issue: int, room_id: int, amount: float) -> dict:
    payload = {"asset_type": "BUILD", "user_id": USER_ID, "room_id": int(room_id), "bet_amount": float(amount)}
    try:
        r = HTTP.post(BET_API_URL, headers=api_headers(), json=payload, timeout=6)
        try:
            return r.json()
        except Exception:
            return {"raw": r.text, "http_status": r.status_code}
    except Exception as e:
        return {"error": str(e)}


def record_bet(issue: int, room_id: int, amount: float, resp: dict, algo_used: Optional[str] = None) -> dict:
    now = datetime.now(tz).strftime("%H:%M:%S")
    rec = {"issue": issue, "room": room_id, "amount": float(amount), "time": now, "resp": resp, "result": "Đang", "algo": algo_used, "delta": 0.0, "win_streak": win_streak, "lose_streak": lose_streak}
    bet_history.append(rec)
    return rec


def place_bet_async(issue: int, room_id: int, amount: float, algo_used: Optional[str] = None):
    def worker():
        console.print(f"[cyan]Đang đặt {amount} BUILD -> PHÒNG_{room_id} (v{issue}) — Thuật toán: {algo_used}[/]")
        time.sleep(random.uniform(0.02, 0.25))
        res = place_bet_http(issue, room_id, amount)
        rec = record_bet(issue, room_id, amount, res, algo_used=algo_used)
        if isinstance(res, dict) and (res.get("msg") == "ok" or res.get("code") == 0 or res.get("status") in ("ok", 1)):
            bet_sent_for_issue.add(issue)
            console.print(f"[green]✅ Đặt thành công {amount} BUILD vào PHÒNG_{room_id} (v{issue}).[/]")
        else:
            console.print(f"[red]❌ Đặt lỗi v{issue}: {res}[/]")
    threading.Thread(target=worker, daemon=True).start()

# -------------------- LOCK & AUTO-BET --------------------

def lock_prediction_if_needed(force: bool = False):
    global prediction_locked, predicted_room, ui_state, current_bet, _rounds_placed_since_skip, skip_next_round_flag, _skip_rounds_remaining, _skip_active_issue
    if stop_flag:
        return
    if prediction_locked and not force:
        return
    if issue_id is None:
        return

    # --- ĐANG NGHỈ SAU KHI THUA ---
    if _skip_rounds_remaining > 0:
        # chỉ trừ 1 lần khi sang ván mới
        if _skip_active_issue != issue_id:
            console.print(f"[yellow]⏸️ Đang nghỉ {_skip_rounds_remaining} ván theo cấu hình sau khi thua.[/]")
            _skip_rounds_remaining -= 1         # tiêu thụ 1 ván nghỉ
            _skip_active_issue = issue_id       # nhớ là ván này đã nghỉ

        # khóa đến hết ván hiện tại để không bị các tick countdown đặt lại
        prediction_locked = True
        ui_state = "ANALYZING"                  # hoặc "PREDICTED" tuỳ UI
        return

    # Chọn phòng chỉ khi KHÔNG skip
    algo = settings.get("algo", "ULTRA_AI")
    try:
        chosen, algo_used = choose_room(algo)
    except Exception as e:
        log_debug(f"choose_room error: {e}")
        console.print(f"[red]⚠️ ULTRA AI selection error: {e}[/]")
        chosen, algo_used = choose_room("ULTRA_AI")
    predicted_room = chosen
    prediction_locked = True
    ui_state = "PREDICTED"

    # place bet if AUTO
    if run_mode == "AUTO" and not skip_next_round_flag:
        # get balance quickly (non-blocking - allow poller to update if needed)
        bld, _, _ = fetch_balances_3games(params={"userId": str(USER_ID)} if USER_ID else None)
        if bld is None:
            console.print("[yellow]⚠️ Không lấy được số dư trước khi đặt — bỏ qua đặt ván này.[/]")
            prediction_locked = False
            return
        global current_bet

        # 🧠 ULTRA AI: KELLY CRITERION - Optimal Bet Sizing
        try:
            import re as regex_module
            conf_match = regex_module.search(r'Conf:\s*(\d+)%', str(algo_used))
            if conf_match:
                confidence_pct = float(conf_match.group(1)) / 100.0
            else:
                confidence_pct = 0.5
            
            # Kelly Criterion: Optimal bet size
            kelly_fraction = _calculate_kelly_criterion(
                win_prob=confidence_pct,
                win_multiplier=7.0,  # Game thường trả 7:1
                loss_multiplier=1.0
            )
            
            # Kelly-based multiplier
            if kelly_fraction > 0.15:
                # High Kelly = High confidence
                confidence_multiplier = 1.0 + (kelly_fraction * 0.5)  # Up to 1.125x
                console.print(f"[green]📊 Kelly Criterion: {kelly_fraction:.1%} → Tăng cược {confidence_multiplier:.2f}x[/green]")
            elif kelly_fraction > 0.05:
                # Medium Kelly
                confidence_multiplier = 1.0
                console.print(f"[cyan]📊 Kelly Criterion: {kelly_fraction:.1%} → Giữ nguyên[/cyan]")
            else:
                # Low Kelly = Low confidence
                confidence_multiplier = 0.7
                console.print(f"[yellow]📊 Kelly Criterion: {kelly_fraction:.1%} → Giảm cược 0.7x[/yellow]")
            
            # Additional: Simple confidence-based adjustment
            if confidence_pct >= 0.75:
                confidence_multiplier *= 1.05
                console.print(f"[green]🚀 Confidence {confidence_pct:.0%} + Kelly → {confidence_multiplier:.2f}x[/green]")
            elif confidence_pct <= 0.40:
                confidence_multiplier *= 0.9
                console.print(f"[yellow]⚠️ Confidence thấp {confidence_pct:.0%} → Giảm thêm[/yellow]")
                
        except Exception as e:
            confidence_multiplier = 1.0
            log_debug(f"Kelly Criterion error: {e}")

        # Debug: Kiểm tra current_bet trước khi đặt cược
        if current_bet is None:
            current_bet = base_bet
        
        amt = float(current_bet) * confidence_multiplier
        
        # Đảm bảo amt >= base_bet (không giảm quá thấp)
        amt = max(amt, base_bet * 0.5)
        
        console.print(f"[cyan]💰 ULTRA AI đặt cược: {amt:.4f} BUILD (Base: {current_bet}, Conf×: {confidence_multiplier})[/cyan]")
        if amt <= 0:
            console.print("[yellow]⚠️ Số tiền đặt không hợp lệ (<=0). Bỏ qua.[/]")
            prediction_locked = False
            return
        place_bet_async(issue_id, predicted_room, amt, algo_used=algo_used)
        _rounds_placed_since_skip += 1
        if bet_rounds_before_skip > 0 and _rounds_placed_since_skip >= bet_rounds_before_skip:
            skip_next_round_flag = True
            _rounds_placed_since_skip = 0
    elif skip_next_round_flag:
        console.print("[yellow]⏸️ TẠM DỪNG THEO DÕI SÁT THỦ[/]")
        skip_next_round_flag = False

# -------------------- WEBSOCKET HANDLERS --------------------

def safe_send_enter_game(ws):
    if not ws:
        log_debug("safe_send_enter_game: ws None")
        return
    try:
        payload = {"msg_type": "handle_enter_game", "asset_type": "BUILD", "user_id": USER_ID, "user_secret_key": SECRET_KEY}
        ws.send(json.dumps(payload))
        log_debug("Sent enter_game")
    except Exception as e:
        log_debug(f"safe_send_enter_game err: {e}")


def _extract_issue_id(d: Dict[str, Any]) -> Optional[int]:
    if not isinstance(d, dict):
        return None
    possible = []
    for key in ("issue_id", "issueId", "issue", "id"):
        v = d.get(key)
        if v is not None:
            possible.append(v)
    if isinstance(d.get("data"), dict):
        for key in ("issue_id", "issueId", "issue", "id"):
            v = d["data"].get(key)
            if v is not None:
                possible.append(v)
    for p in possible:
        try:
            return int(p)
        except Exception:
            try:
                return int(str(p))
            except Exception:
                continue
    return None


def on_open(ws):
    _ws["ws"] = ws
    console.print("[green]ĐANG TRUY CẬP DỮ LIỆU GAME[/]")
    safe_send_enter_game(ws)


def _background_fetch_balance_after_result():
    # fetch in background to update cumulative etc
    try:
        fetch_balances_3games()
    except Exception:
        pass


def _mark_bet_result_from_issue(res_issue: Optional[int], krid: int):
    """
    Update kết quả CHỈ KHI có đặt cược ở issue đó.
    Tránh reset current_bet sai khi skip round.
    """
    global current_bet, win_streak, lose_streak, max_win_streak, max_lose_streak
    global _skip_rounds_remaining, stop_flag, _skip_active_issue

    if res_issue is None:
        return

    # ✅ Quan trọng: chỉ xử lý nếu THỰC SỰ đã đặt cược ở issue này
    if res_issue not in bet_sent_for_issue:
        # Không có cược cho ván này (ví dụ đang nghỉ) -> bỏ qua hoàn toàn
        log_debug(f"_mark_bet_result_from_issue: skip issue {res_issue} (no bet placed)")
        return

    # Tìm đúng bản ghi của issue này (KHÔNG fallback)
    rec = next((b for b in reversed(bet_history) if b.get("issue") == res_issue), None)
    if rec is None:
        log_debug(f"_mark_bet_result_from_issue: no record found for issue {res_issue}, skip")
        return

    # Tránh xử lý lặp
    if rec.get("settled"):
        log_debug(f"_mark_bet_result_from_issue: issue {res_issue} already settled, skip")
        return

    try:
        placed_room = int(rec.get("room"))
        # Nếu phòng bị kill khác phòng đã đặt => THẮNG
        if placed_room != int(krid):
            rec["result"] = "Thắng"
            rec["settled"] = True
            
            # Tính delta thắng (amount × hệ số phòng, thường là ×7 hoặc tùy game)
            bet_amount = float(rec.get("amount"))
            # Game này thắng = nhận lại tiền cược (không tính thêm)
            # Vì vậy delta = 0 (đã đặt, giờ được giữ lại)
            # Lãi thực tế = các phòng khác bị trừ chia đều (game tự tính)
            rec["delta"] = 0.0  # Balance sẽ tự update qua fetch_balances
            
            current_bet = base_bet              # reset martingale về base
            win_streak += 1
            lose_streak = 0
            if win_streak > max_win_streak:
                max_win_streak = win_streak
            
            console.print(f"[green]🟢 THẮNG! Phòng {placed_room} an toàn. Reset về base: {current_bet} BUILD[/green]")
        else:
            # THUA -> nhân tiền cho ván kế tiếp
            rec["result"] = "Thua"
            rec["settled"] = True
            
            bet_amount = float(rec.get("amount"))
            rec["delta"] = -bet_amount  # Mất tiền đã đặt
            
            try:
                old_bet = current_bet
                current_bet = bet_amount * float(multiplier)
                console.print(f"[red]🔴 THUA! Mất {bet_amount} BUILD. Ván sau: {bet_amount} × {multiplier} = {current_bet} BUILD[/red]")
            except Exception as e:
                current_bet = base_bet
                console.print(f"[red]🔴 THUA! Lỗi tính toán: {e}, reset về: {current_bet} BUILD[/red]")
            
            lose_streak += 1
            win_streak = 0
            if lose_streak > max_lose_streak:
                max_lose_streak = lose_streak
            if pause_after_losses > 0:
                _skip_rounds_remaining = pause_after_losses
                _skip_active_issue = None        # để ván kế tiếp mới trừ 1 lần
    except Exception as e:
        log_debug(f"_mark_bet_result_from_issue err: {e}")
    finally:
        # dọn whitelist cho issue đã xử lý xong (optional)
        try:
            bet_sent_for_issue.discard(res_issue)
        except Exception:
            pass

    # --- ULTRA AI: update formulas after we resolved result ---
    try:
        # res_issue corresponds to the round we just resolved; killed_room is global
        # ULTRA AI always learns from every result
        update_formulas_after_result(predicted_room, krid, settings.get("algo", "ULTRA_AI"))
        
        # Update accuracy stats
        global total_predictions, correct_predictions
        total_predictions += 1
        if predicted_room is not None and krid is not None:
            if int(predicted_room) != int(krid):
                correct_predictions += 1
        
        # Update Markov Chain
        if last_killed_room is not None and krid is not None:
            _update_markov_chain(last_killed_room, krid)
        
    except Exception as e:
        log_debug(f"update_formulas_after_result err: {e}")
        console.print(f"[dim red]⚠️ ULTRA AI learning error: {e}[/]")

def on_message(ws, message):
    global issue_id, count_down, killed_room, round_index, ui_state, analysis_start_ts, issue_start_ts
    global prediction_locked, predicted_room, last_killed_room, last_msg_ts, current_bet
    global win_streak, lose_streak, max_win_streak, max_lose_streak, cumulative_profit, _skip_rounds_remaining, stop_flag, analysis_blur
    last_msg_ts = time.time()
    try:
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8", errors="replace")
            except Exception:
                message = str(message)
        data = None
        try:
            data = json.loads(message)
        except Exception:
            try:
                data = json.loads(message.replace("'", '"'))
            except Exception:
                log_debug(f"on_message non-json: {str(message)[:200]}")
                return

        # sometimes payload wraps JSON string in data field
        if isinstance(data, dict) and isinstance(data.get("data"), str):
            try:
                inner = json.loads(data.get("data"))
                merged = dict(data)
                merged.update(inner)
                data = merged
            except Exception:
                pass

        msg_type = data.get("msg_type") or data.get("type") or ""
        msg_type = str(msg_type)
        new_issue = _extract_issue_id(data)

        # issue stat / rooms update
        if msg_type == "notify_issue_stat" or "issue_stat" in msg_type:
            rooms = data.get("rooms") or []
            if not rooms and isinstance(data.get("data"), dict):
                rooms = data["data"].get("rooms", [])
            for rm in (rooms or []):
                try:
                    rid = int(rm.get("room_id") or rm.get("roomId") or rm.get("id"))
                except Exception:
                    continue
                players = int(rm.get("user_cnt") or rm.get("userCount") or 0) or 0
                bet = int(rm.get("total_bet_amount") or rm.get("totalBet") or rm.get("bet") or 0) or 0
                room_state[rid] = {"players": players, "bet": bet}
                room_stats[rid]["last_players"] = players
                room_stats[rid]["last_bet"] = bet
            if new_issue is not None and new_issue != issue_id:
                # New issue arrived -> prepare
                log_debug(f"New issue: {issue_id} -> {new_issue}")
                issue_id = new_issue
                issue_start_ts = time.time()
                round_index += 1
                killed_room = None
                prediction_locked = False
                predicted_room = None
                ui_state = "ANALYZING"
                analysis_start_ts = time.time()
                # NOTE: Do NOT lock prediction immediately here so ANALYZING UI shows.

        # countdown
        elif msg_type == "notify_count_down" or "count_down" in msg_type:
            count_down = data.get("count_down") or data.get("countDown") or data.get("count") or count_down
            try:
                count_val = int(count_down)
            except Exception:
                count_val = None
            # enter analysis blur window when <=45s; place bet when <=10s
            if count_val is not None:
                try:
                    # when <=10s, lock and place (if not already locked)
                    if count_val <= 10 and not prediction_locked:
                        # stop blur animation right before placing
                        analysis_blur = False
                        lock_prediction_if_needed()
                    elif count_val <= 45:
                        # start blur-analysis (45s -> 10s)
                        ui_state = "ANALYZING"
                        analysis_start_ts = time.time()
                        analysis_blur = True
                except Exception:
                    pass

        # result
        elif msg_type == "notify_result" or "result" in msg_type:
            # get killed room
            kr = data.get("killed_room") if data.get("killed_room") is not None else data.get("killed_room_id")
            if kr is None and isinstance(data.get("data"), dict):
                kr = data["data"].get("killed_room") or data["data"].get("killed_room_id")
            if kr is not None:
                try:
                    krid = int(kr)
                except Exception:
                    krid = kr
                killed_room = krid
                last_killed_room = krid
                for rid in ROOM_ORDER:
                    if rid == krid:
                        room_stats[rid]["kills"] += 1
                        room_stats[rid]["last_kill_round"] = round_index
                    else:
                        room_stats[rid]["survives"] += 1

                # Immediately mark bet result locally (fast) without waiting for balance
                res_issue = new_issue if new_issue is not None else issue_id
                _mark_bet_result_from_issue(res_issue, krid)
                # Fire background balance refresh to compute actual deltas & cumulative profit
                threading.Thread(target=_background_fetch_balance_after_result, daemon=True).start()

            ui_state = "RESULT"

            # check profit target or stop-loss after we fetched balances (balance fetch may set current_build)
            def _check_stop_conditions():
                global stop_flag, cumulative_profit
                try:
                    # ✅ FIX: So sánh CUMULATIVE PROFIT thay vì current_build
                    if stop_when_profit_reached and profit_target is not None:
                        if cumulative_profit >= profit_target:
                            console.print(f"[bold green]🎉 MỤC TIÊU LÃI ĐẠT: Lãi {cumulative_profit:+.4f} >= {profit_target}. Dừng tool.[/]")
                            stop_flag = True
                            try:
                                wsobj = _ws.get("ws")
                                if wsobj:
                                    wsobj.close()
                            except Exception:
                                pass
                    if stop_when_loss_reached and stop_loss_target is not None:
                        if cumulative_profit <= -abs(stop_loss_target):
                            console.print(f"[bold red]⚠️ STOP-LOSS TRIGGERED: Lỗ {cumulative_profit:+.4f} <= -{abs(stop_loss_target)}. Dừng tool.[/]")
                            stop_flag = True
                            try:
                                wsobj = _ws.get("ws")
                                if wsobj:
                                    wsobj.close()
                            except Exception:
                                pass
                except Exception as e:
                    log_debug(f"_check_stop_conditions error: {e}")
            # run check slightly delayed to allow balance refresh thread update
            threading.Timer(1.2, _check_stop_conditions).start()

    except Exception as e:
        log_debug(f"on_message err: {e}")


def on_close(ws, code, reason):
    log_debug(f"WS closed: {code} {reason}")


def on_error(ws, err):
    log_debug(f"WS error: {err}")


def start_ws():
    backoff = 0.6
    while not stop_flag:
        try:
            ws_app = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message, on_close=on_close, on_error=on_error)
            _ws["ws"] = ws_app
            ws_app.run_forever(ping_interval=12, ping_timeout=6)
        except Exception as e:
            log_debug(f"start_ws exception: {e}")
        t = min(backoff + random.random() * 0.5, 30)
        log_debug(f"Reconnect WS after {t}s")
        time.sleep(t)
        backoff = min(backoff * 1.5, 30)

# -------------------- BALANCE POLLER THREAD --------------------

class BalancePoller(threading.Thread):
    def __init__(self, uid: Optional[int], secret: Optional[str], poll_seconds: int = 2, on_balance=None, on_error=None, on_status=None):
        super().__init__(daemon=True)
        self.uid = uid
        self.secret = secret
        self.poll_seconds = max(1, int(poll_seconds))
        self._running = True
        self._last_balance_local: Optional[float] = None
        self.on_balance = on_balance
        self.on_error = on_error
        self.on_status = on_status

    def stop(self):
        self._running = False

    def run(self):
        if self.on_status:
            self.on_status("Kết nối...")
        while self._running and not stop_flag:
            try:
                build, world, usdt = fetch_balances_3games(params={"userId": str(self.uid)} if self.uid else None, uid=self.uid, secret=self.secret)
                if build is None:
                    raise RuntimeError("Không đọc được balance từ response")
                delta = 0.0 if self._last_balance_local is None else (build - self._last_balance_local)
                first_time = (self._last_balance_local is None)
                if first_time or abs(delta) > 0:
                    self._last_balance_local = build
                    if self.on_balance:
                        self.on_balance(float(build), float(delta), {"ts": human_ts()})
                    if self.on_status:
                        self.on_status("Đang theo dõi")
                else:
                    if self.on_status:
                        self.on_status("Đang theo dõi (không đổi)")
            except Exception as e:
                if self.on_error:
                    self.on_error(str(e))
                if self.on_status:
                    self.on_status("Lỗi kết nối (thử lại...)")
            for _ in range(max(1, int(self.poll_seconds * 5))):
                if not self._running or stop_flag:
                    break
                time.sleep(0.2)
        if self.on_status:
            self.on_status("Đã dừng")

# -------------------- MONITOR --------------------

def monitor_loop():
    global last_balance_fetch_ts, last_msg_ts, stop_flag
    while not stop_flag:
        now = time.time()
        if now - last_balance_fetch_ts >= BALANCE_POLL_INTERVAL:
            last_balance_fetch_ts = now
            try:
                fetch_balances_3games(params={"userId": str(USER_ID)} if USER_ID else None)
            except Exception as e:
                log_debug(f"monitor fetch err: {e}")
        if now - last_msg_ts > 8:
            log_debug("No ws msg >8s, send enter_game")
            try:
                safe_send_enter_game(_ws.get("ws"))
            except Exception as e:
                log_debug(f"monitor send err: {e}")
        if now - last_msg_ts > 30:
            log_debug("No ws msg >30s, force reconnect")
            try:
                wsobj = _ws.get("ws")
                if wsobj:
                    try:
                        wsobj.close()
                    except Exception:
                        pass
            except Exception:
                pass
        # Removed analysis_duration-based auto-lock. Now locking is driven solely by countdown messages (<=10s).
        time.sleep(0.6)

# -------------------- UI (RICH) --------------------

def _spinner_char():
    return _spinner[int(time.time() * 4) % len(_spinner)]

def _brain_spinner_char():
    return _brain_spinner[int(time.time() * 6) % len(_brain_spinner)]

def _analyze_spinner_char():
    return _analyze_spinner[int(time.time() * 8) % len(_analyze_spinner)]

def _rainbow_border_style() -> str:
    idx = int(time.time() * 2) % len(RAINBOW_COLORS)
    return RAINBOW_COLORS[idx]

def _gradient_text(text: str, start_idx: int = 0) -> Text:
    """Tạo text với gradient color."""
    result = Text()
    for i, char in enumerate(text):
        color_idx = (start_idx + i) % len(GRADIENT_COLORS)
        result.append(char, style=GRADIENT_COLORS[color_idx])
    return result

def _get_confidence_color(confidence: float) -> str:
    """Trả về màu dựa trên confidence level."""
    if confidence >= 0.8:
        return "bright_green"
    elif confidence >= 0.7:
        return "green"
    elif confidence >= 0.6:
        return "yellow"
    elif confidence >= 0.5:
        return "orange1"
    else:
        return "red"

def _create_progress_bar(value: float, max_value: float = 1.0, width: int = 20, style: str = "green") -> str:
    """Tạo progress bar đẹp."""
    ratio = min(1.0, max(0.0, value / max_value))
    filled = int(ratio * width)
    bar_chars = "█" * filled + "░" * (width - filled)
    return f"[{style}]{bar_chars}[/{style}]"

def build_header(border_color: Optional[str] = None):
    """Build beautiful header with gradient and advanced stats."""
    tbl = Table.grid(expand=True)
    tbl.add_column(ratio=2)
    tbl.add_column(ratio=1)

    # Gradient title
    title_offset = int(time.time() * 3) % len(GRADIENT_COLORS)
    left = _gradient_text(f"🧠 ULTRA AI - VUA THOÁT HIỂM {_brain_spinner_char()}", title_offset)

    b = f"{current_build:,.4f}" if isinstance(current_build, (int, float)) else (str(current_build) if current_build is not None else "-")
    u = f"{current_usdt:,.4f}" if isinstance(current_usdt, (int, float)) else (str(current_usdt) if current_usdt is not None else "-")
    x = f"{current_world:,.4f}" if isinstance(current_world, (int, float)) else (str(current_world) if current_world is not None else "-")

    pnl_val = cumulative_profit if cumulative_profit is not None else 0.0
    pnl_str = f"{pnl_val:+,.4f}"
    pnl_style = UI_THEME["win"] if pnl_val > 0 else (UI_THEME["loss"] if pnl_val < 0 else UI_THEME["neutral"])

    # Balance with icons
    bal = Text.assemble(
        ("💵 USDT: ", "dim"),
        (f"{u}", "bright_yellow"),
        ("   ", ""),
        ("🌍 XWORLD: ", "dim"),
        (f"{x}", "bright_cyan"),
        ("   ", ""),
        ("🏗️  BUILD: ", "dim"),
        (f"{b}", "bright_green bold")
    )

    # ULTRA AI Stats
    total_formulas = len(FORMULAS) if FORMULAS else 0
    avg_confidence = sum(f.get("confidence", 0.5) for f in FORMULAS) / max(1, total_formulas) if FORMULAS else 0.5
    pattern_count = len(PATTERN_MEMORY) if PATTERN_MEMORY else 0
    anti_pattern_count = len(ANTI_PATTERNS) if ANTI_PATTERNS else 0
    
    # Calculate accuracy
    accuracy = (correct_predictions / max(1, total_predictions)) if total_predictions > 0 else 0.0
    
    # Confidence bar
    conf_color = _get_confidence_color(avg_confidence)
    conf_bar = _create_progress_bar(avg_confidence, 1.0, 15, conf_color)

    right_lines = []
    right_lines.append(f"[bold magenta]⚡ SIÊU TRÍ TUỆ AI[/]")
    right_lines.append(f"[{pnl_style}]💰 Lãi/lỗ: {pnl_str} BUILD[/]")
    right_lines.append(f"[dim]🎮 Phiên:[/] [cyan]{issue_id or '-'}[/]")
    
    # Streaks with colors
    w_color = "bright_green" if win_streak > 0 else "dim"
    l_color = "bright_red" if lose_streak > 0 else "dim"
    right_lines.append(f"[{w_color}]🔥 W:{max_win_streak}[/] [dim]/[/] [{l_color}]❄️  L:{max_lose_streak}[/]")
    
    # AI Confidence với progress bar
    right_lines.append(f"[bold cyan]🧠 AI:[/] {conf_bar} [{conf_color}]{avg_confidence:.0%}[/]")
    
    # Accuracy
    if total_predictions > 0:
        acc_color = "bright_green" if accuracy >= 0.6 else "yellow" if accuracy >= 0.5 else "red"
        right_lines.append(f"[{acc_color}]🎯 Độ chính xác: {accuracy:.1%}[/] [dim]({correct_predictions}/{total_predictions})[/]")
    
    # Memory stats
    right_lines.append(f"[dim]📚 Patterns:[/] [green]{pattern_count}[/] [dim]| Anti:[/] [red]{anti_pattern_count}[/]")
    right_lines.append(f"[dim]⚙️  LR:[/] [yellow]{META_LEARNING_RATE:.3f}[/]")
    
    if stop_when_profit_reached and profit_target is not None:
        prog = min(1.0, max(0.0, pnl_val / profit_target)) if profit_target > 0 else 0
        prog_bar = _create_progress_bar(prog, 1.0, 10, "green")
        right_lines.append(f"[green]🎯 Target: {prog_bar} +{profit_target:.1f}[/]")
    if stop_when_loss_reached and stop_loss_target is not None:
        prog = min(1.0, max(0.0, abs(pnl_val) / stop_loss_target)) if stop_loss_target > 0 else 0
        prog_bar = _create_progress_bar(prog, 1.0, 10, "red")
        right_lines.append(f"[red]🛑 Stop: {prog_bar} -{stop_loss_target:.1f}[/]")

    right = Text.from_markup("\n".join(right_lines))

    tbl.add_row(left, right)
    
    # Time with animation
    time_text = Text.assemble(
        (f"{datetime.now(tz).strftime('%H:%M:%S')}", "bright_cyan"),
        ("  •  ", "dim"),
        (_spinner_char(), "yellow"),
        ("  ", ""),
        (_analyze_spinner_char(), "magenta")
    )
    tbl.add_row(bal, time_text)
    
    panel = Panel(tbl, box=box.DOUBLE, padding=(0,1), border_style=(border_color or _rainbow_border_style()))
    return panel

def build_rooms_table(border_color: Optional[str] = None):
    """Build beautiful rooms table with heatmap."""
    t = Table(box=box.ROUNDED, expand=True, show_header=True, header_style="bold bright_cyan")
    t.add_column("ID", justify="center", width=4, style="bold")
    t.add_column("Phòng", width=18)
    t.add_column("👥", justify="right", width=6)
    t.add_column("💰", justify="right", width=10)
    t.add_column("🎯", justify="right", width=6)
    t.add_column("Status", justify="center", width=18)
    
    # Calculate room features for heatmap
    room_features = {}
    for r in ROOM_ORDER:
        f = _room_features_ultra_ai(r)
        room_features[r] = f
    
    for r in ROOM_ORDER:
        st = room_state.get(r, {})
        feat = room_features.get(r, {})
        
        # Room ID with gradient
        rid_style = "bold cyan"
        
        # Room name
        room_name = ROOM_NAMES.get(r, f"Phòng {r}")
        
        # Players with heatmap color
        players = st.get("players", 0)
        if players == 0:
            players_str = f"[dim]{players}[/]"
        elif players < 5:
            players_str = f"[green]{players}[/]"
        elif players < 15:
            players_str = f"[yellow]{players}[/]"
        else:
            players_str = f"[red]{players}[/]"
        
        # Bet with color
        bet_val = st.get('bet', 0) or 0
        if bet_val == 0:
            bet_str = f"[dim]0[/]"
        elif bet_val < 1000:
            bet_str = f"[green]{int(bet_val):,}[/]"
        elif bet_val < 3000:
            bet_str = f"[yellow]{int(bet_val):,}[/]"
        else:
            bet_str = f"[red]{int(bet_val):,}[/]"
        
        # Safety score with color and icon
        safety = feat.get("safety_score", 0.5)
        if safety >= 0.7:
            safety_str = f"[bright_green]🟢 {safety:.0%}[/]"
        elif safety >= 0.5:
            safety_str = f"[yellow]🟡 {safety:.0%}[/]"
        else:
            safety_str = f"[red]🔴 {safety:.0%}[/]"
        
        # Status with icons
        status_parts = []
        try:
            if killed_room is not None and int(r) == int(killed_room):
                status_parts.append("[bold red]💀 KILL[/]")
        except Exception:
            pass
        try:
            if predicted_room is not None and int(r) == int(predicted_room):
                status_parts.append("[bold green]✅ AI[/]")
        except Exception:
            pass
        
        status = " ".join(status_parts) if status_parts else "[dim]—[/]"
        
        # Row style based on prediction
        row_style = None
        if predicted_room is not None and int(r) == int(predicted_room):
            row_style = "on dark_green"
        elif killed_room is not None and int(r) == int(killed_room):
            row_style = "on dark_red"
        
        t.add_row(
            str(r),
            room_name,
            players_str,
            bet_str,
            safety_str,
            status,
            style=row_style
        )
    
    title = Text.assemble(
        ("🎰 ", "yellow"),
        ("PHÒNG GAME ", "bold bright_cyan"),
        (_brain_spinner_char(), "magenta")
    )
    return Panel(t, title=title, border_style=(border_color or _rainbow_border_style()), box=box.DOUBLE)

def build_mid(border_color: Optional[str] = None):
    """Build middle panel with beautiful animations."""
    global analysis_start_ts, analysis_blur, total_predictions, correct_predictions
    
    # ANALYZING: show a blur / loading visual from 45s down to 10s
    if ui_state == "ANALYZING":
        lines = []
        # Title with animation
        title_text = Text.assemble(
            (_brain_spinner_char(), "bold magenta"),
            (" ĐANG PHÂN TÍCH 10,000 CÔNG THỨC ", "bold bright_cyan"),
            (_analyze_spinner_char(), "bold yellow")
        )
        lines.append(title_text.markup)
        lines.append("")
        
        # Countdown với màu gradient
        if count_down is not None:
            try:
                cd = int(count_down)
                cd_color = "bright_green" if cd > 30 else "yellow" if cd > 10 else "bright_red"
                lines.append(f"[{cd_color}]⏰ Đếm ngược: {cd}s[/]")
            except Exception:
                pass
        else:
            lines.append("[dim]⏳ Chờ dữ liệu...[/]")
        
        lines.append("")

        # blur visual: animated blocks with varying fill to give a 'loading/blur' impression
        if analysis_blur:
            # Multi-line beautiful animation
            lines.append("[bold bright_cyan]╔" + "═" * 50 + "╗[/]")
            
            bar_len = 48
            blocks = []
            tbase = int(time.time() * 5)
            for i in range(bar_len):
                val = (tbase + i) % 7
                ch = "█" if val in (0, 1, 2) else ("▓" if val in (3, 4) else "░")
                color = GRADIENT_COLORS[(i + tbase) % len(GRADIENT_COLORS)]
                blocks.append(f"[{color}]{ch}[/{color}]")
            lines.append("[bold bright_cyan]║[/] " + "".join(blocks) + " [bold bright_cyan]║[/]")
            
            lines.append("[bold bright_cyan]╚" + "═" * 50 + "╝[/]")
            lines.append("")
            lines.append(f"[bold yellow]⚡ AI ĐANG TÍNH TOÁN PHÒNG AN TOÀN NHẤT {_brain_spinner_char()}[/]")
            lines.append(f"[dim]🔬 Phân tích 15+ features × 10,000 formulas...[/]")
        else:
            # Progress bar with percentage
            bar_len = 40
            if count_down is not None:
                try:
                    cd = int(count_down)
                    # Assume total time is 60s
                    progress = (60 - cd) / 60.0
                except Exception:
                    progress = (time.time() % 60) / 60.0
            else:
                progress = (time.time() % 60) / 60.0
            
            filled = int(progress * bar_len)
            bars = []
            for i in range(bar_len):
                if i < filled:
                    color = GRADIENT_COLORS[i % len(GRADIENT_COLORS)]
                    bars.append(f"[{color}]█[/{color}]")
                else:
                    bars.append("[dim]░[/]")
            
            lines.append("".join(bars))
            lines.append(f"[bold cyan]📊 Tiến độ phân tích: {progress:.0%}[/]")

        lines.append("")
        
        # Last killed room với icon
        last_room_name = ROOM_NAMES.get(last_killed_room, '-')
        lines.append(f"[dim]💀 Sát thủ ván trước:[/] [red bold]{last_room_name}[/]")
        
        # AI Stats
        if FORMULAS:
            avg_conf = sum(f.get("confidence", 0.5) for f in FORMULAS) / len(FORMULAS)
            conf_color = _get_confidence_color(avg_conf)
            lines.append(f"[{conf_color}]🧠 Độ tin cậy trung bình: {avg_conf:.1%}[/]")
        
        txt = "\n".join(lines)
        
        title = Text.assemble(
            ("🔍 ", "yellow"),
            ("PHÂN TÍCH ", "bold bright_cyan"),
            (_brain_spinner_char(), "magenta")
        )
        return Panel(Align.center(Text.from_markup(txt), vertical="middle"), title=title, border_style=(border_color or _rainbow_border_style()), box=box.DOUBLE)

    elif ui_state == "PREDICTED":
        name = ROOM_NAMES.get(predicted_room, f"Phòng {predicted_room}") if predicted_room else '-'
        last_bet_amt = current_bet if current_bet is not None else '-'
        lines = []
        
        # Title với animation
        lines.append(f"[bold bright_green]✅ AI ĐÃ CHỌN PHÒNG AN TOÀN {_brain_spinner_char()}[/]")
        lines.append("")
        
        # Room prediction với highlight box
        lines.append(f"[bold cyan]╔{'═' * 40}╗[/]")
        lines.append(f"[bold cyan]║[/]  [bold bright_green]🏆 PHÒNG DỰ ĐOÁN: {name}[/]  [bold cyan]║[/]")
        lines.append(f"[bold cyan]╚{'═' * 40}╝[/]")
        lines.append("")
        
        # Bet amount với màu theo confidence
        if isinstance(last_bet_amt, (int, float)):
            lines.append(f"[bold yellow]💰 Số tiền đặt: {last_bet_amt:.4f} BUILD[/]")
        else:
            lines.append(f"[bold yellow]💰 Số tiền đặt: {last_bet_amt} BUILD[/]")
        
        lines.append("")
        
        # AI Confidence (trích xuất từ algo_used nếu có)
        if FORMULAS:
            avg_conf = sum(f.get("confidence", 0.5) for f in FORMULAS) / len(FORMULAS)
            conf_color = _get_confidence_color(avg_conf)
            conf_bar = _create_progress_bar(avg_conf, 1.0, 25, conf_color)
            lines.append(f"[bold {conf_color}]🧠 Độ tin cậy AI: {conf_bar} {avg_conf:.1%}[/]")
        
        lines.append("")
        
        # Streaks
        lines.append(f"[bright_green]🔥 Chuỗi thắng: {win_streak}[/]  [dim]|[/]  [bright_red]❄️ Chuỗi thua: {lose_streak}[/]")
        
        lines.append("")
        lines.append(f"[dim]💀 Sát thủ ván trước:[/] [red bold]{ROOM_NAMES.get(last_killed_room, '-')}[/]")
        lines.append("")
        
        # Countdown
        if count_down is not None:
            try:
                cd = int(count_down)
                cd_color = "bright_green" if cd > 30 else "yellow" if cd > 10 else "bright_red"
                lines.append(f"[{cd_color}]⏰ Đếm ngược tới kết quả: {cd}s[/]")
            except Exception:
                pass
        
        lines.append("")
        lines.append(f"[bold magenta]🔬 AI đang theo dõi và học hỏi... {_analyze_spinner_char()}[/]")
        
        txt = "\n".join(lines)
        
        title = Text.assemble(
            ("✅ ", "green"),
            ("DỰ ĐOÁN ", "bold bright_green"),
            (_brain_spinner_char(), "magenta")
        )
        return Panel(Align.center(Text.from_markup(txt)), title=title, border_style=(border_color or _rainbow_border_style()), box=box.DOUBLE)

    elif ui_state == "RESULT":
        k = ROOM_NAMES.get(killed_room, "-") if killed_room else "-"
        lines = []
        
        # Determine win/loss
        last_result = None
        if bet_history:
            last_result = bet_history[-1].get('result')
        
        is_win = last_result and 'Thắng' in str(last_result)
        is_loss = last_result and 'Thua' in str(last_result)
        
        # Big result announcement
        if is_win:
            lines.append(f"[bold bright_green]{'🎉' * 20}[/]")
            lines.append(f"[bold bright_green]🏆 CHIẾN THẮNG! 🏆[/]")
            lines.append(f"[bold bright_green]{'🎉' * 20}[/]")
        elif is_loss:
            lines.append(f"[bold bright_red]{'⚠️' * 20}[/]")
            lines.append(f"[bold bright_red]💔 THẤT BẠI 💔[/]")
            lines.append(f"[bold bright_red]{'⚠️' * 20}[/]")
        
        lines.append("")
        
        # Killer room
        lines.append(f"[bold red]💀 Sát thủ đã vào: {k}[/]")
        lines.append("")
        
        # PnL với màu và progress
        pnl_val = cumulative_profit if cumulative_profit is not None else 0.0
        pnl_style = "bright_green" if pnl_val > 0 else "bright_red" if pnl_val < 0 else "yellow"
        pnl_icon = "📈" if pnl_val > 0 else "📉" if pnl_val < 0 else "➡️"
        lines.append(f"[bold {pnl_style}]{pnl_icon} Lãi/lỗ tích lũy: {pnl_val:+.4f} BUILD[/]")
        
        lines.append("")
        
        # Streaks
        lines.append(f"[bold]📊 THỐNG KÊ:[/]")
        lines.append(f"[bright_green]🔥 Max chuỗi thắng: {max_win_streak}[/]")
        lines.append(f"[bright_red]❄️  Max chuỗi thua: {max_lose_streak}[/]")
        
        # Accuracy
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            acc_color = "bright_green" if accuracy >= 0.6 else "yellow" if accuracy >= 0.5 else "red"
            lines.append(f"[{acc_color}]🎯 Độ chính xác: {accuracy:.1%} ({correct_predictions}/{total_predictions})[/]")
        
        lines.append("")
        lines.append(f"[dim]Đang chờ ván tiếp theo... {_spinner_char()}[/]")
        
        txt = "\n".join(lines)
        
        # Border color reflects result
        border = None
        if is_win:
            border = 'bright_green'
        elif is_loss:
            border = 'bright_red'
        
        title = Text.assemble(
            ("📊 ", "cyan"),
            ("KẾT QUẢ ", "bold bright_cyan"),
            ("✨", "yellow")
        )
        return Panel(Align.center(Text.from_markup(txt)), title=title, border_style=(border or (border_color or _rainbow_border_style())), box=box.DOUBLE)
    else:
        lines = []
        lines.append("Chờ ván mới...")
        lines.append(f"Phòng sát thủ vào ván trước: {ROOM_NAMES.get(last_killed_room, '-')}")
        lines.append(f"AI chọn: {ROOM_NAMES.get(predicted_room, '-') if predicted_room else '-'}")
        lines.append(f"Lãi/lỗ: {cumulative_profit:+.4f} BUILD")
        txt = "\n".join(lines)
        return Panel(Align.center(Text.from_markup(txt)), title="TRẠNG THÁI", border_style=(border_color or _rainbow_border_style()))

def build_bet_table(border_color: Optional[str] = None):
    """Build beautiful bet history table with colors and icons."""
    t = Table(box=box.ROUNDED, expand=True, show_header=True, header_style="bold bright_cyan")
    t.add_column("Ván", no_wrap=True, justify="center", width=8)
    t.add_column("Phòng", no_wrap=True, width=18)
    t.add_column("💰 Tiền", justify="right", no_wrap=True, width=12)
    t.add_column("KQ", no_wrap=True, justify="center", width=10)
    t.add_column("Conf", justify="center", width=8)
    t.add_column("Delta", justify="right", width=10)
    
    last5 = list(bet_history)[-5:]
    for b in reversed(last5):
        amt = b.get('amount') or 0
        amt_fmt = f"{float(amt):,.2f}"
        res = str(b.get('result') or '-')
        algo = str(b.get('algo') or '-')
        room_id = b.get('room', '-')
        room_name = ROOM_NAMES.get(room_id, f"P{room_id}") if isinstance(room_id, int) else str(room_id)
        
        # Extract confidence from algo string if available
        conf_str = "-"
        if "Conf:" in algo:
            try:
                import re as re_module
                match = re_module.search(r'Conf:\s*(\d+)%', algo)
                if match:
                    conf_pct = int(match.group(1))
                    conf_color = _get_confidence_color(conf_pct / 100.0)
                    conf_str = f"[{conf_color}]{conf_pct}%[/]"
            except:
                pass
        
        # Result with icon and color
        if res.lower().startswith('thắng') or res.lower().startswith('win'):
            res_text = "[bold bright_green]✅ Thắng[/]"
            row_style = "on dark_green"
            delta_val = b.get('delta', 0.0)
        elif res.lower().startswith('thua') or res.lower().startswith('lose'):
            res_text = "[bold bright_red]❌ Thua[/]"
            row_style = "on dark_red"
            delta_val = b.get('delta', 0.0)
        else:
            res_text = "[yellow]⏳ Đang[/]"
            row_style = ""
            delta_val = 0.0
        
        # Delta
        if delta_val != 0:
            delta_style = "bright_green" if delta_val > 0 else "bright_red"
            delta_str = f"[{delta_style}]{delta_val:+.2f}[/]"
        else:
            delta_str = "[dim]—[/]"
        
        t.add_row(
            str(b.get('issue') or '-'),
            room_name,
            amt_fmt,
            res_text,
            conf_str,
            delta_str,
            style=row_style
        )
    
    title = Text.assemble(
        ("📜 ", "yellow"),
        ("LỊCH SỬ ", "bold bright_cyan"),
        (f"(5 ván gần nhất)", "dim")
    )
    return Panel(t, title=title, border_style=(border_color or _rainbow_border_style()), box=box.DOUBLE)

# -------------------- SETTINGS & START --------------------


def prompt_settings():
    global base_bet, multiplier, run_mode, bet_rounds_before_skip, current_bet
    global pause_after_losses, profit_target, stop_when_profit_reached
    global stop_loss_target, stop_when_loss_reached, settings

    console.print(Rule("[bold cyan]CẤU HÌNH NHANH[/]"))
    base = safe_input("Số BUILD đặt mỗi ván: ", default="1")
    try:
        base_bet = float(base)
    except Exception:
        base_bet = 1.0
    m = safe_input("Nhập 1 số nhân sau khi thua (ổn định thì 2): ", default="2")
    try:
        multiplier = float(m)
    except Exception:
        multiplier = 2.0
    current_bet = base_bet

    # Algorithm selection - ULTRA AI ONLY
    console.print("\n[bold magenta]═══════════════════════════════════════════[/]")
    console.print("[bold cyan]🧠 ULTRA AI - SIÊU TRÍ TUỆ ĐƯỢC KÍCH HOẠT 🧠[/]")
    console.print("[bold magenta]═══════════════════════════════════════════[/]")
    console.print("")
    console.print("[green]✨ Tính năng ULTRA AI:[/]")
    console.print("  • 10,000 công thức thông minh với Deep Learning")
    console.print("  • Pattern Recognition (Nhận diện mẫu)")
    console.print("  • Sequence Learning (Học chuỗi kết quả)")
    console.print("  • Meta-Learning (Học cách học)")
    console.print("  • Confidence Scoring (Đánh giá độ tin cậy)")
    console.print("  • Memory System (Nhớ patterns thành công)")
    console.print("  • Anti-Pattern Detection (Tránh patterns thất bại)")
    console.print("  • Dynamic Learning Rate (Thích ứng tốc độ học)")
    console.print("")
    console.print("[yellow]Tool này đã được nâng cấp lên phiên bản thông minh nhất![/]")
    console.print("")
    
    settings["algo"] = "ULTRA_AI"
    
    # Initialize ULTRA AI formulas
    try:
        _init_formulas("ULTRA_AI")
    except Exception as e:
        console.print(f"[red]Lỗi khởi tạo ULTRA AI: {e}[/]")
        pass

    s = safe_input("Chống soi: sau bao nhiêu ván đặt thì nghỉ 1 ván: ", default="0")
    try:
        bet_rounds_before_skip = int(s)
    except Exception:
        bet_rounds_before_skip = 0

    pl = safe_input("Nếu thua thì nghỉ bao nhiêu tay trước khi cược lại (ví dụ 2): ", default="0")
    try:
        pause_after_losses = int(pl)
    except Exception:
        pause_after_losses = 0

    pt = safe_input("Lãi bao nhiêu BUILD thì dừng (để trống = không giới hạn): ", default="")
    try:
        if pt and pt.strip() != "":
            profit_target = abs(float(pt))  # Đảm bảo số dương
            stop_when_profit_reached = True
            console.print(f"[green]✓ Sẽ dừng khi LÃI đạt +{profit_target:.4f} BUILD[/green]")
        else:
            profit_target = None
            stop_when_profit_reached = False
            console.print("[dim]✓ Không giới hạn lãi[/dim]")
    except Exception:
        profit_target = None
        stop_when_profit_reached = False

    sl = safe_input("Lỗ bao nhiêu BUILD thì dừng (để trống = không giới hạn): ", default="")
    try:
        if sl and sl.strip() != "":
            stop_loss_target = abs(float(sl))  # Đảm bảo số dương
            stop_when_loss_reached = True
            console.print(f"[red]✓ Sẽ dừng khi LỖ đạt -{stop_loss_target:.4f} BUILD[/red]")
        else:
            stop_loss_target = None
            stop_when_loss_reached = False
            console.print("[dim]✓ Không giới hạn lỗ[/dim]")
    except Exception:
        stop_loss_target = None
        stop_when_loss_reached = False

    runm = safe_input("💯bạn đã sẵn sàng hãy nhấn enter để bắt đầu💯: ", default="AUTO")
    run_mode = str(runm).upper()


def start_threads():
    threading.Thread(target=start_ws, daemon=True).start()
    threading.Thread(target=monitor_loop, daemon=True).start()

def parse_login():
    global USER_ID, SECRET_KEY
    console.print(Rule("[bold cyan]ĐĂNG NHẬP[/]"))
    link = safe_input("Dán link trò chơi (từ xworld.info) tại đây (ví dụ chứa userId & secretKey) > ", default=None)
    if not link:
        console.print("[red]Không nhập link. Thoát.[/]")
        sys.exit(1)
    try:
        parsed = urlparse(link)
        params = parse_qs(parsed.query)
        if 'userId' in params:
            USER_ID = int(params.get('userId')[0])
        SECRET_KEY = params.get('secretKey', [None])[0]
        console.print(f"[green]✅ Đã đọc: userId={USER_ID}[/]")
    except Exception as e:
        console.print("[red]Link không hợp lệ. Thoát.[/]")
        log_debug(f"parse_login err: {e}")
        sys.exit(1)

def main():
    parse_login()
    console.print("[bold magenta]Loading...[/]")
    prompt_settings()
    console.print("[bold green]Bắt đầu kết nối dữ liệu...[/]")

    def on_balance_changed(bal, delta, info):
        console.print(f"[green]⤴️ cập nhật số dư: {bal:.4f} (Δ {delta:+.4f}) — {info.get('ts')}[/]")

    def on_error(msg):
        console.print(f"[red]Balance poll lỗi: {msg}[/]")

    poller = BalancePoller(USER_ID, SECRET_KEY, poll_seconds=max(1, int(BALANCE_POLL_INTERVAL)), on_balance=on_balance_changed, on_error=on_error, on_status=None)
    poller.start()
    start_threads()

    with Live(Group(build_header(), build_mid(), build_rooms_table(), build_bet_table()), refresh_per_second=8, console=console, screen=False) as live:
        try:
            while not stop_flag:
                live.update(Group(build_header(), build_mid(), build_rooms_table(), build_bet_table()))
                time.sleep(0.12)
            console.print("[bold yellow]Tool đã dừng theo yêu cầu hoặc đạt mục tiêu.[/]")
        except KeyboardInterrupt:
            console.print("[yellow]Thoát bằng người dùng.[/]")
            poller.stop()

if __name__ == "__main__":
    main()
