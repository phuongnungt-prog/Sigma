# toolws.py (UPGRADED) - tích hợp VIP50, VIP50+, VIP100, ADAPTIVE
from __future__ import annotations

def show_banner():
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    console.print(Panel(
        "[bold yellow]KH TOOL[/]\n[cyan]Copyright by Duy Hoàng | Chỉnh sửa by Khánh[/]",
        expand=True,
        border_style="green"
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

_num_re = re.compile(r"-?\d+[\d,]*\.?\d*")

RAINBOW_COLORS = ["red", "orange1", "yellow1", "green", "cyan", "blue", "magenta"]

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

def _room_features_ultra_ai(rid: int):
    """
    Phân tích đặc trưng phòng với ULTRA AI - nhiều features hơn rất nhiều.
    """
    st = room_state.get(rid, {})
    stats = room_stats.get(rid, {})
    players = float(st.get("players", 0))
    bet = float(st.get("bet", 0))
    bet_per_player = (bet / players) if players > 0 else bet

    # Basic normalized features
    players_norm = min(1.0, players / 50.0)
    bet_norm = 1.0 / (1.0 + bet / 2000.0)
    bpp_norm = 1.0 / (1.0 + bet_per_player / 1200.0)

    # Statistical features
    kill_count = float(stats.get("kills", 0))
    survive_count = float(stats.get("survives", 0))
    kill_rate = (kill_count + 0.5) / (kill_count + survive_count + 1.0)
    survive_score = 1.0 - kill_rate

    # Recent history analysis (12 ván gần nhất)
    recent_history = list(bet_history)[-12:]
    recent_pen = 0.0
    room_appearances = 0
    for i, rec in enumerate(reversed(recent_history)):
        if rec.get("room") == rid:
            recent_pen += 0.12 * (1.0 / (i + 1))
            room_appearances += 1

    # Last kill penalty
    last_pen = 0.0
    if last_killed_room == rid:
        last_pen = 0.35 if SELECTION_CONFIG.get("avoid_last_kill", True) else 0.0

    # Hot/Cold analysis
    hot_score = max(0.0, survive_score - 0.2)
    cold_score = max(0.0, kill_rate - 0.4)

    # === ULTRA AI FEATURES ===
    
    # Pattern strength (dựa trên tần suất xuất hiện trong history)
    pattern_strength = 1.0 - (room_appearances / max(1, len(recent_history)))
    
    # Sequence correlation (phòng này có xu hướng theo sau phòng nào?)
    sequence_correlation = 0.5
    if len(recent_history) >= 2:
        prev_room = recent_history[-1].get("room")
        if prev_room:
            # Tính xác suất phòng này xuất hiện sau prev_room
            seq_pattern = f"{prev_room}->{rid}"
            if seq_pattern in SEQUENCE_MEMORY:
                mem = SEQUENCE_MEMORY[seq_pattern]
                total = mem["wins"] + mem["losses"]
                if total > 0:
                    sequence_correlation = mem["wins"] / total
    
    # Momentum score (xu hướng gần đây)
    momentum = 0.5
    recent_5 = list(bet_history)[-5:]
    recent_kills_here = sum(1 for r in recent_5 if r.get("room") == rid and "Thua" in r.get("result", ""))
    if len(recent_5) > 0:
        momentum = 1.0 - (recent_kills_here / len(recent_5))
    
    # Variance in betting (độ biến động cược)
    bet_variance = 0.0
    if len(recent_history) >= 3:
        room_bets = [float(r.get("amount", 0)) for r in recent_history[-6:] if r.get("room") == rid]
        if len(room_bets) >= 2:
            mean_bet = sum(room_bets) / len(room_bets)
            variance = sum((b - mean_bet) ** 2 for b in room_bets) / len(room_bets)
            bet_variance = min(1.0, variance / 1000.0)
    
    # Cycle detection (phòng này có chu kỳ không?)
    cycle_score = 0.5
    last_kill_round = stats.get("last_kill_round")
    if last_kill_round is not None and round_index > last_kill_round:
        rounds_since = round_index - last_kill_round
        # Phòng càng lâu không bị kill, càng nguy hiểm
        cycle_score = min(1.0, rounds_since / 20.0)
    
    # Confidence from pattern memory
    current_pattern = _generate_pattern_signature(recent_history)
    pattern_confidence = _calculate_pattern_confidence(current_pattern)
    
    # Risk assessment (đánh giá rủi ro tổng hợp)
    risk_factors = [kill_rate, recent_pen / 2.0, last_pen, 1.0 - pattern_confidence]
    risk_score = sum(risk_factors) / len(risk_factors)
    safety_score = 1.0 - risk_score

    return {
        # Basic features
        "players_norm": players_norm,
        "bet_norm": bet_norm,
        "bpp_norm": bpp_norm,
        "survive_score": survive_score,
        "recent_pen": recent_pen,
        "last_pen": last_pen,
        "hot_score": hot_score,
        "cold_score": cold_score,
        # ULTRA AI features
        "pattern_strength": pattern_strength,
        "sequence_correlation": sequence_correlation,
        "momentum": momentum,
        "bet_variance": bet_variance,
        "cycle_score": cycle_score,
        "pattern_confidence": pattern_confidence,
        "safety_score": safety_score,
        "risk_score": risk_score,
    }

# Backward compatibility
def _room_features_enhanced(rid: int):
    return _room_features_ultra_ai(rid)

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
    🧠 ULTRA AI Room Chooser - Siêu trí tuệ với Deep Learning.
    Returns (room_id, algo_label, confidence_score)
    """
    global FORMULAS, PATTERN_MEMORY, SEQUENCE_MEMORY
    
    # Ensure formulas initialized
    if not FORMULAS or len(FORMULAS) != 10000:
        _init_formulas("ULTRA_AI")

    cand = [r for r in ROOM_ORDER]
    
    # Tính toán scores từ tất cả formulas
    formula_votes = {r: [] for r in cand}  # Lưu (score, confidence) cho mỗi phòng
    
    for idx, fentry in enumerate(FORMULAS):
        weights = fentry["w"]
        adapt = fentry.get("adapt", 1.0)
        formula_confidence = fentry.get("confidence", 0.5)
        noise_scale = fentry.get("noise", 0.02)
        
        room_scores = {}
        for r in cand:
            f = _room_features_ultra_ai(r)
            score = 0.0
            
            # Basic features
            score += weights.get("players", 0.0) * f["players_norm"]
            score += weights.get("bet", 0.0) * f["bet_norm"]
            score += weights.get("bpp", 0.0) * f["bpp_norm"]
            score += weights.get("survive", 0.0) * f["survive_score"]
            score -= weights.get("recent", 0.0) * f["recent_pen"]
            score -= weights.get("last", 0.0) * f["last_pen"]
            score += weights.get("hot", 0.0) * f["hot_score"]
            score -= weights.get("cold", 0.0) * f["cold_score"]
            
            # ULTRA AI features
            score += weights.get("pattern", 0.0) * f["pattern_strength"]
            score += weights.get("sequence", 0.0) * f["sequence_correlation"]
            score += weights.get("momentum", 0.0) * f["momentum"]
            score += weights.get("pattern", 0.0) * f["pattern_confidence"] * 0.5
            score += weights.get("survive", 0.0) * f["safety_score"] * 0.3
            score -= weights.get("recent", 0.0) * f["risk_score"] * 0.2
            
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
    
    # Select best room
    ranked = sorted(final_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_room = ranked[0][0]
    best_confidence = confidence_scores[best_room]
    
    # Logging cho analysis
    log_debug(f"ULTRA_AI chose room {best_room} with confidence {best_confidence:.3f}")
    log_debug(f"Top 3 rooms: {[(r, f'{s:.3f}') for r, s in ranked[:3]]}")
    
    return best_room, f"ULTRA_AI (Conf: {best_confidence:.1%})"

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

        # 🧠 ULTRA AI: Adaptive Bet Sizing based on Confidence
        # Trích xuất confidence từ algo_used (format: "ULTRA_AI (Conf: XX%)")
        try:
            import re as regex_module
            conf_match = regex_module.search(r'Conf:\s*(\d+)%', str(algo_used))
            if conf_match:
                confidence_pct = float(conf_match.group(1)) / 100.0
            else:
                confidence_pct = 0.5  # Default
            
            # Nếu confidence cao (>70%), có thể tăng cược một chút (optional)
            # Nếu confidence thấp (<40%), giảm cược xuống để an toàn
            confidence_multiplier = 1.0
            if confidence_pct >= 0.75:
                confidence_multiplier = 1.1  # Tăng 10% khi rất tự tin
                console.print(f"[green]🚀 Confidence cao ({confidence_pct:.0%}), tăng cược lên {confidence_multiplier}x[/green]")
            elif confidence_pct <= 0.40:
                confidence_multiplier = 0.8  # Giảm 20% khi không chắc chắn
                console.print(f"[yellow]⚠️ Confidence thấp ({confidence_pct:.0%}), giảm cược xuống {confidence_multiplier}x[/yellow]")
        except Exception:
            confidence_multiplier = 1.0

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

def _rainbow_border_style() -> str:
    idx = int(time.time() * 2) % len(RAINBOW_COLORS)
    return RAINBOW_COLORS[idx]

def build_header(border_color: Optional[str] = None):
    tbl = Table.grid(expand=True)
    tbl.add_column(ratio=2)
    tbl.add_column(ratio=1)

    left = Text("🧠 ULTRA AI - VUA THOÁT HIỂM 🧠", style="bold cyan")

    b = f"{current_build:,.4f}" if isinstance(current_build, (int, float)) else (str(current_build) if current_build is not None else "-")
    u = f"{current_usdt:,.4f}" if isinstance(current_usdt, (int, float)) else (str(current_usdt) if current_usdt is not None else "-")
    x = f"{current_world:,.4f}" if isinstance(current_world, (int, float)) else (str(current_world) if current_world is not None else "-")

    pnl_val = cumulative_profit if cumulative_profit is not None else 0.0
    pnl_str = f"{pnl_val:+,.4f}"
    pnl_style = "green bold" if pnl_val > 0 else ("red bold" if pnl_val < 0 else "yellow")

    bal = Text.assemble((f"USDT: {u}", "bold"), ("   "), (f"XWORLD: {x}", "bold"), ("   "), (f"BUILD: {b}", "bold"))

    algo_label = SELECTION_MODES.get(settings.get('algo'), settings.get('algo'))

    # ULTRA AI Stats
    total_formulas = len(FORMULAS) if FORMULAS else 0
    avg_confidence = sum(f.get("confidence", 0.5) for f in FORMULAS) / max(1, total_formulas) if FORMULAS else 0.5
    pattern_count = len(PATTERN_MEMORY) if PATTERN_MEMORY else 0
    anti_pattern_count = len(ANTI_PATTERNS) if ANTI_PATTERNS else 0

    right_lines = []
    right_lines.append(f"🎯 {algo_label}")
    right_lines.append(f"Lãi/lỗ: [{pnl_style}] {pnl_str} BUILD [/{pnl_style}]")
    right_lines.append(f"Phiên: {issue_id or '-'}")
    right_lines.append(f"Chuỗi: W={max_win_streak} / L={max_lose_streak}")
    right_lines.append(f"🧠 AI Conf: {avg_confidence:.1%} | Patterns: {pattern_count}")
    right_lines.append(f"📚 Học: {META_LEARNING_RATE:.3f} | Anti: {anti_pattern_count}")
    if stop_when_profit_reached and profit_target is not None:
        right_lines.append(f"[green]🎯 Target: +{profit_target:.2f} BUILD[/]")
    if stop_when_loss_reached and stop_loss_target is not None:
        right_lines.append(f"[red]🛑 Stop: -{stop_loss_target:.2f} BUILD[/]")

    right = Text.from_markup("\n".join(right_lines))

    tbl.add_row(left, right)
    tbl.add_row(bal, Text(f"{datetime.now(tz).strftime('%H:%M:%S')}  •  {_spinner_char()}", style="dim"))
    panel = Panel(tbl, box=box.ROUNDED, padding=(0,1), border_style=(border_color or _rainbow_border_style()))
    return panel

def build_rooms_table(border_color: Optional[str] = None):
    t = Table(box=box.MINIMAL, expand=True)
    t.add_column("ID", justify="center", width=3)
    t.add_column("Phòng", width=16)
    t.add_column("Ng", justify="right")
    t.add_column("Cược", justify="right")
    t.add_column("TT", justify="center")
    for r in ROOM_ORDER:
        st = room_state.get(r, {})
        status = ""
        try:
            if killed_room is not None and int(r) == int(killed_room):
                status = "[red]☠ Kill[/]"
        except Exception:
            pass
        try:
            if predicted_room is not None and int(r) == int(predicted_room):
                status = (status + " [dim]|[/] [green]✓ Dự đoán[/]") if status else "[green]✓ Dự đoán[/]"
        except Exception:
            pass
        players = str(st.get("players", 0))
        bet_val = st.get('bet', 0) or 0
        bet_fmt = f"{int(bet_val):,}"
        t.add_row(str(r), ROOM_NAMES.get(r, f"Phòng {r}"), players, bet_fmt, status)
    return Panel(t, title="PHÒNG", border_style=(border_color or _rainbow_border_style()))

def build_mid(border_color: Optional[str] = None):
    global analysis_start_ts, analysis_blur
    # ANALYZING: show a blur / loading visual from 45s down to 10s
    if ui_state == "ANALYZING":
        lines = []
        lines.append(f"ĐANG PHÂN TÍCH PHÒNG AN TOÀN NHẤT  {_spinner_char()}")
        # show countdown if available (do not show explicit 'will place at Xs' note)
        if count_down is not None:
            try:
                cd = int(count_down)
                lines.append(f"Đếm ngược tới kết quả: {cd}s")
            except Exception:
                pass
        else:
            lines.append("Chưa nhận được dữ liệu đếm ngược...")

        # blur visual: animated blocks with varying fill to give a 'loading/blur' impression
        if analysis_blur:
            bar_len = 36
            blocks = []
            tbase = int(time.time() * 5)
            for i in range(bar_len):
                # pseudo-random flicker deterministic-ish by tbase + i
                val = (tbase + i) % 7
                ch = "█" if val in (0, 1, 2) else ("▓" if val in (3, 4) else "░")
                color = RAINBOW_COLORS[(i + tbase) % len(RAINBOW_COLORS)]
                blocks.append(f"[{color}]{ch}[/{color}]")
            lines.append("".join(blocks))
            lines.append("")
            lines.append("AI ĐANG TÍNH TOÁN 10S CUỐI VÀO BUID")
        else:
            # fallback compact progress bar (no percent text)
            bar_len = 24
            filled = int((time.time() * 2) % (bar_len + 1))
            bars = []
            for i in range(bar_len):
                if i < filled:
                    color = RAINBOW_COLORS[i % len(RAINBOW_COLORS)]
                    bars.append(f"[{color}]█[/{color}]")
                else:
                    bars.append("·")
            lines.append("".join(bars))

        lines.append("")
        lines.append(f"Phòng sát thủ vào ván trước: {ROOM_NAMES.get(last_killed_room, '-')}")
        txt = "\n".join(lines)
        return Panel(Align.center(Text.from_markup(txt), vertical="middle"), title="PHÂN TÍCH", border_style=(border_color or _rainbow_border_style()))

    elif ui_state == "PREDICTED":
        name = ROOM_NAMES.get(predicted_room, f"Phòng {predicted_room}") if predicted_room else '-'
        last_bet_amt = current_bet if current_bet is not None else '-'
        lines = []
        lines.append(f"AI chọn: {name}  — [green]KẾT QUẢ DỰ ĐOÁN[/]")
        lines.append(f"Số đặt: {last_bet_amt} BUILD")
        lines.append(f"Phòng sát thủ vào ván trước: {ROOM_NAMES.get(last_killed_room, '-')}")
        lines.append(f"Chuỗi thắng: {win_streak}  |  Chuỗi thua: {lose_streak}")
        lines.append("")
        if count_down is not None:
            try:
                cd = int(count_down)
                lines.append(f"Đếm ngược tới kết quả: {cd}s")
            except Exception:
                pass
        lines.append("")
        lines.append(f"đang học hỏi dữ liệu {_spinner_char()}")
        txt = "\n".join(lines)
        return Panel(Align.center(Text.from_markup(txt)), title="DỰ ĐOÁN", border_style=(border_color or _rainbow_border_style()))

    elif ui_state == "RESULT":
        k = ROOM_NAMES.get(killed_room, "-") if killed_room else "-"
        last_success = next((str(b.get('amount')) for b in reversed(bet_history) if b.get('result') in ('Thắng', 'Win')), '-')
        lines = []
        lines.append(f"Sát thủ đã vào: {k}")
        lines.append(f"Lãi/lỗ: {cumulative_profit:+.4f} BUILD")
        lines.append(f"Đặt cược thành công (last): {last_success}")
        lines.append(f"Max Chuỗi: W={max_win_streak} / L={max_lose_streak}")
        txt = "\n".join(lines)
        # border color to reflect last result
        border = None
        last = None
        if bet_history:
            last = bet_history[-1].get('result')
        if last == 'Thắng':
            border = 'green'
        elif last == 'Thua':
            border = 'red'
        return Panel(Align.center(Text.from_markup(txt)), title="KẾT QUẢ", border_style=(border or (border_color or _rainbow_border_style())))
    else:
        lines = []
        lines.append("Chờ ván mới...")
        lines.append(f"Phòng sát thủ vào ván trước: {ROOM_NAMES.get(last_killed_room, '-')}")
        lines.append(f"AI chọn: {ROOM_NAMES.get(predicted_room, '-') if predicted_room else '-'}")
        lines.append(f"Lãi/lỗ: {cumulative_profit:+.4f} BUILD")
        txt = "\n".join(lines)
        return Panel(Align.center(Text.from_markup(txt)), title="TRẠNG THÁI", border_style=(border_color or _rainbow_border_style()))

def build_bet_table(border_color: Optional[str] = None):
    t = Table(title="Lịch sử cược (5 ván gần nhất)", box=box.SIMPLE, expand=True)
    t.add_column("Ván", no_wrap=True)
    t.add_column("Phòng", no_wrap=True)
    t.add_column("Tiền", justify="right", no_wrap=True)
    t.add_column("KQ", no_wrap=True)
    t.add_column("Thuật toán", no_wrap=True)
    last5 = list(bet_history)[-5:]
    for b in reversed(last5):
        amt = b.get('amount') or 0
        amt_fmt = f"{float(amt):,.4f}"
        res = str(b.get('result') or '-')
        algo = str(b.get('algo') or '-')
        # color rows: thắng green, thua red, pending yellow
        if res.lower().startswith('thắng') or res.lower().startswith('win'):
            res_text = Text(res, style="green")
            row_style = ""
        elif res.lower().startswith('thua') or res.lower().startswith('lose'):
            res_text = Text(res, style="red")
            row_style = ""
        else:
            res_text = Text(res, style="yellow")
            row_style = ""
        t.add_row(str(b.get('issue') or '-'), str(b.get('room') or '-'), amt_fmt, res_text, algo)
    return Panel(t, border_style=(border_color or _rainbow_border_style()))

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
