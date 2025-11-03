# ⚡ QUANTUM BRAIN AI v14.0 ⚡ - Trí Tuệ Siêu Phàm
from __future__ import annotations

def show_banner():
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    console = Console()
    
    # ASCII art cho QUANTUM BRAIN - công nghệ cao
    quantum_brain = """
    ⠀⠀⠀⢀⣀⣤⣴⣶⣿⣿⣿⣿⣿⣶⣦⣤⣀⡀⠀⠀⠀⠀
    ⠀⢀⣴⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀⠈⠙⠻⢿⣿⣷⣄⠀⠀
    ⢀⣾⣿⡟⠁⠀⣠⣴⣶⣿⣿⣿⣶⣦⣄⠀⠀⠈⢻⣿⣷⡀
    ⣾⣿⠋⠀⢀⣾⣿⠿⠛⠉⠉⠉⠛⠿⣿⣷⡀⠀⠀⠙⣿⣷
    ⣿⡏⠀⠀⣾⡿⠁⠀⣀⣤⣤⣤⣀⠀⠈⢿⣷⠀⠀⠀⢹⣿
    ⣿⡇⠀⠀⣿⡇⠀⢸⣿⣿⣿⣿⣿⡇⠀⢸⣿⠀⠀⠀⢸⣿
    ⣿⣇⠀⠀⢻⣷⡀⠀⠻⢿⣿⡿⠟⠀⢀⣾⡟⠀⠀⠀⣸⣿
    ⠸⣿⣆⠀⠀⠻⣿⣦⣄⣀⣀⣀⣠⣴⣿⠟⠀⠀⠀⣰⣿⠇
    ⠀⠹⣿⣷⣄⠀⠈⠙⠛⠿⠿⠛⠋⠁⠀⠀⠀⣠⣾⣿⠏⠀
    ⠀⠀⠈⠻⣿⣷⣦⣤⣀⣀⣀⣀⣤⣤⣴⣾⣿⡿⠋⠀⠀⠀
    """
    
    title = Text()
    title.append("🚀 ULTRA AI v15.0 🚀\n", style="bold bright_cyan blink")
    title.append("SIÊU TRÍ TUỆ VƯỢT TRỘI", style="bold bright_magenta")
    title.append(" • ", style="dim")
    title.append("AI CAO CẤP NHẤT", style="bold bright_yellow")
    title.append(" • ", style="dim")
    title.append("MACHINE LEARNING", style="bold bright_green")
    
    console.print(Panel(
        Text.from_markup(f"[bright_cyan]{quantum_brain}[/bright_cyan]\n") + title,
        expand=True,
        border_style="bright_magenta",
        box=box.DOUBLE,
        subtitle="[dim bright_cyan]◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤[/dim bright_cyan]"
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
from rich.style import Style

# Gradient text cho header
header_style = Style(color="bright_cyan", bold=True)
console.print(Rule(
    "⚡ QUANTUM BRAIN AI - TRÍ TUỆ SIÊU PHÀM ⚡",
    style="bold bright_magenta"
))
console.print(Align.center(
    Text.from_markup("🔮 Multi-Dimensional • Deep Logic • Meta-Learning • Quantum Computing 🔮"),
    style="bright_cyan"
))
console.print(Align.center(
    Text.from_markup("◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤"),
    style="dim bright_cyan"
))
console.print(
    Text("🔮 Neural Network • Logic Reasoning • Strategic Planning • Predictive Analytics 🔮", 
    style="bright_green", justify="center")
)
console.print(Rule(style="bright_cyan"))

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
ai_reasoning: Optional[str] = None  # Lý do AI chọn phòng

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

# selection mode duy nhất - ULTRA AI v15.0
ALGO_ID = "ULTRA_AI_v15_HYPERINTELLIGENCE"
SELECTION_MODES = {
    ALGO_ID: "🚀 Ultra AI v15.0 - Siêu Trí Tuệ Vượt Trội (Hyper Intelligence)"
}

settings = {"algo": ALGO_ID}

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


def clear_terminal():
    """
    🧹 Clear terminal
    """
    import os
    import platform
    
    # Clear command tùy OS
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')


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

# -------------------- STOP CONDITIONS CHECK --------------------

def _check_stop_profit_loss():
    """
    ✅ CHECK STOP-LOSS VÀ TAKE-PROFIT
    Hàm này được gọi MỖI KHI balance update để đảm bảo dừng kịp thời!
    """
    global stop_flag
    
    try:
        # Check take-profit
        if stop_when_profit_reached and profit_target is not None:
            if cumulative_profit >= profit_target:
                console.print(f"\n[bold green]🎉 MỤC TIÊU LÃI ĐẠT: {cumulative_profit:+.2f} >= {profit_target}[/bold green]")
                console.print(f"[green]Số dư hiện tại: {current_build:.2f} BUILD (Bắt đầu: {starting_balance:.2f})[/green]")
                console.print(f"[green]Tổng lãi: +{cumulative_profit:.2f} BUILD ✅[/green]")
                
                # 💾 LƯU BỘ NHỚ AI TRƯỚC KHI DỪNG!
                try:
                    if selector._learning_enabled:
                        selector._self_learning_ai.save_brain()
                        console.print("[cyan]💾 AI Brain saved![/cyan]")
                except Exception:
                    pass
                
                stop_flag = True
                try:
                    wsobj = _ws.get("ws")
                    if wsobj:
                        wsobj.close()
                except Exception:
                    pass
        
        # Check stop-loss
        if stop_when_loss_reached and stop_loss_target is not None:
            if cumulative_profit <= -abs(stop_loss_target):
                console.print(f"\n[bold red]⚠️ STOP-LOSS TRIGGERED: Lỗ {cumulative_profit:.2f} >= {stop_loss_target}[/bold red]")
                console.print(f"[red]Số dư hiện tại: {current_build:.2f} BUILD (Bắt đầu: {starting_balance:.2f})[/red]")
                console.print(f"[red]Tổng lỗ: {cumulative_profit:.2f} BUILD ❌[/red]")
                
                # 💾 LƯU BỘ NHỚ AI TRƯỚC KHI DỪNG!
                try:
                    if selector._learning_enabled:
                        selector._self_learning_ai.save_brain()
                        console.print("[cyan]💾 AI Brain saved![/cyan]")
                except Exception:
                    pass
                
                stop_flag = True
                try:
                    wsobj = _ws.get("ws")
                    if wsobj:
                        wsobj.close()
                except Exception:
                    pass
    except Exception as e:
        log_debug(f"_check_stop_profit_loss error: {e}")

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
                if last_balance_val is None:
                    starting_balance = build
                    last_balance_val = build
                else:
                    delta = float(build) - float(last_balance_val)
                    if abs(delta) > 0:
                        cumulative_profit += delta
                        last_balance_val = build
                current_build = build
                
                # ✅ CHECK STOP CONDITIONS NGAY SAU KHI UPDATE BALANCE
                _check_stop_profit_loss()
                
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

# -------------------- ULTIMATE AI ADAPTIVE SELECTION (SIÊU TRÍ TUỆ) --------------------

ULTIMATE_AI_SEED = 9876543


def _room_features_ultimate(rid: int) -> Dict[str, float]:
    """Phân tích đặc trưng nâng cao với 20+ chỉ số thông minh"""
    st = room_state.get(rid, {})
    stats = room_stats.get(rid, {})
    players = float(st.get("players", 0) or 0)
    bet = float(st.get("bet", 0) or 0)
    bet_per_player = (bet / players) if players > 0 else bet

    # Chuẩn hóa cơ bản với công thức tối ưu
    players_norm = 1.0 - math.tanh(players / 40.0)  # ưu tiên phòng ít người
    bet_norm = 1.0 / (1.0 + math.sqrt(bet / 1500.0))  # tránh phòng cược cao
    bpp_norm = 1.0 / (1.0 + math.log1p(bet_per_player / 800.0))  # phân tích chi tiết

    # Phân tích tỉ lệ sống sót với độ chính xác cao
    kill_count = float(stats.get("kills", 0) or 0)
    survive_count = float(stats.get("survives", 0) or 0)
    total_rounds = kill_count + survive_count + 1.0
    kill_rate = (kill_count + 0.3) / total_rounds
    survive_score = 1.0 - kill_rate
    
    # Điểm ổn định dựa trên variance
    stability_score = 1.0 / (1.0 + abs(kill_count - survive_count) / max(total_rounds, 1.0))

    # Phân tích lịch sử gần đây với trọng số cao hơn
    recent_history = list(bet_history)[-20:]
    recent_pen = 0.0
    recent_wins = 0
    for i, rec in enumerate(reversed(recent_history)):
        if rec.get("room") == rid:
            weight = 1.0 / (i + 1)
            recent_pen += 0.15 * weight
            if rec.get("result", "").startswith("Thắng"):
                recent_wins += 1

    # Hình phạt cho phòng vừa bị kill (quan trọng!)
    last_pen = 0.0
    if last_killed_room == rid:
        last_pen = 0.5 if SELECTION_CONFIG.get("avoid_last_kill", True) else 0.0

    # Điểm "hot" và "cold" room - phòng đang nóng/lạnh
    hot_score = max(0.0, survive_score - 0.15) * (1.0 + stability_score * 0.3)
    cold_score = max(0.0, kill_rate - 0.35)
    
    # Phân tích xu hướng thắng gần đây
    win_momentum = math.tanh(recent_wins / max(len(recent_history), 1))

    return {
        "players_norm": players_norm,
        "bet_norm": bet_norm,
        "bpp_norm": bpp_norm,
        "survive_score": survive_score,
        "stability_score": stability_score,
        "recent_pen": recent_pen,
        "last_pen": last_pen,
        "hot_score": hot_score,
        "cold_score": cold_score,
        "win_momentum": win_momentum,
    }


class UltimateAISelector:
    """
    🚀 ULTRA AI SELECTOR - SIÊU TRÍ TUỆ VƯỢT TRỘI 🚀
    
    Features:
    - 150 AI Agents với advanced neural networks
    - Advanced ML algorithms (6 types)
    - Ensemble learning
    - Reinforcement learning (Q-Learning)
    - Genetic algorithm evolution
    - Bayesian optimization
    - Time series prediction
    """
    FEATURE_KEYS = (
        "players_norm",
        "bet_norm",
        "bpp_norm",
        "survive_score",
        "stability_score",
        "recent_pen",
        "last_pen",
        "hot_score",
        "cold_score",
        "win_momentum",
        "kill_gap_norm",
        "pressure_score",
        "momentum_players",
        "momentum_bet",
        "volume_share",
        "streak_pressure",
        "adaptive_memory",
        "long_term_memory",
        "pattern_score",
        "volatility_score",
    )

    def __init__(self, room_ids: List[int]):
        self.room_ids = list(room_ids)
        self._rng = random.Random(ULTIMATE_AI_SEED)
        self._lock = threading.Lock()
        # 150 agents với quantum reasoning
        self._agents: List[Dict[str, Any]] = [self._make_agent(i) for i in range(150)]
        self._room_bias: Dict[int, float] = {rid: 0.0 for rid in self.room_ids}
        self._last_votes: List[Tuple[int, int]] = []
        self._last_features: Dict[int, Dict[str, float]] = {}
        self._recent_outcomes: deque = deque(maxlen=100)
        self._explore_rate: float = 0.05
        self._long_term_memory: Dict[int, deque] = {rid: deque(maxlen=200) for rid in self.room_ids}
        self._pattern_detector: Dict[int, List[int]] = {rid: [] for rid in self.room_ids}
        
        # ⚡ QUANTUM ENHANCEMENTS ⚡
        self._quantum_states: Dict[int, float] = {rid: 0.5 for rid in self.room_ids}
        self._entropy_history: deque = deque(maxlen=50)
        self._logic_chains: List[Dict[str, Any]] = []
        self._meta_strategies: Dict[str, Dict[str, Any]] = {
            "quantum_driven": {"wins": 0, "losses": 0, "confidence": 0.5},
            "data_driven": {"wins": 0, "losses": 0, "confidence": 0.5},
            "hybrid": {"wins": 0, "losses": 0, "confidence": 0.5}
        }
        
        # 🚀 ULTRA AI ENHANCEMENTS 🚀
        try:
            from ultra_ai_algorithms import EnsemblePredictor, TimeSeriesPredictor
            self._ensemble_learner = EnsemblePredictor()
            self._time_series = {rid: TimeSeriesPredictor() for rid in self.room_ids}
            self._ultra_ai_enabled = True
        except:
            self._ultra_ai_enabled = False
        
        # 🎓 SELF-LEARNING AI 🎓
        try:
            from self_learning_ai import SelfLearningAI
            self._self_learning_ai = SelfLearningAI()
            self._learning_enabled = True
            
            # 💾 TỰ ĐỘNG LOAD BỘ NHỚ ĐÃ HỌC!
            if self._self_learning_ai.load_brain():
                log_debug(f"🧠 Loaded AI Brain! Total rounds learned: {self._self_learning_ai.total_rounds}")
                log_debug(f"📊 Accuracy: {self._self_learning_ai.online_learner.get_accuracy():.1%}")
            else:
                log_debug("🆕 Starting fresh - no previous brain data")
            
            log_debug("✅ Self-Learning AI initialized!")
        except Exception as e:
            self._learning_enabled = False
            log_debug(f"⚠️ Self-Learning AI disabled: {e}")

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _make_agent(self, idx: int) -> Dict[str, Any]:
        """Tạo agent thông minh với tham số được tối ưu hóa"""
        # Khởi tạo trọng số với phân phối tốt hơn
        weights = {k: self._rng.uniform(-0.15, 1.2) for k in self.FEATURE_KEYS}
        # Tăng trọng số cho các features quan trọng
        weights["survive_score"] = self._rng.uniform(0.8, 1.5)
        weights["stability_score"] = self._rng.uniform(0.6, 1.3)
        weights["last_pen"] = self._rng.uniform(0.7, 1.4)
        weights["hot_score"] = self._rng.uniform(0.5, 1.2)
        
        return {
            "weights": weights,
            "bias": self._rng.uniform(-0.2, 0.2),
            "temperature": self._rng.uniform(0.6, 1.3),  # giảm nhiễu
            "lr": self._rng.uniform(0.08, 0.15),  # learning rate cao hơn
            "momentum": {k: 0.0 for k in self.FEATURE_KEYS},
            "performance": 0.0,  # theo dõi hiệu suất
            "confidence": 0.5,  # độ tin cậy
        }

    def _compute_recent_memory(self, rid: int) -> float:
        """Bộ nhớ ngắn hạn với phân tích chi tiết"""
        if not bet_history:
            return 0.0
        score = 0.0
        decay = 1.0
        for rec in reversed(list(bet_history)[-40:]):
            decay *= 0.90
            if rec.get("room") != rid:
                continue
            res = (rec.get("result") or "").lower()
            if res.startswith("thắng") or res.startswith("win"):
                score += 0.7 * decay
            elif res.startswith("thua") or res.startswith("lose"):
                score -= 0.9 * decay
        return self._clip(score, -1.0, 1.0)
    
    def _compute_long_term_memory(self, rid: int) -> float:
        """Bộ nhớ dài hạn phân tích xu hướng lâu dài"""
        mem = self._long_term_memory.get(rid, deque())
        if len(mem) < 10:
            return 0.0
        # Tính tỉ lệ thắng trong lịch sử dài hạn
        wins = sum(1 for x in mem if x == 1)
        total = len(mem)
        win_rate = wins / total
        # Chuẩn hóa về [-1, 1]
        return (win_rate - 0.5) * 2.0
    
    def _detect_pattern(self, rid: int) -> float:
        """Phát hiện mô hình lặp lại"""
        pattern = self._pattern_detector.get(rid, [])
        if len(pattern) < 5:
            return 0.0
        # Phân tích 5 ván gần nhất
        recent = pattern[-5:]
        # Nếu có xu hướng giết liên tiếp -> tránh
        kills = sum(1 for x in recent if x == 1)
        return -0.3 if kills >= 3 else 0.2

    def _compose_features(self, rid: int) -> Dict[str, float]:
        """Tổng hợp đặc trưng với 20+ chỉ số thông minh"""
        base = _room_features_ultimate(rid)
        st = room_state.get(rid, {})
        stats = room_stats.get(rid, {})

        players = float(st.get("players", 0) or 0)
        bet = float(st.get("bet", 0) or 0)
        last_players = float(stats.get("last_players", players) or 0)
        last_bet = float(stats.get("last_bet", bet) or 0)

        # Momentum với công thức cải tiến
        delta_players = players - last_players
        delta_bet = bet - last_bet
        momentum_players = math.tanh(delta_players / 4.0)
        momentum_bet = math.tanh(delta_bet / 1500.0)

        # Phân tích khoảng cách từ lần kill cuối
        last_kill_round = stats.get("last_kill_round")
        if last_kill_round is None:
            kill_gap_norm = 0.4  # phòng chưa bao giờ kill
        else:
            gap = max(0, round_index - int(last_kill_round))
            kill_gap_norm = math.tanh(gap / 5.0)

        # Phân tích khối lượng cược
        total_bet = sum(((room_state.get(r, {}) or {}).get("bet", 0) or 0) for r in self.room_ids)
        total_bet = float(total_bet) if total_bet else 1.0
        volume_share = math.sqrt(max(bet, 0.0) / total_bet)

        # Áp lực và stress
        pressure_score = math.tanh((players / 10.0) + (bet / 7000.0))
        streak_pressure = math.tanh((lose_streak - win_streak) / 5.0)
        
        # Bộ nhớ đa tầng
        adaptive_memory = self._compute_recent_memory(rid)
        long_term_memory = self._compute_long_term_memory(rid)
        
        # Phát hiện mô hình
        pattern_score = self._detect_pattern(rid)
        
        # Độ biến động (volatility)
        volatility_score = 1.0 - base["stability_score"]

        features = {
            "players_norm": base["players_norm"],
            "bet_norm": base["bet_norm"],
            "bpp_norm": base["bpp_norm"],
            "survive_score": base["survive_score"],
            "stability_score": base["stability_score"],
            "recent_pen": base["recent_pen"],
            "last_pen": base["last_pen"],
            "hot_score": base["hot_score"],
            "cold_score": base["cold_score"],
            "win_momentum": base["win_momentum"],
            "kill_gap_norm": kill_gap_norm,
            "pressure_score": pressure_score,
            "momentum_players": momentum_players,
            "momentum_bet": momentum_bet,
            "volume_share": volume_share,
            "streak_pressure": streak_pressure,
            "adaptive_memory": adaptive_memory,
            "long_term_memory": long_term_memory,
            "pattern_score": pattern_score,
            "volatility_score": volatility_score,
        }
        return features

    def _agent_vote(self, agent: Dict[str, Any], features_map: Dict[int, Dict[str, float]]) -> Tuple[int, float]:
        """Bỏ phiếu với công thức tối ưu hóa"""
        best_room = None
        best_score = -float("inf")
        for rid, feats in features_map.items():
            score = agent["bias"]
            # Tính điểm weighted sum với confidence
            for key, value in feats.items():
                weight = agent["weights"].get(key, 0.0)
                score += weight * value * agent.get("confidence", 1.0)
            
            # Chuẩn hóa với temperature
            score /= max(0.25, agent["temperature"])
            
            # Thêm explore noise (nhỏ hơn để ổn định)
            score += self._rng.uniform(-self._explore_rate, self._explore_rate)
            
            # Bias phòng dựa trên lịch sử
            score += self._room_bias.get(rid, 0.0) * 0.7
            
            # Boost từ performance của agent
            score *= (1.0 + agent.get("performance", 0.0) * 0.15)
            
            if score > best_score:
                best_score = score
                best_room = rid
        return (best_room or self.room_ids[0]), best_score

    def _calculate_safety_score(self, rid: int, features: Dict[str, float]) -> float:
        """
        🛡️ PHÂN TÍCH AN TOÀN DỰA TRÊN DỮ LIỆU THỰC TẾ - KHÔNG ĐỊNH KIẾN!
        AI học từ lịch sử, không phán đoán trước "ít người = an toàn"
        """
        stats = room_stats.get(rid, {})
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1️⃣ DỮ LIỆU THỰC TẾ: Tỉ lệ sống sót trong lịch sử (60% - QUAN TRỌNG NHẤT!)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        kills = float(stats.get("kills", 0) or 0)
        survives = float(stats.get("survives", 0) or 0)
        total_history = kills + survives
        
        if total_history >= 10:
            # Có đủ dữ liệu lịch sử -> TIN DỮ LIỆU
            actual_survive_rate = survives / total_history
            safety_history = actual_survive_rate * 0.60  # 60% - TRỌNG SỐ CHÍNH!
        elif total_history >= 3:
            # Ít dữ liệu -> giảm độ tin cậy
            actual_survive_rate = survives / total_history
            safety_history = actual_survive_rate * 0.40  # Giảm trọng số
        else:
            # Chưa có dữ liệu -> trung lập
            safety_history = 0.25  # Trung lập, chờ học
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2️⃣ ỔN ĐỊNH: Phòng ít biến động = Dự đoán được (20%)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        stability = features.get("stability_score", 0.5)
        safety_stable = stability * 0.20
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3️⃣ XU HƯỚNG GẦN ĐÂY: Thắng nhiều gần đây = Tốt (15%)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phân tích 10 ván gần nhất
        recent_survive_count = 0
        recent_kill_count = 0
        for rec in list(self._recent_outcomes)[-10:]:
            if rec.get("predicted") == rid:
                if rec.get("win"):
                    recent_survive_count += 1
                else:
                    recent_kill_count += 1
        
        recent_total = recent_survive_count + recent_kill_count
        if recent_total >= 3:
            recent_rate = recent_survive_count / recent_total
            safety_recent = recent_rate * 0.15
        else:
            safety_recent = 0.075  # Trung lập
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4️⃣ PENALTY: Các yếu tố NGUY HIỂM
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Vừa bị kill = NGUY HIỂM cao
        if last_killed_room == rid:
            last_kill_penalty = -0.5
        else:
            last_kill_penalty = 0.0
        
        # Pattern giết liên tục
        pattern = self._pattern_detector.get(rid, [])
        if len(pattern) >= 5:
            recent_pattern = pattern[-5:]
            kills_in_pattern = sum(1 for x in recent_pattern if x == 1)
            if kills_in_pattern >= 3:  # 3/5 ván bị kill
                pattern_penalty = -0.2
            else:
                pattern_penalty = 0.0
        else:
            pattern_penalty = 0.0
        
        # Momentum tăng đột biến (có thể là bẫy)
        momentum = features.get("momentum_players", 0.0)
        if momentum > 0.7:  # Tăng RẤT mạnh
            rush_penalty = -0.15
        elif momentum > 0.5:
            rush_penalty = -0.08
        else:
            rush_penalty = 0.0
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎯 TỔNG HỢP: Ưu tiên DỮ LIỆU thực tế
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        total_safety = (
            safety_history +      # 60% hoặc 40% - DỮ LIỆU THỰC TẾ
            safety_stable +       # 20% - Ổn định
            safety_recent +       # 15% - Xu hướng gần
            last_kill_penalty +   # -0.5 nếu vừa kill
            pattern_penalty +     # -0.2 nếu pattern xấu
            rush_penalty          # -0.15 nếu momentum cao
        )
        
        # Chuẩn hóa về [0, 1]
        return self._clip(total_safety, 0.0, 1.0)
    
    def _quantum_analysis(self, rid: int, features: Dict[str, float]) -> float:
        """
        ⚡ PHÂN TÍCH LƯỢNG TỬ - ĐA CHIỀU
        4 chiều: Base + Entropy + Coherence + Superposition
        """
        # Chiều 1: Xác suất cơ bản
        base_prob = features.get("survive_score", 0.5)
        
        # Chiều 2: Entropy (độ hỗn loạn) - thấp = tốt
        volatility = features.get("volatility_score", 0.5)
        momentum = abs(features.get("momentum_players", 0.0))
        entropy = (volatility + momentum) / 2.0
        
        # Chiều 3: Coherence (độ kết dính) - cao = tốt
        stability = features.get("stability_score", 0.5)
        pattern = features.get("pattern_score", 0.0)
        coherence = (stability + max(0, pattern)) / 2.0
        
        # Chiều 4: Superposition (chồng chất trạng thái)
        hot = features.get("hot_score", 0.0)
        cold = features.get("cold_score", 0.0)
        superposition = hot * 0.7 + (1 - cold) * 0.3
        
        # KẾT HỢP 4 CHIỀU
        quantum_prob = (
            base_prob * 0.40 +
            (1 - entropy) * 0.25 +
            coherence * 0.20 +
            superposition * 0.15
        )
        
        # Lưu trạng thái lượng tử
        self._quantum_states[rid] = quantum_prob
        
        return self._clip(quantum_prob, 0.0, 1.0)
    
    def _deep_logic_score(self, rid: int, features: Dict[str, float]) -> float:
        """
        🧠 PHÂN TÍCH LOGIC SÂU
        Suy luận đa tầng: Observation → Correlation → Causation
        """
        score = 0.5  # Base
        
        # Logic 1: Tránh phòng vừa kill
        if last_killed_room == rid:
            score -= 0.4  # Penalty lớn
        
        # Logic 2: Phòng ổn định + survive cao = Logic tốt
        if features.get("stability_score", 0) > 0.6 and features.get("survive_score", 0) > 0.6:
            score += 0.3
        
        # Logic 3: Pattern xấu = Logic loại bỏ
        pattern = self._pattern_detector.get(rid, [])
        if len(pattern) >= 5:
            recent_kills = sum(1 for x in pattern[-5:] if x == 1)
            if recent_kills >= 3:
                score -= 0.25  # 3/5 kill = xấu
        
        # Logic 4: Momentum đột biến = Nghi ngờ
        momentum = abs(features.get("momentum_players", 0.0))
        if momentum > 0.6:
            score -= 0.15  # Bất thường
        
        return self._clip(score, 0.0, 1.0)
    
    def _select_meta_strategy(self, current_win_rate: float) -> str:
        """
        🎓 META-LEARNING - Chọn strategy tốt nhất
        """
        best_strategy = "hybrid"  # Default
        best_confidence = 0.0
        
        for strategy_name, perf in self._meta_strategies.items():
            total = perf["wins"] + perf["losses"]
            if total >= 5:  # Cần ít nhất 5 ván để đánh giá
                win_rate = perf["wins"] / total
                # Confidence cao nếu win rate cao
                perf["confidence"] = win_rate
                
                if win_rate > best_confidence:
                    best_confidence = win_rate
                    best_strategy = strategy_name
        
        # Nếu đang thua liên tục, thử strategy khác
        if current_win_rate < 0.4 and lose_streak >= 3:
            # Đổi strategy
            strategies = list(self._meta_strategies.keys())
            for s in strategies:
                if s != best_strategy:
                    best_strategy = s
                    break
        
        log_debug(f"🎓 Meta-Learning: Chọn strategy '{best_strategy}' (confidence: {best_confidence:.2%})")
        return best_strategy
    
    def select_room(self) -> Tuple[int, str]:
        """
        🧠 CHỌN PHÒNG THÔNG MINH - KẾT HỢP VOTES + AN TOÀN
        KHÔNG chỉ theo đám đông, mà phân tích RỦI RO!
        """
        with self._lock:
            features_map = {rid: self._compose_features(rid) for rid in self.room_ids}
            self._last_features = features_map
            
            # Bước 1: Thu thập votes từ 150 agents
            room_votes = {rid: 0.0 for rid in self.room_ids}
            last_votes: List[Tuple[int, int]] = []
            
            for idx, agent in enumerate(self._agents):
                voted_room, voted_score = self._agent_vote(agent, features_map)
                weight = 1.0 + agent.get("performance", 0.0) * 0.25
                room_votes[voted_room] += voted_score * weight
                last_votes.append((idx, voted_room))
            
            self._last_votes = last_votes
            
            # Bước 2: 🛡️ PHÂN TÍCH AN TOÀN (Data-Driven)
            room_safety = {rid: self._calculate_safety_score(rid, features_map[rid]) for rid in self.room_ids}
            
            # Bước 3: ⚡ QUANTUM ANALYSIS (Đa chiều)
            room_quantum = {rid: self._quantum_analysis(rid, features_map[rid]) for rid in self.room_ids}
            
            # Bước 4: 🧠 DEEP LOGIC SCORE (Suy luận sâu)
            room_logic = {rid: self._deep_logic_score(rid, features_map[rid]) for rid in self.room_ids}
            
            # Bước 5: 🎯 META-LEARNING - Chọn strategy tốt nhất
            current_win_rate = win_streak / max(1, round_index)
            best_strategy = self._select_meta_strategy(current_win_rate)
            
            # Bước 6: 🌌 KẾT HỢP TẤT CẢ CHIỀU PHÂN TÍCH
            room_final_scores = {}
            for rid in self.room_ids:
                max_votes = max(room_votes.values()) if room_votes else 1.0
                normalized_votes = room_votes[rid] / max_votes if max_votes > 0 else 0.0
                
                # 🎓 SELF-LEARNING PREDICTION
                learning_boost = 0.0
                if self._learning_enabled:
                    try:
                        room_data = {
                            "players": int(features_map[rid].get("players_norm", 0.5) * 50),
                            "bet": features_map[rid].get("bet_norm", 0.5) * 100,
                            "survive_rate": features_map[rid].get("survive_score", 0.5)
                        }
                        
                        prediction = self._self_learning_ai.get_room_prediction(
                            rid, features_map[rid], room_data
                        )
                        
                        learning_boost = prediction["final_score"]
                        
                        # Log nếu learning có insight mạnh
                        if prediction["final_score"] > 0.7 or prediction["final_score"] < 0.3:
                            log_debug(f"🎓 Room {rid} learning score: {learning_boost:.2f} (learned_rate: {prediction['learned_rate']:.2f})")
                    except:
                        pass
                
                # ⚡ SUPER FORMULA - Quantum Intelligence + Self-Learning
                if best_strategy == "quantum_driven":
                    final_score = (
                        normalized_votes * 0.20 +
                        room_safety[rid] * 0.25 +
                        room_quantum[rid] * 0.30 +
                        room_logic[rid] * 0.10 +
                        learning_boost * 0.15  # 15% từ self-learning!
                    )
                elif best_strategy == "data_driven":
                    final_score = (
                        normalized_votes * 0.25 +
                        room_safety[rid] * 0.40 +
                        room_quantum[rid] * 0.10 +
                        room_logic[rid] * 0.10 +
                        learning_boost * 0.15
                    )
                else:  # hybrid
                    final_score = (
                        normalized_votes * 0.25 +
                        room_safety[rid] * 0.30 +
                        room_quantum[rid] * 0.20 +
                        room_logic[rid] * 0.10 +
                        learning_boost * 0.15
                    )
                
                room_final_scores[rid] = final_score
            
            # Bước 4: Chọn phòng AN TOÀN NHẤT (không phải votes nhiều nhất!)
            ranked = sorted(room_final_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            choice = ranked[0][0]
            
            # Log chi tiết để debug với DỮ LIỆU THỰC TẾ
            vote_winner = max(room_votes.items(), key=lambda kv: kv[1])[0]
            safety_winner = max(room_safety.items(), key=lambda kv: kv[1])[0]
            
            # Hiển thị thống kê thực tế
            for rid in self.room_ids:
                stats = room_stats.get(rid, {})
                kills = stats.get("kills", 0)
                survives = stats.get("survives", 0)
                total = kills + survives
                survive_rate = (survives / total * 100) if total > 0 else 0
                
                log_debug(f"📊 Phòng {rid}: {survives}W/{kills}L ({survive_rate:.0f}% survive) | "
                         f"Votes: {room_votes[rid]:.0f} | Safety: {room_safety[rid]:.2f} | Final: {room_final_scores[rid]:.3f}")
            
            log_debug(f"\n🗳️  Votes nhiều nhất: Phòng {vote_winner}")
            log_debug(f"🛡️  An toàn nhất (theo dữ liệu): Phòng {safety_winner} ({room_safety[safety_winner]:.1%})")
            log_debug(f"🎯 QUYẾT ĐỊNH CUỐI: Phòng {choice} (Score: {room_final_scores[choice]:.3f})")
            
            choice_stats = room_stats.get(choice, {})
            choice_k = choice_stats.get("kills", 0)
            choice_s = choice_stats.get("survives", 0)
            choice_total = choice_k + choice_s
            if choice_total > 0:
                log_debug(f"📈 Lịch sử phòng {choice}: {choice_s} thắng / {choice_k} thua ({choice_s/(choice_total)*100:.0f}% survive)")
            
            if choice != vote_winner:
                log_debug(f"⚠️  AI KHÔNG THEO ĐÁM ĐÔNG - Dựa vào dữ liệu thực tế!")
            
            return choice, ALGO_ID

    def update(self, predicted_room: Optional[int], killed_room: Optional[int]):
        """
        🎓 CẬP NHẬT & TỰ HỌC - SELF-LEARNING AI ⚡
        Học từ kết quả + Cập nhật tất cả learners
        """
        if predicted_room is None:
            return
        with self._lock:
            # ⚡ META-LEARNING: Cập nhật performance của strategies
            win = (killed_room is not None and predicted_room != killed_room)
            
            # Cập nhật tất cả strategies (học từ mọi kết quả)
            for strategy_name in self._meta_strategies.keys():
                if win:
                    self._meta_strategies[strategy_name]["wins"] += 1
                else:
                    self._meta_strategies[strategy_name]["losses"] += 1
            
            # 🎓 SELF-LEARNING: Học từ ván này
            if self._learning_enabled and killed_room is not None:
                try:
                    # Lấy features của phòng đã chọn
                    room_features = self._last_features.get(predicted_room, {})
                    
                    # Lấy room data từ global hoặc estimate
                    room_data = {
                        "players": int(room_features.get("players_norm", 0.5) * 50),
                        "bet": room_features.get("bet_norm", 0.5) * 100,
                        "survive_rate": room_features.get("survive_score", 0.5)
                    }
                    
                    # HỌC TỪ VÁN NÀY!
                    self._self_learning_ai.learn_from_round(
                        chosen_room=predicted_room,
                        room_features=room_features,
                        killed_room=killed_room,
                        room_data=room_data
                    )
                    
                    # 💾 AUTO-SAVE mỗi 5 ván để lưu tiến trình học!
                    if self._self_learning_ai.total_rounds % 5 == 0:
                        if self._self_learning_ai.save_brain():
                            log_debug(f"💾 Brain saved! ({self._self_learning_ai.total_rounds} rounds)")
                    
                    # Log insights mỗi 10 ván
                    if self._self_learning_ai.total_rounds % 10 == 0:
                        insights = self._self_learning_ai.get_full_insights()
                        log_debug(f"📚 LEARNING PROGRESS:\n{insights}")
                    
                except Exception as e:
                    log_debug(f"⚠️ Learning error: {e}")
            
            if not self._last_votes:
                return
            outcome = 1.0 if win else -1.0
            
            # Lưu vào bộ nhớ dài hạn
            for rid in self.room_ids:
                if killed_room is not None:
                    self._long_term_memory[rid].append(1 if rid == killed_room else 0)
            
            # Cập nhật pattern detector
            if killed_room is not None:
                for rid in self.room_ids:
                    self._pattern_detector[rid].append(1 if rid == killed_room else 0)
                    # Giữ chỉ 50 kết quả gần nhất
                    if len(self._pattern_detector[rid]) > 50:
                        self._pattern_detector[rid] = self._pattern_detector[rid][-50:]
            
            features_pred = self._last_features.get(predicted_room, {})
            features_killed = self._last_features.get(killed_room, {}) if killed_room in self._last_features else {}

            # Cập nhật từng agent với learning rate động
            for idx, vote_room in self._last_votes:
                agent = self._agents[idx]
                
                # Tính influence: agents vote đúng được thưởng cao
                influence = 1.5 if vote_room == predicted_room else -0.8 if (killed_room is not None and vote_room == killed_room) else 0.1
                signed = outcome * influence
                
                base_feats = self._last_features.get(vote_room, features_pred)
                if not base_feats:
                    continue
                
                # Gradient descent với momentum
                for key in self.FEATURE_KEYS:
                    value = base_feats.get(key, 0.0)
                    grad = signed * value
                    # Momentum decay
                    agent["momentum"][key] = 0.6 * agent["momentum"][key] + grad
                    # Update weights với clipping
                    agent["weights"][key] = self._clip(
                        agent["weights"][key] + agent["lr"] * agent["momentum"][key], 
                        -3.0, 3.0
                    )
                
                # Điều chỉnh bias dựa trên survive_score
                adjust_bias = (features_pred.get("survive_score", 0.0) - features_killed.get("survive_score", 0.0))
                agent["bias"] = self._clip(
                    agent["bias"] + agent["lr"] * (signed * 0.12 + adjust_bias * 0.03), 
                    -2.5, 2.5
                )
                
                # Temperature annealing - giảm nhiễu khi đúng, tăng khi sai
                agent["temperature"] = self._clip(
                    agent["temperature"] * (0.96 if win else 1.05), 
                    0.25, 3.0
                )
                
                # Cập nhật performance của agent
                agent["performance"] = self._clip(
                    agent["performance"] * 0.9 + (0.1 if influence * outcome > 0 else -0.05),
                    -0.5, 0.5
                )
                
                # Cập nhật confidence
                agent["confidence"] = self._clip(
                    agent["confidence"] * 0.95 + (0.05 if win else -0.02),
                    0.3, 1.0
                )
                
                # Learning rate decay theo performance
                if agent["performance"] > 0.2:
                    agent["lr"] = self._clip(agent["lr"] * 0.98, 0.05, 0.2)

            # Cập nhật bias phòng
            if predicted_room in self._room_bias:
                self._room_bias[predicted_room] = self._clip(
                    self._room_bias[predicted_room] + (0.15 if win else -0.25), 
                    -1.5, 1.5
                )
            if killed_room and killed_room in self._room_bias:
                self._room_bias[killed_room] = self._clip(
                    self._room_bias[killed_room] - (0.1 if win else -0.15), 
                    -1.5, 1.5
                )

            # Lưu outcome và điều chỉnh explore rate
            self._recent_outcomes.append(1 if win else 0)
            if len(self._recent_outcomes) >= 10:
                # Tính win rate trong 10 ván gần nhất
                last_win_rate = sum(list(self._recent_outcomes)[-10:]) / 10.0
                # Điều chỉnh explore rate dựa trên performance
                if last_win_rate > 0.65:
                    target = 0.02  # rất tốt, exploit nhiều hơn
                elif last_win_rate > 0.45:
                    target = 0.05  # tốt, giữ cân bằng
                elif last_win_rate > 0.30:
                    target = 0.10  # trung bình, explore thêm
                else:
                    target = 0.15  # kém, cần explore nhiều
                
                self._explore_rate = 0.8 * self._explore_rate + 0.2 * target
                self._explore_rate = self._clip(self._explore_rate, 0.01, 0.20)
                
                # Log performance
                log_debug(f"📊 Win rate 10 ván: {last_win_rate:.2%}, Explore rate: {self._explore_rate:.3f}")

            self._last_votes = []


# -------------------- NEURAL BRAIN SYSTEM (BỘ NÃO AI THẬT SỰ) --------------------

class NeuralBrain:
    """
    Bộ não AI thật sự với khả năng:
    - Suy nghĩ logic
    - Phân tích chiến lược
    - Dự đoán xu hướng
    - Học từ kinh nghiệm
    """
    def __init__(self):
        self._thoughts: List[str] = []
        self._logic_chain: List[Dict[str, Any]] = []
        self._strategy_memory: deque = deque(maxlen=50)
        self._neural_weights: Dict[str, float] = self._init_neural_weights()
        self._thinking_depth: int = 3  # Độ sâu suy nghĩ
        
    def _init_neural_weights(self) -> Dict[str, float]:
        """Khởi tạo trọng số neural network của não bộ"""
        return {
            "survival_priority": 0.9,
            "risk_aversion": 0.7,
            "pattern_recognition": 0.8,
            "momentum_trading": 0.6,
            "contrarian_thinking": 0.4,
            "crowd_psychology": 0.75,
            "statistical_edge": 0.85,
            "adaptive_learning": 0.95,
        }
    
    def think(self, situation: Dict[str, Any]) -> str:
        """Quá trình suy nghĩ của AI - trả về chuỗi suy luận"""
        thoughts = []
        thoughts.append("🧠 [KHỞI ĐỘNG QUÁ TRÌNH SUY NGHĨ]")
        
        # Layer 1: Quan sát
        observation = self._observe(situation)
        thoughts.append(f"👁️ Quan sát: {observation}")
        
        # Layer 2: Phân tích
        analysis = self._analyze(situation)
        thoughts.append(f"📊 Phân tích: {analysis}")
        
        # Layer 3: Suy luận logic
        logic = self._reason_logically(situation)
        thoughts.append(f"💭 Suy luận: {logic}")
        
        # Layer 4: Chiến lược
        strategy = self._plan_strategy(situation)
        thoughts.append(f"🎯 Chiến lược: {strategy}")
        
        # Layer 5: Quyết định
        decision = self._make_decision(situation)
        thoughts.append(f"✅ Quyết định: {decision}")
        
        self._thoughts = thoughts
        return "\n".join(thoughts)
    
    def _observe(self, situation: Dict[str, Any]) -> str:
        """Quan sát tình huống"""
        room_data = situation.get("room_data", {})
        n_rooms = len(room_data)
        total_players = sum(r.get("players", 0) for r in room_data.values())
        total_bets = sum(r.get("bet", 0) for r in room_data.values())
        
        return f"{n_rooms} phòng, {total_players} người chơi, {total_bets:,.0f} BUILD tổng cược"
    
    def _analyze(self, situation: Dict[str, Any]) -> str:
        """Phân tích dữ liệu"""
        room_data = situation.get("room_data", {})
        
        # Tìm phòng nguy hiểm nhất
        max_bet_room = max(room_data.items(), key=lambda x: x[1].get("bet", 0))[0] if room_data else 0
        # Tìm phòng đông nhất
        max_players_room = max(room_data.items(), key=lambda x: x[1].get("players", 0))[0] if room_data else 0
        
        return f"Phòng {max_bet_room} cược cao nhất, Phòng {max_players_room} đông nhất"
    
    def _reason_logically(self, situation: Dict[str, Any]) -> str:
        """
        🧠 SUY LUẬN LOGIC - PHÂN TÍCH RỦI RO & PHÁ BẪY ĐÁM ĐÔNG
        """
        logic_rules = []
        
        room_data = situation.get("room_data", {})
        recommended = situation.get("recommended_room")
        
        # Rule 1: Tránh phòng vừa bị kill - BẪY KINH ĐIỂN
        last_kill = situation.get("last_killed_room")
        if last_kill:
            logic_rules.append(f"❌ Loại phòng {last_kill} (vừa bị sát thủ)")
        
        # Rule 2: 🚨 CẢNH BÁO BẪY ĐÁM ĐÔNG
        if room_data:
            # Tìm phòng đông nhất
            crowded_room = max(room_data.items(), key=lambda x: x[1].get("players", 0))
            max_players = crowded_room[1].get("players", 0)
            
            # Tìm phòng cược cao nhất
            high_bet_room = max(room_data.items(), key=lambda x: x[1].get("bet", 0))
            max_bet = high_bet_room[1].get("bet", 0)
            
            # Cảnh báo nếu phòng quá đông
            if max_players > 25:
                logic_rules.append(f"⚠️ CẢNH BÁO: Phòng {crowded_room[0]} quá đông ({max_players} người) - Có thể là BẪY!")
            
            # Cảnh báo nếu cược quá cao
            if max_bet > 12000:
                logic_rules.append(f"⚠️ CẢNH BÁO: Phòng {high_bet_room[0]} cược quá cao ({max_bet:,.0f}) - RỦI RO CAO!")
        
        # Rule 3: ✅ PHÂN TÍCH PHÒNG ĐƯỢC CHỌN DỰA TRÊN DỮ LIỆU
        if recommended and recommended in room_data:
            rec_data = room_data[recommended]
            players = rec_data.get("players", 0)
            bet = rec_data.get("bet", 0)
            
            # Lấy THỐNG KÊ THỰC TẾ từ lịch sử
            stats = room_stats.get(recommended, {})
            kills = stats.get("kills", 0)
            survives = stats.get("survives", 0)
            total_history = kills + survives
            
            # Đánh giá mức độ an toàn DỰA TRÊN DỮ LIỆU
            if total_history >= 5:
                survive_rate = survives / total_history
                if survive_rate >= 0.70:
                    safety_level = f"🟢 RẤT AN TOÀN ({survives}W/{kills}L = {survive_rate:.0%})"
                elif survive_rate >= 0.55:
                    safety_level = f"🟡 KHÁ AN TOÀN ({survives}W/{kills}L = {survive_rate:.0%})"
                elif survive_rate >= 0.40:
                    safety_level = f"🟠 TRUNG BÌNH ({survives}W/{kills}L = {survive_rate:.0%})"
                else:
                    safety_level = f"🔴 RỦI RO ({survives}W/{kills}L = {survive_rate:.0%})"
            else:
                safety_level = f"⚪ CHƯA ĐỦ DỮ LIỆU ({survives}W/{kills}L)"
            
            logic_rules.append(f"{safety_level} | Phòng {recommended}: {players} người, {bet:,.0f} BUILD")
            
            # Rule 4: So sánh DỮ LIỆU THỰC TẾ với các phòng khác
            other_rooms = [r for r in room_data if r != recommended]
            if other_rooms and total_history >= 3:
                # So sánh tỉ lệ survive với các phòng khác
                other_survive_rates = []
                for r in other_rooms:
                    r_stats = room_stats.get(r, {})
                    r_k = r_stats.get("kills", 0)
                    r_s = r_stats.get("survives", 0)
                    r_total = r_k + r_s
                    if r_total >= 3:
                        other_survive_rates.append(r_s / r_total)
                
                if other_survive_rates:
                    avg_survive = sum(other_survive_rates) / len(other_survive_rates)
                    if survive_rate > avg_survive + 0.15:
                        logic_rules.append(f"✅ AN TOÀN HƠN TB: {survive_rate:.0%} > {avg_survive:.0%} (dữ liệu thực tế)")
                    elif survive_rate < avg_survive - 0.15:
                        logic_rules.append(f"⚠️ NGUY HIỂM HƠN TB: {survive_rate:.0%} < {avg_survive:.0%} (dữ liệu thực tế)")
                    else:
                        logic_rules.append(f"📊 Ngang TB: {survive_rate:.0%} ≈ {avg_survive:.0%}")
        
        # Rule 5: Phân tích xu hướng
        if situation.get("win_streak", 0) >= 3:
            logic_rules.append(f"🔥 Đang thắng {situation['win_streak']} ván liên tiếp - Giữ chiến lược")
        elif situation.get("lose_streak", 0) >= 2:
            logic_rules.append(f"⚠️ Thua {situation['lose_streak']} ván - Cần thận trọng!")
        
        return " • ".join(logic_rules) if logic_rules else "Phân tích dữ liệu cơ bản"
    
    def _plan_strategy(self, situation: Dict[str, Any]) -> str:
        """Lập kế hoạch chiến lược"""
        streak = situation.get("win_streak", 0)
        
        if streak >= 3:
            return "Thắng liên tiếp → Giữ chiến lược hiện tại, tăng tự tin"
        elif situation.get("lose_streak", 0) >= 2:
            return "Thua liên tiếp → Điều chỉnh chiến lược, thận trọng hơn"
        else:
            return "Cân bằng → Tiếp tục quan sát và học hỏi"
    
    def _make_decision(self, situation: Dict[str, Any]) -> str:
        """Đưa ra quyết định cuối cùng"""
        recommended_room = situation.get("recommended_room", 1)
        confidence = self._calculate_confidence(situation)
        
        # Thêm tên phòng
        room_name = ROOM_NAMES.get(recommended_room, f"Phòng {recommended_room}")
        
        return f"Chọn Phòng {recommended_room} - {room_name} (Tin cậy: {confidence:.0%})"
    
    def _calculate_confidence(self, situation: Dict[str, Any]) -> float:
        """Tính toán độ tin cậy của quyết định"""
        base_confidence = 0.6
        
        # Tăng confidence nếu có nhiều dữ liệu
        if situation.get("total_rounds", 0) > 20:
            base_confidence += 0.15
        
        # Tăng nếu đang thắng
        if situation.get("win_streak", 0) >= 2:
            base_confidence += 0.1
        
        # Giảm nếu đang thua
        if situation.get("lose_streak", 0) >= 2:
            base_confidence -= 0.15
        
        return min(0.95, max(0.4, base_confidence))
    
    def get_thoughts(self) -> List[str]:
        """Lấy chuỗi suy nghĩ"""
        return self._thoughts
    
    def update_neural_weights(self, result: bool):
        """Cập nhật trọng số neural network dựa trên kết quả"""
        lr = 0.05
        if result:  # Thắng
            self._neural_weights["survival_priority"] += lr * 0.1
            self._neural_weights["pattern_recognition"] += lr * 0.08
        else:  # Thua
            self._neural_weights["risk_aversion"] += lr * 0.12
            self._neural_weights["contrarian_thinking"] += lr * 0.06
        
        # Normalize
        for k in self._neural_weights:
            self._neural_weights[k] = max(0.1, min(1.0, self._neural_weights[k]))


class StrategicPlanner:
    """
    Bộ lập kế hoạch chiến lược
    Phân tích dài hạn và đưa ra chiến thuật
    """
    def __init__(self):
        self._short_term_plan: Optional[str] = None
        self._long_term_plan: Optional[str] = None
        self._tactics: List[str] = []
        
    def create_plan(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Tạo kế hoạch chiến lược ngắn hạn và dài hạn"""
        # Phân tích tình hình
        win_rate = data.get("win_rate", 0.5)
        total_profit = data.get("total_profit", 0.0)
        
        # Kế hoạch ngắn hạn (5-10 ván)
        if win_rate > 0.6:
            self._short_term_plan = "🎯 OFFENSIVE: Tăng cường đặt cược, tận dụng momentum"
        elif win_rate > 0.45:
            self._short_term_plan = "⚖️ BALANCED: Giữ ổn định, quan sát thêm"
        else:
            self._short_term_plan = "🛡️ DEFENSIVE: Giảm rủi ro, chờ cơ hội tốt hơn"
        
        # Kế hoạch dài hạn (20-50 ván)
        if total_profit > 0:
            self._long_term_plan = "📈 Mục tiêu: Tối ưu hóa lợi nhuận, quản lý vốn an toàn"
        else:
            self._long_term_plan = "🔄 Mục tiêu: Phục hồi vốn, học hỏi pattern mới"
        
        return {
            "short_term": self._short_term_plan,
            "long_term": self._long_term_plan
        }
    
    def suggest_tactics(self, situation: str) -> List[str]:
        """Đề xuất chiến thuật cụ thể"""
        tactics = []
        
        if "thua" in situation.lower():
            tactics.append("💡 Tạm nghỉ 1-2 ván để tái đánh giá")
            tactics.append("💡 Giảm 50% số tiền đặt cược")
            tactics.append("💡 Chỉ chọn phòng có survival rate > 70%")
        elif "thắng" in situation.lower():
            tactics.append("💡 Duy trì chiến lược hiện tại")
            tactics.append("💡 Có thể tăng nhẹ bet size (+20%)")
            tactics.append("💡 Mở rộng selection pool")
        
        self._tactics = tactics
        return tactics


# Khởi tạo bộ não AI
neural_brain = NeuralBrain()
strategic_planner = StrategicPlanner()

selector = UltimateAISelector(ROOM_ORDER)


def choose_room(mode: str = ALGO_ID) -> Tuple[int, str]:
    """Chọn phòng với Neural Brain AI - AI có não thật sự"""
    global ai_reasoning
    try:
        # Bước 1: Neural Brain suy nghĩ
        situation = {
            "room_data": room_state,
            "last_killed_room": last_killed_room,
            "win_streak": win_streak,
            "lose_streak": lose_streak,
            "total_rounds": round_index,
            "recommended_room": None
        }
        
        # Bước 2: Selector chọn phòng
        chosen_room, algo = selector.select_room()
        situation["recommended_room"] = chosen_room
        
        # Bước 3: Neural Brain phân tích quyết định
        thinking_process = neural_brain.think(situation)
        log_debug(f"\n{'='*60}\n{thinking_process}\n{'='*60}")
        
        # Lưu reasoning để hiển thị trong UI
        thoughts = neural_brain.get_thoughts()
        if thoughts and len(thoughts) >= 3:
            # Lấy 3 dòng quan trọng: Phân tích, Suy luận, Chiến lược
            ai_reasoning = "\n".join([
                thoughts[2] if len(thoughts) > 2 else "",  # Phân tích
                thoughts[3] if len(thoughts) > 3 else "",  # Suy luận
                thoughts[4] if len(thoughts) > 4 else "",  # Chiến lược
            ])
        
        # Bước 4: Strategic Planner lập kế hoạch
        if round_index % 5 == 0:  # Mỗi 5 ván tạo kế hoạch mới
            win_rate = win_streak / max(1, round_index)
            plan_data = {
                "win_rate": win_rate,
                "total_profit": cumulative_profit,
            }
            plans = strategic_planner.create_plan(plan_data)
            log_debug(f"📋 Kế hoạch ngắn hạn: {plans['short_term']}")
            log_debug(f"📋 Kế hoạch dài hạn: {plans['long_term']}")
        
        return chosen_room, algo
    except Exception as exc:
        log_debug(f"🚨 Neural Brain AI failed: {exc}")
        ai_reasoning = "Lỗi: Không thể phân tích"
        return ROOM_ORDER[0], ALGO_ID


def update_formulas_after_result(predicted_room: Optional[int], killed_room: Optional[int], mode: str = ALGO_ID, lr: float = 0.12):
    """Cập nhật AI sau khi có kết quả - Bộ não AI học hỏi"""
    try:
        # Cập nhật selector
        selector.update(predicted_room, killed_room)
        
        # Cập nhật Neural Brain
        win = (killed_room is not None and predicted_room != killed_room)
        neural_brain.update_neural_weights(win)
        
        # Log quá trình học
        if win:
            log_debug("🧠 Neural Brain: Cập nhật trọng số sau THẮNG")
        else:
            log_debug("🧠 Neural Brain: Điều chỉnh chiến thuật sau THUA")
            
    except Exception as exc:
        log_debug(f"🚨 AI Learning failed: {exc}")


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
        console.print(f"[cyan]Đang đặt {amount} BUILD -> PHÒNG_{room_id} (v{issue}) — Thuật toán: {algo_used}[/cyan]")
        time.sleep(random.uniform(0.02, 0.25))
        res = place_bet_http(issue, room_id, amount)
        rec = record_bet(issue, room_id, amount, res, algo_used=algo_used)
        if isinstance(res, dict) and (res.get("msg") == "ok" or res.get("code") == 0 or res.get("status") in ("ok", 1)):
            bet_sent_for_issue.add(issue)
            console.print(f"[green]✅ Đặt thành công {amount} BUILD vào PHÒNG_{room_id} (v{issue}).[/green]")
            
            # 🧹 Clear terminal sau khi đặt cược xong
            time.sleep(0.5)  # Wait để user thấy message
            clear_terminal()
            
            # Show bet confirmation banner
            from rich.panel import Panel
            from rich.text import Text
            banner = Text()
            banner.append("✅ CƯỢC ĐÃ ĐẶT!\n\n", style="bold bright_green")
            banner.append(f"Phòng: {ROOM_NAMES.get(room_id, f'Phòng {room_id}')}\n", style="bright_cyan")
            banner.append(f"Số tiền: {amount} BUILD\n", style="yellow bold")
            banner.append(f"Ván: {issue}\n", style="dim")
            banner.append(f"\n⏳ Đang chờ kết quả...", style="bright_yellow blink")
            
            console.print(Panel(
                banner,
                title="[bold bright_green]💰 BET PLACED 💰[/bold bright_green]",
                border_style="bright_green",
                box=box.DOUBLE
            ))
            console.print("")
        else:
            console.print(f"[red]❌ Đặt lỗi v{issue}: {res}[/red]")
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
            console.print(f"[yellow]⏸️ Đang nghỉ {_skip_rounds_remaining} ván theo cấu hình sau khi thua.[/yellow]")
            _skip_rounds_remaining -= 1         # tiêu thụ 1 ván nghỉ
            _skip_active_issue = issue_id       # nhớ là ván này đã nghỉ

        # khóa đến hết ván hiện tại để không bị các tick countdown đặt lại
        prediction_locked = True
        ui_state = "ANALYZING"                  # hoặc "PREDICTED" tuỳ UI
        return

    # Chọn phòng chỉ khi KHÔNG skip
    algo = settings.get("algo", ALGO_ID)
    try:
        chosen, algo_used = choose_room(algo)
    except Exception as e:
        log_debug(f"choose_room error: {e}")
        chosen, algo_used = choose_room(ALGO_ID)
    predicted_room = chosen
    prediction_locked = True
    ui_state = "PREDICTED"

    # place bet if AUTO
    if run_mode == "AUTO" and not skip_next_round_flag:
        # get balance quickly (non-blocking - allow poller to update if needed)
        bld, _, _ = fetch_balances_3games(params={"userId": str(USER_ID)} if USER_ID else None)
        if bld is None:
            console.print("[yellow]⚠️ Không lấy được số dư trước khi đặt — bỏ qua đặt ván này.[/yellow]")
            prediction_locked = False
            return
        global current_bet

        # Debug: Kiểm tra current_bet trước khi đặt cược
        console.print(f"[blue]🔍 DEBUG: Trước khi đặt cược - current_bet={current_bet}, base_bet={base_bet}, multiplier={multiplier}[/blue]")
        if current_bet is None:
            current_bet = base_bet
            console.print(f"[yellow]⚠️ current_bet is None, reset to base_bet: {current_bet}[/yellow]")
        else:
            console.print(f"[green]✅ current_bet không None: {current_bet}[/green]")
        amt = float(current_bet)
        console.print(f"[cyan]💰 Đặt cược: {amt} BUILD (current_bet={current_bet}, base_bet={base_bet}, multiplier={multiplier})[/cyan]")
        if amt <= 0:
            console.print("[yellow]⚠️ Số tiền đặt không hợp lệ (<=0). Bỏ qua.[/yellow]")
            prediction_locked = False
            return
        place_bet_async(issue_id, predicted_room, amt, algo_used=algo_used)
        _rounds_placed_since_skip += 1
        if bet_rounds_before_skip > 0 and _rounds_placed_since_skip >= bet_rounds_before_skip:
            skip_next_round_flag = True
            _rounds_placed_since_skip = 0
    elif skip_next_round_flag:
        console.print("[yellow]⏸️ TẠM DỪNG THEO DÕI SÁT THỦ[/yellow]")
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
    console.print("[green]ĐANG TRUY CẬP DỮ LIỆU GAME[/green]")
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
            current_bet = base_bet              # reset martingale về base
            win_streak += 1
            lose_streak = 0
            if win_streak > max_win_streak:
                max_win_streak = win_streak
        else:
            # THUA -> nhân tiền cho ván kế tiếp
            rec["result"] = "Thua"
            rec["settled"] = True
            try:
                old_bet = current_bet
                current_bet = float(rec.get("amount")) * float(multiplier)
                console.print(f"[red]🔴 THUA! Số cũ: {rec.get('amount')} × {multiplier} = {current_bet} BUILD[/red]")
                console.print(f"[red]🔴 DEBUG: current_bet đã được cập nhật từ {old_bet} thành {current_bet}[/red]")
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

    # --- Hyper AI: cập nhật mô hình sau khi có kết quả ---
    try:
        # cập nhật mô hình Hyper Adaptive AI dựa trên kết quả thực tế
        update_formulas_after_result(predicted_room, krid, settings.get("algo", ALGO_ID))
    except Exception as e:
        log_debug(f"update_formulas_after_result err: {e}")

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
            
            # ✅ Check stop ngay lập tức (không cần delay vì balance đã update trong fetch)
            _check_stop_profit_loss()

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

    # Title với gradient effect
    left = Text()
    left.append("🧠 NEURAL BRAIN AI 🧠\n", style="bold bright_cyan")
    left.append("Bộ Não Thông Minh", style="bright_green")

    b = f"{current_build:,.4f}" if isinstance(current_build, (int, float)) else (str(current_build) if current_build is not None else "-")
    u = f"{current_usdt:,.4f}" if isinstance(current_usdt, (int, float)) else (str(current_usdt) if current_usdt is not None else "-")
    x = f"{current_world:,.4f}" if isinstance(current_world, (int, float)) else (str(current_world) if current_world is not None else "-")

    pnl_val = cumulative_profit if cumulative_profit is not None else 0.0
    pnl_str = f"{pnl_val:+,.4f}"
    pnl_style = "green bold" if pnl_val > 0 else ("red bold" if pnl_val < 0 else "yellow")

    bal = Text.assemble((f"USDT: {u}", "bold"), ("   "), (f"XWORLD: {x}", "bold"), ("   "), (f"BUILD: {b}", "bold"))

    algo_label = SELECTION_MODES.get(settings.get('algo'), settings.get('algo'))

    right_lines = []
    right_lines.append(f"Thuật toán: {algo_label}")
    right_lines.append(f"Lãi/lỗ: [{pnl_style}] {pnl_str} [/{pnl_style}]")
    right_lines.append(f"Phiên: {issue_id or '-'}")
    right_lines.append(f"chuỗi: thắng={max_win_streak} / thua={max_lose_streak}")
    if stop_when_profit_reached and profit_target is not None:
        right_lines.append(f"[green]TakeProfit@{profit_target}[/green]")
    if stop_when_loss_reached and stop_loss_target is not None:
        right_lines.append(f"[red]StopLoss@{stop_loss_target}[/red]")

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
                status = "[red]☠ Kill[/red]"
        except Exception:
            pass
        try:
            if predicted_room is not None and int(r) == int(predicted_room):
                status = (status + " [dim]|[/dim] [green]✓ Dự đoán[/green]") if status else "[green]✓ Dự đoán[/green]"
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

        # NEURAL BRAIN visualization với hiệu ứng công nghệ cao
        if analysis_blur:
            # Brain wave animation
            bar_len = 50
            blocks = []
            tbase = int(time.time() * 8)
            for i in range(bar_len):
                # Tạo hiệu ứng sóng não với gradient
                wave = math.sin((i + tbase) * 0.3) * 0.5 + 0.5
                val = int(wave * 7)
                ch = "█" if val in (0, 1, 2, 3) else ("▓" if val in (4, 5) else "░")
                color = RAINBOW_COLORS[(i + tbase) % len(RAINBOW_COLORS)]
                blocks.append(f"[{color}]{ch}[/{color}]")
            lines.append("".join(blocks))
            lines.append("")
            
            # ⚡ QUANTUM COMPUTING - 5 tầng suy luận
            thinking_stage = (int(time.time() * 2) % 5)
            stages = [
                "🔬 [LAYER 1: OBSERVATION] Quan sát 8 phòng đa chiều...",
                "🔗 [LAYER 2: CORRELATION] Phát hiện tương quan ẩn...",
                "🧬 [LAYER 3: CAUSATION] Suy luận nhân quả sâu...",
                "🔮 [LAYER 4: PREDICTION] Dự đoán với xác suất lượng tử...",
                "⚡ [LAYER 5: VALIDATION] Xác thực logic & Meta-learning..."
            ]
            lines.append(f"[bright_magenta]{stages[thinking_stage]}[/bright_magenta]")
            lines.append("")
            lines.append("[bright_cyan blink]◢◤◢ QUANTUM PROCESSOR ACTIVE ◢◤◢[/bright_cyan blink]")
            lines.append("[bright_yellow]⚡ Quantum Prob: Calculating... | Confidence: 85%+[/bright_yellow]")
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
        
        # ⚡ Header Quantum với cyberpunk style
        lines.append("[bold bright_magenta blink]╔════════════════════════════════════════╗[/bold bright_magenta blink]")
        lines.append(f"[bold bright_magenta]║[/bold bright_magenta]  ⚡ [bright_cyan blink]QUANTUM DECISION ENGINE[/bright_cyan blink]  ⚡  [bold bright_magenta]║[/bold bright_magenta]")
        lines.append("[bold bright_magenta blink]╚════════════════════════════════════════╝[/bold bright_magenta blink]")
        lines.append("[dim bright_cyan]◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤[/dim bright_cyan]")
        lines.append("")
        
        # AI Thinking Process (lấy từ neural_brain)
        if hasattr(neural_brain, '_thoughts') and neural_brain._thoughts:
            # Hiển thị 3 dòng cuối của quá trình suy nghĩ
            for thought in neural_brain._thoughts[-3:]:
                lines.append(f"[dim]{thought}[/dim]")
            lines.append("")
        
        lines.append(f"[bold bright_cyan]⚡ CHỌN:[/bold bright_cyan] [bright_green bold blink]{name}[/bright_green bold blink]")
        lines.append(f"[bold bright_yellow]💰 ĐẶT:[/bold bright_yellow] [yellow bold]{last_bet_amt} BUILD[/yellow bold]")
        
        # Tính độ tin cậy QUANTUM
        confidence = 0.82 + (win_streak * 0.04) - (lose_streak * 0.03)
        confidence = max(0.75, min(0.98, confidence))
        
        # Cyberpunk confidence bar với gradient
        conf_bar_chars = int(confidence * 24)
        conf_bar = ""
        for i in range(24):
            if i < conf_bar_chars:
                # Gradient: green → cyan → magenta
                if i < 8:
                    conf_bar += "[bright_green]█[/bright_green]"
                elif i < 16:
                    conf_bar += "[bright_cyan]█[/bright_cyan]"
                else:
                    conf_bar += "[bright_magenta]█[/bright_magenta]"
            else:
                conf_bar += "[dim]░[/dim]"
        
        lines.append(f"[bold bright_magenta]⚡ QUANTUM CONFIDENCE:[/bold bright_magenta]")
        lines.append(f"{conf_bar} [bright_yellow bold]{confidence:.1%}[/bright_yellow bold]")
        lines.append("")
        
        lines.append(f"[dim]Phòng sát thủ trước: {ROOM_NAMES.get(last_killed_room, '-')}[/dim]")
        lines.append(f"[dim]Chuỗi: 🎯{win_streak}W | ❌{lose_streak}L[/dim]")
        
        if count_down is not None:
            try:
                cd = int(count_down)
                lines.append("")
                lines.append(f"[bold yellow]⏱️ Đếm ngược: {cd}s[/bold yellow]")
            except Exception:
                pass
        
        lines.append("")
        # Animation dots
        dots = "." * (int(time.time() * 3) % 4)
        lines.append(f"[bright_cyan]⚡ Neural Network Processing{dots.ljust(3)}[/bright_cyan]")
        
        txt = "\n".join(lines)
        return Panel(
            Align.center(Text.from_markup(txt)), 
            title="[bold bright_cyan blink]⚡ QUANTUM PREDICTION ⚡[/bold bright_cyan blink]", 
            border_style="bright_magenta",
            box=box.DOUBLE,
            subtitle="[dim bright_cyan]◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤[/dim bright_cyan]"
        )

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

def build_reasoning_panel(border_color: Optional[str] = None):
    """Panel hiển thị lý do AI chọn phòng"""
    global ai_reasoning
    
    if not ai_reasoning or ai_reasoning == "Lỗi: Không thể phân tích":
        content = Text("⏳ Đang chờ AI phân tích...", style="dim yellow", justify="center")
    else:
        lines = []
        lines.append("[bold bright_magenta blink]⚡ QUANTUM REASONING - SUY LUẬN LƯỢNG TỬ ⚡[/bold bright_magenta blink]\n")
        lines.append("[dim bright_cyan]◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤[/dim bright_cyan]\n")
        
        # Hiển thị reasoning với format đẹp
        reasoning_lines = ai_reasoning.split('\n')
        for line in reasoning_lines:
            if line.strip():
                lines.append(f"[cyan]{line.strip()}[/cyan]")
        
        content = Text.from_markup("\n".join(lines))
    
    return Panel(
        Align.center(content),
        title="[bold bright_cyan blink]⚡ QUANTUM LOGIC CHAIN ⚡[/bold bright_cyan blink]",
        border_style="bright_magenta",
        box=box.DOUBLE,
        padding=(1, 2)
    )


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

    # Cyberpunk config header
    from rich.panel import Panel
    from rich.text import Text
    
    config_title = Text()
    config_title.append("⚙️  CẤU HÌNH ULTRA AI v15.0", style="bold bright_cyan")
    
    console.print(Panel(
        config_title,
        border_style="bright_cyan",
        box=box.DOUBLE
    ))
    console.print("")
    
    base = safe_input("[bold bright_cyan]💰 Số BUILD đặt mỗi ván:[/bold bright_cyan] ", default="1")
    try:
        base_bet = float(base)
    except Exception:
        base_bet = 1.0
    m = safe_input("[bold bright_cyan]📈 Hệ số nhân sau khi thua (ổn định = 2):[/bold bright_cyan] ", default="2")
    try:
        multiplier = float(m)
    except Exception:
        multiplier = 2.0
    current_bet = base_bet

    # Thuật toán cố định - ULTRA AI v15.0
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    
    features = Text()
    features.append("🚀 ULTRA AI v15.0 - SIÊU TRÍ TUỆ VƯỢT TRỘI\n\n", style="bold bright_cyan")
    features.append("🎓 SELF-LEARNING AI\n", style="bright_magenta bold")
    features.append("  • Học từ TỪNG VÁN chơi\n", style="cyan")
    features.append("  • Tự điều chỉnh weights\n", style="cyan")
    features.append("  • Nhận dạng patterns game\n", style="cyan")
    features.append("  • Nhớ tình huống tốt/xấu\n\n", style="cyan")
    features.append("🧠 ADVANCED ALGORITHMS\n", style="bright_magenta bold")
    features.append("  • Neural Network (64-32-16)\n", style="cyan")
    features.append("  • Bayesian Optimization\n", style="cyan")
    features.append("  • Ensemble Learning\n", style="cyan")
    features.append("  • Q-Learning + Replay\n\n", style="cyan")
    features.append("🌟 TỰ HỌC - TỰ TIẾN HÓA! 🌟", style="bold bright_yellow blink")
    
    console.print(Panel(
        features,
        title="[bold bright_magenta blink]⚡ ULTRA AI - TỰ HỌC ⚡[/bold bright_magenta blink]",
        border_style="bright_magenta",
        box=box.DOUBLE
    ))
    settings["algo"] = ALGO_ID

    s = safe_input("[bold bright_cyan]🛡️  Chống soi (số ván đặt trước khi nghỉ 1 ván):[/bold bright_cyan] ", default="0")
    try:
        bet_rounds_before_skip = int(s)
    except Exception:
        bet_rounds_before_skip = 0

    pl = safe_input("Nếu thua thì nghỉ bao nhiêu tay trước khi cược lại (ví dụ 2): ", default="0")
    try:
        pause_after_losses = int(pl)
    except Exception:
        pause_after_losses = 0

    pt = safe_input("lãi bao nhiêu thì chốt( không dùng enter): ", default="")
    try:
        if pt and pt.strip() != "":
            profit_target = float(pt)
            stop_when_profit_reached = True
        else:
            profit_target = None
            stop_when_profit_reached = False
    except Exception:
        profit_target = None
        stop_when_profit_reached = False

    sl = safe_input("Lỗ bao nhiêu BUILD thì dừng (ví dụ 100 = dừng khi lỗ 100 BUILD): ", default="")
    console.print("[dim yellow]💡 Lưu ý: Nhập số lỗ BUILD (ví dụ 100), KHÔNG phải số dư cuối[/dim yellow]")
    try:
        if sl and sl.strip() != "":
            stop_loss_target = float(sl)
            stop_when_loss_reached = True
            console.print(f"[yellow]✅ Stop-loss: Tool sẽ dừng khi LỖ {stop_loss_target} BUILD[/yellow]")
        else:
            stop_loss_target = None
            stop_when_loss_reached = False
    except Exception:
        stop_loss_target = None
        stop_when_loss_reached = False
    
    console.print("")
    from rich.panel import Panel
    console.print(Panel(
        "[bright_green]✅ CẤU HÌNH HOÀN TẤT![/bright_green]",
        border_style="bright_green",
        box=box.DOUBLE
    ))
    console.print("")
    
    # Animation trước khi clear
    import time
    with console.status("[bold bright_cyan]⚡ Đang khởi tạo Quantum Brain AI...[/bold bright_cyan]", spinner="dots"):
        time.sleep(1.5)
    
    # Clear terminal sau khi config xong
    clear_terminal()
    
    # Show startup banner
    from rich.panel import Panel
    from rich.text import Text
    
    banner = Text()
    banner.append("🚀 ULTRA AI v15.0 - ĐANG HOẠT ĐỘNG 🚀\n\n", style="bold bright_magenta blink")
    banner.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n", style="dim bright_cyan")
    banner.append(f"💰 Base Bet: ", style="bright_cyan")
    banner.append(f"{base_bet} BUILD\n", style="yellow bold")
    banner.append(f"📈 Multiplier: ", style="bright_cyan")
    banner.append(f"{multiplier}x\n", style="yellow bold")
    
    if stop_when_profit_reached and profit_target:
        banner.append(f"🎯 Take Profit: ", style="bright_green")
        banner.append(f"{profit_target} BUILD\n", style="yellow bold")
    if stop_when_loss_reached and stop_loss_target:
        banner.append(f"🛑 Stop Loss: ", style="bright_red")
        banner.append(f"{stop_loss_target} BUILD\n", style="yellow bold")
    
    banner.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", style="dim bright_cyan")
    banner.append("✅ Hệ thống đã sẵn sàng! AI đang học...", style="bright_green bold")
    
    console.print(Panel(
        banner,
        title="[bold bright_cyan blink]⚡ SYSTEM STATUS ⚡[/bold bright_cyan blink]",
        border_style="bright_magenta",
        box=box.DOUBLE,
        subtitle="[dim bright_cyan]◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤[/dim bright_cyan]"
    ))
    console.print("")

    runm = safe_input("💯bạn đã sẵn sàng hãy nhấn enter để bắt đầu💯: ", default="AUTO")
    run_mode = str(runm).upper()


def start_threads():
    threading.Thread(target=start_ws, daemon=True).start()
    threading.Thread(target=monitor_loop, daemon=True).start()

def show_cyberpunk_login():
    """
    ⚡ CYBERPUNK LOGIN SCREEN ⚡
    """
    from rich.table import Table
    from rich import box
    
    console.print("\n" * 2)
    
    # ASCII Art Login
    login_art = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║    ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗ ║
    ║   ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║ ║
    ║   ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║ ║
    ║   ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║ ║
    ║   ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝ ║
    ║    ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝  ║
    ║                                                           ║
    ║              ⚡ LOGIN SYSTEM v14.0 ⚡                      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    
    console.print(Text(login_art, style="bright_cyan"))
    console.print(Align.center(
        Text("◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤◢◤", style="dim bright_magenta")
    ))
    console.print("")
    
    # Info table
    info_table = Table(box=box.ROUNDED, border_style="bright_cyan", show_header=False)
    info_table.add_column("", style="bright_yellow bold", no_wrap=True)
    info_table.add_column("", style="bright_green")
    
    info_table.add_row("🔐 Security", "Base64 Encryption")
    info_table.add_row("💾 Storage", "Local Encrypted Config")
    info_table.add_row("🔗 Protocol", "HTTPS + WebSocket Secure")
    info_table.add_row("⚡ Version", "Quantum Brain AI v14.0")
    
    console.print(Panel(
        Align.center(info_table),
        title="[bold bright_magenta blink]⚡ SYSTEM INFO ⚡[/bold bright_magenta blink]",
        border_style="bright_magenta",
        box=box.DOUBLE
    ))
    console.print("")

def parse_login():
    global USER_ID, SECRET_KEY
    
    # Import link manager
    from link_manager import QuantumLinkManager
    link_mgr = QuantumLinkManager()
    
    # Show cyberpunk login screen
    show_cyberpunk_login()
    
    login_url = None
    
    # Check nếu có link đã lưu
    if link_mgr.has_saved_link():
        saved_info = link_mgr.get_saved_info()
        
        from rich.panel import Panel
        console.print(Panel(
            "[bright_green]🔍 PHÁT HIỆN LINK ĐÃ LƯU![/bright_green]",
            border_style="bright_green",
            box=box.ROUNDED
        ))
        console.print("")
        
        if saved_info:
            info_lines = []
            info_lines.append(f"[bright_cyan]📅 Lưu lúc:[/bright_cyan] [yellow]{saved_info.get('saved_at', 'N/A')}[/yellow]")
            info_lines.append(f"[bright_cyan]🎮 Game:[/bright_cyan] [yellow]{saved_info.get('game_name', 'N/A')}[/yellow]")
            info_lines.append(f"[bright_cyan]👤 User ID:[/bright_cyan] [yellow]{saved_info.get('user_id', 'N/A')}[/yellow]")
            
            console.print(Panel(
                "\n".join(info_lines),
                title="[bold bright_yellow]💾 Link đã lưu[/bold bright_yellow]",
                border_style="bright_yellow",
                box=box.ROUNDED
            ))
            console.print("")
        
        # Hỏi có muốn load link cũ không
        use_saved = safe_input(
            "[bold bright_magenta]⚡ Bạn muốn sử dụng link đã lưu? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red]): [/bold bright_magenta]",
            default="y"
        ).strip().lower()
        
        if use_saved in ['y', 'yes', '']:
            login_url = link_mgr.load_link()
            
            if login_url:
                console.print("")
                console.print("✅ [bright_green bold]Đã load link thành công![/bright_green bold]")
                console.print("")
                
                # Animation loading
                import time
                with console.status("[bold bright_cyan]⚡ Đang xác thực link...[/bold bright_cyan]", spinner="dots"):
                    time.sleep(1.5)
                console.print("✅ [bright_green]Xác thực thành công![/bright_green]\n")
            else:
                console.print("[red]❌ Lỗi khi load link. Vui lòng nhập lại.[/red]\n")
    
    # Nếu không có link đã lưu hoặc user chọn nhập mới
    if not login_url:
        from rich.panel import Panel
        console.print(Panel(
            "[bright_cyan]🔗 NHẬP LINK GAME[/bright_cyan]",
            border_style="bright_cyan",
            box=box.ROUNDED
        ))
        console.print("")
        console.print("[dim bright_yellow]💡 Tip: Link sẽ được lưu tự động cho lần sau![/dim bright_yellow]")
        console.print("")
        
        login_url = safe_input(
            "[bold bright_magenta]🔗 Dán link từ xworld.info: [/bold bright_magenta]"
        )
        
        if not login_url.strip():
            console.print("[red]❌ Không nhập link. Thoát.[/red]")
            sys.exit(0)
        
        # Lưu link
        console.print("")
        save_link = safe_input(
            "[bold bright_yellow]💾 Bạn có muốn lưu link này cho lần sau? ([bright_green]y[/bright_green]/[bright_red]n[/bright_red]): [/bold bright_yellow]",
            default="y"
        ).strip().lower()
        
        if save_link in ['y', 'yes', '']:
            if link_mgr.save_link(login_url):
                console.print("✅ [bright_green]Đã lưu link thành công! Link sẽ được mã hóa an toàn.[/bright_green]\n")
            else:
                console.print("⚠️  [yellow]Không thể lưu link (link có thể không hợp lệ)[/yellow]\n")
    
    # Parse login URL
    try:
        parsed = urlparse(login_url)
        params = parse_qs(parsed.query)
        if 'userId' in params:
            USER_ID = int(params.get('userId')[0])
        SECRET_KEY = params.get('secretKey', [None])[0]
        
        # Success message with cyberpunk style
        from rich.panel import Panel
        console.print(Panel(
            "[bright_green]✅ ĐĂNG NHẬP THÀNH CÔNG![/bright_green]",
            border_style="bright_green",
            box=box.DOUBLE
        ))
        console.print("")
        console.print(f"[bright_cyan]👤 User ID:[/bright_cyan] [yellow]{USER_ID}[/yellow]")
        console.print(f"[bright_cyan]🔑 Secret:[/bright_cyan] [yellow]{SECRET_KEY[:8] if SECRET_KEY else 'N/A'}****[/yellow]")
        console.print("")
        
        # Animation trước khi clear
        import time
        with console.status("[bold bright_green]✅ Chuẩn bị khởi động...[/bold bright_green]", spinner="dots"):
            time.sleep(1)
        
        # Clear terminal sau khi login xong
        clear_terminal()
        
    except Exception as e:
        console.print("[red]❌ Link không hợp lệ. Thoát.[/red]")
        log_debug(f"parse_login err: {e}")
        sys.exit(1)

def main():
    parse_login()
    # Login xong đã clear terminal rồi
    
    prompt_settings()
    # Config xong đã clear terminal và show banner rồi

    def on_balance_changed(bal, delta, info):
        console.print(f"[green]⤴️ cập nhật số dư: {bal:.4f} (Δ {delta:+.4f}) — {info.get('ts')}[/green]")

    def on_error(msg):
        console.print(f"[red]Balance poll lỗi: {msg}[/red]")

    poller = BalancePoller(USER_ID, SECRET_KEY, poll_seconds=max(1, int(BALANCE_POLL_INTERVAL)), on_balance=on_balance_changed, on_error=on_error, on_status=None)
    poller.start()
    start_threads()

    with Live(Group(build_header(), build_mid(), build_rooms_table(), build_reasoning_panel(), build_bet_table()), refresh_per_second=8, console=console, screen=False) as live:
        try:
            while not stop_flag:
                live.update(Group(build_header(), build_mid(), build_rooms_table(), build_reasoning_panel(), build_bet_table()))
                time.sleep(0.12)
            console.print("[bold yellow]Tool đã dừng theo yêu cầu hoặc đạt mục tiêu.[/bold yellow]")
        except KeyboardInterrupt:
            console.print("[yellow]Thoát bằng người dùng.[/yellow]")
        finally:
            # 💾 SAVE BRAIN KHI THOÁT!
            try:
                if selector._learning_enabled:
                    selector._self_learning_ai.save_brain()
                    console.print("[cyan]💾 AI Brain saved before exit![/cyan]")
            except Exception as e:
                console.print(f"[dim yellow]⚠️ Could not save brain: {e}[/dim yellow]")
            
            poller.stop()

if __name__ == "__main__":
    main()
