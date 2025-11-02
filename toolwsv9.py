# toolws.py (OMEGA) - Omega AI siêu trí tuệ tập trung 1 chế độ
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
import statistics
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
console.print("[bold magenta]Ω Omega AI Supreme: Một chế độ - Tối đa hóa tỷ lệ sống sót[/]")
console.print(Rule())

tz = pytz.timezone("Asia/Ho_Chi_Minh")

logger = logging.getLogger("omega_ai_supreme")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("omega_ai_supreme.log", encoding="utf-8"))

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
room_stats: Dict[int, Dict[str, Any]] = {
    r: {
        "kills": 0,
        "survives": 0,
        "last_kill_round": None,
        "last_players": 0,
        "last_bet": 0,
        "player_history": deque(maxlen=120),
        "bet_history": deque(maxlen=120),
        "outcome_history": deque(maxlen=120),
    }
    for r in ROOM_ORDER
}
room_outcome_history: Dict[int, deque] = {r: deque(maxlen=60) for r in ROOM_ORDER}

predicted_room: Optional[int] = None
last_killed_room: Optional[int] = None
prediction_locked: bool = False
prediction_confidence: float = 0.5
formula_votes: Dict[int, int] = {}
omega_meta: Dict[str, Any] = {
    "win": 0,
    "lose": 0,
    "streak": 0,
    "version": "omega-1.1",
    "history": deque(maxlen=200),
    "last_choice": None,
    "last_confidence": prediction_confidence,
}

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

# selection modes
SUPER_MODE = "OMEGA_AI"
OMEGA_LABEL = "Omega AI (siêu trí tuệ)"
OMEGA_FORMULA_COUNT = 240

SELECTION_MODES = {
    SUPER_MODE: OMEGA_LABEL,
}

settings = {"algo": SUPER_MODE}

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
                if last_balance_val is None:
                    starting_balance = build
                    last_balance_val = build
                else:
                    delta = float(build) - float(last_balance_val)
                    if abs(delta) > 0:
                        cumulative_profit += delta
                        last_balance_val = build
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

# -------------------- OMEGA AI SELECTION (SUPER MODE) --------------------

# FORMULAS storage and generator seed
FORMULAS: List[Dict[str, Any]] = []
FORMULA_SEED = 1234567

def _room_features_enhanced(rid: int):
    st = room_state.get(rid, {})
    stats = room_stats.get(rid, {})

    players = float(st.get("players", 0))
    bet = float(st.get("bet", 0))
    bet_per_player = (bet / players) if players > 0 else bet

    players_norm = min(1.0, players / 50.0)
    bet_norm = 1.0 / (1.0 + bet / 2000.0)
    bpp_norm = 1.0 / (1.0 + bet_per_player / 1200.0)

    kill_count = float(stats.get("kills", 0))
    survive_count = float(stats.get("survives", 0))
    kill_rate = (kill_count + 0.5) / (kill_count + survive_count + 1.0)
    survive_score = 1.0 - kill_rate

    recent_history = list(bet_history)[-12:]
    recent_pen = 0.0
    for i, rec in enumerate(reversed(recent_history)):
        if rec.get("room") == rid:
            recent_pen += 0.12 * (1.0 / (i + 1))

    last_pen = 0.0
    if last_killed_room == rid and SELECTION_CONFIG.get("avoid_last_kill", True):
        last_pen = 0.35

    outcome_hist = stats.get("outcome_history") or room_outcome_history.get(rid, deque())
    recent_window = list(outcome_hist)[-12:]
    survival_recent_ratio = (sum(recent_window) + 0.5) / (len(recent_window) + 1.0)

    survival_streak = 0
    for val in reversed(outcome_hist):
        if val == 1:
            survival_streak += 1
        else:
            break
    kill_streak = 0
    for val in reversed(outcome_hist):
        if val == 0:
            kill_streak += 1
        else:
            break

    survival_streak_norm = min(survival_streak / 6.0, 1.0)
    kill_streak_norm = min(kill_streak / 4.0, 1.0)
    momentum = survival_streak_norm - kill_streak_norm

    player_hist = list(stats.get("player_history", []))
    bet_hist = list(stats.get("bet_history", []))

    avg_players = statistics.mean(player_hist) if player_hist else players
    avg_bet = statistics.mean(bet_hist) if bet_hist else bet

    player_trend_raw = players - avg_players
    player_trend = math.tanh(player_trend_raw / 25.0) if avg_players else 0.0

    bet_pressure_raw = (bet - avg_bet) / (avg_bet + 1.0)
    bet_pressure = math.tanh(bet_pressure_raw / 4.0)

    bet_volatility = 0.0
    if len(bet_hist) >= 3:
        try:
            bet_volatility = statistics.pstdev(bet_hist) / (statistics.mean(bet_hist) + 1.0)
        except statistics.StatisticsError:
            bet_volatility = 0.0
    bet_volatility = min(max(bet_volatility, 0.0), 2.0)
    volatility = math.tanh(bet_volatility)
    stability = 1.0 - min(volatility, 0.95)

    last_kill_round = stats.get("last_kill_round")
    if last_kill_round is None:
        kill_gap_norm = 1.0
    else:
        gap = max(0, round_index - int(last_kill_round))
        kill_gap_norm = min(gap / 8.0, 1.0)

    adaptive_focus = max(-1.0, min(1.0, (survival_recent_ratio - kill_rate)))
    resilience = (sum(recent_window[-6:]) + 0.5) / (len(recent_window[-6:]) + 1.0)

    hot_score = max(0.0, survive_score - 0.2)
    cold_score = max(0.0, kill_rate - 0.4)

    return {
        "players_norm": players_norm,
        "bet_norm": bet_norm,
        "bpp_norm": bpp_norm,
        "survive_score": survive_score,
        "recent_pen": recent_pen,
        "last_pen": last_pen,
        "hot_score": hot_score,
        "cold_score": cold_score,
        "kill_gap_norm": kill_gap_norm,
        "survival_streak_norm": survival_streak_norm,
        "kill_streak_norm": kill_streak_norm,
        "momentum": momentum,
        "resilience": resilience,
        "volatility": volatility,
        "stability": stability,
        "bet_pressure": bet_pressure,
        "player_trend": player_trend,
        "adaptive_focus": adaptive_focus,
    }

def _init_formulas(mode: str = SUPER_MODE):
    """
    Initialize the Omega AI ensemble. Even if mode is provided, only SUPER_MODE is supported.
    """
    global FORMULAS
    rng = random.Random(FORMULA_SEED if mode == SUPER_MODE else FORMULA_SEED + 9999)
    formulas: List[Dict[str, Any]] = []

    def mk_formula(focus: str = "balanced") -> Dict[str, Any]:
        w = {
            "players": rng.uniform(0.15, 0.45),
            "bet": rng.uniform(0.08, 0.28),
            "bpp": rng.uniform(0.06, 0.32),
            "survive": rng.uniform(0.24, 0.48),
            "recent": rng.uniform(0.05, 0.25),
            "last": rng.uniform(0.08, 0.3),
            "hot": rng.uniform(0.1, 0.42),
            "cold": rng.uniform(0.08, 0.35),
            "kill_gap": rng.uniform(0.25, 0.6),
            "survival_streak": rng.uniform(0.2, 0.55),
            "kill_streak": rng.uniform(0.15, 0.45),
            "momentum": rng.uniform(0.15, 0.4),
            "resilience": rng.uniform(0.22, 0.5),
            "volatility": rng.uniform(0.12, 0.28),
            "stability": rng.uniform(0.18, 0.42),
            "bet_pressure": rng.uniform(0.05, 0.24),
            "player_trend": rng.uniform(0.05, 0.26),
            "adaptive_focus": rng.uniform(0.22, 0.45),
        }

        if focus == "safe":
            w["survive"] += rng.uniform(0.3, 0.55)
            w["resilience"] += rng.uniform(0.22, 0.38)
            w["volatility"] += rng.uniform(0.12, 0.2)
            w["kill_streak"] += rng.uniform(0.18, 0.28)
        elif focus == "aggressive":
            w["momentum"] += rng.uniform(0.25, 0.45)
            w["player_trend"] += rng.uniform(0.3, 0.5)
            w["bet_pressure"] += rng.uniform(0.2, 0.38)
        elif focus == "counter":
            w["cold"] += rng.uniform(0.3, 0.5)
            w["last"] += rng.uniform(0.2, 0.4)
            w["adaptive_focus"] += rng.uniform(0.25, 0.38)
        elif focus == "streak":
            w["survival_streak"] += rng.uniform(0.25, 0.4)
            w["kill_gap"] += rng.uniform(0.3, 0.45)

        temperature = rng.uniform(0.7, 1.35)
        if focus == "aggressive":
            temperature += rng.uniform(0.05, 0.15)
        elif focus == "safe":
            temperature -= rng.uniform(0.05, 0.12)

        formula = {
            "w": w,
            "noise": rng.uniform(0.01, 0.06),
            "adapt": 1.0,
            "temperature": max(0.5, min(1.6, temperature)),
            "score_bias": rng.uniform(-0.05, 0.05),
            "stats": {"win": 0, "lose": 0, "recent": deque(maxlen=24)},
        }
        return formula

    focus_pool = ["balanced", "safe", "aggressive", "counter", "streak"]
    for idx in range(OMEGA_FORMULA_COUNT):
        focus = focus_pool[idx % len(focus_pool)]
        formulas.append(mk_formula(focus=focus))

    FORMULAS = formulas
    formula_votes.clear()

# initialize default formulas
_init_formulas(SUPER_MODE)

def choose_room(mode: str = SUPER_MODE) -> Tuple[int, str]:
    """
    Master chooser for Omega AI. Returns (room_id, algo_label).
    """
    global FORMULAS, prediction_confidence, formula_votes

    if mode != SUPER_MODE or not FORMULAS:
        _init_formulas(SUPER_MODE)

    cand = [r for r in ROOM_ORDER]
    agg_scores: Dict[int, float] = {r: 0.0 for r in cand}
    confidence_tracker: Dict[int, List[float]] = {r: [] for r in cand}
    formula_votes.clear()

    for idx, fentry in enumerate(FORMULAS):
        weights = fentry["w"]
        adapt = fentry.get("adapt", 1.0)
        noise_scale = fentry.get("noise", 0.02)
        temperature = fentry.get("temperature", 1.0)
        stats = fentry.get("stats", {})

        best_room = None
        best_score = -1e9

        for r in cand:
            feat = _room_features_enhanced(r)
            score = 0.0
            score += weights.get("players", 0.0) * feat["players_norm"]
            score += weights.get("bet", 0.0) * feat["bet_norm"]
            score += weights.get("bpp", 0.0) * feat["bpp_norm"]
            score += weights.get("survive", 0.0) * feat["survive_score"]
            score -= weights.get("recent", 0.0) * feat["recent_pen"]
            score -= weights.get("last", 0.0) * feat["last_pen"]
            score += weights.get("hot", 0.0) * feat["hot_score"]
            score -= weights.get("cold", 0.0) * feat["cold_score"]
            score += weights.get("kill_gap", 0.0) * feat["kill_gap_norm"]
            score += weights.get("survival_streak", 0.0) * feat["survival_streak_norm"]
            score -= weights.get("kill_streak", 0.0) * feat["kill_streak_norm"]
            score += weights.get("momentum", 0.0) * feat["momentum"]
            score += weights.get("resilience", 0.0) * feat["resilience"]
            score -= weights.get("volatility", 0.0) * feat["volatility"]
            score += weights.get("stability", 0.0) * feat["stability"]
            score -= weights.get("bet_pressure", 0.0) * feat["bet_pressure"]
            score += weights.get("player_trend", 0.0) * feat["player_trend"]
            score += weights.get("adaptive_focus", 0.0) * feat["adaptive_focus"]

            score *= adapt

            noise = (math.sin((idx + 1) * (r + 1) * 12.9898) * 43758.5453) % 1.0
            noise = (noise - 0.5) * (noise_scale * 2.0)
            score += noise

            recent_perf = stats.get("recent") if isinstance(stats.get("recent"), deque) else None
            if recent_perf:
                win_rate = (sum(recent_perf) / len(recent_perf)) if len(recent_perf) else 0.5
                score *= 1.0 + (win_rate - 0.5) * 0.35

            score *= 1.0 + (temperature - 1.0) * 0.22

            if score > best_score:
                best_score = score
                best_room = r

        if best_room is None:
            continue

        agg_scores[best_room] += best_score
        confidence_tracker[best_room].append(best_score)
        formula_votes[idx] = best_room
        fentry["last_score"] = best_score

    n_formulas = max(1, len(FORMULAS))
    for r in agg_scores:
        agg_scores[r] /= n_formulas

    for r in cand:
        feat = _room_features_enhanced(r)
        agg_scores[r] += 0.03 * feat["survival_streak_norm"]
        agg_scores[r] += 0.02 * feat["kill_gap_norm"]
        agg_scores[r] -= 0.03 * feat["kill_streak_norm"]
        agg_scores[r] -= 0.02 * feat["recent_pen"]

    ranked = sorted(agg_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_room, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else best_score

    confidence_delta = max(0.0, best_score - second_score)
    confidence = 0.55 + confidence_delta * 0.9
    if confidence_tracker[best_room]:
        if len(confidence_tracker[best_room]) >= 2:
            try:
                local_std = statistics.pstdev(confidence_tracker[best_room])
                confidence -= min(0.2, local_std * 0.25)
            except statistics.StatisticsError:
                pass
    prediction_confidence = max(0.05, min(0.95, confidence))
    omega_meta["history"].append({"scores": ranked, "ts": time.time(), "confidence": prediction_confidence})
    omega_meta["last_choice"] = best_room
    omega_meta["last_confidence"] = prediction_confidence

    return best_room, SUPER_MODE

def update_formulas_after_result(predicted_room: Optional[int], killed_room: Optional[int], mode: str = SUPER_MODE, lr: float = 0.12):
    """
    Reinforcement loop for Omega AI after mỗi kết quả.
    """
    global FORMULAS, prediction_confidence, omega_meta

    if not FORMULAS or killed_room is None:
        return

    win = (predicted_room is not None and killed_room is not None and predicted_room != killed_room)

    for idx, fentry in enumerate(FORMULAS):
        vote = formula_votes.get(idx)
        stats = fentry.setdefault("stats", {"win": 0, "lose": 0, "recent": deque(maxlen=24)})
        recent = stats.setdefault("recent", deque(maxlen=24))
        current_adapt = fentry.get("adapt", 1.0)
        current_noise = fentry.get("noise", 0.02)

        if vote is None:
            # Không tham gia vote -> nhẹ nhàng khuyến khích khám phá
            fentry["adapt"] = min(4.0, current_adapt * 1.01)
            continue

        if vote == killed_room:
            stats["lose"] = stats.get("lose", 0) + 1
            recent.append(0)
            new_adapt = max(0.2, current_adapt * (1.0 - lr * 0.7))
            fentry["adapt"] = new_adapt
            fentry["score_bias"] = fentry.get("score_bias", 0.0) - lr * 0.12
            fentry["noise"] = min(0.08, max(0.006, current_noise * (1.0 + lr * 0.5)))
        else:
            stats["win"] = stats.get("win", 0) + 1
            recent.append(1)
            new_adapt = min(4.5, current_adapt * (1.0 + lr * 0.6))
            fentry["adapt"] = new_adapt
            fentry["score_bias"] = fentry.get("score_bias", 0.0) + lr * 0.1
            fentry["noise"] = max(0.005, current_noise * (1.0 - lr * 0.4))

        if recent:
            win_rate_recent = sum(recent) / len(recent)
        else:
            win_rate_recent = 0.5

        temp = fentry.get("temperature", 1.0)
        temp_delta = (win_rate_recent - 0.5) * 0.2
        fentry["temperature"] = max(0.5, min(1.6, temp + temp_delta))

    formula_votes.clear()

    if win:
        omega_meta["win"] = omega_meta.get("win", 0) + 1
        streak = omega_meta.get("streak", 0)
        omega_meta["streak"] = streak + 1 if streak >= 0 else 1
        prediction_confidence = min(0.99, prediction_confidence + 0.05)
    else:
        omega_meta["lose"] = omega_meta.get("lose", 0) + 1
        streak = omega_meta.get("streak", 0)
        omega_meta["streak"] = streak - 1 if streak <= 0 else -1
        prediction_confidence = max(0.05, prediction_confidence - 0.07)

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
    algo = settings.get("algo", SUPER_MODE)
    try:
        chosen, algo_used = choose_room(algo)
    except Exception as e:
        log_debug(f"choose_room error: {e}")
        chosen, algo_used = choose_room(SUPER_MODE)
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

    # --- OMEGA AI: update ensemble sau khi chốt kết quả ---
    try:
        # res_issue corresponds to the round we just resolved; killed_room is global
        # Omega AI tự điều chỉnh trọng số sau mỗi kết quả
        update_formulas_after_result(predicted_room, krid, settings.get("algo", SUPER_MODE))
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
                stats = room_stats[rid]
                stats["last_players"] = players
                stats["last_bet"] = bet
                stats.setdefault("player_history", deque(maxlen=120)).append(players)
                stats.setdefault("bet_history", deque(maxlen=120)).append(bet)
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
                        room_stats[rid].setdefault("outcome_history", deque(maxlen=120)).append(0)
                        room_outcome_history[rid].append(0)
                    else:
                        room_stats[rid]["survives"] += 1
                        room_stats[rid].setdefault("outcome_history", deque(maxlen=120)).append(1)
                        room_outcome_history[rid].append(1)

                # Immediately mark bet result locally (fast) without waiting for balance
                res_issue = new_issue if new_issue is not None else issue_id
                _mark_bet_result_from_issue(res_issue, krid)
                # Fire background balance refresh to compute actual deltas & cumulative profit
                threading.Thread(target=_background_fetch_balance_after_result, daemon=True).start()

            ui_state = "RESULT"

            # check profit target or stop-loss after we fetched balances (balance fetch may set current_build)
            def _check_stop_conditions():
                global stop_flag
                try:
                    if stop_when_profit_reached and profit_target is not None and isinstance(current_build, (int, float)) and current_build >= profit_target:
                        console.print(f"[bold green]🎉 MỤC TIÊU LÃI ĐẠT: {current_build} >= {profit_target}. Dừng tool.[/]")
                        stop_flag = True
                        try:
                            wsobj = _ws.get("ws")
                            if wsobj:
                                wsobj.close()
                        except Exception:
                            pass
                    if stop_when_loss_reached and stop_loss_target is not None and isinstance(current_build, (int, float)) and current_build <= stop_loss_target:
                        console.print(f"[bold red]⚠️ STOP-LOSS TRIGGED: {current_build} <= {stop_loss_target}. Dừng tool.[/]")
                        stop_flag = True
                        try:
                            wsobj = _ws.get("ws")
                            if wsobj:
                                wsobj.close()
                        except Exception:
                            pass
                except Exception:
                    pass
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

    left = Text("VUA THOÁT HIỂM VIP", style="bold cyan")

    b = f"{current_build:,.4f}" if isinstance(current_build, (int, float)) else (str(current_build) if current_build is not None else "-")
    u = f"{current_usdt:,.4f}" if isinstance(current_usdt, (int, float)) else (str(current_usdt) if current_usdt is not None else "-")
    x = f"{current_world:,.4f}" if isinstance(current_world, (int, float)) else (str(current_world) if current_world is not None else "-")

    pnl_val = cumulative_profit if cumulative_profit is not None else 0.0
    pnl_str = f"{pnl_val:+,.4f}"
    pnl_style = "green bold" if pnl_val > 0 else ("red bold" if pnl_val < 0 else "yellow")

    bal = Text.assemble((f"USDT: {u}", "bold"), ("   "), (f"XWORLD: {x}", "bold"), ("   "), (f"BUILD: {b}", "bold"))

    algo_label = SELECTION_MODES.get(settings.get('algo'), settings.get('algo'))

    right_lines = []
    right_lines.append(f"Thuật toán: {algo_label} - v{omega_meta.get('version')}")
    right_lines.append(f"Lãi/lỗ: [{pnl_style}] {pnl_str} [/{pnl_style}]")
    right_lines.append(f"Phiên: {issue_id or '-'}")
    right_lines.append(f"chuỗi: thắng={max_win_streak} / thua={max_lose_streak}")
    omega_summary = f"{omega_meta.get('win', 0)}W/{omega_meta.get('lose', 0)}L"
    right_lines.append(f"Độ tin cậy: {prediction_confidence * 100:.1f}%")
    right_lines.append(f"Omega track: {omega_summary} | streak {omega_meta.get('streak', 0)}")
    if stop_when_profit_reached and profit_target is not None:
        right_lines.append(f"[green]TakeProfit@{profit_target}[/]")
    if stop_when_loss_reached and stop_loss_target is not None:
        right_lines.append(f"[red]StopLoss@{stop_loss_target}[/]")

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
        lines.append(f"Ω-AI: {len(FORMULAS)} công thức đang hội ý")
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
        lines.append(f"Độ tự tin Ω: {prediction_confidence * 100:.1f}%")
        lines.append(f"Thành tích Ω: {omega_meta.get('win', 0)}W/{omega_meta.get('lose', 0)}L")
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
        lines.append(f"Độ tin cậy Ω hiện tại: {prediction_confidence * 100:.1f}%")
        lines.append(f"Thành tích Ω: {omega_meta.get('win', 0)}W/{omega_meta.get('lose', 0)}L")
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
        lines.append(f"Ω tracking: {omega_meta.get('win', 0)}W/{omega_meta.get('lose', 0)}L | Độ tin cậy {prediction_confidence * 100:.1f}%")
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

    console.print("\n[bold green]Ω AI siêu trí tuệ đã kích hoạt — tự động chọn chế độ tối ưu.[/]")
    settings["algo"] = SUPER_MODE
    try:
        _init_formulas(SUPER_MODE)
    except Exception:
        log_debug("Không thể tái khởi tạo Omega AI")

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

    sl = safe_input("lỗ bao nhiêu thì chốt( không dùng enter): ", default="")
    try:
        if sl and sl.strip() != "":
            stop_loss_target = float(sl)
            stop_when_loss_reached = True
        else:
            stop_loss_target = None
            stop_when_loss_reached = False
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
