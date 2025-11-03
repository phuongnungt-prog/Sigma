"""
?? SELF-LEARNING AI v15.0 ??
AI t? h?c h?i t? t?ng v?n ch?i - Online Learning
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque, defaultdict
import statistics
import json
import pickle


class OnlineLearner:
    """
    ?? H?C TR?C TUY?N
    H?c v? c?p nh?t model sau m?i v?n ch?i
    """
    
    def __init__(self):
        # Feature weights - s? ???c h?c
        self.feature_weights = {
            "players": 0.0,
            "bet": 0.0,
            "survive_rate": 1.0,  # Start with survive rate as most important
            "stability": 0.5,
            "pattern": 0.0,
            "momentum": 0.0,
            "recent_kills": -0.5  # Penalty for recent kills
        }
        
        # Learning rate
        self.learning_rate = 0.01
        
        # Experience buffer
        self.experiences = deque(maxlen=500)
        
        # Performance tracking
        self.round_count = 0
        self.correct_predictions = 0
        
    def predict_room_quality(self, features: Dict[str, float]) -> float:
        """
        D? ?o?n ch?t l??ng ph?ng d?a tr?n features ?? h?c
        """
        score = 0.0
        for feature_name, weight in self.feature_weights.items():
            if feature_name in features:
                score += weight * features[feature_name]
        
        # Sigmoid to normalize to 0-1
        return 1 / (1 + np.exp(-score))
    
    def learn_from_result(self, room_features: Dict[str, float], survived: bool):
        """
        H?C T? K?T QU?
        C?p nh?t weights d?a tr?n k?t qu? th?c t?
        """
        # Store experience
        self.experiences.append({
            "features": room_features.copy(),
            "survived": survived
        })
        
        self.round_count += 1
        
        # Predict what we thought would happen
        prediction = self.predict_room_quality(room_features)
        
        # Calculate error
        target = 1.0 if survived else 0.0
        error = target - prediction
        
        # Track accuracy
        if (prediction > 0.5 and survived) or (prediction <= 0.5 and not survived):
            self.correct_predictions += 1
        
        # Update weights using gradient descent
        for feature_name, value in room_features.items():
            if feature_name in self.feature_weights:
                # Gradient descent update
                gradient = error * value * prediction * (1 - prediction)
                self.feature_weights[feature_name] += self.learning_rate * gradient
        
        # Decay learning rate over time (converge)
        if self.round_count % 50 == 0:
            self.learning_rate *= 0.95
            self.learning_rate = max(0.001, self.learning_rate)
    
    def get_accuracy(self) -> float:
        """L?y ?? ch?nh x?c hi?n t?i"""
        if self.round_count == 0:
            return 0.0
        return self.correct_predictions / self.round_count
    
    def get_insights(self) -> str:
        """L?y insights v? nh?ng g? AI ?? h?c"""
        insights = []
        
        # Top 3 most important features (positive)
        positive_features = {k: v for k, v in self.feature_weights.items() if v > 0}
        if positive_features:
            top_positive = sorted(positive_features.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append("? Features t?t:")
            for feat, weight in top_positive:
                insights.append(f"  ? {feat}: {weight:.3f}")
        
        # Top 3 most negative features
        negative_features = {k: v for k, v in self.feature_weights.items() if v < 0}
        if negative_features:
            top_negative = sorted(negative_features.items(), key=lambda x: x[1])[:3]
            insights.append("? Features x?u:")
            for feat, weight in top_negative:
                insights.append(f"  ? {feat}: {weight:.3f}")
        
        return "\n".join(insights)


class PatternLearner:
    """
    ?? H?C PATTERN
    Nh?n d?ng v? h?c c?c pattern l?p l?i trong game
    """
    
    def __init__(self):
        self.kill_sequences = defaultdict(lambda: {"count": 0, "total": 0})
        self.room_transitions = defaultdict(lambda: defaultdict(int))
        self.time_patterns = defaultdict(list)
        
    def record_kill(self, killed_room: int, prev_killed: int = None, time_of_day: str = ""):
        """Ghi nh?n kill v? h?c pattern"""
        # Record sequence if we have previous
        if prev_killed is not None:
            sequence = (prev_killed, killed_room)
            self.kill_sequences[sequence]["count"] += 1
        
        # Record time pattern
        if time_of_day:
            self.time_patterns[time_of_day].append(killed_room)
    
    def predict_next_kill(self, last_killed: int, time_of_day: str = "") -> Dict[int, float]:
        """
        D? ?o?n ph?ng n?o c? kh? n?ng b? kill ti?p theo
        D?a tr?n pattern ?? h?c
        """
        predictions = defaultdict(float)
        
        # Check sequence patterns
        for (prev, next_room), data in self.kill_sequences.items():
            if prev == last_killed and data["count"] > 2:
                # Confidence based on frequency
                confidence = data["count"] / sum(
                    d["count"] for (p, _), d in self.kill_sequences.items() if p == last_killed
                )
                predictions[next_room] = confidence
        
        # Check time patterns
        if time_of_day and time_of_day in self.time_patterns:
            time_kills = self.time_patterns[time_of_day]
            if len(time_kills) > 5:
                # Most common room killed at this time
                from collections import Counter
                common = Counter(time_kills).most_common(3)
                for room, count in common:
                    predictions[room] += (count / len(time_kills)) * 0.3
        
        return dict(predictions)
    
    def get_pattern_insights(self) -> str:
        """L?y insights v? patterns ?? h?c"""
        insights = []
        
        # Most common sequences
        if self.kill_sequences:
            top_sequences = sorted(
                self.kill_sequences.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:3]
            
            insights.append("?? Sequences ph? bi?n:")
            for (prev, next_room), data in top_sequences:
                if data["count"] >= 3:
                    insights.append(f"  ? Ph?ng {prev} ? Ph?ng {next_room} (x{data['count']})")
        
        return "\n".join(insights) if insights else "?? Ch?a ?? d? li?u ?? ph?t hi?n pattern"


class AdaptiveStrategy:
    """
    ?? CHI?N L??C TH?CH ?NG
    T? ??ng ?i?u ch?nh strategy d?a tr?n hi?u su?t
    """
    
    def __init__(self):
        self.strategies = {
            "conservative": {
                "min_survive_rate": 0.65,
                "max_players": 25,
                "avoid_recent_kill_rounds": 2,
                "wins": 0,
                "losses": 0,
                "weight": 0.33
            },
            "balanced": {
                "min_survive_rate": 0.55,
                "max_players": 35,
                "avoid_recent_kill_rounds": 1,
                "wins": 0,
                "losses": 0,
                "weight": 0.34
            },
            "aggressive": {
                "min_survive_rate": 0.45,
                "max_players": 50,
                "avoid_recent_kill_rounds": 0,
                "wins": 0,
                "losses": 0,
                "weight": 0.33
            }
        }
        
        self.current_strategy = "balanced"
        self.rounds_since_change = 0
        self.change_threshold = 10  # ??i strategy sau 10 v?n n?u t?
        
    def get_current_params(self) -> Dict[str, Any]:
        """L?y parameters c?a strategy hi?n t?i"""
        return self.strategies[self.current_strategy].copy()
    
    def record_result(self, strategy_name: str, won: bool):
        """Ghi nh?n k?t qu? v? h?c"""
        if strategy_name in self.strategies:
            if won:
                self.strategies[strategy_name]["wins"] += 1
            else:
                self.strategies[strategy_name]["losses"] += 1
        
        self.rounds_since_change += 1
        
        # Auto-adjust weights based on performance
        self._update_weights()
        
        # Consider switching strategy if performing poorly
        if self.rounds_since_change >= self.change_threshold:
            self._consider_strategy_change()
    
    def _update_weights(self):
        """C?p nh?t weights d?a tr?n performance"""
        total_performance = 0
        
        for strategy, data in self.strategies.items():
            total = data["wins"] + data["losses"]
            if total > 0:
                win_rate = data["wins"] / total
                total_performance += win_rate
        
        if total_performance > 0:
            for strategy, data in self.strategies.items():
                total = data["wins"] + data["losses"]
                if total > 0:
                    win_rate = data["wins"] / total
                    data["weight"] = win_rate / total_performance
    
    def _consider_strategy_change(self):
        """Xem x?t ??i strategy n?u c?n"""
        current = self.strategies[self.current_strategy]
        current_total = current["wins"] + current["losses"]
        
        if current_total >= 5:
            current_win_rate = current["wins"] / current_total
            
            # Find best performing strategy
            best_strategy = None
            best_win_rate = current_win_rate
            
            for name, data in self.strategies.items():
                if name != self.current_strategy:
                    total = data["wins"] + data["losses"]
                    if total >= 5:
                        win_rate = data["wins"] / total
                        if win_rate > best_win_rate + 0.1:  # Must be 10% better
                            best_win_rate = win_rate
                            best_strategy = name
            
            if best_strategy:
                self.current_strategy = best_strategy
                self.rounds_since_change = 0
    
    def get_insights(self) -> str:
        """L?y insights v? strategies"""
        insights = []
        insights.append(f"?? Current Strategy: {self.current_strategy}")
        
        for name, data in self.strategies.items():
            total = data["wins"] + data["losses"]
            if total > 0:
                win_rate = (data["wins"] / total) * 100
                marker = "?" if name == self.current_strategy else "  "
                insights.append(f"{marker} {name}: {win_rate:.1f}% ({data['wins']}W/{data['losses']}L)")
        
        return "\n".join(insights)


class MemoryBasedLearner:
    """
    ?? H?C D?A TR?N K? ?C
    Nh? c?c t?nh hu?ng t?t/x?u v? h?c t? ch?ng
    """
    
    def __init__(self):
        self.good_situations = deque(maxlen=100)
        self.bad_situations = deque(maxlen=100)
        self.room_history = defaultdict(lambda: {
            "total_rounds": 0,
            "survived": 0,
            "killed": 0,
            "avg_players": [],
            "avg_bet": []
        })
    
    def record_situation(self, room_id: int, room_data: Dict[str, Any], survived: bool):
        """Ghi nh?n t?nh hu?ng"""
        situation = {
            "room_id": room_id,
            "players": room_data.get("players", 0),
            "bet": room_data.get("bet", 0),
            "survive_rate": room_data.get("survive_rate", 0.5),
            "survived": survived
        }
        
        if survived:
            self.good_situations.append(situation)
        else:
            self.bad_situations.append(situation)
        
        # Update room history
        hist = self.room_history[room_id]
        hist["total_rounds"] += 1
        if survived:
            hist["survived"] += 1
        else:
            hist["killed"] += 1
        hist["avg_players"].append(room_data.get("players", 0))
        hist["avg_bet"].append(room_data.get("bet", 0))
        
        # Keep only recent 50
        if len(hist["avg_players"]) > 50:
            hist["avg_players"] = hist["avg_players"][-50:]
        if len(hist["avg_bet"]) > 50:
            hist["avg_bet"] = hist["avg_bet"][-50:]
    
    def find_similar_situation(self, current_room_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        T?m t?nh hu?ng t??ng t? trong qu? kh?
        Returns: ("good" ho?c "bad", similarity_score)
        """
        current_players = current_room_data.get("players", 0)
        current_bet = current_room_data.get("bet", 0)
        
        best_similarity = 0
        best_outcome = "unknown"
        
        # Check good situations
        for situation in self.good_situations:
            similarity = self._calculate_similarity(
                current_players, current_bet,
                situation["players"], situation["bet"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_outcome = "good"
        
        # Check bad situations
        for situation in self.bad_situations:
            similarity = self._calculate_similarity(
                current_players, current_bet,
                situation["players"], situation["bet"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_outcome = "bad"
        
        return best_outcome, best_similarity
    
    def _calculate_similarity(self, p1: float, b1: float, p2: float, b2: float) -> float:
        """T?nh ?? t??ng ??ng gi?a 2 t?nh hu?ng"""
        if p2 == 0 or b2 == 0:
            return 0.0
        
        player_diff = abs(p1 - p2) / max(p1, p2, 1)
        bet_diff = abs(b1 - b2) / max(b1, b2, 1)
        
        # Similarity score (0-1)
        similarity = 1 - (player_diff + bet_diff) / 2
        return max(0, similarity)
    
    def get_room_learned_survival_rate(self, room_id: int) -> float:
        """L?y survival rate ?? h?c cho ph?ng"""
        hist = self.room_history[room_id]
        if hist["total_rounds"] == 0:
            return 0.5  # Unknown
        
        return hist["survived"] / hist["total_rounds"]
    
    def get_insights(self) -> str:
        """L?y insights t? memory"""
        insights = []
        
        insights.append(f"?? Learned from {len(self.good_situations)} good + {len(self.bad_situations)} bad situations")
        
        # Best rooms learned
        best_rooms = sorted(
            [(rid, data) for rid, data in self.room_history.items() if data["total_rounds"] >= 5],
            key=lambda x: x[1]["survived"] / x[1]["total_rounds"] if x[1]["total_rounds"] > 0 else 0,
            reverse=True
        )[:3]
        
        if best_rooms:
            insights.append("?? Top rooms learned:")
            for room_id, data in best_rooms:
                rate = (data["survived"] / data["total_rounds"]) * 100
                insights.append(f"  ? Room {room_id}: {rate:.0f}% ({data['survived']}/{data['total_rounds']})")
        
        return "\n".join(insights)


class SelfLearningAI:
    """
    ?? SELF-LEARNING AI MASTER
    K?t h?p t?t c? c?c learners
    """
    
    def __init__(self):
        self.online_learner = OnlineLearner()
        self.pattern_learner = PatternLearner()
        self.adaptive_strategy = AdaptiveStrategy()
        self.memory_learner = MemoryBasedLearner()
        
        self.total_rounds = 0
        self.last_killed_room = None
        
    def learn_from_round(self, 
                         chosen_room: int,
                         room_features: Dict[str, float],
                         killed_room: int,
                         room_data: Dict[str, Any]):
        """
        H?C T? M?I V?N CH?I
        """
        survived = (chosen_room != killed_room)
        
        # Online learning
        self.online_learner.learn_from_result(room_features, survived)
        
        # Pattern learning
        self.pattern_learner.record_kill(
            killed_room, 
            self.last_killed_room,
            self._get_time_category()
        )
        
        # Strategy adaptation
        self.adaptive_strategy.record_result(
            self.adaptive_strategy.current_strategy,
            survived
        )
        
        # Memory learning
        self.memory_learner.record_situation(
            chosen_room,
            room_data,
            survived
        )
        
        self.last_killed_room = killed_room
        self.total_rounds += 1
    
    def get_room_prediction(self, room_id: int, room_features: Dict[str, float], 
                           room_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        D? ?O?N CHO PH?NG d?a tr?n nh?ng g? ?? h?c
        """
        # Online learning prediction
        online_score = self.online_learner.predict_room_quality(room_features)
        
        # Memory-based prediction
        similar_outcome, similarity = self.memory_learner.find_similar_situation(room_data)
        memory_score = similarity if similar_outcome == "good" else (1 - similarity)
        
        # Learned survival rate
        learned_rate = self.memory_learner.get_room_learned_survival_rate(room_id)
        
        # Pattern-based prediction (avoid predicted kills)
        kill_predictions = self.pattern_learner.predict_next_kill(
            self.last_killed_room if self.last_killed_room else 0
        )
        pattern_penalty = kill_predictions.get(room_id, 0)
        
        # Combine predictions
        final_score = (
            online_score * 0.30 +
            memory_score * 0.25 +
            learned_rate * 0.35 +
            (1 - pattern_penalty) * 0.10
        )
        
        return {
            "final_score": final_score,
            "online_score": online_score,
            "memory_score": memory_score,
            "learned_rate": learned_rate,
            "pattern_penalty": pattern_penalty
        }
    
    def _get_time_category(self) -> str:
        """L?y category th?i gian"""
        from datetime import datetime
        hour = datetime.now().hour
        
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def get_full_insights(self) -> str:
        """L?y t?t c? insights"""
        insights = []
        
        insights.append(f"?? SELF-LEARNING AI STATUS")
        insights.append(f"   Total Rounds Learned: {self.total_rounds}")
        insights.append(f"   Accuracy: {self.online_learner.get_accuracy():.1%}")
        insights.append("")
        
        insights.append(self.online_learner.get_insights())
        insights.append("")
        insights.append(self.pattern_learner.get_pattern_insights())
        insights.append("")
        insights.append(self.adaptive_strategy.get_insights())
        insights.append("")
        insights.append(self.memory_learner.get_insights())
        
        return "\n".join(insights)
    
    def save_brain(self, filepath: str = "ai_brain_memory.json"):
        """
        ?? L?U B? N?O AI
        Luu t?t c? ki?n th?c ?? h?c vao file
        """
        from datetime import datetime
        
        brain_data = {
            "metadata": {
                "version": "15.0",
                "last_updated": datetime.now().isoformat(),
                "total_rounds": self.total_rounds,
                "accuracy": self.online_learner.get_accuracy()
            },
            
            # Online Learner Data
            "online_learner": {
                "feature_weights": self.online_learner.feature_weights,
                "learning_rate": self.online_learner.learning_rate,
                "round_count": self.online_learner.round_count,
                "correct_predictions": self.online_learner.correct_predictions,
                "experiences": list(self.online_learner.experiences)
            },
            
            # Pattern Learner Data
            "pattern_learner": {
                "kill_sequences": {
                    str(k): v for k, v in self.pattern_learner.kill_sequences.items()
                },
                "time_patterns": dict(self.pattern_learner.time_patterns)
            },
            
            # Adaptive Strategy Data
            "adaptive_strategy": {
                "strategies": self.adaptive_strategy.strategies,
                "current_strategy": self.adaptive_strategy.current_strategy,
                "rounds_since_change": self.adaptive_strategy.rounds_since_change
            },
            
            # Memory Learner Data
            "memory_learner": {
                "good_situations": list(self.memory_learner.good_situations),
                "bad_situations": list(self.memory_learner.bad_situations),
                "room_history": {
                    str(k): v for k, v in self.memory_learner.room_history.items()
                }
            },
            
            # Global State
            "last_killed_room": self.last_killed_room
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(brain_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"? Error saving brain: {e}")
            return False
    
    def load_brain(self, filepath: str = "ai_brain_memory.json") -> bool:
        """
        ?? T?I B? N?O AI
        T?i l?i ki?n th?c ?? h?c t? file
        """
        import os
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                brain_data = json.load(f)
            
            # Load metadata
            self.total_rounds = brain_data["metadata"]["total_rounds"]
            
            # Load Online Learner
            ol_data = brain_data["online_learner"]
            self.online_learner.feature_weights = ol_data["feature_weights"]
            self.online_learner.learning_rate = ol_data["learning_rate"]
            self.online_learner.round_count = ol_data["round_count"]
            self.online_learner.correct_predictions = ol_data["correct_predictions"]
            self.online_learner.experiences = deque(ol_data["experiences"], maxlen=500)
            
            # Load Pattern Learner
            pl_data = brain_data["pattern_learner"]
            self.pattern_learner.kill_sequences = defaultdict(
                lambda: {"count": 0, "total": 0},
                {eval(k): v for k, v in pl_data["kill_sequences"].items()}
            )
            self.pattern_learner.time_patterns = defaultdict(
                list,
                pl_data["time_patterns"]
            )
            
            # Load Adaptive Strategy
            as_data = brain_data["adaptive_strategy"]
            self.adaptive_strategy.strategies = as_data["strategies"]
            self.adaptive_strategy.current_strategy = as_data["current_strategy"]
            self.adaptive_strategy.rounds_since_change = as_data["rounds_since_change"]
            
            # Load Memory Learner
            ml_data = brain_data["memory_learner"]
            self.memory_learner.good_situations = deque(ml_data["good_situations"], maxlen=100)
            self.memory_learner.bad_situations = deque(ml_data["bad_situations"], maxlen=100)
            
            # Load room history
            room_hist = {}
            for room_str, hist in ml_data["room_history"].items():
                room_hist[int(room_str)] = hist
            self.memory_learner.room_history = defaultdict(
                lambda: {
                    "total_rounds": 0,
                    "survived": 0,
                    "killed": 0,
                    "avg_players": [],
                    "avg_bet": []
                },
                room_hist
            )
            
            # Load global state
            self.last_killed_room = brain_data.get("last_killed_room")
            
            return True
            
        except Exception as e:
            print(f"? Error loading brain: {e}")
            return False
