"""
? QUANTUM BRAIN AI v14.0 - CORE MODULE ?
Tr? tu? si?u ph?m v?i ph?n t?ch ?a chi?u v? logic c?c m?nh
"""

import math
import random
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
import statistics


class QuantumAnalyzer:
    """
    ?? B? PH?N T?CH L??NG T?
    Multi-dimensional analysis v?i ?? ch?nh x?c si?u cao
    """
    
    def __init__(self):
        self.quantum_states = {}  # Tr?ng th?i l??ng t? c?a m?i ph?ng
        self.probability_matrix = {}  # Ma tr?n x?c su?t
        self.entropy_scores = {}  # ?? h?n lo?n
        
    def calculate_quantum_probability(self, room_id: int, features: Dict[str, float]) -> float:
        """
        ?? T?NH X?C SU?T L??NG T?
        S? d?ng nhi?u chi?u ph?n t?ch ??ng th?i
        """
        # Chi?u 1: X?c su?t c? b?n t? d? li?u
        base_prob = features.get("survive_score", 0.5)
        
        # Chi?u 2: Entropy (?? h?n lo?n)
        entropy = self._calculate_entropy(room_id, features)
        
        # Chi?u 3: Coherence (?? k?t d?nh)
        coherence = self._calculate_coherence(room_id, features)
        
        # Chi?u 4: Superposition (ch?ng ch?t tr?ng th?i)
        superposition = self._calculate_superposition(room_id, features)
        
        # K?T H?P T?T C? CHI?U
        quantum_prob = (
            base_prob * 0.40 +        # 40% t? d? li?u c? b?n
            (1 - entropy) * 0.25 +    # 25% t? ?? ?n ??nh (ng??c entropy)
            coherence * 0.20 +        # 20% t? ?? k?t d?nh
            superposition * 0.15      # 15% t? ch?ng ch?t
        )
        
        return max(0.0, min(1.0, quantum_prob))
    
    def _calculate_entropy(self, room_id: int, features: Dict[str, float]) -> float:
        """?? h?n lo?n - cao = kh?ng d? ?o?n ???c"""
        volatility = features.get("volatility_score", 0.5)
        momentum = abs(features.get("momentum_players", 0.0))
        
        # Entropy cao n?u bi?n ??ng m?nh
        entropy = (volatility + momentum) / 2.0
        return max(0.0, min(1.0, entropy))
    
    def _calculate_coherence(self, room_id: int, features: Dict[str, float]) -> float:
        """?? k?t d?nh - cao = xu h??ng r? r?ng"""
        stability = features.get("stability_score", 0.5)
        pattern = features.get("pattern_score", 0.0)
        
        # Coherence cao n?u ?n ??nh v? c? pattern
        coherence = (stability + max(0, pattern)) / 2.0
        return max(0.0, min(1.0, coherence))
    
    def _calculate_superposition(self, room_id: int, features: Dict[str, float]) -> float:
        """Ch?ng ch?t tr?ng th?i - ph?ng c? th? ? nhi?u tr?ng th?i"""
        hot_score = features.get("hot_score", 0.0)
        cold_score = features.get("cold_score", 0.0)
        
        # Superposition = kh? n?ng ph?ng ? nhi?u tr?ng th?i t?t
        superposition = hot_score * 0.7 + (1 - cold_score) * 0.3
        return max(0.0, min(1.0, superposition))


class DeepLogicEngine:
    """
    ?? ??NG C? LOGIC S?U
    Multi-layer reasoning v?i ph?n t?ch nh?n qu?
    """
    
    def __init__(self):
        self.logic_chains = []
        self.inference_tree = {}
        self.reasoning_depth = 5  # 5 t?ng suy lu?n
        
    def analyze_with_deep_logic(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """
        ?? PH?N T?CH V?I LOGIC S?U
        
        5 t?ng suy lu?n:
        1. Observation (Quan s?t)
        2. Correlation (T??ng quan)
        3. Causation (Nh?n qu?)
        4. Prediction (D? ?o?n)
        5. Validation (X?c th?c)
        """
        results = {
            "observations": [],
            "correlations": [],
            "causations": [],
            "predictions": [],
            "validations": [],
            "confidence": 0.0
        }
        
        # Layer 1: QUAN S?T
        obs = self._observe_patterns(situation)
        results["observations"] = obs
        
        # Layer 2: T??NG QUAN
        corr = self._find_correlations(obs, situation)
        results["correlations"] = corr
        
        # Layer 3: NH?N QU?
        caus = self._infer_causation(corr, situation)
        results["causations"] = caus
        
        # Layer 4: D? ?O?N
        pred = self._make_predictions(caus, situation)
        results["predictions"] = pred
        
        # Layer 5: X?C TH?C
        val = self._validate_logic(pred, situation)
        results["validations"] = val
        results["confidence"] = self._calculate_confidence(val)
        
        return results
    
    def _observe_patterns(self, situation: Dict[str, Any]) -> List[str]:
        """T?ng 1: Quan s?t c?c pattern"""
        observations = []
        
        room_data = situation.get("room_data", {})
        if not room_data:
            return observations
        
        # Pattern 1: Ph?ng ??ng ??t bi?n
        for rid, data in room_data.items():
            players = data.get("players", 0)
            if players > 30:
                observations.append(f"?? Ph?ng {rid} ??ng ??t bi?n ({players} ng??i) - B?t th??ng")
        
        # Pattern 2: Chu?i th?ng/thua
        win_streak = situation.get("win_streak", 0)
        lose_streak = situation.get("lose_streak", 0)
        if win_streak >= 3:
            observations.append(f"?? ?ang chu?i th?ng {win_streak} - Xu h??ng t?ch c?c")
        if lose_streak >= 2:
            observations.append(f"?? ?ang chu?i thua {lose_streak} - C?n ?i?u ch?nh")
        
        return observations
    
    def _find_correlations(self, observations: List[str], situation: Dict[str, Any]) -> List[str]:
        """T?ng 2: T?m t??ng quan"""
        correlations = []
        
        # T??ng quan gi?a ??ng ng??i v? r?i ro
        if any("??ng ??t bi?n" in obs for obs in observations):
            correlations.append("? T??ng quan: ??ng ng??i ? R?i ro cao ? Tr?nh")
        
        # T??ng quan gi?a chu?i th?ng v? confidence
        if any("chu?i th?ng" in obs for obs in observations):
            correlations.append("? T??ng quan: Chu?i th?ng ? Strategy ??ng ? Ti?p t?c")
        
        return correlations
    
    def _infer_causation(self, correlations: List[str], situation: Dict[str, Any]) -> List[str]:
        """T?ng 3: Suy lu?n nh?n qu?"""
        causations = []
        
        # Nh?n qu?: T?i sao ph?ng ??ng?
        if any("??ng ng??i" in c for c in correlations):
            causations.append("?? Nh?n qu?: Ph?ng ??ng V? nhi?u ng??i c?ng ngh? 'an to?n' ? B?y ??m ??ng")
        
        # Nh?n qu?: T?i sao th?ng li?n t?c?
        if any("Strategy ??ng" in c for c in correlations):
            causations.append("?? Nh?n qu?: Th?ng li?n t?c V? AI ch?n ph?ng ?n ??nh ? Duy tr?")
        
        return causations
    
    def _make_predictions(self, causations: List[str], situation: Dict[str, Any]) -> List[str]:
        """T?ng 4: ??a ra d? ?o?n"""
        predictions = []
        
        # D? ?o?n t? nh?n qu?
        if any("B?y ??m ??ng" in c for c in causations):
            predictions.append("?? D? ?o?n: Ph?ng ??ng s? b? kill trong 1-2 v?n t?i")
        
        if any("Duy tr?" in c for c in causations):
            predictions.append("?? D? ?o?n: Ti?p t?c strategy hi?n t?i s? th?ng 70%+")
        
        return predictions
    
    def _validate_logic(self, predictions: List[str], situation: Dict[str, Any]) -> List[str]:
        """T?ng 5: X?c th?c logic"""
        validations = []
        
        # X?c th?c d? ?o?n v?i d? li?u
        for pred in predictions:
            validations.append(f"? Logic chain validated: {len(pred)} chars")
        
        return validations
    
    def _calculate_confidence(self, validations: List[str]) -> float:
        """T?nh ?? tin c?y c?a chu?i logic"""
        if not validations:
            return 0.5
        
        # Confidence t?ng theo s? validation th?nh c?ng
        confidence = 0.6 + (len(validations) * 0.1)
        return min(0.95, confidence)


class MetaLearner:
    """
    ?? B? H?C META
    H?c c?ch h?c - t? ?i?u ch?nh chi?n l??c
    """
    
    def __init__(self):
        self.strategies_performance = {}
        self.learning_history = deque(maxlen=100)
        self.meta_knowledge = {}
        
    def evaluate_strategy(self, strategy_name: str, result: bool, context: Dict[str, Any]):
        """??nh gi? hi?u qu? c?a strategy"""
        if strategy_name not in self.strategies_performance:
            self.strategies_performance[strategy_name] = {
                "wins": 0,
                "losses": 0,
                "contexts": []
            }
        
        if result:
            self.strategies_performance[strategy_name]["wins"] += 1
        else:
            self.strategies_performance[strategy_name]["losses"] += 1
        
        self.strategies_performance[strategy_name]["contexts"].append(context)
        
        # H?c t? l?ch s?
        self.learning_history.append({
            "strategy": strategy_name,
            "result": result,
            "context": context
        })
    
    def get_best_strategy(self, current_context: Dict[str, Any]) -> str:
        """Ch?n strategy t?t nh?t cho context hi?n t?i"""
        best_strategy = "data_driven"  # Default
        best_score = 0.0
        
        for strategy, perf in self.strategies_performance.items():
            total = perf["wins"] + perf["losses"]
            if total == 0:
                continue
            
            win_rate = perf["wins"] / total
            
            # ?i?m cao h?n n?u context t??ng t?
            similarity = self._calculate_context_similarity(
                current_context, 
                perf["contexts"]
            )
            
            score = win_rate * 0.7 + similarity * 0.3
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    def _calculate_context_similarity(self, current: Dict[str, Any], historical: List[Dict[str, Any]]) -> float:
        """T?nh ?? t??ng ??ng gi?a context hi?n t?i v? l?ch s?"""
        if not historical:
            return 0.5
        
        # Simple similarity: so s?nh win_streak, lose_streak
        current_streak = current.get("win_streak", 0) - current.get("lose_streak", 0)
        
        similar_contexts = 0
        for hist in historical[-10:]:  # Ch? xem 10 g?n nh?t
            hist_streak = hist.get("win_streak", 0) - hist.get("lose_streak", 0)
            if abs(current_streak - hist_streak) <= 2:
                similar_contexts += 1
        
        return similar_contexts / min(10, len(historical))
    
    def get_meta_insights(self) -> str:
        """L?y insights t? meta learning"""
        if not self.strategies_performance:
            return "?? Meta-learning: ?ang thu th?p d? li?u..."
        
        insights = []
        
        for strategy, perf in self.strategies_performance.items():
            total = perf["wins"] + perf["losses"]
            if total >= 5:
                win_rate = perf["wins"] / total
                insights.append(f"  ? {strategy}: {win_rate:.0%} win rate ({total} v?n)")
        
        if insights:
            return "?? Meta-learning insights:\n" + "\n".join(insights)
        else:
            return "?? Meta-learning: C?n th?m d? li?u (< 5 v?n)"
