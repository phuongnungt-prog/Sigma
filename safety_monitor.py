"""
??? SAFETY MONITOR v16.0  
Gi?m s?t an to?n v? c?nh b?o r?i ro real-time
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import time


class SafetyMonitor:
    """
    ??? SAFETY MONITOR
    Ph?t hi?n v? c?nh b?o c?c t?nh hu?ng nguy hi?m
    """
    
    def __init__(self):
        # Loss tracking
        self.consecutive_losses = 0
        self.losses_in_window = deque(maxlen=10)  # 10 v?n g?n nh?t
        
        # Betting pattern
        self.bet_history = deque(maxlen=20)
        self.max_bet_threshold = None  # S? ???c set t? config
        
        # Balance tracking
        self.balance_history = deque(maxlen=50)
        self.starting_balance = None
        self.lowest_balance = None
        
        # Warning history
        self.warnings = []
        self.critical_warnings = []
        
        # Auto-pause conditions
        self.auto_pause_enabled = True
        self.pause_reasons = []
        
    def set_starting_balance(self, balance: float):
        """Set s? d? ban ??u"""
        self.starting_balance = balance
        self.lowest_balance = balance
        
    def set_max_bet_threshold(self, threshold: float):
        """Set ng??ng bet t?i ?a an to?n"""
        self.max_bet_threshold = threshold
    
    def check_before_bet(self,
                        current_balance: float,
                        bet_amount: float,
                        room_risk: str,
                        ai_confidence: float,
                        recent_losses: int) -> Dict[str, Any]:
        """
        Ki?m tra an to?n TR??C KHI ??t c??c
        Returns: {
            'safe': bool,
            'warnings': List[str],
            'should_pause': bool,
            'risk_level': str
        }
        """
        warnings = []
        critical = False
        should_pause = False
        
        # 1. Balance check
        if current_balance < bet_amount * 2:
            warnings.append("?? Balance th?p! Ch? ?? c??c ~1 v?n n?a")
            critical = True
        
        if self.starting_balance and current_balance < self.starting_balance * 0.3:
            warnings.append("?? ?? M?T 70% V?N! C?n nh?c d?ng l?i")
            critical = True
            should_pause = True
        
        # 2. Bet amount check
        if self.max_bet_threshold and bet_amount > self.max_bet_threshold:
            warnings.append(f"?? Bet qu? l?n! ({bet_amount} > {self.max_bet_threshold})")
            critical = True
        
        if self.starting_balance and bet_amount > self.starting_balance * 0.2:
            warnings.append("?? Bet > 20% v?n! R?t r?i ro")
        
        # 3. Consecutive losses
        if recent_losses >= 5:
            warnings.append(f"?? ?? thua {recent_losses} v?n li?n ti?p! Ngh? ng?i")
            should_pause = True
            critical = True
        elif recent_losses >= 3:
            warnings.append(f"?? Thua {recent_losses} v?n li?n ti?p. C?n th?n!")
        
        # 4. Room risk check
        if room_risk in ['high', 'very_high'] and ai_confidence < 0.7:
            warnings.append("?? Ph?ng r?i ro CAO + AI kh?ng ch?c ch?n!")
        
        # 5. AI confidence check
        if ai_confidence < 0.5:
            warnings.append(f"?? AI kh?ng t? tin ({ai_confidence:.0%})")
        
        # 6. Recent performance check
        if len(self.losses_in_window) >= 8:
            loss_rate = sum(1 for x in self.losses_in_window if not x) / len(self.losses_in_window)
            if loss_rate >= 0.7:  # Thua 70% trong 10 v?n g?n
                warnings.append("?? Win rate k?m! Thua 7/10 v?n g?n nh?t")
                should_pause = True
                critical = True
        
        # Determine overall risk level
        if critical:
            risk_level = "CRITICAL"
        elif len(warnings) >= 2:
            risk_level = "HIGH"
        elif len(warnings) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'safe': not critical,
            'warnings': warnings,
            'should_pause': should_pause,
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(risk_level, warnings)
        }
    
    def record_result(self,
                     won: bool,
                     bet_amount: float,
                     current_balance: float):
        """
        Ghi nh?n k?t qu? sau khi c??c
        """
        # Update tracking
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        self.losses_in_window.append(won)
        self.bet_history.append(bet_amount)
        self.balance_history.append(current_balance)
        
        # Update lowest balance
        if self.lowest_balance is None or current_balance < self.lowest_balance:
            self.lowest_balance = current_balance
    
    def get_safety_score(self) -> float:
        """
        T?nh safety score (0-100)
        100 = R?t an to?n
        0 = C?c k? nguy hi?m
        """
        score = 100.0
        
        # Consecutive losses penalty
        if self.consecutive_losses >= 5:
            score -= 50
        elif self.consecutive_losses >= 3:
            score -= 30
        elif self.consecutive_losses >= 2:
            score -= 15
        
        # Recent win rate
        if len(self.losses_in_window) >= 5:
            loss_rate = sum(1 for x in self.losses_in_window if not x) / len(self.losses_in_window)
            score -= loss_rate * 30  # Max -30 points
        
        # Balance health
        if self.starting_balance and len(self.balance_history) > 0:
            current = self.balance_history[-1]
            balance_pct = current / self.starting_balance
            
            if balance_pct < 0.3:
                score -= 20  # M?t > 70% v?n
            elif balance_pct < 0.5:
                score -= 10  # M?t > 50% v?n
        
        return max(0.0, min(100.0, score))
    
    def get_safety_status(self) -> Dict[str, Any]:
        """
        L?y status an to?n t?ng th?
        """
        score = self.get_safety_score()
        
        if score >= 80:
            status = "SAFE"
            emoji = "??"
            description = "An to?n - Ti?p t?c c??c"
        elif score >= 60:
            status = "CAUTION"
            emoji = "??"
            description = "C?n th?n - Theo d?i ch?t"
        elif score >= 40:
            status = "WARNING"
            emoji = "??"
            description = "C?nh b?o - C?n nh?c d?ng"
        else:
            status = "DANGER"
            emoji = "??"
            description = "Nguy hi?m - N?N D?NG!"
        
        return {
            'score': score,
            'status': status,
            'emoji': emoji,
            'description': description,
            'consecutive_losses': self.consecutive_losses,
            'recommendations': self._get_safety_recommendations(score)
        }
    
    def _get_recommendation(self, risk_level: str, warnings: List[str]) -> str:
        """T?o recommendation d?a tr?n risk level"""
        if risk_level == "CRITICAL":
            return "?? KH?NG N?N C??C! R?i ro c?c cao!"
        elif risk_level == "HIGH":
            return "?? C?n th?n! Nhi?u r?i ro"
        elif risk_level == "MEDIUM":
            return "?? C? r?i ro, c?n nh?c k?"
        else:
            return "? An to?n, c? th? c??c"
    
    def _get_safety_recommendations(self, score: float) -> List[str]:
        """T?o recommendations d?a tr?n safety score"""
        recs = []
        
        if score < 40:
            recs.append("?? N?N D?NG ngay! Safety score qu? th?p")
            recs.append("?? Ngh? ng?i ?t nh?t 30 ph?t")
        elif score < 60:
            recs.append("?? Gi?m bet amount xu?ng 50%")
            recs.append("?? Ch? c??c v?o ph?ng confidence > 80%")
        elif score < 80:
            recs.append("?? Theo d?i ch?t ch?")
            recs.append("?? C?n nh?c gi?m bet n?u ti?p t?c thua")
        else:
            recs.append("? Ti?p t?c theo strategy hi?n t?i")
        
        if self.consecutive_losses >= 3:
            recs.append(f"?? ?? thua {self.consecutive_losses} v?n - C?n nh?c PAUSE")
        
        return recs
    
    def should_auto_pause(self) -> Tuple[bool, List[str]]:
        """
        Ki?m tra xem c? n?n t? ??ng pause kh?ng
        Returns: (should_pause, reasons)
        """
        reasons = []
        
        # Consecutive losses
        if self.consecutive_losses >= 5:
            reasons.append(f"Thua {self.consecutive_losses} v?n li?n ti?p")
        
        # Balance too low
        if self.starting_balance and len(self.balance_history) > 0:
            current = self.balance_history[-1]
            if current < self.starting_balance * 0.3:
                reasons.append("M?t > 70% v?n ban ??u")
        
        # Safety score too low
        if self.get_safety_score() < 30:
            reasons.append("Safety score < 30 (nguy hi?m)")
        
        return (len(reasons) > 0 and self.auto_pause_enabled, reasons)
