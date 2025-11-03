"""
?? PERFORMANCE METRICS & STATISTICS v16.0
Real-time tracking and analysis of tool performance
"""

import time
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics


class PerformanceMetrics:
    """
    ?? PERFORMANCE METRICS TRACKER
    Track all performance metrics in real-time
    """
    
    def __init__(self):
        # Win/Loss tracking
        self.total_rounds = 0
        self.total_wins = 0
        self.total_losses = 0
        
        # Streak tracking
        self.current_streak = 0
        self.current_streak_type = None  # 'win' or 'loss'
        self.best_win_streak = 0
        self.worst_loss_streak = 0
        
        # Financial tracking
        self.total_profit = 0.0
        self.total_bet_amount = 0.0
        self.biggest_win = 0.0
        self.biggest_loss = 0.0
        self.roi = 0.0  # Return on Investment
        
        # Time tracking
        self.session_start_time = time.time()
        self.round_times = deque(maxlen=50)  # Last 50 round durations
        
        # Room performance
        self.room_stats = defaultdict(lambda: {
            'played': 0,
            'won': 0,
            'lost': 0,
            'win_rate': 0.0,
            'profit': 0.0
        })
        
        # AI confidence tracking
        self.confidence_history = deque(maxlen=100)
        self.high_confidence_wins = 0
        self.high_confidence_losses = 0
        
        # Recent performance (sliding window)
        self.recent_results = deque(maxlen=20)  # Last 20 rounds
        
    def record_round(self, 
                     won: bool,
                     room_id: int,
                     bet_amount: float,
                     profit_delta: float,
                     ai_confidence: float = 0.0,
                     round_duration: float = 0.0):
        """
        Ghi nh?n m?t v?n ch?i
        """
        self.total_rounds += 1
        
        # Win/Loss
        if won:
            self.total_wins += 1
            if self.current_streak_type == 'win':
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.current_streak_type = 'win'
            
            self.best_win_streak = max(self.best_win_streak, self.current_streak)
            
            if ai_confidence >= 0.8:
                self.high_confidence_wins += 1
        else:
            self.total_losses += 1
            if self.current_streak_type == 'loss':
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.current_streak_type = 'loss'
            
            self.worst_loss_streak = max(self.worst_loss_streak, self.current_streak)
            
            if ai_confidence >= 0.8:
                self.high_confidence_losses += 1
        
        # Financial
        self.total_bet_amount += bet_amount
        self.total_profit += profit_delta
        
        if profit_delta > 0:
            self.biggest_win = max(self.biggest_win, profit_delta)
        else:
            self.biggest_loss = min(self.biggest_loss, profit_delta)
        
        # ROI
        if self.total_bet_amount > 0:
            self.roi = (self.total_profit / self.total_bet_amount) * 100
        
        # Room stats
        room_stat = self.room_stats[room_id]
        room_stat['played'] += 1
        if won:
            room_stat['won'] += 1
        else:
            room_stat['lost'] += 1
        room_stat['win_rate'] = (room_stat['won'] / room_stat['played']) * 100
        room_stat['profit'] += profit_delta
        
        # Time
        if round_duration > 0:
            self.round_times.append(round_duration)
        
        # Confidence
        self.confidence_history.append(ai_confidence)
        
        # Recent results
        self.recent_results.append(won)
    
    def get_win_rate(self) -> float:
        """T? l? th?ng t?ng th?"""
        if self.total_rounds == 0:
            return 0.0
        return (self.total_wins / self.total_rounds) * 100
    
    def get_recent_win_rate(self) -> float:
        """T? l? th?ng 20 v?n g?n nh?t"""
        if len(self.recent_results) == 0:
            return 0.0
        wins = sum(1 for r in self.recent_results if r)
        return (wins / len(self.recent_results)) * 100
    
    def get_average_round_time(self) -> float:
        """Th?i gian trung b?nh m?i v?n (gi?y)"""
        if len(self.round_times) == 0:
            return 0.0
        return statistics.mean(self.round_times)
    
    def get_session_duration(self) -> float:
        """Th?i gian ch?i (gi?y)"""
        return time.time() - self.session_start_time
    
    def get_average_confidence(self) -> float:
        """?? tin c?y trung b?nh c?a AI"""
        if len(self.confidence_history) == 0:
            return 0.0
        return statistics.mean(self.confidence_history)
    
    def get_high_confidence_accuracy(self) -> float:
        """Accuracy khi AI c? confidence cao (>=80%)"""
        total = self.high_confidence_wins + self.high_confidence_losses
        if total == 0:
            return 0.0
        return (self.high_confidence_wins / total) * 100
    
    def get_best_room(self) -> Optional[int]:
        """Ph?ng c? win rate cao nh?t"""
        if not self.room_stats:
            return None
        
        best_room = None
        best_wr = 0.0
        
        for room_id, stats in self.room_stats.items():
            if stats['played'] >= 5 and stats['win_rate'] > best_wr:
                best_wr = stats['win_rate']
                best_room = room_id
        
        return best_room
    
    def get_worst_room(self) -> Optional[int]:
        """Ph?ng c? win rate th?p nh?t"""
        if not self.room_stats:
            return None
        
        worst_room = None
        worst_wr = 100.0
        
        for room_id, stats in self.room_stats.items():
            if stats['played'] >= 5 and stats['win_rate'] < worst_wr:
                worst_wr = stats['win_rate']
                worst_room = room_id
        
        return worst_room
    
    def get_summary(self) -> Dict[str, Any]:
        """L?y summary ??y ??"""
        duration = self.get_session_duration()
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        return {
            'session': {
                'duration_seconds': duration,
                'duration_text': f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m",
                'rounds_played': self.total_rounds,
                'avg_round_time': self.get_average_round_time()
            },
            'performance': {
                'total_wins': self.total_wins,
                'total_losses': self.total_losses,
                'win_rate': self.get_win_rate(),
                'recent_win_rate': self.get_recent_win_rate(),
                'current_streak': self.current_streak,
                'current_streak_type': self.current_streak_type,
                'best_win_streak': self.best_win_streak,
                'worst_loss_streak': self.worst_loss_streak
            },
            'financial': {
                'total_profit': self.total_profit,
                'total_bet': self.total_bet_amount,
                'roi': self.roi,
                'biggest_win': self.biggest_win,
                'biggest_loss': self.biggest_loss
            },
            'ai': {
                'avg_confidence': self.get_average_confidence(),
                'high_confidence_accuracy': self.get_high_confidence_accuracy(),
                'high_conf_wins': self.high_confidence_wins,
                'high_conf_losses': self.high_confidence_losses
            },
            'rooms': {
                'best_room': self.get_best_room(),
                'worst_room': self.get_worst_room(),
                'room_details': dict(self.room_stats)
            }
        }
    
    def get_performance_grade(self) -> str:
        """??nh gi? hi?u su?t (S, A, B, C, D, F)"""
        wr = self.get_win_rate()
        
        if wr >= 90:
            return "S"  # Legendary
        elif wr >= 80:
            return "A"  # Excellent
        elif wr >= 70:
            return "B"  # Good
        elif wr >= 60:
            return "C"  # Average
        elif wr >= 50:
            return "D"  # Below Average
        else:
            return "F"  # Poor
    
    def get_performance_trend(self) -> str:
        """Xu h??ng hi?u su?t (improving, declining, stable)"""
        if len(self.recent_results) < 10:
            return "insufficient_data"
        
        # So s?nh 10 v?n ??u vs 10 v?n cu?i
        first_half = list(self.recent_results)[:10]
        second_half = list(self.recent_results)[10:]
        
        first_wr = (sum(1 for r in first_half if r) / len(first_half)) * 100
        second_wr = (sum(1 for r in second_half if r) / len(second_half)) * 100
        
        diff = second_wr - first_wr
        
        if diff >= 10:
            return "improving"  # ?ang c?i thi?n
        elif diff <= -10:
            return "declining"  # ?ang gi?m s?t
        else:
            return "stable"  # ?n ??nh
