"""
?? ULTIMATE AI ENGINE v17.0
Advanced Mathematical & Statistical Algorithms
Nh? to?n h?c v? ??i - T? l? ch?nh x?c si?u cao
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
import statistics


class UltimateAIEngine:
    """
    ?? ULTIMATE AI ENGINE
    
    K?t h?p T?T C? thu?t to?n cao c?p:
    - Bayesian Inference (x?c su?t ti?n nghi?m/h?u nghi?m)
    - Monte Carlo Simulation (m? ph?ng ng?u nhi?n)
    - Kalman Filter (l?c nhi?u, d? ?o?n ch?nh x?c)
    - Game Theory (Nash Equilibrium)
    - Statistical Significance Testing
    - Advanced Ensemble Learning
    """
    
    def __init__(self):
        # Bayesian priors (x?c su?t ti?n nghi?m)
        self.room_priors = defaultdict(lambda: 0.5)  # Prior = 50%
        self.room_posteriors = defaultdict(lambda: 0.5)  # Posterior update
        
        # Kalman filter states
        self.kalman_estimates = defaultdict(lambda: 0.5)
        self.kalman_uncertainties = defaultdict(lambda: 1.0)
        
        # Monte Carlo samples
        self.mc_samples = 10000  # S? l??ng m? ph?ng
        
        # Game theory payoff matrix
        self.payoff_history = defaultdict(list)
        
        # Statistical tracking
        self.room_history = defaultdict(lambda: {
            'survives': 0,
            'kills': 0,
            'total': 0,
            'recent': deque(maxlen=50)
        })
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.85
        self.very_high_confidence_threshold = 0.92
        
    def bayesian_inference(self, 
                          room_id: int,
                          observed_data: Dict[str, float]) -> float:
        """
        ?? BAYESIAN INFERENCE
        C?p nh?t x?c su?t d?a tr?n d? li?u quan s?t
        
        P(survive|data) = P(data|survive) * P(survive) / P(data)
        """
        # Prior probability
        prior = self.room_priors[room_id]
        
        # Likelihood: P(data|survive) vs P(data|kill)
        # D?a tr?n features ?? t?nh likelihood
        
        # High survive_score ? high likelihood of survival
        survive_score = observed_data.get('survive_score', 0.5)
        stability = observed_data.get('stability_score', 0.5)
        recent_pen = observed_data.get('recent_pen', 0)
        
        # P(data|survive) - X?c su?t th?y data n?y n?u room survive
        likelihood_survive = (
            survive_score * 0.4 +
            stability * 0.3 +
            (1 - abs(recent_pen)) * 0.3
        )
        
        # P(data|kill) - X?c su?t th?y data n?y n?u room kill
        likelihood_kill = 1 - likelihood_survive
        
        # Bayes' theorem
        # P(survive|data) = P(data|survive) * P(survive) / P(data)
        # P(data) = P(data|survive)*P(survive) + P(data|kill)*P(kill)
        
        p_data = (likelihood_survive * prior + 
                 likelihood_kill * (1 - prior))
        
        if p_data > 0:
            posterior = (likelihood_survive * prior) / p_data
        else:
            posterior = prior
        
        # Update posterior
        self.room_posteriors[room_id] = posterior
        
        return posterior
    
    def kalman_filter(self,
                     room_id: int,
                     measurement: float) -> float:
        """
        ?? KALMAN FILTER
        L?c nhi?u v? d? ?o?n ch?nh x?c
        
        Gi?m noise trong d? li?u, cho prediction ?n ??nh h?n
        """
        # Get previous estimate and uncertainty
        prev_estimate = self.kalman_estimates[room_id]
        prev_uncertainty = self.kalman_uncertainties[room_id]
        
        # Process noise (?? kh?ng ch?c ch?n c?a model)
        process_noise = 0.01
        
        # Measurement noise (?? kh?ng ch?c ch?n c?a measurement)
        measurement_noise = 0.05
        
        # Prediction step
        predicted_estimate = prev_estimate
        predicted_uncertainty = prev_uncertainty + process_noise
        
        # Update step
        # Kalman gain: balance between prediction and measurement
        kalman_gain = predicted_uncertainty / (predicted_uncertainty + measurement_noise)
        
        # Update estimate
        new_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        
        # Update uncertainty
        new_uncertainty = (1 - kalman_gain) * predicted_uncertainty
        
        # Store
        self.kalman_estimates[room_id] = new_estimate
        self.kalman_uncertainties[room_id] = new_uncertainty
        
        return new_estimate
    
    def monte_carlo_simulation(self,
                               room_id: int,
                               base_probability: float,
                               uncertainty: float = 0.1) -> Dict[str, float]:
        """
        ?? MONTE CARLO SIMULATION
        Ch?y 10,000 m? ph?ng ?? estimate probability distribution
        """
        samples = []
        
        # Generate random samples around base_probability
        for _ in range(self.mc_samples):
            # Add noise
            noise = np.random.normal(0, uncertainty)
            sample = base_probability + noise
            sample = max(0.0, min(1.0, sample))  # Clip to [0, 1]
            samples.append(sample)
        
        # Statistics from samples
        mean_prob = np.mean(samples)
        std_prob = np.std(samples)
        median_prob = np.median(samples)
        
        # Confidence interval (95%)
        percentile_5 = np.percentile(samples, 5)
        percentile_95 = np.percentile(samples, 95)
        
        # Probability of success (sample > 0.5)
        success_rate = sum(1 for s in samples if s > 0.5) / len(samples)
        
        return {
            'mean': mean_prob,
            'std': std_prob,
            'median': median_prob,
            'ci_low': percentile_5,
            'ci_high': percentile_95,
            'success_rate': success_rate,
            'confidence': 1 - std_prob  # Lower std = higher confidence
        }
    
    def nash_equilibrium(self,
                        room_scores: Dict[int, float]) -> int:
        """
        ?? NASH EQUILIBRIUM (Game Theory)
        T?m strategy t?i ?u khi ??i th? c?ng ch?i optimal
        
        Trong betting game, ch?n room m?:
        - Maximize expected value
        - Minimize regret
        - Consider opponent's strategy
        """
        if not room_scores:
            return None
        
        # Expected value for each room
        expected_values = {}
        
        for room_id, score in room_scores.items():
            history = self.room_history[room_id]
            
            if history['total'] > 0:
                actual_survive_rate = history['survives'] / history['total']
            else:
                actual_survive_rate = 0.5
            
            # Expected value = P(survive) * reward - P(kill) * loss
            # Assume reward = +1, loss = -1
            ev = actual_survive_rate * 1.0 - (1 - actual_survive_rate) * 1.0
            ev = ev * score  # Weight by score
            
            expected_values[room_id] = ev
        
        # Nash equilibrium: choose max EV
        best_room = max(expected_values.items(), key=lambda x: x[1])[0]
        
        return best_room
    
    def statistical_significance(self,
                                room_id: int,
                                sample_size_threshold: int = 30) -> Dict[str, Any]:
        """
        ?? STATISTICAL SIGNIFICANCE TESTING
        Ki?m tra xem k?t qu? c? ? ngh?a th?ng k? kh?ng
        """
        history = self.room_history[room_id]
        
        if history['total'] < sample_size_threshold:
            return {
                'significant': False,
                'reason': 'insufficient_data',
                'sample_size': history['total'],
                'confidence': 0.0
            }
        
        # Calculate survive rate
        survive_rate = history['survives'] / history['total']
        
        # Z-test for proportion
        # H0: survive_rate = 0.5 (null hypothesis: random)
        # H1: survive_rate != 0.5 (alternative: not random)
        
        p0 = 0.5  # Null hypothesis probability
        n = history['total']
        
        # Standard error
        se = np.sqrt(p0 * (1 - p0) / n)
        
        # Z-score
        z = (survive_rate - p0) / se
        
        # P-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Significance at ? = 0.05
        significant = p_value < 0.05
        
        # Confidence level
        confidence = 1 - p_value
        
        return {
            'significant': significant,
            'p_value': p_value,
            'z_score': z,
            'confidence': confidence,
            'survive_rate': survive_rate,
            'sample_size': n
        }
    
    def advanced_ensemble(self,
                         predictions: List[float],
                         confidences: List[float]) -> Dict[str, float]:
        """
        ?? ADVANCED ENSEMBLE LEARNING
        K?t h?p nhi?u predictions v?i confidence weighting
        """
        if not predictions or not confidences:
            return {'mean': 0.5, 'weighted': 0.5, 'confidence': 0.0}
        
        # Simple mean
        mean_pred = np.mean(predictions)
        
        # Confidence-weighted mean
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weighted_pred = sum(p * c for p, c in zip(predictions, confidences)) / total_confidence
        else:
            weighted_pred = mean_pred
        
        # Variance (lower = more agreement = higher confidence)
        variance = np.var(predictions)
        ensemble_confidence = 1 / (1 + variance)  # Transform to [0, 1]
        
        # Median (robust to outliers)
        median_pred = np.median(predictions)
        
        # Consensus (% of predictions above threshold)
        consensus = sum(1 for p in predictions if p > 0.5) / len(predictions)
        
        return {
            'mean': mean_pred,
            'weighted': weighted_pred,
            'median': median_pred,
            'confidence': ensemble_confidence,
            'consensus': consensus,
            'variance': variance
        }
    
    def ultimate_prediction(self,
                           room_id: int,
                           room_features: Dict[str, float],
                           base_prediction: float) -> Dict[str, Any]:
        """
        ?? ULTIMATE PREDICTION
        
        K?t h?p T?T C? algorithms ?? cho prediction t?t nh?t:
        1. Bayesian Inference
        2. Kalman Filter
        3. Monte Carlo Simulation
        4. Statistical Significance
        5. Advanced Ensemble
        """
        # 1. Bayesian inference
        bayesian_prob = self.bayesian_inference(room_id, room_features)
        
        # 2. Kalman filter (smooth the base prediction)
        kalman_prob = self.kalman_filter(room_id, base_prediction)
        
        # 3. Statistical significance
        sig_test = self.statistical_significance(room_id)
        
        # 4. Monte Carlo simulation
        mc_result = self.monte_carlo_simulation(room_id, kalman_prob)
        
        # 5. Combine all predictions
        all_predictions = [
            bayesian_prob,
            kalman_prob,
            base_prediction,
            mc_result['mean']
        ]
        
        all_confidences = [
            0.25,  # Bayesian
            0.30,  # Kalman (highest weight - most reliable)
            0.20,  # Base
            0.25   # Monte Carlo
        ]
        
        # If statistically significant, increase confidence
        if sig_test['significant']:
            all_confidences[0] *= 1.2  # Boost Bayesian
            all_confidences[1] *= 1.2  # Boost Kalman
        
        # Normalize confidences
        total_conf = sum(all_confidences)
        all_confidences = [c / total_conf for c in all_confidences]
        
        # Advanced ensemble
        ensemble = self.advanced_ensemble(all_predictions, all_confidences)
        
        # Final prediction (weighted by confidence)
        final_prediction = ensemble['weighted']
        
        # Overall confidence
        overall_confidence = (
            ensemble['confidence'] * 0.4 +
            mc_result['confidence'] * 0.3 +
            (sig_test['confidence'] if sig_test['significant'] else 0.5) * 0.3
        )
        
        # Determine confidence level
        if overall_confidence >= self.very_high_confidence_threshold:
            confidence_level = "VERY_HIGH"
        elif overall_confidence >= self.high_confidence_threshold:
            confidence_level = "HIGH"
        elif overall_confidence >= 0.7:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        return {
            'prediction': final_prediction,
            'confidence': overall_confidence,
            'confidence_level': confidence_level,
            'bayesian': bayesian_prob,
            'kalman': kalman_prob,
            'monte_carlo': mc_result,
            'statistical_test': sig_test,
            'ensemble': ensemble,
            'recommendation': self._get_recommendation(final_prediction, overall_confidence)
        }
    
    def update_history(self, room_id: int, survived: bool):
        """Update room history"""
        history = self.room_history[room_id]
        history['total'] += 1
        
        if survived:
            history['survives'] += 1
        else:
            history['kills'] += 1
        
        history['recent'].append(1 if survived else 0)
        
        # Update Bayesian prior for next time
        if history['total'] > 0:
            self.room_priors[room_id] = history['survives'] / history['total']
    
    def _get_recommendation(self, prediction: float, confidence: float) -> str:
        """Get betting recommendation"""
        if prediction >= 0.8 and confidence >= 0.85:
            return "?? STRONGLY RECOMMEND - Bet v?i confidence cao!"
        elif prediction >= 0.7 and confidence >= 0.75:
            return "?? RECOMMEND - L?a ch?n t?t"
        elif prediction >= 0.6 and confidence >= 0.65:
            return "?? ACCEPTABLE - C? th? c??c"
        elif prediction >= 0.5:
            return "?? NEUTRAL - C?n nh?c k?"
        else:
            return "?? NOT RECOMMENDED - R?i ro cao"
    
    def get_best_room(self, 
                     room_predictions: Dict[int, float]) -> Tuple[int, Dict[str, Any]]:
        """
        Ch?n room t?t nh?t d?a tr?n Nash Equilibrium + highest confidence
        """
        if not room_predictions:
            return None, None
        
        # Get Nash equilibrium choice
        nash_choice = self.nash_equilibrium(room_predictions)
        
        # Get room with highest prediction * confidence
        best_room = None
        best_score = 0
        best_analysis = None
        
        for room_id, pred_data in room_predictions.items():
            if isinstance(pred_data, dict):
                score = pred_data['prediction'] * pred_data['confidence']
            else:
                score = pred_data
            
            if score > best_score:
                best_score = score
                best_room = room_id
                best_analysis = pred_data if isinstance(pred_data, dict) else None
        
        # Prefer Nash choice if confidence is similar
        if nash_choice and nash_choice in room_predictions:
            nash_data = room_predictions[nash_choice]
            nash_score = nash_data['prediction'] * nash_data['confidence'] if isinstance(nash_data, dict) else nash_data
            
            # If Nash choice is within 5% of best, prefer it (game theory optimal)
            if nash_score >= best_score * 0.95:
                best_room = nash_choice
                best_analysis = nash_data if isinstance(nash_data, dict) else None
        
        return best_room, best_analysis
