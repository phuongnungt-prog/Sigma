"""
?? AI DECISION EXPLAINER v16.0
Gi?i th?ch chi ti?t t?i sao AI ch?n ph?ng ??
"""

from typing import Dict, List, Any, Tuple
from collections import defaultdict


class AIExplainer:
    """
    ?? AI DECISION EXPLAINER
    Gi?i th?ch chi ti?t quy?t ??nh c?a AI
    """
    
    def __init__(self):
        self.decision_history = []
        
    def explain_decision(self,
                        chosen_room: int,
                        room_scores: Dict[int, float],
                        room_features: Dict[int, Dict[str, float]],
                        ai_votes: List[Tuple[int, int]],  # [(agent_id, room_voted)]
                        learning_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        T?o gi?i th?ch chi ti?t cho quy?t ??nh
        """
        # 1. Vote Analysis
        vote_counts = defaultdict(int)
        for _, room in ai_votes:
            vote_counts[room] += 1
        
        total_votes = len(ai_votes)
        chosen_vote_pct = (vote_counts.get(chosen_room, 0) / total_votes) * 100 if total_votes > 0 else 0
        
        # 2. Feature Analysis
        chosen_features = room_features.get(chosen_room, {})
        
        # Top 3 strongest features
        top_features = sorted(
            chosen_features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        # 3. Comparison with other rooms
        alternatives = []
        for room_id, score in sorted(room_scores.items(), key=lambda x: x[1], reverse=True):
            if room_id != chosen_room:
                alternatives.append({
                    'room_id': room_id,
                    'score': score,
                    'score_diff': room_scores[chosen_room] - score,
                    'vote_pct': (vote_counts.get(room_id, 0) / total_votes) * 100 if total_votes > 0 else 0
                })
                if len(alternatives) >= 2:  # Top 2 alternatives
                    break
        
        # 4. Risk Assessment
        risk_level = self._assess_risk(chosen_features, room_scores[chosen_room])
        
        # 5. Learning Insights (n?u c?)
        learning_insights = []
        if learning_data:
            if 'learned_survival_rate' in learning_data:
                rate = learning_data['learned_survival_rate']
                learning_insights.append(
                    f"Ph?ng n?y survive {rate:.0%} trong l?ch s? ({learning_data.get('total_rounds', 0)} v?n)"
                )
            
            if 'pattern_warning' in learning_data:
                learning_insights.append(f"?? {learning_data['pattern_warning']}")
            
            if 'similar_situation' in learning_data:
                sim = learning_data['similar_situation']
                learning_insights.append(
                    f"T?nh hu?ng t??ng t?: {sim['outcome']} ({sim['similarity']:.0%} gi?ng)"
                )
        
        # 6. Confidence explanation
        confidence_factors = self._explain_confidence(
            room_scores[chosen_room],
            chosen_vote_pct,
            risk_level,
            learning_data
        )
        
        explanation = {
            'chosen_room': chosen_room,
            'final_score': room_scores[chosen_room],
            'confidence': self._calculate_confidence(room_scores[chosen_room], chosen_vote_pct),
            
            'vote_analysis': {
                'chosen_votes': vote_counts.get(chosen_room, 0),
                'total_votes': total_votes,
                'vote_percentage': chosen_vote_pct,
                'consensus': 'strong' if chosen_vote_pct >= 60 else 'moderate' if chosen_vote_pct >= 40 else 'weak'
            },
            
            'feature_analysis': {
                'top_strengths': [
                    {'feature': name, 'value': value, 'interpretation': self._interpret_feature(name, value)}
                    for name, value in top_features
                ]
            },
            
            'risk_assessment': {
                'level': risk_level,
                'description': self._get_risk_description(risk_level)
            },
            
            'alternatives': alternatives,
            
            'learning_insights': learning_insights,
            
            'confidence_factors': confidence_factors,
            
            'recommendation': self._generate_recommendation(
                room_scores[chosen_room],
                chosen_vote_pct,
                risk_level,
                learning_insights
            )
        }
        
        # L?u v?o history
        self.decision_history.append({
            'room': chosen_room,
            'score': room_scores[chosen_room],
            'confidence': explanation['confidence'],
            'risk': risk_level
        })
        
        return explanation
    
    def _assess_risk(self, features: Dict[str, float], final_score: float) -> str:
        """??nh gi? m?c ?? r?i ro"""
        risk_score = 0
        
        # Check negative indicators
        if features.get('recent_pen', 0) < -0.5:
            risk_score += 2  # V?a b? kill g?n ??y
        
        if features.get('players_norm', 0.5) > 0.8:
            risk_score += 1  # Qu? ??ng ng??i
        
        if features.get('volatility_score', 0.5) > 0.7:
            risk_score += 1  # Bi?n ??ng cao
        
        if final_score < 0.6:
            risk_score += 1  # Score th?p
        
        # Assess
        if risk_score >= 4:
            return "very_high"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 2:
            return "moderate"
        elif risk_score >= 1:
            return "low"
        else:
            return "very_low"
    
    def _get_risk_description(self, risk_level: str) -> str:
        """M? t? risk level"""
        descriptions = {
            'very_low': '? C?c k? an to?n - Confidence cao',
            'low': '? An to?n - R?i ro th?p',
            'moderate': '?? Trung b?nh - C?n nh?c c?n th?n',
            'high': '?? Cao - C?n th?n!',
            'very_high': '?? R?t cao - Kh?ng khuy?n ngh?!'
        }
        return descriptions.get(risk_level, 'Unknown')
    
    def _calculate_confidence(self, final_score: float, vote_pct: float) -> float:
        """T?nh confidence t?ng h?p"""
        # K?t h?p score v? vote percentage
        confidence = (final_score * 0.6 + (vote_pct / 100) * 0.4)
        return max(0.0, min(1.0, confidence))
    
    def _explain_confidence(self,
                           final_score: float,
                           vote_pct: float,
                           risk_level: str,
                           learning_data: Dict[str, Any] = None) -> List[str]:
        """Gi?i th?ch c?c y?u t? ?nh h??ng confidence"""
        factors = []
        
        # Score
        if final_score >= 0.8:
            factors.append(f"? Score cao ({final_score:.1%})")
        elif final_score >= 0.6:
            factors.append(f"?? Score trung b?nh ({final_score:.1%})")
        else:
            factors.append(f"?? Score th?p ({final_score:.1%})")
        
        # Votes
        if vote_pct >= 60:
            factors.append(f"? ??ng thu?n m?nh ({vote_pct:.0f}% votes)")
        elif vote_pct >= 40:
            factors.append(f"?? ??ng thu?n v?a ({vote_pct:.0f}% votes)")
        else:
            factors.append(f"?? ??ng thu?n y?u ({vote_pct:.0f}% votes)")
        
        # Risk
        if risk_level in ['very_low', 'low']:
            factors.append("? R?i ro th?p")
        elif risk_level == 'moderate':
            factors.append("?? R?i ro trung b?nh")
        else:
            factors.append("?? R?i ro cao")
        
        # Learning data
        if learning_data and 'learned_survival_rate' in learning_data:
            rate = learning_data['learned_survival_rate']
            if rate >= 0.7:
                factors.append(f"? L?ch s? t?t ({rate:.0%} survive)")
            elif rate >= 0.5:
                factors.append(f"?? L?ch s? trung b?nh ({rate:.0%})")
            else:
                factors.append(f"?? L?ch s? x?u ({rate:.0%})")
        
        return factors
    
    def _interpret_feature(self, feature_name: str, value: float) -> str:
        """Gi?i th?ch ? ngh?a c?a feature"""
        interpretations = {
            'survive_score': 'T? l? s?ng s?t' if value > 0.5 else 'T? l? s?ng s?t th?p',
            'stability_score': '?n ??nh' if value > 0.5 else 'Kh?ng ?n ??nh',
            'recent_pen': 'Kh?ng b? kill g?n ??y' if value > -0.3 else 'V?a b? kill',
            'players_norm': 'S? ng??i v?a ph?i' if 0.3 < value < 0.7 else 'S? ng??i b?t th??ng',
            'hot_score': '?ang hot' if value > 0.6 else 'B?nh th??ng',
            'cold_score': 'L?nh (an to?n)' if value > 0.6 else 'B?nh th??ng',
        }
        return interpretations.get(feature_name, 'N/A')
    
    def _generate_recommendation(self,
                                 final_score: float,
                                 vote_pct: float,
                                 risk_level: str,
                                 learning_insights: List[str]) -> str:
        """T?o recommendation t?ng h?p"""
        if final_score >= 0.8 and vote_pct >= 60 and risk_level in ['very_low', 'low']:
            return "?? HIGHLY RECOMMENDED - L?a ch?n tuy?t v?i! AI r?t t? tin."
        
        elif final_score >= 0.7 and vote_pct >= 50:
            return "?? RECOMMENDED - L?a ch?n t?t, n?n ??t c??c."
        
        elif final_score >= 0.6 and vote_pct >= 40:
            return "?? ACCEPTABLE - L?a ch?n ch?p nh?n ???c, c?n nh?c."
        
        elif risk_level in ['high', 'very_high']:
            return "?? RISKY - R?i ro cao! C?n th?n!"
        
        else:
            return "?? MODERATE - L?a ch?n trung b?nh, c?n nh?c th?m."
    
    def format_explanation_panel(self, explanation: Dict[str, Any]) -> str:
        """
        Format th?nh text ??p ?? hi?n th?
        """
        lines = []
        
        # Header
        lines.append(f"?? QUY?T ??NH: PH?NG {explanation['chosen_room']}")
        lines.append(f"?? Score: {explanation['final_score']:.1%} | Confidence: {explanation['confidence']:.1%}")
        lines.append("")
        
        # Recommendation
        lines.append(explanation['recommendation'])
        lines.append("")
        
        # Vote Analysis
        vote = explanation['vote_analysis']
        lines.append(f"???  ??NG THU?N: {vote['chosen_votes']}/{vote['total_votes']} votes ({vote['vote_percentage']:.0f}%) - {vote['consensus'].upper()}")
        lines.append("")
        
        # Risk
        risk = explanation['risk_assessment']
        lines.append(f"??  R?I RO: {risk['level'].upper()} - {risk['description']}")
        lines.append("")
        
        # Top Features
        lines.append("?? ?I?M M?NH:")
        for feat in explanation['feature_analysis']['top_strengths']:
            lines.append(f"  ? {feat['feature']}: {feat['value']:.3f} ({feat['interpretation']})")
        lines.append("")
        
        # Learning Insights
        if explanation['learning_insights']:
            lines.append("?? AI ?? H?C:")
            for insight in explanation['learning_insights']:
                lines.append(f"  ? {insight}")
            lines.append("")
        
        # Confidence Factors
        lines.append("?? Y?U T?:")
        for factor in explanation['confidence_factors']:
            lines.append(f"  {factor}")
        
        return "\n".join(lines)
