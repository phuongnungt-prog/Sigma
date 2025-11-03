"""
?? ULTRA AI ALGORITHMS v15.0 ??
Thu?t to?n AI cao c?p nh?t - SI?U TR? TU?
"""

import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
import statistics


class AdvancedNeuralNetwork:
    """
    ?? M?NG NEURAL TI?N TI?N
    Multi-layer perceptron v?i backpropagation v? adaptive learning
    """
    
    def __init__(self, input_size: int = 20, hidden_layers: List[int] = None):
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]  # 3 hidden layers
        
        self.layers = [input_size] + hidden_layers + [1]  # Output: probability
        self.weights = []
        self.biases = []
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.velocity_w = []
        self.velocity_b = []
        
        # Initialize weights (Xavier initialization)
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # ReLU for hidden layers, sigmoid for output
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        """Backward propagation with momentum"""
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer error
        deltas[-1] = output - y
        
        # Backpropagate errors
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(deltas[i+1], self.weights[i+1].T) * self.relu_derivative(self.z_values[i])
            deltas[i] = delta
        
        # Update weights with momentum
        for i in range(len(self.weights)):
            dw = np.dot(self.activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            # Momentum update
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dw
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db
            
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]
    
    def train(self, X, y):
        """Train the network"""
        X = np.array(X).reshape(1, -1)
        y = np.array([[y]])
        
        output = self.forward(X)
        self.backward(X, y, output)
        
        return output[0][0]
    
    def predict(self, X):
        """Predict probability"""
        X = np.array(X).reshape(1, -1)
        return self.forward(X)[0][0]


class BayesianOptimizer:
    """
    ?? BAYESIAN OPTIMIZATION
    T?i ?u h?a chi?n l??c d?a tr?n x?c su?t Bayes
    """
    
    def __init__(self):
        self.observations = []
        self.prior = {
            "alpha": 1.0,  # Prior successes
            "beta": 1.0    # Prior failures
        }
    
    def update(self, success: bool):
        """C?p nh?t Bayesian posterior"""
        if success:
            self.prior["alpha"] += 1
        else:
            self.prior["beta"] += 1
        
        self.observations.append(1 if success else 0)
    
    def get_probability(self) -> float:
        """L?y x?c su?t ??c t?nh (Beta distribution mean)"""
        return self.prior["alpha"] / (self.prior["alpha"] + self.prior["beta"])
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """L?y confidence interval"""
        from scipy import stats
        alpha = self.prior["alpha"]
        beta = self.prior["beta"]
        
        lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
        upper = stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
        
        return (lower, upper)


class TimeSeriesPredictor:
    """
    ?? TIME SERIES PREDICTION
    D? ?o?n xu h??ng d?a tr?n chu?i th?i gian v?i ARIMA-like model
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.trend_weights = np.array([0.4, 0.3, 0.2, 0.1])  # Recent more important
    
    def add_observation(self, value: float):
        """Th?m quan s?t m?i"""
        self.history.append(value)
    
    def predict_next(self) -> float:
        """D? ?o?n gi? tr? ti?p theo"""
        if len(self.history) < 2:
            return 0.5
        
        # Exponential moving average
        history_array = np.array(list(self.history))
        
        # Calculate trend
        if len(history_array) >= 4:
            recent_trend = np.dot(history_array[-4:], self.trend_weights)
        else:
            recent_trend = np.mean(history_array)
        
        # Calculate volatility
        if len(history_array) >= 2:
            volatility = np.std(history_array)
        else:
            volatility = 0
        
        # Adjust prediction based on trend and volatility
        prediction = recent_trend * (1 - volatility * 0.1)
        
        return max(0, min(1, prediction))
    
    def detect_pattern(self) -> str:
        """Ph?t hi?n pattern: trending up, down, or stable"""
        if len(self.history) < 3:
            return "unknown"
        
        history_array = np.array(list(self.history))
        
        # Calculate slope
        x = np.arange(len(history_array))
        slope = np.polyfit(x, history_array, 1)[0]
        
        if slope > 0.05:
            return "up"
        elif slope < -0.05:
            return "down"
        else:
            return "stable"


class EnsemblePredictor:
    """
    ?? ENSEMBLE LEARNING
    K?t h?p nhi?u models ?? d? ?o?n ch?nh x?c h?n
    """
    
    def __init__(self):
        self.models = {
            "neural": AdvancedNeuralNetwork(),
            "bayesian": BayesianOptimizer(),
            "timeseries": TimeSeriesPredictor()
        }
        self.model_weights = {
            "neural": 0.4,
            "bayesian": 0.35,
            "timeseries": 0.25
        }
        self.model_performance = {
            "neural": deque(maxlen=50),
            "bayesian": deque(maxlen=50),
            "timeseries": deque(maxlen=50)
        }
    
    def predict(self, features: List[float]) -> float:
        """D? ?o?n ensemble"""
        predictions = {}
        
        # Neural network prediction
        try:
            predictions["neural"] = self.models["neural"].predict(features)
        except:
            predictions["neural"] = 0.5
        
        # Bayesian prediction
        predictions["bayesian"] = self.models["bayesian"].get_probability()
        
        # Time series prediction
        predictions["timeseries"] = self.models["timeseries"].predict_next()
        
        # Weighted average
        ensemble_pred = sum(
            predictions[model] * self.model_weights[model]
            for model in predictions
        )
        
        return max(0, min(1, ensemble_pred))
    
    def update(self, features: List[float], result: bool):
        """C?p nh?t t?t c? models"""
        # Update neural network
        try:
            self.models["neural"].train(features, 1.0 if result else 0.0)
        except:
            pass
        
        # Update Bayesian
        self.models["bayesian"].update(result)
        
        # Update time series
        self.models["timeseries"].add_observation(1.0 if result else 0.0)
        
        # Track performance (for adaptive weights)
        for model in self.models:
            self.model_performance[model].append(1 if result else 0)
        
        # Adaptive weights based on recent performance
        self._update_weights()
    
    def _update_weights(self):
        """T? ??ng ?i?u ch?nh weights d?a tr?n performance"""
        if all(len(perf) >= 10 for perf in self.model_performance.values()):
            performances = {}
            for model, perf in self.model_performance.items():
                performances[model] = np.mean(list(perf))
            
            # Softmax to get weights
            total_perf = sum(performances.values())
            if total_perf > 0:
                for model in self.model_weights:
                    self.model_weights[model] = performances[model] / total_perf


class GeneticAlgorithm:
    """
    ?? GENETIC ALGORITHM
    Ti?n h?a chi?n l??c qua nhi?u th? h?
    """
    
    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = -float('inf')
    
    def initialize_population(self, gene_length: int = 10):
        """Kh?i t?o qu?n th?"""
        self.population = [
            {
                "genes": np.random.rand(gene_length),
                "fitness": 0.0
            }
            for _ in range(self.population_size)
        ]
    
    def evaluate_fitness(self, individual: Dict, result: bool):
        """??nh gi? fitness"""
        if result:
            individual["fitness"] += 1
        else:
            individual["fitness"] -= 0.5
        
        # Update best
        if individual["fitness"] > self.best_fitness:
            self.best_fitness = individual["fitness"]
            self.best_individual = individual.copy()
    
    def selection(self) -> List[Dict]:
        """Ch?n l?c t? nhi?n - tournament selection"""
        selected = []
        for _ in range(self.population_size // 2):
            tournament = np.random.choice(self.population, size=3, replace=False)
            winner = max(tournament, key=lambda x: x["fitness"])
            selected.append(winner)
        return selected
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Lai gh?p"""
        crossover_point = len(parent1["genes"]) // 2
        child_genes = np.concatenate([
            parent1["genes"][:crossover_point],
            parent2["genes"][crossover_point:]
        ])
        return {"genes": child_genes, "fitness": 0.0}
    
    def mutate(self, individual: Dict, mutation_rate: float = 0.1):
        """??t bi?n"""
        for i in range(len(individual["genes"])):
            if np.random.rand() < mutation_rate:
                individual["genes"][i] = np.random.rand()
    
    def evolve(self):
        """Ti?n h?a m?t th? h?"""
        # Selection
        parents = self.selection()
        
        # Crossover
        children = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child = self.crossover(parents[i], parents[i+1])
                self.mutate(child)
                children.append(child)
        
        # Replace population (elitism: keep best 10%)
        elite_count = self.population_size // 10
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        self.population = self.population[:elite_count] + children
        
        # Fill up if needed
        while len(self.population) < self.population_size:
            self.population.append({
                "genes": np.random.rand(len(self.population[0]["genes"])),
                "fitness": 0.0
            })
        
        self.generation += 1


class ReinforcementLearner:
    """
    ?? REINFORCEMENT LEARNING
    Q-Learning v?i experience replay
    """
    
    def __init__(self, state_size: int = 20, action_size: int = 8):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.experience_replay = deque(maxlen=1000)
    
    def get_state_key(self, state: List[float]) -> str:
        """Convert state to hashable key"""
        return str([round(x, 2) for x in state])
    
    def get_q_value(self, state: str, action: int) -> float:
        """Get Q value"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size
        return self.q_table[state][action]
    
    def choose_action(self, state: List[float]) -> int:
        """Epsilon-greedy action selection"""
        state_key = self.get_state_key(state)
        
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(self.action_size)
        else:
            # Exploit
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * self.action_size
            return np.argmax(self.q_table[state_key])
    
    def update(self, state: List[float], action: int, reward: float, next_state: List[float]):
        """Update Q-value"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Store experience
        self.experience_replay.append((state_key, action, reward, next_state_key))
        
        # Q-learning update
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.action_size
        
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        # Experience replay (learn from past)
        if len(self.experience_replay) >= 32:
            batch = np.random.choice(len(self.experience_replay), 32, replace=False)
            for idx in batch:
                s, a, r, ns = self.experience_replay[idx]
                self._replay_update(s, a, r, ns)
    
    def _replay_update(self, state_key: str, action: int, reward: float, next_state_key: str):
        """Update from experience replay"""
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * 0.5 * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
