from typing import Dict, Any, List
import numpy as np

class MABConfig:
    """
    Configuration class for MAB experiments
    """
    def __init__(self):
        # Environment settings
        self.n_arms = 10
        self.n_trials = 1000
        self.seed = 42
        
        # Reward distributions (optional - will use default Bernoulli if None)
        self.reward_distributions = None
        
        # Algorithm-specific parameters
        self.algorithm_params = {
            'epsilon_greedy': {'epsilon': 0.1},
            'ucb': {'c': 2.0},
            'exploration_only': {},
            'exploitation_only': {}
        }
        
    def get_bernoulli_config(self, n_arms: int = 10, n_trials: int = 1000):
        """Get configuration for Bernoulli bandits"""
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.reward_distributions = [
            {'type': 'bernoulli', 'p': np.random.uniform(0.1, 0.9)} 
            for _ in range(n_arms)
        ]
        return self
    
    def get_normal_config(self, n_arms: int = 10, n_trials: int = 1000):
        """Get configuration for Normal bandits"""
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.reward_distributions = [
            {'type': 'normal', 'mu': np.random.uniform(0, 1), 'sigma': 0.1} 
            for _ in range(n_arms)
        ]
        return self
    
    def set_algorithm_params(self, algorithm: str, params: Dict[str, Any]):
        """Set parameters for a specific algorithm"""
        self.algorithm_params[algorithm] = params 