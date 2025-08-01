import numpy as np
from typing import List, Dict, Any
import random

class MABEnvironment:
    """
    Multi-Armed Bandit Environment
    """
    def __init__(self, n_arms: int, reward_distributions: List[Dict] = None, seed: int = None):
        """
        Initialize MAB environment
        
        Args:
            n_arms: Number of arms/actions
            reward_distributions: List of dicts with 'type' and parameters for each arm
            seed: Random seed for reproducibility
        """
        self.n_arms = n_arms
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # Default to Bernoulli distributions if none provided
        if reward_distributions is None:
            self.reward_distributions = [
                {'type': 'bernoulli', 'p': np.random.uniform(0.1, 0.9)} 
                for _ in range(n_arms)
            ]
        else:
            self.reward_distributions = reward_distributions
            
        # Track true expected rewards
        self.true_expected_rewards = self._compute_expected_rewards()
        
    def _compute_expected_rewards(self) -> List[float]:
        """Compute true expected rewards for each arm"""
        expected_rewards = []
        for dist in self.reward_distributions:
            if dist['type'] == 'bernoulli':
                expected_rewards.append(dist['p'])
            elif dist['type'] == 'normal':
                expected_rewards.append(dist['mu'])
            elif dist['type'] == 'uniform':
                expected_rewards.append((dist['low'] + dist['high']) / 2)
        return expected_rewards
    
    def pull(self, arm: int) -> float:
        """
        Pull an arm and get reward
        
        Args:
            arm: Arm index to pull
            
        Returns:
            Reward from the arm
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm {arm}. Must be 0 <= arm < {self.n_arms}")
            
        dist = self.reward_distributions[arm]
        
        if dist['type'] == 'bernoulli':
            return np.random.binomial(1, dist['p'])
        elif dist['type'] == 'normal':
            return np.random.normal(dist['mu'], dist['sigma'])
        elif dist['type'] == 'uniform':
            return np.random.uniform(dist['low'], dist['high'])
        else:
            raise ValueError(f"Unknown distribution type: {dist['type']}")
    
    def get_optimal_arm(self) -> int:
        """Get the arm with highest expected reward"""
        return np.argmax(self.true_expected_rewards)
    
    def get_optimal_reward(self) -> float:
        """Get the optimal expected reward"""
        return max(self.true_expected_rewards)
    
    def get_regret(self, arm: int) -> float:
        """Get regret for pulling a specific arm"""
        return self.get_optimal_reward() - self.true_expected_rewards[arm] 