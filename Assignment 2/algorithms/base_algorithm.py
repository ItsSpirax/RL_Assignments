from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any
from environment.mab_environment import MABEnvironment

class BaseMABAlgorithm(ABC):
    """
    Base class for all MAB algorithms
    """
    def __init__(self, n_arms: int, **kwargs):
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms, dtype=int)  # Number of times each arm pulled
        self.rewards = np.zeros(n_arms)  # Cumulative rewards for each arm
        self.estimates = np.zeros(n_arms)  # Current estimates of expected rewards
        
    @abstractmethod
    def select_arm(self) -> int:
        """
        Select which arm to pull next
        
        Returns:
            Index of the arm to pull
        """
        pass
    
    def update(self, arm: int, reward: float):
        """
        Update algorithm with observed reward
        
        Args:
            arm: Arm that was pulled
            reward: Observed reward
        """
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        
        # Update estimate (sample mean)
        self.estimates[arm] = self.rewards[arm] / self.pulls[arm]
    
    def get_estimated_optimal_arm(self) -> int:
        """Get arm with highest estimated reward"""
        return np.argmax(self.estimates)
    
    def get_cumulative_regret(self, environment: MABEnvironment, history: List[int]) -> float:
        """Calculate cumulative regret"""
        optimal_reward = environment.get_optimal_reward()
        cumulative_reward = sum(environment.true_expected_rewards[arm] for arm in history)
        return optimal_reward * len(history) - cumulative_reward
    
    def reset(self):
        """Reset algorithm state"""
        self.pulls = np.zeros(self.n_arms, dtype=int)
        self.rewards = np.zeros(self.n_arms)
        self.estimates = np.zeros(self.n_arms) 