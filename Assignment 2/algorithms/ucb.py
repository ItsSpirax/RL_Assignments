import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class UCB(BaseMABAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm
    Balances exploration and exploitation using confidence bounds
    """
    def __init__(self, n_arms: int, c: float = 2.0, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.c = c  # Exploration parameter
        
    def select_arm(self) -> int:
        """
        ## IMPLEMENT UCB ALGORITHM HERE ##
        
        Input: None (uses self.estimates, self.pulls, self.c)
        Output: int - selected arm index
        
        Strategy: UCB balances exploration and exploitation using confidence bounds
        - If some arms haven't been pulled: pull one of them
        - Otherwise: select arm with highest UCB value
        - UCB formula: estimate + c * sqrt(log(total_pulls) / arm_pulls)
        """
        # TODO: Implement UCB algorithm
        # Hint: 
        # 1. Check for unpulled arms using np.where(self.pulls == 0)[0]
        # 2. If unpulled arms exist, return the first one
        # 3. Calculate total_pulls = np.sum(self.pulls)
        # 4. Calculate UCB values: estimate + c * sqrt(log(total_pulls) / arm_pulls)
        # 5. Return arm with highest UCB value
        
        # YOUR CODE HERE
        unpulled = np.where(self.pulls == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        
        total_pulls = np.sum(self.pulls)
        confidence_bonus = self.c * np.sqrt((np.log(total_pulls) + 1e-10) / (self.pulls + 1e-10))
        ucb_values = self.estimates + confidence_bonus
        return int(np.argmax(ucb_values))
