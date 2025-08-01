import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class EpsilonGreedy(BaseMABAlgorithm):
    """
    Epsilon-Greedy algorithm
    With probability epsilon: explore (random arm)
    With probability 1-epsilon: exploit (best estimated arm)
    """
    def __init__(self, n_arms: int, epsilon: float = 0.1, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.epsilon = epsilon
        
    def select_arm(self) -> int:
        """
        ## IMPLEMENT EPSILON GREEDY ALGORITHM HERE ##
        
        Input: None (uses self.epsilon, self.estimates, self.pulls)
        Output: int - selected arm index
        
        Strategy: 
        - With probability epsilon: explore (random arm)
        - With probability 1-epsilon: exploit (best estimated arm)
        - If no arm pulled yet, select arm 0
        """
        # TODO: Implement epsilon greedy algorithm
        # Hint: Use np.random.random() to generate random number between 0 and 1
        # If random number < epsilon: explore (random arm)
        # Else: exploit (best estimated arm)
        # Check if no pulls yet and handle that case
        
        # YOUR CODE HERE
        # pass
        
        # SOLUTION
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        if np.sum(self.pulls) == 0:
            return 0
        return np.argmax(self.estimates) 