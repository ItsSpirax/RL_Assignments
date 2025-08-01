import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from environment.mab_environment import MABEnvironment
from algorithms.base_algorithm import BaseMABAlgorithm
from utils.config import MABConfig

class MABExperimentRunner:
    """
    Main experiment runner for MAB algorithms
    """
    def __init__(self, config: MABConfig):
        self.config = config
        self.environment = MABEnvironment(
            n_arms=config.n_arms,
            reward_distributions=config.reward_distributions,
            seed=config.seed
        )
        
    def run_experiment(self, algorithm: BaseMABAlgorithm, n_trials: int = None) -> Dict[str, Any]:
        """
        Run a single experiment with the given algorithm
        
        Args:
            algorithm: Algorithm to test
            n_trials: Number of trials (uses config if None)
            
        Returns:
            Dictionary with experiment results
        """
        if n_trials is None:
            n_trials = self.config.n_trials
            
        # Reset algorithm
        algorithm.reset()
        
        # Track results
        rewards = []
        regrets = []
        arm_history = []
        cumulative_regret = 0
        
        for trial in range(n_trials):
            # Select arm
            arm = algorithm.select_arm()
            arm_history.append(arm)
            
            # Get reward
            reward = self.environment.pull(arm)
            rewards.append(reward)
            
            # Update algorithm
            algorithm.update(arm, reward)
            
            # Calculate regret
            regret = self.environment.get_regret(arm)
            cumulative_regret += regret
            regrets.append(cumulative_regret)
        
        return {
            'rewards': rewards,
            'regrets': regrets,
            'arm_history': arm_history,
            'final_estimates': algorithm.estimates.copy(),
            'final_pulls': algorithm.pulls.copy(),
            'optimal_arm': self.environment.get_optimal_arm(),
            'estimated_optimal_arm': algorithm.get_estimated_optimal_arm()
        }
    
    def compare_algorithms(self, algorithms: Dict[str, BaseMABAlgorithm]) -> Dict[str, Dict]:
        """
        Compare multiple algorithms
        
        Args:
            algorithms: Dictionary mapping algorithm names to algorithm instances
            
        Returns:
            Dictionary with results for each algorithm
        """
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"Running {name}...")
            results[name] = self.run_experiment(algorithm)
            
        return results
    
    def plot_results(self, results: Dict[str, Dict], save_path: str = None):
        """
        Plot comparison results
        
        Args:
            results: Results from compare_algorithms
            save_path: Path to save plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot cumulative regret
        for name, result in results.items():
            ax1.plot(result['regrets'], label=name)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Cumulative Regret')
        ax1.set_title('Cumulative Regret Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot average reward
        for name, result in results.items():
            avg_rewards = np.cumsum(result['rewards']) / np.arange(1, len(result['rewards']) + 1)
            ax2.plot(avg_rewards, label=name)
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def print_summary(self, results: Dict[str, Dict]):
        """Print summary statistics"""
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Final Cumulative Regret: {result['regrets'][-1]:.2f}")
            print(f"  Final Average Reward: {np.mean(result['rewards'][-100:]):.3f}")
            print(f"  Optimal Arm: {result['optimal_arm']}")
            print(f"  Estimated Optimal Arm: {result['estimated_optimal_arm']}")
            print(f"  Arm Pulls: {dict(enumerate(result['final_pulls']))}") 