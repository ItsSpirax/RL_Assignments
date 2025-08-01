#!/usr/bin/env python3
"""
Main script demonstrating the modular MAB framework
"""

from utils.config import MABConfig
from experiments.experiment_runner import MABExperimentRunner
from algorithms.exploration_only import ExplorationOnly
from algorithms.exploitation_only import ExploitationOnly
from algorithms.epsilon_greedy import EpsilonGreedy
from algorithms.ucb import UCB

def test_algorithm_implementation(algorithm, name):
    """Test if an algorithm is properly implemented"""
    try:
        # Try to select an arm
        arm = algorithm.select_arm()
        if arm is None:
            return False, f"{name}: ❌ Not implemented (returns None)"
        if not isinstance(arm, int):
            return False, f"{name}: ❌ Not implemented (returns {type(arm)})"
        if arm < 0 or arm >= algorithm.n_arms:
            return False, f"{name}: ❌ Not implemented (invalid arm index {arm})"
        return True, f"{name}: ✅ Implemented"
    except Exception as e:
        return False, f"{name}: ❌ Not implemented (error: {str(e)})"

def main():
    # Create configuration
    config = MABConfig()
    config.get_bernoulli_config(n_arms=5, n_trials=1000)
    config.set_algorithm_params('epsilon_greedy', {'epsilon': 0.1})
    config.set_algorithm_params('ucb', {'c': 1.5})
    
    # Create experiment runner
    runner = MABExperimentRunner(config)
    
    # Create algorithms
    algorithms = {
        'Exploration Only': ExplorationOnly(config.n_arms),
        'Exploitation Only': ExploitationOnly(config.n_arms),
        'Epsilon-Greedy': EpsilonGreedy(config.n_arms, epsilon=0.1),
        'UCB': UCB(config.n_arms, c=2.0)
    }
    
    # Test which algorithms are implemented
    print("=== MAB Algorithm Implementation Status ===")
    implemented_algorithms = {}
    unimplemented_algorithms = []
    
    for name, algorithm in algorithms.items():
        is_implemented, status = test_algorithm_implementation(algorithm, name)
        print(status)
        
        if is_implemented:
            implemented_algorithms[name] = algorithm
        else:
            unimplemented_algorithms.append(name)
    
    print(f"\n✅ Ready to run: {len(implemented_algorithms)} algorithms")
    if unimplemented_algorithms:
        print(f"⏳ Pending implementation: {', '.join(unimplemented_algorithms)}")
    
    # Run comparison only for implemented algorithms
    if implemented_algorithms:
        print(f"\n=== Running Comparison ({len(implemented_algorithms)} algorithms) ===")
        results = runner.compare_algorithms(implemented_algorithms)
        
        # Print summary
        runner.print_summary(results)
        
        # Plot results
        runner.plot_results(results, save_path='mab_comparison.png')
        print("\nPlot saved as 'mab_comparison.png'")
    else:
        print("\n❌ No algorithms implemented yet!")
        print("Please implement at least one algorithm to see results.")
    
    # Show what students need to do
    if unimplemented_algorithms:
        print(f"\n=== Student Assignment ===")
        print("To complete the assignment, implement these algorithms:")
        for algo in unimplemented_algorithms:
            print(f"  - {algo}")
        print("\nFiles to modify:")
        print("  - src/algorithms/exploration_only.py")
        print("  - src/algorithms/exploitation_only.py") 
        print("  - src/algorithms/ucb.py")
        print("\nEpsilon-Greedy is already implemented as a demo!")

if __name__ == "__main__":
    main() 