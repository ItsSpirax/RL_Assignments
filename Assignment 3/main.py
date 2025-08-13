from __future__ import annotations

from gridworld import make_default_grid
from rtdp import RTDP, RTDPConfig, LinearDecay
from mcts import MCTS, MCTSConfig


def run_rtdp():
    env = make_default_grid()
    cfg = RTDPConfig(
        gamma=0.95,
        episodes=50,
        max_steps=1000,
        epsilon_schedule=LinearDecay(start=0.5, end=0.05, steps=50),
    )
    agent = RTDP(env, cfg)
    agent.run()  # Will raise NotImplementedError until students implement


def run_mcts():
    env = make_default_grid()
    cfg = MCTSConfig(gamma=0.95, c_uct=1.4, rollouts=200, max_depth=200)
    agent = MCTS(env, cfg)
    a = agent.search(env.initial_state())  # Will raise NotImplementedError
    print("MCTS chose:", a)


if __name__ == "__main__":
    # Choose one to test
    run_rtdp()
    # run_mcts()
    pass

