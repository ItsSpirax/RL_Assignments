from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .mdp import MDP
from .types import Action, HeuristicFn, State


@dataclass
class MCTSConfig:
    gamma: float = 0.95
    c_uct: float = 1.4
    rollouts_per_search: int = 500
    max_depth: int = 200


@dataclass
class Node:
    state: State
    parent: Optional[Tuple["Node", Action]] = None
    children: Dict[Action, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


@dataclass
class MCTSSolver:
    mdp: MDP[State, Action]
    config: MCTSConfig
    heuristic: HeuristicFn | None = None
    rng: any = field(default=None)

    def __post_init__(self) -> None:
        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def search(self, root_state: State) -> Action:
        root = Node(state=root_state)

        for _ in range(self.config.rollouts_per_search):
            # YOUR CODE HERE: one MCTS iteration
            # Suggested decomposition:
            # 1) Selection: walk down using UCT until leaf
            # 2) Expansion: add a child for one unexpanded action
            # 3) Rollout: simulate until terminal or depth using heuristic when needed
            # 4) Backprop: propagate discounted returns up the path
            raise NotImplementedError("Implement MCTS iteration (selection, expansion, rollout, backprop)")

        # After search, return the most visited action from root
        best_action = None
        best_visits = -1
        for a, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = a
        if best_action is None:
            # No children expanded (e.g., terminal), just pick any available action
            actions = list(self.mdp.actions(root_state))
            if not actions:
                raise RuntimeError("MCTS called on terminal state")
            best_action = actions[0]
        return best_action

