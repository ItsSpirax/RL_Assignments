from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


State = Tuple[int, int]
Action = str  # "U", "D", "L", "R"


@dataclass(frozen=True)
class Transition:
    next_state: State
    probability: float
    reward: float


class MDP:
    def initial_state(self) -> State:
        raise NotImplementedError

    def actions(self, state: State) -> Sequence[Action]:
        raise NotImplementedError

    def is_terminal(self, state: State) -> bool:
        raise NotImplementedError

    def transitions(self, state: State, action: Action) -> Iterable[Transition]:
        raise NotImplementedError


class GridWorld(MDP):
    ACTIONS: Sequence[Action] = ("U", "D", "L", "R")
    DELTAS: Dict[Action, Tuple[int, int]] = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1),
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        start: State,
        goal: State,
        obstacles: Sequence[State] | None = None,
        step_cost: float = -1.0,
        goal_reward: float = 0.0,
        slip: float = 0.1,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.slip = float(min(max(slip, 0.0), 1.0))

    def initial_state(self) -> State:
        return self.start

    def actions(self, state: State) -> Sequence[Action]:
        if self.is_terminal(state):
            return ()
        return self.ACTIONS

    def is_terminal(self, state: State) -> bool:
        return state == self.goal

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _blocked(self, r: int, c: int) -> bool:
        return (r, c) in self.obstacles

    def _move(self, state: State, action: Action) -> State:
        dr, dc = self.DELTAS[action]
        nr, nc = state[0] + dr, state[1] + dc
        if not self._in_bounds(nr, nc) or self._blocked(nr, nc):
            return state
        return (nr, nc)

    def transitions(self, state: State, action: Action) -> Iterable[Transition]:
        if self.is_terminal(state):
            yield Transition(next_state=state, probability=1.0, reward=0.0)
            return

        intended_next = self._move(state, action)

        if action in ("U", "D"):
            perpendiculars: List[Action] = ["L", "R"]
        else:
            perpendiculars = ["U", "D"]

        slip_each = self.slip / 2.0
        stay_prob = 1.0 - self.slip

        reward = self.goal_reward if intended_next == self.goal else self.step_cost
        yield Transition(intended_next, stay_prob, reward)

        for perp in perpendiculars:
            next_s = self._move(state, perp)
            reward = self.goal_reward if next_s == self.goal else self.step_cost
            yield Transition(next_s, slip_each, reward)


def sample_next_state_and_reward(mdp: MDP, state: State, action: Action, rng) -> Tuple[State, float]:
    r = rng.random()
    acc = 0.0
    chosen_next = state
    chosen_reward = 0.0
    for t in mdp.transitions(state, action):
        acc += t.probability
        if r <= acc:
            chosen_next = t.next_state
            chosen_reward = t.reward
            break
    return chosen_next, chosen_reward


def make_default_grid() -> GridWorld:
    rows, cols = 5, 6
    start = (4, 0)
    goal = (0, 5)
    obstacles = [(1, 1), (1, 2), (3, 3)]
    return GridWorld(
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
        obstacles=obstacles,
        step_cost=-1.0,
        goal_reward=0.0,
        slip=0.2,
    )

