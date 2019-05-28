import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional


Position = Tuple[int, int]
Reward = float
Rewards = Dict[Position, Reward]
Action = str
ActionsList = List[Action]
Actions = Dict[Position, ActionsList]


class Grid(object):
    def __init__(self, width: int, height: int, start: Position) -> None:
        self.width = width
        self.height = height
        self.row = start[0]
        self.col = start[1]

    def set_rewards_actions(self, rewards: Rewards, actions: Actions) -> None:
        self.rewards = rewards
        self.actions = actions

    def set_state(self, state: Position) -> None:
        self.row = state[0]
        self.col = state[1]

    def current_state(self) -> Position:
        return (self.row, self.col)

    def is_terminal(self, state: Position) -> bool:
        """
        Returns whether a given state is a terminal state or not.
        By checking to see if the state is in our available actions,
        we will know whether or not there is some action we can perform
        to get out of the current state.
        """
        return state not in self.actions

    def move(self, action: Action) -> Reward:
        # Is the action available in our ActionList given our current position?
        if action in self.actions[(self.row, self.col)]:
            if action == 'U':
                self.row -= 1
            elif action == 'D':
                self.row += 1
            elif action == 'R':
                self.col += 1
            elif action == 'L':
                self.col -= 1
        # If the action is not there, get the reward (0 if it's not there)
        return self.rewards.get((self.row, self.col), 0)

    def undo_move(self, action: Action) -> None:
        if action == 'U':
            self.row += 1
        elif action == 'D':
            self.row -= 1
        elif action == 'R':
            self.col -= 1
        elif action == 'L':
            self.col += 1

        # Should never get here
        assert(self.current_state() in self.all_states())
        
    def is_game_over(self) -> bool:
        """
        Check to see if we are in a terminal state
        """
        return (self.row, self.col) not in self.actions

    def all_states(self):
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_grid() -> Grid:
    """
    Define a grid that describes the rewrad for arriving at each state
    and possible actions at each state. The grid looks like the following:

    .  .  .  1
    .  X  . -1
    s  .  .  .

    where:
        "." is a grid location you could get to eventually
        "s" is the start position
        "X" is a wall and you cannot get there
    """
    grid = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U')
    }
    grid.set_rewards_actions(rewards, actions)
    return grid


def negative_grid(step_cost=-.1) -> Grid:
    grid = standard_grid()
    grid.rewards.update({k: step_cost for k in grid.actions.keys()})
    return grid

