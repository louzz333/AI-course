#%%
""" importing necessary packages """

from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


#%%

""" Modeling MDPs """


@dataclass(frozen=True)
class Step:
    action: Any
    prob: float
    reward: float
    new_state: Any


class MDP(ABC):
    """
    Abstract class
    Formally represents an MDP.
    """

    @abstractmethod
    def start_state(self) -> Any:
        pass

    @abstractmethod
    def successors(self, state: Any) -> list[Step]:
        pass

    @abstractmethod
    def is_end(self, state: Any) -> bool:
        pass

    @abstractmethod
    def discount(self) -> float:
        pass

   # @abstractmethod
   # def actions(self, state: Any) -> list[Any]:
   #     pass


#%%

class WalkTramMDP(MDP):
    """
    Street with locations numbered 1 to n 'num_locs'.
    Walking from i to i+1 takes 1 minute.
    Taking a magic tram from i to 2*i takes 2 minutes.
    The tram can break down with probability 'failure_prob'
    How to travel from 1 to n in the least time in expectation?
    """

    def __init__(self, num_locs, failure_prob=0, discount=1):
        self.num_locs = num_locs
        self.failure_prob = failure_prob
        self.gamma = discount

    def start_state(self) -> Any:
        return 1

    def successors(self, state: Any) -> list[Step]:
        steps = []

        if state + 1 <= self.num_locs:
            steps.append(Step("walk", 1.0, -1.0, state + 1))

        if state * 2 <= self.num_locs:
            steps.append(Step("tram", self.failure_prob, -2, state))
            steps.append(Step("tram", 1-self.failure_prob, -2, state*2))

        return steps

    def is_end(self, state: Any) -> bool:
        return state == self.num_locs

    def discount(self) -> float:
        return self.gamma


def walk_policy(state):
    return "walk"


def tram_policy(state):
    return "tram"


def compute_utility(steps: list[Step], discount: float) -> float:
    """
    Computes the utility (discounted sum of rewards) of a path.
    """
    utility = 0

    for i in range(len(steps)):
        utility += (discount ** i) * steps[i].reward

    return utility


def run_policy(mdp: MDP, policy) -> list[Step]:
    """
    Run the `policy` from start state in the `mdp` and
    return the path with corresponding utility.
    """
    steps = []
    state = mdp.start_state()

    while not mdp.is_end(state):
        action = policy(state)
        s = [step for step in mdp.successors(state) if step.action == action]

        if s == []:
            break

        probs = [step.prob for step in s]
        

        i = np.random.choice(range(len(s)), p=probs)
        suivant = s[i]

        steps.append(suivant)
        state = suivant.new_state

    return steps


def Monte_Carlo(mdp: MDP, policy, num_rollout) -> float:
    value = 0.0

    for i in range(num_rollout):
        value += compute_utility(run_policy(mdp, policy), mdp.discount())

    return value / num_rollout

#%%

class DiceGameMDP(MDP):

    def __init__(self, discount=1):
        self.gamma = discount

    def start_state(self) -> Any:
        return "start"

    def actions(self, state: Any) -> list[Any]:
        if self.is_end(state):
            return []
        return ["stay", "quit"]

    def successors(self, state: Any) -> list[Step]:
        steps = []

        if state == "start" or state not in ["END", 1, 2]:

            # Option 1: Quit
            steps.append(Step("quit", 1.0, 10, "END"))

            # Option 2: Stay and roll the die
            for dice in range(1, 7):
                steps.append(
                    Step("stay", 1/6, 4, dice)
                )

        return steps

    def is_end(self, state: Any) -> bool:
        return state in [1, 2, "END"]

    def discount(self) -> float:
        return self.gamma

#%%

"""
Main
"""
mdp = WalkTramMDP(10,0.2,1)
v = Monte_Carlo(mdp, tram_policy, 100)
print("Value = ", v)


