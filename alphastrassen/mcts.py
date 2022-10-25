from typing import *
from tqdm import tqdm
import numpy as np

from .nnet import NeuralNet
from .environment import State, Environment


class MCTS:
    """This class handles a MCTS tree.
    """

    def __init__(
            self,
            environment: Environment,
            nnet: NeuralNet,
            num_simulations: int,
            max_num_steps: int,
            cpuct: float = 1.0,
    ):
        self.environment = environment
        self.nnet = nnet
        self.num_simulations = num_simulations
        self.max_num_steps = max_num_steps
        self.cpuct = cpuct

        # transposition tables
        self.N: Dict[str, np.ndarray] = {}
        self.W: Dict[str, np.ndarray] = {}
        self.Q: Dict[str, np.ndarray] = {}
        self.P: Dict[str, np.ndarray] = {}

    def policy(self, state: State) -> np.ndarray:
        """This function performs ``num_mcts_simulations`` simulations of MCTS
        starting from ``state``.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)] ** (1 / tau)
        """
        for _ in range(self.num_simulations):
            self.run_simulation(state)

        s = state.to_str()

        return self.N[s] / np.sum(self.N[s])  # TODO: add temperature

    def run_simulation(self, state):
        assert not self.environment.is_terminal(state)

        states = []
        actions = []
        cumulative_rewards = []
        cumulative_reward = 0
        num_steps = 0
        while True:
            s = state.to_str()

            if s not in self.P:
                # leaf state
                p, v = self.nnet.predict(state)
                cumulative_reward += v
                self.P[s] = p
                self.N[s] = np.zeros(len(p), dtype=int)
                self.W[s] = np.zeros(len(p), dtype=float)
                self.Q[s] = np.zeros(len(p), dtype=float)
                break

            # pick the action with the highest upper confidence bound
            a = np.argmax(self.Q[s] + self.cpuct * self.P[s] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s]))

            states.append(s)
            actions.append(a)
            cumulative_rewards.append(cumulative_reward)  # current cumulative reward, before applying action a

            cumulative_reward += self.environment.get_intermediate_reward(state, a)
            state = self.environment.get_next_state(state, a)
            num_steps += 1

            if num_steps >= self.max_num_steps or self.environment.is_terminal(state):
                cumulative_reward += self.environment.get_final_reward(state)
                break

        # back up
        for s, a, r in zip(states, actions, cumulative_rewards):
            self.N[s][a] = self.N[s][a] + 1
            self.W[s][a] = self.W[s][a] + cumulative_reward - r
            self.Q[s][a] = self.W[s][a] / self.N[s][a]
