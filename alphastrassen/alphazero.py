from typing import *
import os
from tqdm import tqdm
import numpy as np

from .environment import Environment, State
from .mcts import MCTS


class AlphaZero:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Environment and NeuralNet. args are specified in main.py.
    """

    def __init__(self, environment: Environment, nnet, args):
        self.environment = environment
        self.nnet = nnet
        self.args = args

    def play(self) -> List[Tuple[State, np.ndarray, float]]:
        mcts = MCTS(self.environment, self.nnet, self.args)  # reset tree for each self-play game
        train_examples = []
        cumulative_rewards = []
        state = self.environment.get_init_state()
        cumulative_reward = 0
        num_steps = 0
        while True:
            pi = mcts.policy(state)
            train_examples.append([state, pi, None])
            cumulative_rewards.append(cumulative_reward)  # current cumulative reward before applying policy pi

            action_idx = np.random.choice(len(pi), p=pi)
            cumulative_reward += self.environment.get_intermediate_reward(state, action_idx)  # -1 in AlphaTensor case
            state = self.environment.get_next_state(state, action_idx)
            num_steps += 1

            if num_steps >= self.args.max_num_steps or self.environment.is_terminal(state):
                cumulative_reward += self.environment.get_final_reward(state)
                break

        return [(s, pi, cumulative_reward - r) for (s, pi, _), r in zip(train_examples, cumulative_rewards)]

    def learn(self):
        for i in range(1, self.args.num_iter + 1):
            train_examples = []
            for _ in tqdm(range(self.args.num_self_play_games), desc='Self-playing'):
                train_examples += self.play()

            self.nnet.train(train_examples)
            self.nnet.save_checkpoint(path=os.path.join(self.args.ckpt_dir, f'checkpoint_{i}.pth.tar'))
