from typing import *
import collections
from tqdm import tqdm
import numpy as np
import random

import pytorch_lightning as pl

from .environment import Environment, State
from .mcts import MCTS
from .nnet import NeuralNet, DataModule


class AlphaZero:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Environment and NeuralNet.
    """

    def __init__(
            self,
            environment: Environment,
            nnet: NeuralNet,
            num_iter: int,
            num_self_play_games: int,
            num_mcts_simulations: int,
            max_num_steps: int,
            max_num_examples: int,
            num_epochs: int,
            batch_size: int,
            logs_dir: str,
            device: int,
    ):
        self.environment = environment
        self.nnet = nnet.to(device=device)
        self.num_iter = num_iter
        self.num_self_play_games = num_self_play_games
        self.num_mcts_simulations = num_mcts_simulations
        self.max_num_steps = max_num_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.logs_dir = logs_dir
        self.device = device

        self.examples = collections.deque([], maxlen=max_num_examples)

    def play(self) -> List[Tuple[State, np.ndarray, float]]:
        mcts = MCTS(self.environment, self.nnet, self.num_mcts_simulations, self.max_num_steps)  # reset tree for each self-play game
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

            if num_steps >= self.max_num_steps or self.environment.is_terminal(state):
                cumulative_reward += self.environment.get_final_reward(state)
                break

        return [(s, pi, cumulative_reward - r) for (s, pi, _), r in zip(train_examples, cumulative_rewards)], cumulative_reward

    def learn(self):
        cumulative_reward_logger = pl.loggers.TensorBoardLogger(save_dir=self.logs_dir, name='cumulative_rewards')
        for i in range(1, self.num_iter + 1):
            cumulative_rewards = []
            for _ in tqdm(range(self.num_self_play_games), desc='Self-playing'):
                examples, cumulative_reward = self.play()
                self.examples.extend(examples)
                cumulative_rewards.append(cumulative_reward)

            cumulative_reward_logger.log_metrics({
                'avg_cumulative_reward': np.mean(cumulative_rewards),
                'max_cumulative_reward': np.max(cumulative_rewards)
            }, step=i)

            # for _ in range(self.num_self_play_games):
            #     self.examples.extend(self.environment.generate_synthetic_examples(self.max_num_steps))

            data_module = DataModule(self.examples, batch_size=self.batch_size)

            logger = pl.loggers.TensorBoardLogger(save_dir=self.logs_dir, name=f'iter_{i}')
            trainer = pl.Trainer(
                logger=logger,
                accelerator='gpu',
                devices=self.device,
                max_epochs=self.num_epochs,
            )
            trainer.fit(self.nnet, data_module)
