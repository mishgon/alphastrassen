import numpy as np
import random


class State:
    def __init__(self, tensor: np.ndarray):
        self.tensor = np.array(tensor, dtype=int)

    def to_str(self) -> str:
        return str(self.tensor)

    @property
    def rank_upper_bound(self) -> int:
        return float(np.sum(np.linalg.matrix_rank(self.tensor)))


class Action:
    def __init__(self, u, v, w):
        u = np.array(u, dtype=int)
        v = np.array(v, dtype=int)
        w = np.array(w, dtype=int)

        self.u = u
        self.v = v
        self.w = w
        self.tensor = u[:, None, None] * v[None, :, None] * w[None, None]

    @classmethod
    def random(cls):
        u = np.random.choice([-1, 0, 1], size=4)
        v = np.random.choice([-1, 0, 1], size=4)
        w = np.random.choice([-1, 0, 1], size=4)
        return cls(u, v, w)

    def __eq__(self, other):
        return np.allclose(self.tensor, other.tensor)


REQUIRED_ACTIONS = (
    Action([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    Action([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
    Action([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
    Action([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
    Action([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    Action([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    Action([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
)
RANDOM_ACTIONS = tuple(Action.random() for _ in range(7))
ACTIONS = REQUIRED_ACTIONS + tuple(a for a in RANDOM_ACTIONS if a not in REQUIRED_ACTIONS)


# INIT_STATE = State(np.sum([ACTIONS[i].tensor for i in [0, 1, 2]], axis=0))
INIT_STATE = State([
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0 ,0]],
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0]],
    [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
])


class Environment:
    """This class have no state. De facto it is a namespace of functions which
    define the environment.
    """

    @property
    def num_actions(self):
        return len(ACTIONS)

    def is_terminal(self, state) -> bool:
        return np.all(state.tensor == 0)

    def get_init_state(self) -> State:
        return INIT_STATE

    def get_next_state(self, state: State, action_idx: int) -> State:
        return State(state.tensor - ACTIONS[action_idx].tensor)

    def get_intermediate_reward(self, state: State, action_idx: int) -> float:
        return -1.0

    def get_final_reward(self, state: State) -> float:
        if self.is_terminal(state):
            return 0.0

        return -state.rank_upper_bound

    def generate_synthetic_examples(self, max_num_steps: int):
        n = min(self.num_actions, max_num_steps)
        indices = np.random.choice(self.num_actions, size=n)
        state = State(np.sum([ACTIONS[i].tensor for i in indices], axis=0))
        examples = []
        cumulative_rewards = []
        cumulative_reward = 0
        for action_idx in indices:
            examples.append([state, np.eye(self.num_actions)[action_idx], None])
            cumulative_rewards.append(cumulative_reward)

            cumulative_reward += self.get_intermediate_reward(state, action_idx)
            state = self.get_next_state(state, action_idx)

        assert self.is_terminal(state)
        cumulative_reward += self.get_final_reward(state)

        return [(s, pi, cumulative_reward - r) for (s, pi, _), r in zip(examples, cumulative_rewards)]
