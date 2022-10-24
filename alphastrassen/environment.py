import numpy as np


class State:
    def __init__(self, tensor: np.ndarray):
        self.tensor = np.array(tensor)

    def to_str(self) -> str:
        return str(self.tensor)

    @property
    def rank_upper_bound(self) -> int:
        return np.sum(np.linalg.matrix_rank(self.tensor))


class Action:
    def __init__(self, u, v, w):
        u = np.array(u)
        v = np.array(v)
        w = np.array(w)
        
        self.u = u
        self.v = v
        self.w = w
        self.tensor = u[:, None, None] * v[None, :, None] * w[None, None]


class Environment:
    """This class have no state. De facto it is a namespace of functions which
    define the environment.
    """
    init_state = State([
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

    actions = (
        Action([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        Action([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        Action([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        Action([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        Action([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        Action([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        Action([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    )

    @property
    def num_actions(self):
        return len(self.actions)

    def is_terminal(self, state) -> bool:
        return np.all(state.tensor == 0)

    def get_init_state(self) -> State:
        return self.init_state

    def get_next_state(self, state: State, action_idx: int) -> State:
        return State(state.tensor - self.actions[action_idx].tensor)

    def get_intermediate_reward(self, state: State, action_idx: int) -> float:
        return -1

    def get_final_reward(self, state: State) -> float:
        if self.is_terminal(state):
            return 0

        return -state.rank_upper_bound
