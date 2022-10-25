from alphastrassen.alphazero import AlphaZero
from alphastrassen.environment import Environment
from alphastrassen.nnet import NeuralNet


def main():
    env = Environment()
    
    nnet = NeuralNet(
        input_size=env.get_init_state().tensor.shape,
        num_actions=env.num_actions,
        lr=1e-3
    )

    alphazero = AlphaZero(
        env,
        nnet,
        num_iter=10_000,  # number of alternations between self-playing and nnet training
        num_self_play_games=100,  # number of self-play games per iteration
        num_mcts_simulations=100,  # number of simulations during "thinking over the move"
        max_num_steps=8,  # per game
        max_num_examples=1000,  # max length of history of moves, which network is training on
        num_epochs=10,  # number of epochs of nnet training per iteration
        batch_size=50,
        logs_dir='/shared/personal/mgoncharov/alphastrassen/strassen_14act/',
        device=1
    )

    alphazero.learn()


if __name__ == '__main__':
    main()
