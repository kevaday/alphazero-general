import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tictactoe.TicTacToeGame import TicTacToeGame as Game
from alphazero.GenericPlayers import RandomPlayer


args = get_args(
    run_name='tictactoe',
    cpuct=2,
    numWarmupIters=1,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    baselineTester=RandomPlayer,
    process_batch_size=128,
    train_batch_size=2048,
    gamesPerIteration=128*4,
    lr=0.01,
    num_channels=32,
    depth=4,
    value_head_channels=4,
    policy_head_channels=4,
    value_dense_layers=[64, 32],
    policy_dense_layers=[64]
)


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
