import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.othello.OthelloGame import OthelloGame as Game
from alphazero.envs.othello.OthelloPlayers import GreedyOthelloPlayer


args = get_args(
    run_name='othello',
    cpuct=2,
    numWarmupIters=1,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    baselineTester=GreedyOthelloPlayer,
    process_batch_size=128,
    train_batch_size=2048,
    gamesPerIteration=128*4,
    lr=0.01,
    num_channels=64,
    depth=8,
    value_head_channels=8,
    policy_head_channels=8,
    value_dense_layers=[512, 256],
    policy_dense_layers=[512]
)


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
