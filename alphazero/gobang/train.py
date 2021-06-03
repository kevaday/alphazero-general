import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.gobang.GobangGame import GobangGame as Game
from alphazero.gobang.GobangPlayers import GreedyGobangPlayer


args = get_args(
    run_name='gobang',
    cpuct=3,
    numWarmupIters=1,
    testCompareFreq=3,
    pastCompareFreq=3,
    compareTester=GreedyGobangPlayer,
    process_batch_size=128,
    train_batch_size=2048,
    train_steps_per_iteration=800,
    gamesPerIteration=512,
    lr=0.01,
    num_channels=128,
    depth=16,
    value_head_channels=2,
    policy_head_channels=4,
    value_dense_layers=[128, 128],
    policy_dense_layers=[1024]
)


if __name__ == "__main__":
    g = Game()
    nnet = nn(g, args)
    c = Coach(g, nnet, args)
    c.learn()
