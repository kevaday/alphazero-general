import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.gobang.GobangGame import GobangGame as Game
from alphazero.envs.gobang.GobangPlayers import GreedyGobangPlayer


args = get_args(
    run_name='gobang',
    cpuct=3,
    skipSelfPlayIters=1,
    numWarmupIters=1,
    baselineCompareFreq=3,
    pastCompareFreq=3,
    baselineTester=GreedyGobangPlayer,
    process_batch_size=256,
    train_batch_size=512,
    gamesPerIteration=4 * 256,
    lr=0.01,
    num_channels=128,
    depth=16,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[2048, 1024],
    policy_dense_layers=[2048]
)


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
