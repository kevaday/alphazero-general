import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tictactoe.tictactoe import Game


args = get_args(
    run_name='tictactoe',
    workers=2,
    cpuct=2,
    numMCTSSims=100,
    probFastSim=0.5,
    numWarmupIters=1,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    arenaBatchSize=2048,
    arenaCompare=2*2048,
    arenaCompareBaseline=2*2048,
    process_batch_size=512,
    train_batch_size=2048,
    gamesPerIteration=2*512,
    lr=0.01,
    num_channels=32,
    depth=4,
    value_head_channels=4,
    policy_head_channels=4,
    value_dense_layers=[128, 64],
    policy_dense_layers=[128],
    skipSelfPlayIters=1,
)


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
