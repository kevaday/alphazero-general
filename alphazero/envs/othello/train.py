import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.othello.othello import Game
#from alphazero.envs.othello.OthelloPlayers import GreedyOthelloPlayer


args = get_args(
    run_name='othello',
    workers=2,
    cpuct=4,
    numWarmupIters=1,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    numMCTSSims=100,
    numFastSims=25,
    probFastSim=0.75,
    numWarmupSims=1,
    #baselineTester=GreedyOthelloPlayer,
    process_batch_size=128,
    train_batch_size=1024,
    gamesPerIteration=128*4,
    lr=0.01,
    num_channels=64,
    depth=4,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[512, 256],
    policy_dense_layers=[512]
)


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
