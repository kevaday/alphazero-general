import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tictactoe.tictactoe import Game


args = get_args(
    run_name='tictactoe_ordo',
    workers=7,
    cpuct=2,
    numMCTSSims=25,
    probFastSim=0.5,
    numWarmupIters=1,
    baselineCompareFreq=5,
    pastCompareFreq=5,
    arenaBatchSize=512,
    arenaCompare=128,
    arenaCompareBaseline=128,
    process_batch_size=512,
    train_batch_size=512,
    gamesPerIteration=2*512,
    lr=0.01,
    num_channels=32,
    depth=4,
    value_head_channels=4,
    policy_head_channels=4,
    value_dense_layers=[128, 64],
    policy_dense_layers=[128],
    compareWithBaseline=False,
    compareWithPast=False,

    eloMCTS=25,
    eloGames=10,
    eloMatches=10,
    #arenaBatched=False
)


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
