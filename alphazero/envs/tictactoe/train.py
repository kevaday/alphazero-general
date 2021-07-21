import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tictactoe.TicTacToeGame import TicTacToeGame as Game
from alphazero.GenericPlayers import RawMCTSPlayer


args = get_args(
    run_name='tictactoe',
    # workers=2,
    cpuct=2,
    numMCTSSims=100,
    probFastSim=0.5,
    numWarmupIters=1,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    arenaBatchSize=128,
    arenaCompare=512,
    arenaCompareBaseline=512,
    process_batch_size=2048,
    train_batch_size=2048,
    gamesPerIteration=4*2048,
    lr=0.01,
    num_channels=32,
    depth=4,
    value_head_channels=4,
    policy_head_channels=4,
    value_dense_layers=[128, 64],
    policy_dense_layers=[128]
)


def raw_mcts_player():
    return RawMCTSPlayer(Game, args)


args.baselineTester = raw_mcts_player


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
