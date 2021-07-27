import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.gobang.GobangGame import GobangGame as Game
#from alphazero.envs.gobang.GobangPlayers import GreedyGobangPlayer


args = get_args(
    run_name='gobang',
    cpuct=1.25,
    symmetricSamples=True,
    numMCTSSims=250,
    numFastSims=40,
    numWarmupSims=5,
    probFastSim=0.75,
    #skipSelfPlayIters=1,
    numWarmupIters=1,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    process_batch_size=512,
    train_batch_size=1024,
    arena_batch_size=64,
    arenaCompare=64*4,
    arenaCompareBaseline=64*4,
    gamesPerIteration=512*4,
    train_steps_per_iteration=32,
    
    lr=0.01,
    depth=12,
    num_channels=64,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[2048, 256],
    policy_dense_layers=[2048]
)


def raw_mcts_player():
    return RawMCTSPlayer(Game, args)


args.scheduler_args.milestones = [75, 100]
args.baselineTester = raw_mcts_player


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
