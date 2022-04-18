import numpy, pyximport
pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.gobang.gobang import Game
from alphazero.GenericPlayers import RawMCTSPlayer
#from alphazero.envs.gobang.GobangPlayers import GreedyGobangPlayer


args = get_args(
    run_name='gobang',
    max_moves=225,
    cpuct=2,
    fpu_reduction=0.1,
    symmetricSamples=True,
    numMCTSSims=250,
    numFastSims=50,
    numWarmupSims=5,
    probFastSim=0.75,
    #skipSelfPlayIters=1,
    train_on_past_data=True,
    past_data_run_name='gobang',
    numWarmupIters=1,
    baselineCompareFreq=3,
    pastCompareFreq=1,
    process_batch_size=128,
    train_batch_size=512,
    arena_batch_size=64,
    arenaCompare=64*4,
    arenaCompareBaseline=64*4,
    gamesPerIteration=128*4,
    #train_steps_per_iteration=150,
    autoTrainSteps=True,
    train_sample_ratio=1,
    
    lr=0.01,
    depth=8,
    num_channels=128,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[2048, 128],
    policy_dense_layers=[2048]
)
args.scheduler_args.milestones = [75, 100]


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
