import pyximport, numpy

pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.hnefatafl.fastafl import Game as Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

args = get_args(
    run_name='hnefatafl_fastafl',
    #workers=1,
    max_moves=512,
    num_stacked_observations=1,
    cpuct=1.25,
    symmetricSamples=True,
    numMCTSSims=250,
    numFastSims=50,
    numWarmupSims=5,
    probFastSim=0.8,

    selfPlayModelIter=None,
    skipSelfPlayIters=None,
    #train_on_past_data=True,
    #past_data_run_name='hnefatafl_fastafl',
    model_gating=True,
    max_gating_iters=None,
    numWarmupIters=1,
    arenaMCTS=True,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    train_steps_per_iteration=80,
    min_next_model_winrate=0.52,
    use_draws_for_winrate=True,

    process_batch_size=128,
    train_batch_size=2048,
    arena_batch_size=32,
    arenaCompare=32 * 4,
    arenaCompareBaseline=32 * 4,
    gamesPerIteration=128 * 4,

    lr=1e-2,
    optimizer_args=dotdict({
        'momentum': 0.9,
        'weight_decay': 1e-3
    }),

    depth=10,
    num_channels=128,
    value_head_channels=32,
    policy_head_channels=32,
    value_dense_layers=[4096, 128],
    policy_dense_layers=[4096]
)
args.scheduler_args.milestones = [75, 150]

if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
