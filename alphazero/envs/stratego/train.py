import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.stratego.stratego import Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

args = get_args(
    run_name='stratego',
    workers=6,
    max_moves=512,
    num_stacked_observations=1,
    cpuct=1.25,
    symmetricSamples=True,
    numMCTSSims=100,
    numFastSims=15,
    numWarmupSims=5,
    probFastSim=0.75,
    
    selfPlayModelIter=None,
    skipSelfPlayIters=None,
    #train_on_past_data=True,
    #past_data_run_name='brandubh',
    model_gating=True,
    max_gating_iters=None,
    numWarmupIters=2,
    arenaMCTS=True,
    baselineCompareFreq=5,
    pastCompareFreq=5,
    #train_steps_per_iteration=40,
    min_next_model_winrate=0.52,
    use_draws_for_winrate=True,
    
    process_batch_size=64,
    train_batch_size=256,
    arena_batch_size=32,
    arenaCompare=32*4,
    arenaCompareBaseline=32*4,
    gamesPerIteration=64*6,

    lr=1e-2,
    optimizer_args=dotdict({
        'momentum': 0.9,
        'weight_decay': 1e-3
    }),

    depth=4,
    num_channels=64,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[1024, 128],
    policy_dense_layers=[1024]
)
args.scheduler_args.milestones = [75, 150]


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
