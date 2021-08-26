import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tafl.tafl_old import Game, NUM_STACKED_OBSERVATIONS, DRAW_MOVE_COUNT
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

args = get_args(
    run_name='hnefatafl',
    max_moves=DRAW_MOVE_COUNT,
    num_stacked_observations=NUM_STACKED_OBSERVATIONS,
    cpuct=1.25,
    symmetricSamples=False,
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
    baselineCompareFreq=3,
    pastCompareFreq=3,
    train_steps_per_iteration=40,
    min_next_model_winrate=0.52,
    use_draws_for_winrate=True,
    
    process_batch_size=64,
    train_batch_size=4096,
    arena_batch_size=32,
    arenaCompare=32*4,
    arenaCompareBaseline=32*4,
    gamesPerIteration=64*4,

    lr=1e-2,
    optimizer_args=dotdict({
        'momentum': 0.9,
        'weight_decay': 1e-3
    }),

    depth=8,
    num_channels=64,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[2048, 256],
    policy_dense_layers=[2048]
)


def raw_mcts_player():
    return RawMCTSPlayer(Game, args)


args.scheduler_args.milestones = [75, 150]
args.baselineTester = raw_mcts_player


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
