import pyximport;

pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tafl.tafl import TaflGame as Game, NUM_STACKED_OBSERVATIONS, DRAW_MOVE_COUNT
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

args = get_args(
    run_name='hnefatafl_test',
    max_moves=DRAW_MOVE_COUNT,
    num_stacked_observations=NUM_STACKED_OBSERVATIONS,
    cpuct=1.25,
    symmetricSamples=False,
    numMCTSSims=5,
    numFastSims=5,
    numWarmupSims=5,
    probFastSim=0.75,

    selfPlayModelIter=None,
    skipSelfPlayIters=None,
    model_gating=True,
    max_gating_iters=None,
    numWarmupIters=1,
    arenaMCTS=False,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    train_sample_ratio=3,
    min_next_model_winrate=0.52,
    use_draws_for_winrate=False,

    process_batch_size=16,
    train_batch_size=1024,
    arena_batch_size=32,
    arenaCompare=32 * 4,
    arenaCompareBaseline=32 * 4,
    gamesPerIteration=1,

    lr=1e-2,
    optimizer_args=dotdict({
        'momentum': 0.9,
        'weight_decay': 1e-4
    }),

    depth=5,
    num_channels=32,
    value_head_channels=1,
    policy_head_channels=2,
    value_dense_layers=[32],
    policy_dense_layers=[32]
)
args.scheduler_args.milestones = [75, 150]
args.baselineTester = lambda: RawMCTSPlayer(Game, args)

if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
