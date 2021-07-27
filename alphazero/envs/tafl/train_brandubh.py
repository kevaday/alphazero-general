import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tafl.brandubh import TaflGame as Game, NUM_STACKED_OBSERVATIONS, DRAW_MOVE_COUNT
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

args = get_args(
    run_name='brandubh_new',
    max_moves=DRAW_MOVE_COUNT,
    num_stacked_observations=NUM_STACKED_OBSERVATIONS,
    cpuct=1.25,
    symmetricSamples=False,
    numMCTSSims=100,
    numFastSims=15,
    numWarmupSims=5,
    probFastSim=0.5,
    
    selfPlayModelIter=None,
    skipSelfPlayIters=None,
    model_gating=True,
    max_gating_iters=None,
    numWarmupIters=1,
    arenaMCTS=True,
    baselineCompareFreq=1,
    pastCompareFreq=1,
    #train_sample_ratio=3,
    train_steps_per_iteration=64,
    min_next_model_winrate=0.52,
    use_draws_for_winrate=True,
    
    process_batch_size=512,
    train_batch_size=4096,
    arena_batch_size=64,
    arenaCompare=64*4,
    arenaCompareBaseline=64*4,
    gamesPerIteration=512*4,

    lr=1e-2,
    optimizer_args=dotdict({
        'momentum': 0.9,
        'weight_decay': 1e-3
    }),

    depth=8,
    num_channels=64,
    value_head_channels=8,
    policy_head_channels=8,
    value_dense_layers=[512, 128],
    policy_dense_layers=[512]
)


def raw_mcts_player():
    return RawMCTSPlayer(Game, args)


args.scheduler_args.milestones = [75, 125]
args.baselineTester = raw_mcts_player


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
