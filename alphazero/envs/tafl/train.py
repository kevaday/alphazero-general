import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.tafl.tafl import TaflGame as Game, NUM_STACKED_OBSERVATIONS, DRAW_MOVE_COUNT
from alphazero.utils import dotdict

args = get_args(
    run_name='hnefatafl',
    max_moves=DRAW_MOVE_COUNT,
    num_stacked_observations=NUM_STACKED_OBSERVATIONS,
    cpuct=4,
    symmetricSamples=True,
    numMCTSSims=50,
    numFastSims=5,
    numWarmupSims=5,
    probFastSim=0.75,
    # mctsResetThreshold=DRAW_MOVE_COUNT // 2,
    tempThreshold=int(DRAW_MOVE_COUNT*0.7),
    
    skipSelfPlayIters=None,
    model_gating=True,
    max_gating_iters=3,
    numWarmupIters=1,
    arenaCompareBaseline=8,
    baselineCompareFreq=3,
    pastCompareFreq=3,
    # baselineTester=GreedyTaflPlayer,
    min_next_model_winrate=0.52,
    use_draws_for_winrate=False,

    minTrainHistoryWindow=1,
    maxTrainHistoryWindow=2,
    trainHistoryIncrementIters=10,
    
    process_batch_size=32,
    train_batch_size=1024,
    arena_batch_size=32,
    arenaCompare=32*4,
    gamesPerIteration=32*4,
    
    scheduler_args=dotdict({
        'min_lr': 1e-4,
        'patience': 2,
        'cooldown': 1,
        'verbose': True
    }),
    optimizer_args=dotdict({
        'lr': 1e-2,
        'momentum': 0.9,
        'weight_decay': 1e-4
    }),

    num_channels=128,
    depth=16,
    value_head_channels=32,
    policy_head_channels=32,
    value_dense_layers=[2048, 1024],
    policy_dense_layers=[2048]
)


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
