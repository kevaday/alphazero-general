import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.tafl.tafl import TaflGame as Game, NUM_STACKED_OBSERVATIONS, DRAW_MOVE_COUNT
from alphazero.tafl.players import GreedyTaflPlayer

from hnefatafl.engine import variants

"""
args = dotdict({
    'run_name': 'hnefatafl_run2',
    'cuda': cuda.is_available(),
    'workers': mp.cpu_count(),
    'startIter': 0,
    'numIters': 1000,
    'process_batch_size': 128,
    'train_batch_size': 512,
    'arena_batch_size': 32,
    'train_steps_per_iteration': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512,
    'numItersForTrainExamplesHistory': 10,
    'max_moves': DRAW_MOVE_COUNT,
    'num_stacked_observations': NUM_STACKED_OBSERVATIONS,
    'numWarmupIters': 1,  # Iterations where games are played randomly, 0 for none
    'skipSelfPlayIters': 0,
    'symmetricSamples': True,
    'numMCTSSims': 50,
    'numFastSims': 15,
    'numWarmupSims': 10,
    'probFastSim': 0.75,
    'tempThreshold': 10,
    'temp': 1,
    'compareWithTester': True,
    'arenaCompareTester': 16,
    'arenaCompare': 32*4,
    'arenaTemp': 0.25,
    'arenaMCTS': True,
    'arenaBatched': True,
    'testCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'model_gating': True,
    'min_next_model_winrate': 0.52,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 0,
        'iterations': 35
    }),
    'load_model': True,
    'cpuct': 1.25,
    'checkpoint': 'checkpoint/hnefatafl_run2',
    'data': 'data/hnefatafl_run2',

    'lr': 0.01,
    'num_channels': 64,
    'depth': 6,
})
"""


args = get_args(
    run_name='brandubh',
    max_moves=DRAW_MOVE_COUNT,
    num_stacked_observations=NUM_STACKED_OBSERVATIONS,
    cpuct=3,
    numWarmupIters=1,
    testCompareFreq=3,
    pastCompareFreq=3,
    compareTester=GreedyTaflPlayer,
    process_batch_size=512,
    train_batch_size=4096,
    arena_batch_size=64,
    train_steps_per_iteration=256,
    gamesPerIteration=2048,
    lr=0.01,
    num_channels=128,
    depth=16,
    value_head_channels=2,
    policy_head_channels=4,
    value_dense_layers=[128, 64],
    policy_dense_layers=[1024]
)


if __name__ == "__main__":
    g = Game(variants.brandubh, max_moves=args.max_moves, num_stacked_obs=args.num_stacked_observations)
    nnet = nn(g, args)
    # nnet.save_checkpoint(args.checkpoint, f'iteration-0000.pkl')
    c = Coach(g, nnet, args)
    c.learn()
