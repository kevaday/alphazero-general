import numpy, pyximport
pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.gobang.gobang import Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict
#from alphazero.envs.gobang.GobangPlayers import GreedyGobangPlayer

args = get_args(dotdict({
    'run_name': 'gomoku_standard',
    'workers': 7,
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 512,
    'train_batch_size': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * 5,
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 500,
    'numFastSims': 40,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 128,
    'arenaCompare': 128,
    'arena_batch_size': 128,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'baselineCompareFreq': 3,
    'compareWithPast': True,
    'pastCompareFreq': 3,
    'cpuct': 5,
    'fpu_reduction': -0.01,
    'load_model': True,
    'root_noise_frac': 0.3,
    'min_next_model_winrate': 0.56,

}),
    model_gating=True,
    max_gating_iters=100,
    max_moves=15*15,

    lr=0.001,
    num_channels=128,
    depth=10,
    value_head_channels=128,
    policy_head_channels=128,
    value_dense_layers=[2048, 512],
    policy_dense_layers=[2048]
    )
args.scheduler_args.milestones = [75, 150]



if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
