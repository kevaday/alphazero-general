import numpy, pyximport
pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.gobang.gobang import Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict
#from alphazero.envs.gobang.GobangPlayers import GreedyGobangPlayer


args = get_args(dotdict({
    'run_name': 'gobang',
    'workers': 7,
    'startIter': 0,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 512,
    'train_batch_size': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * 7,
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 300,
    'numFastSims': 100,
    'probFastSim': 0.65,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 10,
    'arenaCompare': 32,
    'arena_batch_size': 512,
    'arenaTemp': 0.25,
    'arenaMCTS': True,
    'baselineCompareFreq': 2,
    'compareWithPast': False,
    'pastCompareFreq': 10,
    'cpuct': 4,
    'fpu_reduction': 0.4,
    'load_model': True,
    'root_noise_frac': 0.3,
    'min_next_model_winrate': 0.54,
    'root_policy_temp': 1.3,
    'train_on_past_data': False,
    'past_data_chunk_size': 25,
    'past_data_run_name': 'gobang',
    'use_draws_for_winrate': False,
    "_num_players":2,

    'eloMCTS': 15,
    'eloGames':10,
    'eloMatches':10,
    'calculateElo': True,

}),
    model_gating=False,
    max_gating_iters=100,
    max_moves=36,

    lr=0.02,
    num_channels=128,
    depth=10,
    value_head_channels=64,
    policy_head_channels=64,
    value_dense_layers=[2048, 1024, 512],
    policy_dense_layers=[2048, 1024, 512]
    )
args.scheduler_args.milestones = [150, 225]




if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
