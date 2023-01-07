import numpy, pyximport
pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.gobang.gobang import Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict


args = get_args(dotdict({
    'run_name': 'reversi_othello',
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
    'numMCTSSims': 1300,
    'numFastSims': 40,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 128,
    'arenaCompare': 128,
    'arena_batch_size': 128,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'baselineCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'cpuct': 3.85,
    'fpu_reduction': 0.16,
    'load_model': True,
    'root_noise_frac': 0.03,
    'min_next_model_winrate': 0.56,
    '_num_players': 2

}),
    model_gating=True,
    max_gating_iters=100,
    max_moves=64,

    lr=0.01,
    num_channels=128,
    depth=10,
    value_head_channels=128,
    policy_head_channels=128,
    value_dense_layers=[1024, 512],
    policy_dense_layers=[1024]
    )
args.scheduler_args.milestones = [75, 150]


P = RawMCTSPlayer(Game, args)
G = Game()
G.display()

for i in range(15**2):
  m = P(G)
  G.play_action(m)
  G.display()
  P.update(G, m)