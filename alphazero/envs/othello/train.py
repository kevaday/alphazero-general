import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.reversi_othello.othello import Game
from alphazero.utils import dotdict

#from alphazero.envs.othello.OthelloPlayers import GreedyOthelloPlayer


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
    'numMCTSSims': 300,
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
    'cpuct': 1.25,
    'fpu_reduction': 0.13,
    'load_model': True,
    'root_noise_frac': 0.25,
    'min_next_model_winrate': 0.6,

}),
    model_gating=False,
    max_gating_iters=100,
    max_moves=64,

    lr=0.01,
    num_channels=256,
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
