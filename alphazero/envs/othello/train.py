import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.othello.othello import Game
from alphazero.utils import dotdict

#from alphazero.envs.othello.OthelloPlayers import GreedyOthelloPlayer


args = get_args(dotdict({
    'run_name': 'othello',
    'workers': 7,
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 512,
    'train_batch_size': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * 3,
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
    'cpuct': 4,
    'fpu_reduction': 0,
    'load_model': True,
    'startTemp': 1,
    '_num_players': 2
}),
    model_gating=True,
    max_gating_iters=10,
    max_moves=64,

    lr=0.01,
    num_channels=128,
    depth=8,
    value_head_channels=32,
    policy_head_channels=32,
    value_dense_layers=[2048, 256],
    policy_dense_layers=[2048]
    )
args.scheduler_args.milestones = [75, 150]



if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
