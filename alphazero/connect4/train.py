import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.connect4.Connect4Game import Connect4Game as Game
from alphazero.utils import dotdict

args = get_args(dotdict({
    'run_name': 'connect4',
    'workers': mp.cpu_count(),
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 128,
    'train_batch_size': 512,
    'train_steps_per_iteration': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 4*128*mp.cpu_count(),
    '': 100,
    'symmetricSamples': True,
    'numMCTSSims': 50,
    'numFastSims': 5,
    'probFastSim': 0.75,
    'tempThreshold': 20,
    'temp': 1,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 16,
    'arenaCompare': 256,
    'arenaTemp': 0.1,
    'arenaMCTS': True,
    'baselineCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 3,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 0,
        'iterations': 35
    }),
    'cpuct': 3,
    'load_model': False,
    'checkpoint': 'checkpoint',
    'data': 'data',
}))

if __name__ == "__main__":
    g = Game()
    nnet = nn(g, args)
    c = Coach(g, nnet, args)
    c.learn()
