import pyximport; pyximport.install()

from torch import multiprocessing as mp

from Coach import Coach
from NNetWrapper import NNetWrapper as nn
from connect4.Connect4Game import Connect4Game as Game
from utils import *

args = dotdict({
    'run_name': 'connect4_hardcore',
    'workers': mp.cpu_count() - 1,
    'startIter': 1,
    'numIters': 1000,
    'process_batch_size': 128,
    'train_batch_size': 512,
    'train_steps_per_iteration': 500,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 4*128*(mp.cpu_count()-1),
    'numItersForTrainExamplesHistory': 100,
    'symmetricSamples': False,
    'numMCTSSims': 50,
    'numFastSims': 5,
    'probFastSim': 0.75,
    'tempThreshold': 10,
    'temp': 1,
    'compareWithRandom': True,
    'arenaCompareRandom': 500,
    'arenaCompare': 500,
    'arenaTemp': 0.1,
    'arenaMCTS': False,
    'randomCompareFreq': 1,
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
})

if __name__ == "__main__":
    g = Game()
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
