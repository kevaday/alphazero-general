import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.gobang.GobangGame import GobangGame as Game
from alphazero.utils import dotdict

args = dotdict({
    'run_name': 'gobang',
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
    'compareWithTester': True,
    'arenaCompareTester': 500,
    'arenaCompare': 500,
    'arenaTemp': 0.1,
    'arenaMCTS': False,
    'testCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 3,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 0,
        'iterations': 35
    }),
    'cpuct': 3,
    'load_model': False,
    'checkpoint': 'checkpoint/gobang',
    'data': 'data/gobang',
})

if __name__ == "__main__":
    g = Game()
    nnet = nn(g, args)
    c = Coach(g, nnet, args)
    c.learn()
