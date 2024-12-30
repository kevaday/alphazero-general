import pyximport
pyximport.install()
from alphazero.envs.othello.othello import Game
#from alphazero.envs.connect4.connect4 import *
import numpy as np
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.utils import dotdict
from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as NNet



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
    'numMCTSSims': 200,
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
    value_dense_layers=[1024, 256],
    policy_dense_layers=[1024]
    )
args.scheduler_args.milestones = [75, 150]

#nn = NNet(Game, args)
#nn.load_checkpoint('./checkpoint/othello', 'iteration-0024.pkl')
#P = MCTSPlayer(nn, args=args)
P = RawMCTSPlayer(Game, args)
G = Game()


turn = 1
while True:
    G.display()
    m = 0
    v = G.valid_moves()
    for i in range(len(v)):
        if v[i] == 1:
            print(i)
    if turn == 1:
        m = int(input(">>>"))
    else:
        m = P.play(G)
    print("\n\n")
    print(m)
    G.play_action(m)
    turn *= -1
