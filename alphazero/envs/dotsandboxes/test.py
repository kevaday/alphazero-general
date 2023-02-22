import pyximport
pyximport.install()
from alphazero.envs.dotsandboxes.dab import *
#from alphazero.envs.connect4.connect4 import *
import numpy as np
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.utils import dotdict
from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as NNet
import time


g = Game()
args = get_args(dotdict({
    'run_name': 'dotsandboxes',
    'workers': 7,
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 512,
    'train_batch_size': 1024,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * 7,
    'symmetricSamples': False,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 300,
    'numFastSims': 40,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompare': 16 * 7,
    'arena_batch_size': 16,
    'arenaTemp': 0.25,
    'arenaMCTS': True,
    'baselineCompareFreq': 10,
    'compareWithPast': False, #elo caclulation enabled to this is not needed
    'pastCompareFreq': 10,
    'cpuct': 3.5,
    'fpu_reduction': 0.3,
    'load_model': True,
    'root_policy_temp': 1.3,
    'root_noise_frac': 0.3,
    "_num_players": 2,
    #Elo
    'eloMCTS': 25,
    'eloGames':10,
    'eloMatches':10,
    'calculateElo': True,
}),
    model_gating=False,
    max_gating_iters=None,
    max_moves=42,

    lr=0.01,
    num_channels=128,
    depth=8,
    value_head_channels=32,
    policy_head_channels=32,
    value_dense_layers=[1024, 256],
    policy_dense_layers=[1024]
)
args.scheduler_args.milestones = [75, 150]
args.temp_scaling_fn = lambda x, y, z: 0
args.root_noise_frac = 0
args.add_root_noise = args.add_root_temp = False
args.fpu_reduction = 0

G = Game()
nn = NNet(Game, args)
nn.load_checkpoint('./checkpoint/dotsandboxes', 'iteration-0006.pkl')

P = MCTSPlayer(nn, args=args)
P2 = RawMCTSPlayer(Game, args)
turn = 1

#G.play_action(0)
for i in range(81):
    if turn == 1:
        a = P.play(G)
        G.play_action(a)
    else:
        v = G.valid_moves()
        for i in range(len(v)):
            if v[i] == 1:
                print(i, end=" ")
        a = int(input(">>>"))
        G.play_action(a)
        
    P.update(G, a)
    G.display()

    if np.any(G.win_state()):
        print(np.asarray(G._board.pieces))
        print(G.win_state())
        break
    turn *= -1

