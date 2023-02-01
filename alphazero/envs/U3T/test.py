import pyximport; pyximport.install()
from alphazero.envs.U3T.U3T import Game
import numpy as np
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.utils import dotdict
from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as NNet
# B = Board()
# # B.place_piece(0,2, 2,0,-1)
# # B.place_piece(0,2, 1,1,-1)
# # B.place_piece(0,2, 0,2,-1)

# # B.place_piece(1,1, 2,0,-1)
# # B.place_piece(1,1, 1,1,-1)
# # B.place_piece(1,1, 0,2,-1)

# # B.place_piece(2,0, 2,0,-1)
# # B.place_piece(2,0, 1,1,-1)
# # B.place_piece(2,0, 0,2,-1)

# print(np.asarray(B.pieces))
# print(B.num_to_point(B.point_to_num(2, 1, 1, 2)))

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
    'numMCTSSims': 500,
    'numFastSims': 40,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 128,
    'arenaCompare': 128,
    'arena_batch_size': 128,
    'arenaTemp': 0.25,
    'arenaMCTS': True,
    'baselineCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'cpuct': 2.5,
    'fpu_reduction': 0,
    'load_model': True,
    '_num_players': 2
}),
    model_gating=True,
    max_gating_iters=100,
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

args.numMCTSSims = 500
args.temp_scaling_fn = lambda x, y, z: 0
args.root_noise_frac = 0
args.add_root_noise = args.add_root_temp = False
args.fpu_reduction = 0

G = Game()
print(G.observation())
nn = NNet(Game, args)
nn.load_checkpoint('./checkpoint/U3T', 'iteration-0060.pkl')

P = MCTSPlayer(nn, args=args)
P2 = RawMCTSPlayer(Game, args)
turn = 1

for i in range(81):
    if turn == 1:
        a = P.play(G)
        G.play_action(a)
        POINT = G._board.num_to_point(a)
        print("AZ played", POINT)
        B = np.asarray(G._board.pieces)
        print(B[POINT[0]][POINT[1]])
        #P.update(G, a)
    else:
        #a = P2.play(G)
        v = G.valid_moves()
        for i in range(len(v)):
            if v[i] == 1:
                print(G._board.num_to_point(i), i)
        a = int(input(">>>"))#P2.play(G)
        G.play_action(a)
        


    P.update(G, a)
    
    #print(G.observation())
    G.display()
	
    if np.any(G.win_state()):
        print(np.asarray(G._board.pieces))
        print(G.win_state())
        break
    turn *= -1
