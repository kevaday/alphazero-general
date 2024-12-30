import pyximport
pyximport.install()
from alphazero.envs.reversi_othello.othello import *
#from alphazero.envs.connect4.connect4 import *
import numpy as np
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.utils import dotdict
from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as NNet
import time
from subprocess import PIPE, STDOUT, Popen

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

args.numMCTSSims = 100
#args.arena_batch_size = 64
args.temp_scaling_fn = lambda x, y, z: 0
args.add_root_noise = args.add_root_temp = False

B = Game()

print(B.observation())
#P = RawMCTSPlayer(Game, args)

nn = NNet(Game, args)
nn.load_checkpoint('./checkpoint/reversi_othello', 'iteration-0016.pkl')
P = MCTSPlayer(nn, args=args)
#args.numMCTSSims = 2000
P2 = RawMCTSPlayer(Game, args)
#P.reset()

letters = "A B C D E F G H".split(" ")

def RUN(cpuct):
  B = Game()
  nn = NNet(Game, args)
  nn.load_checkpoint('D:/Programming Projects/Python/Machine Learning/AlphaZero/alphazero-general2/alphazero-general/checkpoint/reversi_othello', 'iteration-0016.pkl')
  P2.reset()
  args.cpuct = cpuct
  P = MCTSPlayer(nn, args=args)
  P.reset()
  turn = -1

  a_played = False
  b_played = False
  for N in range(100):
    #print(N)
    #B.display()
    #print(B.observation()[0])
    #print(B.win_state())
    v = B.valid_moves()
    rs = []
    for j, a in enumerate(v):
      if a == 1:
          rs.append(j)
    
    # for i, a in enumerate(B.valid_moves()):
    #   if a == 1:
    #     print((i // 8, i % 8), i)

    if turn == 1:
      a_played = True
      m = P.play(B)
      if m != 64:
        MOVE = str(letters[m // 8])+str(m % 8 + 1)
        #print(letters[m // 8], m % 8 + 1)
      else:
        MOVE = "pass"
      
    else:
      b_played = True
      m = P2.play(B)
      #print(m[0], m[1])
      # A = letters.index(m[0])
      # T = int(m[1])
      # m = A * 8 + T - 1
      #print(m)
    B.play_action(m)

    if B.win_state().any():
      break

    if a_played:
      P.update(B, m)
    if b_played:
      P2.update(B, m)
    turn *= -1

  return np.asarray(B.win_state())


def objective(config):
  GAMES = 10
  wins = 0
  for i in range(GAMES):
    print("GAME", i)
    if RUN(config["cpuct"])[1] == 1:
      wins += 1
  return wins/GAMES


for i in np.linspace(1, 4, 25):
  print("RESULT:", i, objective({"cpuct": i}))