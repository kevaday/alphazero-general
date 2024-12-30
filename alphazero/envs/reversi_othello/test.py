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

args.numMCTSSims = 500
args.arena_batch_size = 64
args.temp_scaling_fn = lambda x, y, z: 0
args.root_noise_frac = 0
args.add_root_noise = args.add_root_temp = False
args.fpu_reduction = 0

B = Game()

print(B.observation())
#P = RawMCTSPlayer(Game, args)

nn = NNet(Game, args)
nn.load_checkpoint('./checkpoint/reversi_othello', 'iteration-0029.pkl')
P = MCTSPlayer(nn, args=args)
#args.numMCTSSims = 2000
#P2 = RawMCTSPlayer(Game, args)
#P.reset()

#Iter 20 beats Level 0, 1000 MCTS
#Iter 22 beats level 1, 1000 MCTS
#Iter 47 beats level 2, 1000 MCTS
#Iter 47 bears level 3, 100 MCTS

edax = Popen("alphazero\\envs\\reversi_othello\\edax\\bin\\wEdax-x64-modern.exe --level 1", shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
def read_stdout():
  out = b''
  while True:
      next_b = edax.stdout.read(1)
      if next_b == b'>' and ((len(out) > 0 and out[-1] == 10) or len(out) == 0):
          break
      else:
          out += next_b
  return out.decode("utf-8")
def write_stdin(command):
  edax.stdin.write(str.encode(command + "\n"))
  edax.stdin.flush()
def get_edax_move():
  #read_stdout()
  write_stdin("go")
  return read_stdout().split("plays ")[-1][:2]
def play_edax(m):
  write_stdin(m)
  read_stdout()
  

read_stdout()
letters = "A B C D E F G H".split(" ")
turn = -1

MOVES = []
for N in range(100):
  B.display()
  #print(B.observation()[0])
  az = B._board.get_total(1)
  ED = B._board.get_total(-1)
  print(f"AlphaZero:{az}, Edax:{ED}")
  v = B.valid_moves()
  rs = []
  for j, a in enumerate(v):
    if a == 1:
        rs.append(j)
  
  # for i, a in enumerate(B.valid_moves()):
  #   if a == 1:
  #     print((i // 8, i % 8), i)

  if turn == 1:
    m = P.play(B)
    if m != 64:
      MOVE = str(letters[m // 8])+str(m % 8 + 1)
      #print(letters[m // 8], m % 8 + 1)
    else:
      MOVE = "pass"
    play_edax(MOVE)
    print(MOVE)
    
  else:
    m = get_edax_move()
    #print(m[0], m[1])
    print(m)
    if m[0] == "P":
      m = 64
    else:
      A = letters.index(m[0])
      T = int(m[1])
      m = A * 8 + T - 1
  
  MOVES.append(m)
  B.play_action(m)
  turn *= -1

  P.update(B, m)
  if B.win_state().any():
    
    break

B.display()
az = B._board.get_total(1)
ED = B._board.get_total(-1)
print(f"AlphaZero:{az}, Edax:{ED}")
print(MOVES)
print(B.win_state())
print(B.player)
# if B.player == 0 and B.win_state()[1] == 1:
#   print("AlphaZero wins!")
# elif B.player == 0 and B.win_state()[0] == 1:
#   print("Edax wins!")
# else:
#   print("Draw!")