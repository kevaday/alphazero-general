import pyximport; pyximport.install()

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.connect4.connect4 import Game
from alphazero.envs.reversi_othello.othello import Game as Game2
from alphazero.utils import dotdict
import numpy as np
from alphazero.envs.reversi_othello.train import args
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.Arena import Arena
GG = Game2()


args.numMCTSSims = 15
# args.temp_scaling_fn = lambda x, y, z: 0
# args.root_noise_frac = 0
# args.add_root_noise = args.add_root_temp = False
# args.fpu_reduction = 0

nnet = nn(Game2, args)
nnet2 = nn(Game2, args)
cls = MCTSPlayer
nplayer = cls(nnet, Game2, args)
nplayer2 = cls(nnet2, Game2, args)


players = [nplayer] + [nplayer2] * (Game2.num_players() - 1)
arena = Arena(players, Game2, use_batched_mcts=False, args=args)
a, b, c = arena.play_games(5, verbose=False)
print(a, b, c)