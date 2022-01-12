from pyximport import install
from numpy import get_include
install(setup_args={'include_dirs': get_include()})

from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.NNetWrapper import NNetWrapper as NNet
from alphazero.envs.stratego.stratego import Game as Game, display
from alphazero.envs.stratego.train import args
from alphazero.envs.stratego.players import HumanStrategoPlayer
import random

batched_arena = False
args.arenaCompare = 1
#args.arena_batch_size = 8
#args.workers = 4
#args.arenaCompare = args.workers * args.arena_batch_size
#args.numMCTSSims = 2000
#args.arena_batch_size = 64
#args.temp_scaling_fn = lambda x,y,z:0.25
#args2.temp_scaling_fn = args.temp_scaling_fn
#args.cuda = False
#args.add_root_noise = args.add_root_temp = False

# nnet players
#nn1 = NNet(Game, args)
#nn1.load_checkpoint('./checkpoint/' + args.run_name, 'iteration-0001.pkl')
#nn2 = NNet(Game, args)
#nn2.load_checkpoint('./checkpoint/brandubh2', 'iteration-0112.pkl')
#player1 = nn1.process
#player2 = nn1.process

#player1 = MCTSPlayer(Game, args, nn1, verbose=False)
#player2 = MCTSPlayer(Game, args, nn1, verbose=False)
#player2 = RandomPlayer()
#player2 = GreedyTaflPlayer()
#player2 = RandomPlayer()
#player2 = OneStepLookaheadConnect4Player()
player1 = RawMCTSPlayer(Game, args)
player2 = RawMCTSPlayer(Game, args)
#player2 = HumanFastaflPlayer()

players = [player1, player2]
#random.shuffle(players)

arena = Arena(players, Game, use_batched_mcts=batched_arena, args=args, display=display)
if batched_arena:
    wins, draws, winrates = arena.play_games(args.arenaCompare)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
else:
    arena.play_game(verbose=True)
