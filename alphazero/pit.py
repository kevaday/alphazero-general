import pyximport
pyximport.install()

from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.NNetWrapper import NNetWrapper as NNet


"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
if __name__ == '__main__':
    from alphazero.tafl.tafl import TaflGame as Game
    from alphazero.tafl.players import HumanTaflPlayer, GreedyTaflPlayer
    from alphazero.tafl.train_test import args, variants

    g = Game(variants.brandubh, max_moves=args.max_moves, num_stacked_obs=args.num_stacked_observations)
    args.numMCTSSims = 1000
    args.tempThreshold = 0
    args.temp = 0.1
    args.arena_batch_size = 64

    # all players
    # rp = RandomPlayer(g).play
    # gp = OneStepLookaheadConnect4Player(g).play
    # hp = HumanTaflPlayer(g).play

    # nnet players
    nn1 = NNet(g, args)
    nn1.load_checkpoint('./checkpoint/hnefatafl_run2', 'iteration-0015.pkl')
    nn2 = NNet(g, args)
    nn2.load_checkpoint('./checkpoint/hnefatafl_run2', 'iteration-0000.pkl')
    player1 = nn1.process
    player2 = nn2.process

    #player1 = MCTSPlayer(g, nn1, reset_mcts=True, args=args).play
    #player2 = MCTSPlayer(g, nn2, reset_mcts=True, args=args).play
    #player2 = RandomPlayer(g).play
    #player2 = GreedyTaflPlayer(g).play

    players = [player1, player2]
    arena = Arena(players, g, use_batched_mcts=True, args=args, display=lambda b: print(b))
    wins, draws, winrates = arena.play_games(256)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
