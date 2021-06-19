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
    from alphazero.connect4.Connect4Game import Connect4Game as Game, display
    from alphazero.connect4.Connect4Players import HumanConnect4Player, OneStepLookaheadConnect4Player
    from alphazero.connect4.train import args

    import random

    args.numMCTSSims = 2000
    args.tempThreshold = 42
    args.temp = 1
    args.arena_batch_size = 64

    # all players
    # rp = RandomPlayer(g).play
    # gp = OneStepLookaheadConnect4Player(g).play
    player1 = HumanConnect4Player()

    # nnet players
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('./checkpoint/connect4', 'iteration-0173.pkl')
    # nn2 = NNet(g, args)
    # nn2.load_checkpoint('./checkpoint/connect4', 'iteration-0080.pkl')
    # player1 = nn1.process
    # player2 = nn2.process

    # player2 = NNPlayer(g, nn1, args=args, verbose=True).play
    player2 = MCTSPlayer(Game, nn1, args=args, verbose=True)
    # player2 = MCTSPlayer(g, nn2, reset_mcts=True, args=args).play
    # player2 = RandomPlayer()
    # player2 = GreedyTaflPlayer(g).play

    players = [player1, player2]
    random.shuffle(players)
    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=display)
    """
    wins, draws, winrates = arena.play_games(64)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
    """
    _, result = arena.play_game(verbose=True)
    print('Game result:', result)
