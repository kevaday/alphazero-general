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

    g = Game
    args.numMCTSSims = 100
    args.tempThreshold = 20
    args.temp = 0
    args.arena_batch_size = 64

    # all players
    # rp = RandomPlayer()
    # gp = OneStepLookaheadConnect4Player()
    # player2 = HumanConnect4Player()

    # nnet players
    nn1 = NNet(g, args)
    nn1.load_checkpoint('./checkpoint/connect4', 'iteration-0001.pkl')
    nn2 = NNet(g, args)
    nn2.load_checkpoint('./checkpoint/connect4', 'iteration-0000.pkl')
    #player1 = nn1.process
    #player2 = nn2.process

    player1 = MCTSPlayer(nn1, args=args)
    player2 = MCTSPlayer(nn2, args=args)
    #player2 = RandomPlayer()
    #player2 = GreedyTaflPlayer()

    players = [player1, player2]
    arena = Arena(players, g, use_batched_mcts=False, args=args, display=display)

    wins, draws, winrates = arena.play_games(args.arenaCompare)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
    # _, result = arena.play_game(verbose=True)
    # print(f'Player {result} won.')
