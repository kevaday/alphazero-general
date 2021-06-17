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

    g = Game()
    args.numMCTSSims = 100
    args.tempThreshold = 20
    args.temp = 0
    args.arena_batch_size = 64

    # all players
    # rp = RandomPlayer(g).play
    # gp = OneStepLookaheadConnect4Player(g).play
    # player2 = HumanConnect4Player(g).play

    # nnet players
    nn1 = NNet(g, args)
    nn1.load_checkpoint('./checkpoint/connect4', 'iteration-0097.pkl')
    nn2 = NNet(g, args)
    nn2.load_checkpoint('./checkpoint/connect4', 'iteration-0080.pkl')
    #player1 = nn1.process
    #player2 = nn2.process

    player1 = MCTSPlayer(g, nn1, reset_mcts=True, args=args).play
    player2 = MCTSPlayer(g, nn2, reset_mcts=True, args=args).play
    #player2 = RandomPlayer(g).play
    #player2 = GreedyTaflPlayer(g).play

    players = [player1, player2]
    arena = Arena(players, g, use_batched_mcts=False, args=args, display=display)
    wins, draws, winrates = arena.play_games(64)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
    # result, player, board = arena.play_game(verbose=True)
    # print(result, player)
