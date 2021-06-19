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
    from alphazero.tafl.train import args

    args.numMCTSSims = 100
    args.tempThreshold = 100
    args.temp = 0.1
    args.arena_batch_size = 64

    # all players
    # rp = RandomPlayer(g).play
    # gp = OneStepLookaheadConnect4Player(g).play
    # hp = HumanTaflPlayer(g).play

    # nnet players
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('./checkpoint/hnefatafl', 'iteration-0001.pkl')
    #nn2 = NNet(Game, args)
    #nn2.load_checkpoint('./checkpoint/hnefatafl', 'iteration-0000.pkl')
    #player1 = nn1.process
    #player2 = nn2.process

    player1 = MCTSPlayer(nn1, args=args)
    #player2 = MCTSPlayer(nn2, args=args)
    #player2 = RandomPlayer()
    player2 = GreedyTaflPlayer()

    players = [player1, player2]
    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=print)
    wins, draws, winrates = arena.play_game(verbose=True)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
