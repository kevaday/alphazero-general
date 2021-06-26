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
    from alphazero.envs.tafl.tafl import TaflGame as Game
    from alphazero.envs.tafl.train import args

    args.numMCTSSims = 2000
    args.tempThreshold = 100
    args.temp = 1
    args.arena_batch_size = 64

    # nnet players
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('./checkpoint/hnefatafl', 'iteration-0058.pkl')
    nn2 = NNet(Game, args)
    nn2.load_checkpoint('./checkpoint/hnefatafl', 'iteration-0058.pkl')
    #player1 = nn1.process
    #player2 = nn2.process

    player1 = MCTSPlayer(nn1, args=args)
    player2 = MCTSPlayer(nn2, args=args)
    #player2 = RandomPlayer()
    #player2 = GreedyTaflPlayer()
    #player2 = RandomPlayer()
    #player2 = OneStepLookaheadConnect4Player()
    #player2 = HumanTaflPlayer()

    players = [player1, player2]
    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=print)
    wins, draws, winrates = arena.play_game(verbose=True)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
