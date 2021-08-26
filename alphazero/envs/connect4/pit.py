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
    from alphazero.envs.connect4.Connect4Game import Game as Game, display
    from alphazero.envs.connect4.Connect4Players import HumanConnect4Player
    from alphazero.envs.connect4.train import args

    import random

    args.numMCTSSims = 2000
    #args.arena_batch_size = 64
    args.temp_scaling_fn = lambda x, y, z: 0
    args.add_root_noise = args.add_root_temp = False

    # all players
    # rp = RandomPlayer(g).play
    # gp = OneStepLookaheadConnect4Player(g).play
    player2 = HumanConnect4Player()

    # nnet players
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('./checkpoint/connect4_fpu', 'iteration-0035.pkl')
    #nn2 = NNet(Game, args)
    #nn2.load_checkpoint('./checkpoint/connect4', 'iteration-0094.pkl')
    #player1 = nn1.process
    #player2 = nn1.process

    # player2 = NNPlayer(g, nn1, args=args, verbose=True).play
    player1 = MCTSPlayer(Game, nn1, args=args, verbose=True, print_policy=True)#, draw_mcts=True, draw_depth=3)
    #args2 = args.copy()
    #args2.numMCTSSims = 10
    #player2 = MCTSPlayer(Game, nn1, args=args, verbose=True, draw_mcts=True, draw_depth=3)
    #player2 = RandomPlayer()
    #player2 = RawMCTSPlayer(Game, args).process

    players = [player2, player1]
    #random.shuffle(players)
    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=display)

    """
    wins, draws, winrates = arena.play_games(256)
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
    """

    _, result = arena.play_game(verbose=True)
    print('Game result:', result)
