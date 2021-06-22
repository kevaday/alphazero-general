import pyximport; pyximport.install()

from alphazero.GenericPlayers import *
from alphazero.MCTS import MCTS
from alphazero.Arena import Arena
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.othello import NNetSpecialWrapper as nns
from alphazero.envs.othello import OthelloGame as Game
from alphazero.utils import *
from tensorboardX import SummaryWriter
from pathlib import Path
from glob import glob

import numpy as np
import pprint


"""
use this script to play every x agents against a single agent and graph win rate.
"""

args = dotdict({
    'run_name': 'othello_better_teacher',
    'arenaCompare': 100,
    'arenaTemp': 0,
    'temp': 1,
    'tempThreshold': 10,
    # use zero if no montecarlo
    'numMCTSSims': 50,
    'cpuct': 1,
    'x': 10,
})

if __name__ == '__main__':
    print('Args:')
    pprint.pprint(args)
    benchmark_agent = "othello/special/6x6_153checkpoints_best.pth.tar"
    
    if args.run_name != '':
        writer = SummaryWriter(log_dir='runs/'+args.run_name)
    else:
        writer = SummaryWriter()
    if not Path('checkpoint').exists():
        Path('checkpoint').mkdir()
    print('Beginning comparison')
    networks = sorted(glob('checkpoint/*'))
    temp = networks[::args['x']]
    if temp[-1] != networks[-1]:
        temp.append(networks[-1])
    
    networks = temp
    model_count = len(networks)

    if model_count < 1:
        print(
            "Too few models for pit multi.")
        exit()

    total_games = model_count * args.arenaCompare
    print(
        f'Comparing {model_count} different models in {total_games} total games')

    g = Game(6)
    nnet1 = nns(g)
    nnet2 = nn(g)

    nnet1.load_checkpoint(folder="", filename=benchmark_agent)
    short_name = Path(benchmark_agent).stem

    if args.numMCTSSims <= 0:
        p1 = NNPlayer(g, nnet1, args.arenaTemp).play
    else:
        mcts1 = MCTS(g, nnet1, args)

        def p1(x, turn):
            if turn <= 2:
                mcts1.reset()
            temp = args.temp if turn <= args.tempThreshold else args.arenaTemp
            policy = mcts1.getActionProb(x, temp=temp)
            return np.random.choice(len(policy), p=policy)
    
    for i in range(model_count):
        file = Path(networks[i])
        print(f'{short_name} vs {file.stem}')

        nnet2.load_checkpoint(folder='checkpoint', filename=file.name)
        if args.numMCTSSims <= 0:
            p2 = NNPlayer(g, nnet2, args.arenaTemp).play
        else:
            mcts2 = MCTS(g, nnet2, args)

            def p2(x, turn):
                if turn <= 2:
                    mcts2.reset()
                temp = args.temp if turn <= args.tempThreshold else args.arenaTemp
                policy = mcts2.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

        arena = Arena(p1, p2, g)
        p1wins, p2wins, draws = arena.play_games(args.arenaCompare)
        writer.add_scalar(
            f'Win Rate vs {short_name}', (p2wins + 0.5*draws)/args.arenaCompare, i*args.x)
        print(f'wins: {p1wins}, ties: {draws}, losses:{p2wins}\n')
    writer.close()