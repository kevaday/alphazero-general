import pyximport
pyximport.install()

from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.GenericPlayers import *
from alphazero.Arena import Arena
from pathlib import Path
from glob import glob

import numpy as np
import pprint
import choix

if __name__ == '__main__':
    from alphazero.envs.tafl.fastafl import Game as Game
    from alphazero.envs.tafl.train_fastafl import args

    print('Args:')
    pprint.pprint(args)
    if not Path('roundrobin').exists():
        Path('roundrobin').mkdir()
    print('Beginning round robin')
    networks = sorted(glob('roundrobin/*'), reverse=True)
    model_count = len(networks) + int(args.compareWithBaseline)

    if model_count <= 2:
        print(
            "Too few models for round robin. Please add models to the roundrobin/ directory"
        )
        exit()

    total_games = 0
    for i in range(model_count):
        total_games += i
    total_games *= args.arenaCompare
    print(
        f'Comparing {model_count} different models in {total_games} total games')
    win_matrix = np.zeros((model_count, model_count))

    nnet1 = nn(Game, args)
    nnet2 = nn(Game, args)

    for i in range(model_count - 1):
        for j in range(i + 1, model_count):
            file1 = Path(networks[i])
            file2 = Path('random' if args.compareWithBaseline and j == model_count - 1 else networks[j])
            print(f'{file1.stem} vs {file2.stem}')
            nnet1.load_checkpoint(folder='roundrobin', filename=file1.name)

            if file2.name != 'random':
                nnet2.load_checkpoint(folder='roundrobin', filename=file2.name)

                if args.arenaBatched:
                    if not args.arenaMCTS:
                        args.arenaMCTS = True
                        raise UserWarning(
                            'Batched arena comparison is enabled which uses MCTS, but arena MCTS is set to False.'
                            ' Ignoring this, and continuing with batched MCTS in arena.')

                    p1 = nnet1.process
                    p2 = nnet2.process
                else:
                    cls = MCTSPlayer if args.arenaMCTS else NNPlayer
                    p1 = cls(nnet1, args=args)
                    p2 = cls(nnet2, args=args)
            else:
                p1 = nnet1.process  #(MCTSPlayer if args.arenaMCTS else NNPlayer)(Game, nnet1, args=args)
                p2 = RawMCTSPlayer(Game, args).process

            arena = Arena([p1, p2], Game, use_batched_mcts=args.arenaBatched, args=args)
            wins, draws, winrates = arena.play_games(args.arenaCompare)
            win_matrix[i, j] = wins[0] + 0.5 * draws
            win_matrix[j, i] = wins[1] + 0.5 * draws
            print(f'wins: {wins[0]}, ties: {draws}, losses:{wins[1]}\n')

    print("\nWin Matrix(row beat column):")
    print(win_matrix)
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            params = choix.ilsr_pairwise_dense(win_matrix)
        print("\nRankings:")
        for i, player in enumerate(np.argsort(params)[::-1]):
            name = 'random' if args.compareWithBaseline and player == model_count - \
                               1 else Path(networks[player]).stem
            print(f"{i + 1}. {name} with {params[player]:0.2f} rating")
        print(
            "\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
    except Exception:
        print("\nNot Enough data to calculate rankings")
