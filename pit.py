"""Pit generic or environment-specific players against one another.

Examples:
    python pit.py --env connect4 --players mcts human
    python pit.py --env brandubh --players RawMCTSPlayer random --games 20
"""

import argparse
import importlib

import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from alphazero import load_args_file, save_args_file
from alphazero.Arena import Arena
from alphazero.GenericPlayers import BasePlayer, MCTSPlayer, NNPlayer, RandomPlayer, RawMCTSPlayer
from alphazero.NNetWrapper import NNetWrapper
from alphazero.Coach import get_args
from train import ENVIRONMENTS, _args_from_namespace, _parser as train_parser, load_game


def _parser():
    parser = train_parser()
    parser.description = __doc__
    parser.add_argument(
        '--players', nargs='+', metavar='PLAYER',
        help='Player names in seat order. Defaults to raw_mcts for every seat.',
    )
    parser.add_argument(
        '--player-checkpoint', dest='player_checkpoints', nargs=2, action='append',
        metavar=('FOLDER', 'FILE'),
        help='Checkpoint for a player, repeated in seat order.',
    )
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--batched', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False)
    return parser


def _find_player_class(env, name):
    if ':' in name:
        module_name, class_name = name.split(':', 1)
        return getattr(importlib.import_module(module_name), class_name)
    wanted = name.casefold()
    module = importlib.import_module(f'alphazero.envs.{env}.players')
    for attribute in dir(module):
        candidate = getattr(module, attribute)
        if (isinstance(candidate, type) and issubclass(candidate, BasePlayer)
                and candidate is not BasePlayer
                and (attribute.casefold() == wanted or (
                    wanted == 'human' and attribute.casefold().startswith('human')))):
            return candidate
    raise ValueError(f'Unknown player {name!r} for environment {env!r}')


def _model(game, args, checkpoint):
    model = NNetWrapper(game, args)
    if checkpoint:
        model.load_checkpoint(checkpoint[0], checkpoint[1])
    return model


def create_player(spec, game, args, checkpoint=None):
    generic = {
        'random': RandomPlayer,
        'mcts': MCTSPlayer,
        'nn': NNPlayer,
        'raw_mcts': RawMCTSPlayer,
        'rawmcts': RawMCTSPlayer,
    }
    player_class = generic.get(spec.casefold()) or _find_player_class(args.env, spec)
    if player_class is RandomPlayer:
        return player_class(game_cls=game, args=args)
    if player_class is RawMCTSPlayer:
        return player_class(game_cls=game, args=args)
    if player_class in (MCTSPlayer, NNPlayer):
        model = _model(game, args, checkpoint)
        return player_class(model, game_cls=game, args=args)
    return player_class()


def main(argv=None):
    parsed = _parser().parse_args(argv)
    overrides, game_module = _args_from_namespace(parsed)
    for option in ('players', 'player_checkpoints', 'games', 'batched', 'verbose'):
        overrides.pop(option, None)
    args = load_args_file(parsed.load_args) if parsed.load_args else get_args()
    args.update(overrides)
    args.env = parsed.env
    if parsed.save_args:
        save_args_file(args, parsed.save_args)

    game = load_game(parsed.env, game_module)
    num_players = game.num_players()
    player_specs = parsed.players or ['raw_mcts'] * num_players
    if len(player_specs) != num_players:
        raise ValueError(
            f'Environment {parsed.env!r} requires {num_players} players, '
            f'but {len(player_specs)} were supplied.'
        )
    checkpoints = parsed.player_checkpoints or []
    if len(checkpoints) > num_players:
        raise ValueError(f'At most {num_players} checkpoints may be supplied.')
    checkpoints += [None] * (num_players - len(checkpoints))
    players = [
        create_player(spec, game, args, checkpoint)
        for spec, checkpoint in zip(player_specs, checkpoints)
    ]
    display = getattr(importlib.import_module(game_module or ENVIRONMENTS[parsed.env]), 'display', print)
    arena = Arena(players, game, use_batched_mcts=parsed.batched, args=args, display=display)
    if parsed.games == 1:
        arena.play_game(verbose=parsed.verbose)
    else:
        wins, draws, winrates = arena.play_games(parsed.games, verbose=parsed.verbose)
        print(f'wins: {wins}\ndraws: {draws}\nwinrates: {winrates}')
    return args


if __name__ == '__main__':
    main()
