"""Train an AlphaZero model for any environment in ``alphazero.envs``.

Examples:
    python train.py --env connect4 --numMCTSSims 200
    python train.py --env tictactoe --load-args args.json --save-args args.json
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from alphazero import load_args_file, save_args_file
from alphazero.Coach import Coach, DEFAULT_ARGS, get_args
from alphazero.NNetWrapper import NNetWrapper


def discover_environments():
    env_root = Path(__file__).resolve().parent / 'alphazero' / 'envs'
    environments = {}
    for directory in env_root.iterdir():
        if not directory.is_dir() or directory.name.startswith('_'):
            continue
        module_file = directory / f'{directory.name}.py'
        cython_file = directory / f'{directory.name}.pyx'
        if module_file.exists() or cython_file.exists():
            environments[directory.name] = f'alphazero.envs.{directory.name}.{directory.name}'
    return environments


ENVIRONMENTS = discover_environments()


def _json_value(value):
    if isinstance(value, (list, dict)):
        return json.loads(value) if isinstance(value, str) else value
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=sorted(ENVIRONMENTS), required=True)
    parser.add_argument('--load-args', metavar='PATH')
    parser.add_argument('--save-args', metavar='PATH')
    parser.add_argument('--game-module', help='Override the module containing Game')
    for name, default in DEFAULT_ARGS.items():
        option = '--' + name.replace('_', '-')
        kwargs = {'dest': name, 'default': argparse.SUPPRESS}
        if isinstance(default, bool):
            kwargs['action'] = argparse.BooleanOptionalAction
        elif isinstance(default, (list, dict)):
            kwargs['type'] = json.loads
        else:
            kwargs['type'] = type(default) if default is not None else str
        parser.add_argument(option, **kwargs)
    parser.add_argument(
        '--arg', action='append', metavar='NAME=VALUE',
        help='Set an argument not present in the default argument set (JSON values are supported).',
    )
    return parser


def _args_from_namespace(namespace):
    values = vars(namespace).copy()
    values.pop('env', None)
    values.pop('load_args', None)
    values.pop('save_args', None)
    game_module = values.pop('game_module', None)
    extra = values.pop('arg', None) or []
    for item in extra:
        if '=' not in item:
            raise ValueError(f'Invalid --arg value {item!r}; expected NAME=VALUE')
        key, value = item.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        values[key] = value
    return values, game_module


def load_game(env, module_name=None):
    module = importlib.import_module(module_name or ENVIRONMENTS[env])
    try:
        return module.Game
    except AttributeError as exc:
        raise ValueError(f'Environment module {module.__name__!r} does not define Game') from exc


def main(argv=None):
    parsed = _parser().parse_args(argv)
    overrides, game_module = _args_from_namespace(parsed)
    args = load_args_file(parsed.load_args) if parsed.load_args else get_args()
    args.update(overrides)
    if parsed.save_args:
        save_args_file(args, parsed.save_args)

    game = load_game(parsed.env, game_module)
    Coach(game, NNetWrapper(game, args), args).learn()
    return args


if __name__ == '__main__':
    main()
