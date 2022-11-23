from pyximport import install as pyxinstall
from numpy import get_include

pyxinstall(setup_args={'include_dirs': get_include()})

from alphazero.Coach import DEFAULT_ARGS
from alphazero.Game import GameState

# Options for args eval
from torch.optim import *
from torch.optim.lr_scheduler import *
from alphazero.GenericPlayers import *
from alphazero.utils import default_temp_scaling, const_temp_scaling

import json
import os

CALLABLE_PREFIX = '__CALLABLE__'


def load_args_file(filepath: str) -> dotdict:
    new_args = dotdict()
    raw_args = json.load(open(filepath, 'r'))

    for k, v in raw_args.items():
        if isinstance(v, str) and CALLABLE_PREFIX in v:
            try:
                v = eval(v.replace(CALLABLE_PREFIX, ''))
            except Exception as e:
                raise RuntimeError('Failed to parse argument file: ' + str(e))

        elif isinstance(v, dict):
            v = dotdict(v)

        new_args.update({k: v})

    return new_args


def save_args_file(args: dotdict or dict, filepath, replace=True):
    if not replace and os.path.exists(filepath): return

    save_args = dict()
    for k, v in args.items():
        if callable(v):
            v = CALLABLE_PREFIX + v.__name__
        save_args.update({k: v})

    with open(filepath, 'w') as f:
        json.dump(save_args, f)

    return save_args
