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
    def restore(value):
        if isinstance(value, str) and value.startswith(CALLABLE_PREFIX):
            try:
                return eval(value[len(CALLABLE_PREFIX):])
            except Exception as e:
                raise RuntimeError('Failed to parse argument file: ' + str(e)) from e
        if isinstance(value, dict):
            return dotdict({k: restore(v) for k, v in value.items()})
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value

    with open(filepath, 'r') as f:
        return restore(json.load(f))


def save_args_file(args: dotdict or dict, filepath, replace=True):
    if not replace and os.path.exists(filepath): return

    def serialise(value):
        if callable(value):
            return CALLABLE_PREFIX + value.__name__
        if isinstance(value, dict):
            return {k: serialise(item) for k, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serialise(item) for item in value]
        return value

    save_args = serialise(dict(args))

    with open(filepath, 'w') as f:
        json.dump(save_args, f)

    return save_args
