from fastafl import variants, errors
from enum import IntEnum
from collections import namedtuple
from typing import Union, NamedTuple, Tuple, Generator

import numpy as np


# region Tiles and pieces
class Piece(IntEnum):
    attacker = 1
    defender = 2
    king = 3
    king_on_throne = 6
    king_on_escape = 7


class Tile(IntEnum):
    normal = 0
    throne = 4
    escape = 5


class Square(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __get_tuple(self):
        return self.x, self.y

    def __str__(self):
        return self.__class__.__name__ + str(self.__get_tuple())

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __getitem__(self, index):
        return self.__get_tuple()[index]

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self.__get_tuple())

"""
class SquareType(IntEnum):
    normal = 0
    attacker = 1
    defender = 2
    king = 3
    throne = 4
    escape = 5
    king_on_throne = 6
    king_on_escape = 7
"""
# endregion


# region Constants
ALL_SQUARES = list(map(int, Piece))
ALL_SQUARES.extend(list(map(int, Tile)))
#Square = namedtuple('Square', 'x y')
#SquareT = NamedTuple[int, int]
RawState = Union[str, np.ndarray]

DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
# endregion


# region Helper functions
def _raise_invalid_board(board_data: RawState):
    raise errors.InvalidBoardState(f'An attempt was made to load an invalid board state. Got data:\n{board_data}')
# endregion


class Board(object):
    # region Load board
    def __init__(self, state: RawState = variants.hnefatafl, _copy_state=False):
        self.state = self.width = self.height = None
        if isinstance(state, str):
            self._load_str(state)
        elif isinstance(state, np.ndarray):
            self.state = np.copy(state) if _copy_state else state
        else:
            raise ValueError(f'Invalid type provided for state. Expected {RawState}, got {type(state)}')

        self.num_turns = 0

    def _load_str_inner(self, data: str) -> None:
        self.state = np.array([[int(tile) for tile in row] for row in data.splitlines()], dtype=np.intc)
        self.width, self.height = self.state.shape

    def _load_str(self, data: str, _skip_error_check=False) -> None:
        if _skip_error_check:
            self._load_str_inner(data)
            return

        import warnings
        with warnings.catch_warnings(record=True) as w:
            try:
                self._load_str_inner(data)
            except ValueError:
                _raise_invalid_board(data)

            if len(w) and isinstance(w[-1].category, np.VisibleDeprecationWarning):
                _raise_invalid_board(data)

            if not np.isin(self.state, ALL_SQUARES).all():
                _raise_invalid_board(data)
    #endregion

    #region Magic methods
    def __str__(self) -> str:
        return '\n'.join([' '.join([str(tile) for tile in row]) for row in self.state])

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}>{self.__dict__}'

    def __getitem__(self, key) -> int:
        if isinstance(key, Square):
            return self.state[key.x, key.y]
        else:
            return self.state[key]
    #endregion

    # region Valid moves
    def _in_bounds(self, square: Square) -> bool:
        return self.width - 1 >= square.x >= 0 and self.height - 1 >= square.y >= 0

    def _is_valid(self, square: Square, is_king=False):
        # No piece can go on a square out of bounds
        if not self._in_bounds(square):
            return False

        # Only king can go on escape square
        if self[square] == Tile.escape.value:
            return is_king

        # Allowed if the square is empty
        if self[square] == Tile.normal.value:
            return True

    def legal_moves(self, *pieces, piece_type: Piece = None) -> Generator[Tuple[Square, Square]]:
    # endregion

    # region Misc methods
    @staticmethod
    def _relative_tile(source: Square, direction: Tuple[int, int]) -> Square:
        return source + Square(*direction)

    def to_play(self) -> Piece:
        return Piece(2 - self.num_turns % 2)
    # endregion
