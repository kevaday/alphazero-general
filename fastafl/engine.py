from fastafl import variants, errors
from enum import IntEnum
from collections import namedtuple
from typing import Union, NamedTuple, Tuple, Optional, Callable, Iterator, Iterable, List

import numpy as np


# region Tiles and pieces
class Piece(IntEnum):
    attacker = 1
    defender = 2
    king = 3
    king_on_throne = 7
    king_on_escape = 8


class Tile(IntEnum):
    normal = 0
    throne = 4
    escape = 5


class Square(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def _get_tuple(self):
        return self.x, self.y

    def __str__(self):
        return self.__class__.__name__ + str(self._get_tuple())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self._get_tuple() == other._get_tuple()

    def __hash__(self):
        return np.prod(self._get_tuple())

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __getitem__(self, index):
        return self._get_tuple()[index]

    def __setitem__(self, index, value):
        vals = list(self._get_tuple())
        vals[index] = vals
        self.x, self.y = vals

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self._get_tuple())


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
ALL_PIECES = list(map(int, Piece))
KING_ON_TILE = (Piece.king_on_throne, Piece.king_on_escape)
BASE_PIECES = [piece for piece in ALL_PIECES if piece not in KING_ON_TILE]
ALL_TILES = list(map(int, Tile))
ALL_SQUARES = ALL_PIECES + ALL_TILES
KING_VALUES = (Piece.king, *KING_ON_TILE)
ATTACKERS = (Piece.attacker, *KING_VALUES)
SPECIAL_TILES = (Tile.throne, Tile.escape)
KING_CAPTURE = (Piece.defender, *SPECIAL_TILES)

# Square = namedtuple('Square', 'x y')
# SquareT = NamedTuple[int, int]
RawStateT = Union[str, np.ndarray]
MoveT = Tuple[Square, Square]

DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


# endregion


# region Helper functions
def _raise_invalid_board(board_data: RawStateT):
    raise errors.InvalidBoardState(f'An attempt was made to load an invalid board state. Got data:\n{board_data}')


def _get_key(key) -> Square:
    if not isinstance(key, Square):
        if not isinstance(key, Iterable):
            raise TypeError('Key for setitem must be iterable.')
        return Square(*key)
    return key


# endregion


# region Board class
class Board(object):
    # region Load board
    def __init__(self, state: RawStateT = variants.hnefatafl, king_two_sided_capture=False, _copy_state=False):
        self._state = self.width = self.height = None
        if isinstance(state, str):
            self._load_str(state)
        elif isinstance(state, np.ndarray):
            self._state = np.copy(state) if _copy_state else state
        else:
            raise ValueError(f'Invalid type provided for state. Expected {RawStateT}, got {type(state)}')

        self.king_two_sided_capture = king_two_sided_capture
        self.num_turns = 0
        self._king_captured = False
        self._king_escaped = False

    def _load_str_inner(self, data: str) -> None:
        self._state = np.array([[int(tile) for tile in row] for row in data.splitlines()], dtype=np.intc)
        self.height, self.width = self._state.shape

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

            if not np.isin(self._state, ALL_SQUARES).all():
                _raise_invalid_board(data)

    def copy(self):
        return self.__copy__()

    # endregion

    # region Magic methods
    def __str__(self) -> str:
        return '\n'.join([' '.join([str(tile) for tile in row]) for row in self._state])

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}>{self.__dict__}'

    def __getitem__(self, key) -> int:
        key = _get_key(key)
        return self._state[key.y, key.x]

    def __setitem__(self, key, value):
        key = _get_key(key)
        self._state[key.y, key.x] = value

    def __eq__(self, other):
        return (
            (self._state == other._state).all()
            and self.num_turns == other.num_turns
            and self.king_two_sided_capture == other.king_two_sided_capture
        )

    def __hash__(self):
        return hash(self._state) + self.num_turns + int(self.king_two_sided_capture)

    def __copy__(self):
        new_dict = self.__dict__.copy()
        new_dict['_state'] = np.copy(self._state)
        board = self.__new__(self.__class__)
        board.__dict__.update(new_dict)
        return board

    def __deepcopy__(self, memodict):
        return self.__copy__()

    # endregion

    # region Legal moves
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

    def _get_piece_squares(self, condition: Callable[[int], bool]) -> Iterator[Square]:
        for x in range(self.width):
            for y in range(self.height):
                square = Square(x, y)
                if condition(self[square]):
                    yield square

    def legal_moves(self, *pieces: Optional[Tuple[Square, ...]], piece_type: Piece = None) \
            -> Iterator[MoveT]:
        def iter_pieces() -> Iterator[Square]:
            for piece in pieces:
                if self[piece] not in ALL_PIECES:
                    continue
                yield piece

            if piece_type:
                for piece in self._get_piece_squares(lambda x: x == piece_type.value):
                    yield piece
            elif not pieces:
                for piece in self._get_piece_squares(lambda x: x not in ALL_TILES):
                    yield piece

        for piece_square in iter_pieces():
            is_king = self[piece_square] in KING_VALUES
            for move_dir in DIRECTIONS:
                cur_square = self._relative_square(piece_square, move_dir)
                while self._is_valid(cur_square, is_king):
                    yield piece_square, cur_square
                    cur_square = self._relative_square(cur_square, move_dir)

    def has_legal_moves(self, *pieces: Optional[Tuple[Square, ...]], piece_type: Piece = None) -> bool:
        try:
            next(self.legal_moves(*pieces, piece_type=piece_type))
        except StopIteration:
            return False
        else:
            return True

    # endregion

    # region Win state
    def king_escaped(self) -> bool:
        if self._king_escaped:
            return True
        return (self._state == Piece.king_on_escape.value).any()

    def king_captured(self) -> bool:
        if self._king_captured:
            return True
        elif self.king_two_sided_capture:
            return False
        king_mask = (self._state == Piece.king.value) | (self._state == Piece.king_on_throne.value) \
            | (self._state == Piece.king_on_escape.value)

        return any([all(map(lambda x: self[x] in KING_CAPTURE, self._surrounding_squares(Square(*reversed(pos)))))
                    for pos in zip(*np.where(king_mask))])

    def get_winner(self) -> Piece:
        if self.king_escaped() or not self.has_legal_moves(piece_type=Piece.defender):
            return Piece.attacker
        elif self.king_captured() or not self.has_legal_moves(piece_type=Piece.attacker):
            return Piece.defender

    def is_game_over(self) -> bool:
        return self.get_winner() is not None

    # endregion

    # region Capture and move logic
    def _check_capture(self, moved_piece: Square):
        piece_val = self[moved_piece]
        friendly = ATTACKERS if piece_val in ATTACKERS else (piece_val,)
        enemy = 3 - piece_val if piece_val != Piece.king.value else Piece.defender.value

        for check_dir in DIRECTIONS:
            enemy_square = self._relative_square(moved_piece, check_dir)
            if not self._in_bounds(enemy_square): continue

            value = self[enemy_square]
            enemy_is_king = value in KING_VALUES
            if value == enemy or (self.king_two_sided_capture and enemy_is_king):
                friendly_square = self._relative_square(enemy_square, check_dir)
                if not self._in_bounds(friendly_square): continue

                value = self[friendly_square]
                if value in friendly or value in SPECIAL_TILES:
                    if self.king_two_sided_capture and enemy_is_king:
                        self._king_captured = True
                    else:
                        self._set_square(enemy_square, Tile.normal)

    def _check_surround(self, moved_piece: Square, _check_for_error=True):
        enemy = ATTACKERS if self[moved_piece] == Piece.defender.value else (Piece.defender.value,)

        def next_check_squares(squares: Iterable[Square]) -> Tuple[Square, ...]:
            return tuple(filter(lambda sq: self[sq] in enemy, squares))

        def _blocked(square: Square) -> bool:
            return self[square] != Tile.normal.value

        def recurse_check(square: Square, checked: List[Square]) -> Tuple[bool, bool]:
            checked.append(square)
            squares = tuple(self._surrounding_squares(square))

            if all(tuple(map(_blocked, squares))):
                captured_check = []
                exit_recurse = False

                for sq in filter(lambda x: x not in checked, next_check_squares(squares)):
                    is_captured, exit_recurse = recurse_check(sq, checked)
                    captured_check.append(is_captured)
                    if exit_recurse:
                        break

                return all(captured_check), exit_recurse
            else:
                return False, True

        start_squares = next_check_squares(self._surrounding_squares(moved_piece))
        if not start_squares: return

        checked_squares = []
        for s in start_squares:
            if s not in checked_squares:
                to_capture = []

                if recurse_check(s, to_capture)[0]:
                    for captured in to_capture:
                        if self[captured] in KING_VALUES:
                            self._king_captured = True
                        else:
                            self.remove_piece(captured, raise_no_piece=_check_for_error)

                checked_squares.extend(to_capture)

    def move(self, source: Square, dest: Square, _check_valid=True, _check_win=True):
        source_val = self[source]
        if _check_valid:
            if (source, dest) not in self.legal_moves(source):
                raise errors.InvalidMoveError(
                    f'The move {source}->{dest} is illegal. Source value: {source_val}, dest. value: {self[dest]}'
                )

        self[dest] = self.remove_piece(source, raise_no_piece=_check_valid)
        self._check_capture(dest)
        self._check_surround(dest, _check_for_error=_check_valid)
        self.num_turns += 1

        if _check_win:
            self._king_escaped = self.king_escaped()
            self._king_captured = self.king_captured()

    def move_(self, *args, **kwargs) -> 'Board':
        b = self.copy()
        b.move(*args, **kwargs)
        return b

    def random_move(self):
        import random
        self.move(*random.choice(list(self.legal_moves(piece_type=self.to_play()))), _check_valid=False)

    # endregion

    # region Misc methods
    @staticmethod
    def _relative_square(source: Square, direction: Tuple[int, int]) -> Square:
        return source + Square(*direction)

    def _surrounding_squares(self, source: Square) -> Iterator[Square]:
        for check_dir in DIRECTIONS:
            square = self._relative_square(source, check_dir)
            if self._in_bounds(square):
                yield square

    def _set_square(self, square: Square, new_val: Union[Tile, Piece]):
        self[square] = new_val.value

    def add_piece(self, square: Square, piece: Piece, replace=True):
        if piece.value not in BASE_PIECES:
            raise ValueError(f'{piece} is not a valid piece to add to the board.')

        dest_value = self[square]

        if not replace and dest_value in ALL_PIECES:
            raise errors.PositionError(
                f"Can't set square {square} to piece {piece} because argument replace is set to False, and the square "
                f"contains the piece {Piece(dest_value)}."
            )

        if dest_value == Tile.escape.value or dest_value == Tile.throne.value:
            if piece == Piece.king:
                self[square] = piece.value + dest_value
                return
            else:
                raise errors.PositionError('Only a king can be placed on the throne or an escape square.')

        self[square] = piece.value

    def remove_piece(self, square: Square, raise_no_piece=True) -> int:
        dest_value = self[square]
        if raise_no_piece and dest_value not in ALL_PIECES:
            raise errors.PositionError(f"There's no piece on the square {square}. Found tile {Tile(dest_value)}")

        new_value = Tile.normal
        piece = dest_value
        if dest_value == Piece.king_on_throne.value:
            new_value = Tile.throne
            piece = Piece.king.value
        elif dest_value == Piece.king_on_escape.value:
            new_value = Tile.escape
            piece = Piece.king.value

        self[square] = new_value
        return piece

    def to_play(self) -> Piece:
        return Piece(2 - self.num_turns % 2)
    # endregion
# endregion


# region Variants
class BrandubhBoard(Board):
    def __init__(self, *args, **kwargs):
        super().__init__(variants.brandubh_args, *args, **kwargs)


class HnefataflBoard(Board):
    def __init__(self, *args, **kwargs):
        super().__init__(variants.hnefatafl_args, *args, **kwargs)
# endregion
