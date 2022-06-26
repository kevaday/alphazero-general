# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

import warnings
import numpy as np
cimport numpy as np

from boardgame cimport board
from boardgame import errors

np.import_array()


cdef tuple DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
cdef int HASH_CONST = 0x1f1f1f1f


cdef class Square:
    """A simple class which represents coordinates on a board."""

    def __init__(self, Py_ssize_t x, Py_ssize_t y):
        self.x, self.y = x, y

    cpdef tuple _get_tuple(self):
        return self.x, self.y

    def __str__(self):
        return self.__class__.__name__ + str(self._get_tuple())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self._get_tuple() == other._get_tuple()

    def __hash__(self):
        return (self.x * HASH_CONST) ^ self.y

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __getitem__(self, int index):
        return self._get_tuple()[index]

    def __setitem__(self, int index, int value):
        vals = list(self._get_tuple())
        vals[index] = vals
        self.x, self.y = vals

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self._get_tuple())


cdef class Team:
    """A simple class to manage the types of pieces on a team (ex. black and white pieces in chess)"""

    def __init__(self, int team_colour, tuple team_pieces):
        self.colour = team_colour
        self.pieces = team_pieces

    def __iter__(self):
        return iter(self.pieces)

    def __len__(self):
        return len(self.pieces)

    def __reversed__(self):
        return reversed(self.pieces)


cpdef void _raise_invalid_board(board.RawState_T board_data):
    raise errors.InvalidBoardState(f'An attempt was made to load an invalid board state. Got data:\n{board_data}')


cdef class BaseBoard:
    """A base board class assuming 2 players (can be modified by overiding methods)"""

    def __init__(self, state, tuple valid_squares, tuple all_pieces, *teams, int empty_square_value=0, use_load_whitespace=False):
        self._valid_squares = valid_squares
        self._all_pieces = all_pieces
        self._empty_square = empty_square_value
        self.use_load_whitespace = use_load_whitespace
        self.teams = teams
        self.num_turns = 0

        if isinstance(state, str):
            self._load_str(state)
        else:
            self._state = state
        self.height = self._state.shape[0]
        self.width = self._state.shape[1]

    cpdef void _load_str_inner(self, str data):
        self._state = np.array(
            [[int(tile) for tile in (row.split() if self.use_load_whitespace else row)]
            for row in data.splitlines()], dtype=np.uint8
        )

    cpdef void _load_str(self, str data, bint _skip_error_check=False):
        if _skip_error_check:
            self._load_str_inner(data)
            return

        with warnings.catch_warnings(record=True) as w:
            try:
                self._load_str_inner(data)
            except ValueError:
                _raise_invalid_board(data)

            if len(w) and isinstance(w[0].category, np.VisibleDeprecationWarning):
                _raise_invalid_board(data)

            if not np.isin(self._state, self._valid_squares).all():
                _raise_invalid_board(data)

    cpdef BaseBoard copy(self):
        return self.__copy__()

    # endregion

    # region Magic methods
    def __str__(self) -> str:
        format_string = ' '.join(['{:>' + str(len(str(np.max(self._state)))) + '}'] * self.width)
        return '\n'.join(
            [format_string.format(*[str(tile) for tile in row]) for row in self._state]
        )

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, Square key) -> int:
        return self._state[key.y, key.x]

    def __setitem__(self, Square key, int value):
        self._state[key.y, key.x] = value

    def __eq__(self, other):
        return (self._state == other._state).all() and self.num_turns == other.num_turns

    def __hash__(self):
        return (hash(self._state.tostring()) * 0x1f1f1f1f) ^ self.num_turns

    def __copy__(self):
        cdef BaseBoard board = self.__new__(self.__class__)
        board._state = np.copy(self._state)
        board.width = self.width
        board.height = self.height
        board.num_turns = self.num_turns
        board._valid_squares = self._valid_squares
        board._all_pieces = self._all_pieces
        board._empty_square = self._empty_square
        board.use_load_whitespace = self.use_load_whitespace
        board.teams = self.teams
        return board

    def __deepcopy__(self, memodict):
        return self.__copy__()

    cpdef bint _in_bounds(self, Square square):
        return self.width - 1 >= square.x >= 0 and self.height - 1 >= square.y >= 0

    cpdef list _iter_pieces(self, tuple pieces, int piece_type=0):
        cdef list ret = [piece for piece in pieces if self[piece] in self._all_pieces]
        cdef tuple piece_types = ()
        cdef tuple pos

        if piece_type != 0:
            piece_types = self._get_team(piece_type)
        elif not pieces:
            piece_types = self._all_pieces

        if piece_types:
            ret.extend(self.get_squares(piece_types))

        return ret

    cpdef list legal_moves(self, tuple pieces=(), int piece_type=0):
        """Abstract method for getting legal moves"""
        pass

    cpdef bint _has_legals_check(self, Square piece_square):
        """Abstract method for determining if a piece on piece_square has legal moves."""
        pass

    cpdef bint has_legal_moves(self, tuple pieces=(), int piece_type=0):
        cdef Square piece_square
        cdef tuple piece_types = ()
        cdef tuple pos
        cdef int piece

        if pieces:
            for piece_square in pieces:
                #if self[piece_square] in self._all_pieces and self._has_legals_check(piece_square):
                if self._has_legals_check(piece_square):  # faster
                    return True

        if piece_type != 0:
            piece_types = self._get_team(piece_type)
        elif not pieces:
            piece_types = self._all_pieces

        if piece_types:
            # re-wrote self.get_squares for speed
            for piece in piece_types:
                for pos in zip(*np.where(self._state == piece)):
                    if self._has_legals_check(Square(*reversed(pos))):
                        return True

        return False

    cpdef int get_winner(self):
        """Abstract method for getting the winner"""
        pass

    cpdef bint is_game_over(self):
        return self.get_winner() != 0

    cpdef void move(self, source, dest, bint check_turn=True, bint _check_valid=True, bint _check_win=True):
        cdef int source_val = self[source]
        if _check_valid:
            if source_val not in self._all_pieces:
                raise errors.InvalidMoveError(f'The source square has no piece on it. Found value: {source_val}')

            if check_turn:
                colour = self._team_colour(source_val)
                if colour != self.to_play():
                    raise errors.InvalidMoveError(f"It is not player {colour}'s turn.")

            if (source, dest) not in self.legal_moves((source,)):
                raise errors.InvalidMoveError(
                    f'The move {source}->{dest} is illegal. Source value: {source_val}, dest. value: {self[dest]}'
                )

        self.add_piece(dest, self.remove_piece(source, raise_no_piece=_check_valid), replace=True, _check_valid=_check_valid)
        self.num_turns += 1

    cpdef BaseBoard move_(self, source, dest, bint check_turn=True, bint _check_valid=True, bint _check_win=True):
        cdef BaseBoard b = self.copy()
        b.move(source, dest, check_turn, _check_valid, _check_win)
        return b

    cpdef void random_move(self, bint print_move=False):
        import random
        cdef tuple move = random.choice(self.legal_moves(piece_type=self.to_play()))

        if print_move: print(move)
        self.move(*move, _check_valid=False)

    cpdef BaseBoard random_move_(self, bint print_move=False):
        cdef BaseBoard b = self.copy()
        b.random_move(print_move)
        return b

    cpdef Square _relative_square(self, Square source, tuple direction):
        return Square(source.x + direction[0], source.y + direction[1])

    cpdef list _surrounding_squares(self, Square source):
        cdef tuple check_dir
        cdef Square square
        cdef list squares = []

        for check_dir in DIRECTIONS:
            square = self._relative_square(source, check_dir)
            if self._in_bounds(square):
                squares.append(square)

        return squares

    cpdef void _set_square(self, Square square, int new_val):
        self[square] = new_val

    cpdef tuple _get_team(self, int piece_type, bint enemy=False):
        if enemy and len(self.teams) != 2:
            raise ValueError(
                f'Cannot find enemy of piece type {piece_type} because there are {len(self.teams)} teams (expected 2).'
            )

        cdef Team team
        cdef Py_ssize_t i
        for i, team in enumerate(self.teams):
            if piece_type in team:
                if enemy:
                    return self.teams[0 if i == 1 else 1].pieces
                else:
                    return team.pieces

    cpdef int _team_colour(self, int piece_type):
        cdef Team team

        for team in self.teams:
            if piece_type in team:
                return team.colour

        return 0

    cpdef void add_piece(self, Square square, int piece, bint replace=False, _check_valid=True):
        if _check_valid:
            if piece not in self._all_pieces:
                raise ValueError(f'{piece} is not a valid piece to add to the board.')
            if not replace and self[square] in self._all_pieces:
                raise errors.PositionError(
                    f"Can't set square {square} to piece {piece} because argument replace is set to False, and the square "
                    f"contains the piece {self[square]}."
                )

        self[square] = piece

    cpdef int remove_piece(self, Square square, bint raise_no_piece=True):
        cdef int dest_value = self[square]
        if raise_no_piece and dest_value not in self._all_pieces:
            raise errors.PositionError(f"There's no piece on the square {square}. Found tile {dest_value}")

        self[square] = self._empty_square
        return dest_value

    cpdef np.ndarray get_mask(self, tuple piece_types):
        cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask = self._state == piece_types[0]
        cdef int piece
        for piece in piece_types[1:]:
            mask |= (self._state == piece)
        return mask

    cpdef list get_squares(self, tuple piece_types):
        cdef list squares = []
        cdef tuple pos
        for pos in zip(*np.where(self.get_mask(piece_types))):
            squares.append(Square(*reversed(pos)))
        return squares

    cpdef int to_play(self):
        return self.num_turns % len(self.teams)

    cpdef bint is_turn(self, Square square):
        return self[square] in self._get_team(self.to_play())
