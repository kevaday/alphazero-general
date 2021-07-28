# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from fastafl import variants, errors

import warnings
import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t


# region Tiles and pieces

cdef public int piece_attacker = 1
cdef public int piece_defender = 2
cdef public int piece_king = 3
cdef public int piece_king_on_throne = 7
cdef public int piece_king_on_escape = 8

cdef public int tile_normal = 0
cdef public int tile_throne = 4
cdef public int tile_escape = 5


cdef class Square:
    cdef public int x
    cdef public int y
    def __init__(self, int x, int y):
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
        return np.prod(self._get_tuple())

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
# endregion


# region Constants
cdef tuple ALL_PIECES = (piece_attacker, piece_defender, piece_king, piece_king_on_throne, piece_king_on_escape)
cdef tuple KING_ON_TILE = (piece_king_on_throne, piece_king_on_escape)
cdef tuple BASE_PIECES = tuple(piece for piece in ALL_PIECES if piece not in KING_ON_TILE)
cdef tuple ALL_TILES = (tile_normal, tile_throne, tile_escape)
cdef tuple ALL_SQUARES = (*ALL_PIECES, *ALL_TILES)
cdef tuple KING_VALUES = (piece_king, *KING_ON_TILE)
cdef tuple ATTACKERS = (piece_attacker, *KING_VALUES)
cdef tuple SPECIAL_TILES = (tile_throne, tile_escape)
cdef tuple KING_CAPTURE = (piece_defender, *SPECIAL_TILES)

ctypedef fused RawState_T:
    str
    np.ndarray

cdef tuple DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))

# endregion


# region Helper functions
cpdef _raise_invalid_board(RawState_T board_data):
    raise errors.InvalidBoardState(f'An attempt was made to load an invalid board state. Got data:\n{board_data}')

# endregion


# region Board class
cdef class Board:
    cdef public np.ndarray _state
    cdef public int width
    cdef public int height
    cdef public bint king_two_sided_capture
    cdef public int num_turns
    cdef public bint _king_captured
    cdef public bint _king_escaped

    # region Load board
    def __init__(self, str state = variants.hnefatafl, bint king_two_sided_capture=False):
        self._load_str(state)
        self.king_two_sided_capture = king_two_sided_capture
        self.num_turns = 0
        self._king_captured = False
        self._king_escaped = False

    cpdef _load_str_inner(self, str data):
        self._state = np.array([[int(tile) for tile in row] for row in data.splitlines()], dtype=np.int32)
        self.height = self._state.shape[0]
        self.width = self._state.shape[1]

    cpdef _load_str(self, str data, bint _skip_error_check=False):
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

            if not np.isin(self._state, ALL_SQUARES).all():
                _raise_invalid_board(data)

    cpdef Board copy(self):
        return self.__copy__()

    # endregion

    # region Magic methods
    def __str__(self) -> str:
        return '\n'.join([' '.join([str(tile) for tile in row]) for row in self._state])

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, Square key) -> int:
        return self._state[key.y, key.x]

    def __setitem__(self, Square key, int value):
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
        board = self.__new__(self.__class__)
        board._state = np.copy(self._state)
        board.width = self.width
        board.height = self.height
        board.num_turns = self.num_turns
        board.king_two_sided_capture = self.king_two_sided_capture
        board._king_captured = self._king_captured
        board._king_escaped = self._king_escaped
        return board

    def __deepcopy__(self, memodict):
        return self.__copy__()

    # endregion

    # region Legal moves
    cpdef bint _in_bounds(self, Square square):
        return self.width - 1 >= square.x >= 0 and self.height - 1 >= square.y >= 0

    cpdef bint _is_valid(self, Square square, bint is_king=False):
        # No piece can go on a square out of bounds
        if not self._in_bounds(square):
            return False

        # Only king can go on escape square
        if self[square] == tile_escape:
            return is_king

        # Allowed if the square is empty
        if self[square] == tile_normal:
            return True

    cpdef list _get_piece_squares(self, condition):
        cdef int x, y
        cdef Square square
        cdef list squares = []

        for x in range(self.width):
            for y in range(self.height):
                square = Square(x, y)
                if condition(self[square]):
                    squares.append(square)

        return squares

    cpdef list __iter_pieces(self, tuple pieces, int piece_type):
        cdef Square piece
        cdef int x, y
        cdef list ret = [piece for piece in pieces if self[piece] in ALL_PIECES]

        if piece_type != piece_king:
            for x in range(self.width):
                for y in range(self.height):
                    piece = Square(x, y)
                    if self[piece] == piece_type:
                        ret.append(piece)
        elif not pieces:
            for x in range(self.width):
                for y in range(self.height):
                    piece = Square(x, y)
                    if self[piece] not in ALL_TILES:
                        ret.append(piece)

        return ret

    cpdef list legal_moves(self, tuple pieces = (), int piece_type=piece_king):
        cdef Square piece_square, cur_square
        cdef bint is_king
        cdef tuple move_dir
        cdef list legals = []

        for piece_square in self.__iter_pieces(pieces, piece_type):
            is_king = self[piece_square] in KING_VALUES
            for move_dir in DIRECTIONS:
                cur_square = self._relative_square(piece_square, move_dir)
                while self._is_valid(cur_square, is_king):
                    legals.append((piece_square, cur_square))
                    cur_square = self._relative_square(cur_square, move_dir)

        return legals

    cpdef bint has_legal_moves(self, tuple pieces = (), int piece_type=piece_king):
        try:
            next(iter(self.legal_moves(pieces=pieces, piece_type=piece_type)))
        except StopIteration:
            return False
        else:
            return True

    # endregion

    # region Win state
    cdef bint king_escaped(self):
        if self._king_escaped:
            return True
        return (self._state == piece_king_on_escape).any()

    cpdef bint __king_captured_lambda(self, Square x):
        return self[x] in KING_CAPTURE

    cpdef bint king_captured(self):
        if self._king_captured:
            return True
        elif self.king_two_sided_capture:
            return False
        cdef np.ndarray[np.uint8_t, ndim=2, cast=True] king_mask = (self._state == piece_king) \
            | (self._state == piece_king_on_throne) | (self._state == piece_king_on_escape)

        return any([all(map(self.__king_captured_lambda, self._surrounding_squares(Square(*reversed(pos)))))
                    for pos in zip(*np.where(king_mask))])

    cpdef get_winner(self):
        if self.king_escaped() or not self.has_legal_moves(piece_type=piece_defender):
            return piece_attacker
        elif self.king_captured() or not self.has_legal_moves(piece_type=piece_attacker):
            return piece_defender

    cpdef bint is_game_over(self):
        return self.get_winner() is not None

    # endregion

    # region Capture and move logic
    cpdef _check_capture(self, Square moved_piece):
        cdef int piece_val = self[moved_piece]
        cdef tuple friendly = ATTACKERS if piece_val in ATTACKERS else (piece_val,)
        cdef int enemy = 3 - piece_val if piece_val != piece_king else piece_defender

        cdef tuple check_dir
        cdef Square enemy_square, friendly_square
        cdef int value
        cdef bint enemy_is_king

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
                        self._set_square(enemy_square, tile_normal)

    cpdef tuple __next_check_squares(self, tuple enemy, squares):
        return tuple([sq for sq in squares if self[sq] in enemy])

    cpdef bint __blocked(self, Square square):
        return self[square] != tile_normal

    cpdef tuple __recurse_check(self, Square square, list checked, tuple enemy):
        checked.append(square)
        cdef tuple squares = tuple(self._surrounding_squares(square))
        cdef list captured_check
        cdef bint exit_recurse, is_captured
        cdef Square sq

        if all(tuple(map(self.__blocked, squares))):
            captured_check = []
            exit_recurse = False

            for sq in [x for x in self.__next_check_squares(enemy, squares) if x not in checked]:
                is_captured, exit_recurse = self.__recurse_check(sq, checked, enemy)
                captured_check.append(is_captured)
                if exit_recurse:
                    break

            return all(captured_check), exit_recurse
        else:
            return False, True

    cpdef _check_surround(self, Square moved_piece):
        cdef tuple enemy = ATTACKERS if self[moved_piece] == piece_defender else (piece_defender,)
        cdef tuple start_squares = self.__next_check_squares(enemy, self._surrounding_squares(moved_piece))
        if not start_squares: return

        cdef list checked_squares = []
        cdef list to_capture = []
        cdef Square s, captured
        for s in start_squares:
            if s not in checked_squares:
                to_capture = []

                if self.__recurse_check(s, to_capture, enemy)[0]:
                    for captured in to_capture:
                        if self[captured] in KING_VALUES:
                            self._king_captured = True
                        else:
                            self.remove_piece(captured, raise_no_piece=False)

                checked_squares.extend(to_capture)

    cpdef move(self, Square source, Square dest, bint _check_valid=True, bint _check_win=True):
        cdef int source_val = self[source]
        if _check_valid:
            if (source, dest) not in self.legal_moves((source,)):
                raise errors.InvalidMoveError(
                    f'The move {source}->{dest} is illegal. Source value: {source_val}, dest. value: {self[dest]}'
                )

        self[dest] = self.remove_piece(source, raise_no_piece=_check_valid)
        self._check_capture(dest)
        self._check_surround(dest)
        self.num_turns += 1

        if _check_win:
            self._king_escaped = self.king_escaped()
            self._king_captured = self.king_captured()

    cpdef Board move_(self, Square source, Square dest, bint _check_valid=True, bint _check_win=True):
        cdef Board b = self.copy()
        b.move(source, dest, _check_valid, _check_win)
        return b

    cpdef random_move(self):
        import random
        self.move(*random.choice(list(self.legal_moves(piece_type=self.to_play()))), _check_valid=False)

    # endregion

    # region Misc methods
    cpdef Square _relative_square(self, Square source, tuple direction):
        return source + Square(*direction)

    cpdef list _surrounding_squares(self, Square source):
        cdef tuple check_dir
        cdef Square square
        cdef list squares = []

        for check_dir in DIRECTIONS:
            square = self._relative_square(source, check_dir)
            if self._in_bounds(square):
                squares.append(square)

        return squares

    cpdef _set_square(self, Square square, int new_val):
        self[square] = new_val

    cpdef add_piece(self, Square square, int piece, bint replace=True):
        if piece not in BASE_PIECES:
            raise ValueError(f'{piece} is not a valid piece to add to the board.')

        cdef int dest_value = self[square]

        if not replace and dest_value in ALL_PIECES:
            raise errors.PositionError(
                f"Can't set square {square} to piece {piece} because argument replace is set to False, and the square "
                f"contains the piece {dest_value}."
            )

        if dest_value == tile_escape or dest_value == tile_throne:
            if piece == piece_king:
                self[square] = piece + dest_value
                return
            else:
                raise errors.PositionError('Only a king can be placed on the throne or an escape square.')

        self[square] = piece

    cpdef int remove_piece(self, Square square, bint raise_no_piece=True):
        cdef int dest_value = self[square]
        if raise_no_piece and dest_value not in ALL_PIECES:
            raise errors.PositionError(f"There's no piece on the square {square}. Found tile {dest_value}")

        cdef int new_value = tile_normal
        cdef int piece = dest_value
        if dest_value == piece_king_on_throne:
            new_value = tile_throne
            piece = piece_king
        elif dest_value == piece_king_on_escape:
            new_value = tile_escape
            piece = piece_king

        self[square] = new_value
        return piece

    cpdef int to_play(self):
        return 2 - self.num_turns % 2
    # endregion
# endregion


# region Variants
cdef class BrandubhBoard(Board):
    def __init__(self, *args, **kwargs):
        super().__init__(variants.brandubh_args, *args, **kwargs)


cdef class HnefataflBoard(Board):
    def __init__(self, *args, **kwargs):
        super().__init__(variants.hnefatafl_args, *args, **kwargs)
# endregion
