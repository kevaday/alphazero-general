# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

import numpy as np
cimport numpy as np

from boardgame.board cimport BaseBoard, Team, Square
from boardgame import errors
from alphazero.envs.stratego cimport engine

np.import_array()


DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

ctypedef fused SquareOrPiece:
    Square
    int


cdef int tile_normal = 0
cdef int tile_blocked = 13

cdef int piece_spy = 1
cdef int piece_scout = 2
cdef int piece_miner = 3
cdef int piece_sergeant = 4
cdef int piece_lieutenant = 5
cdef int piece_captain = 6
cdef int piece_major = 7
cdef int piece_colonel = 8
cdef int piece_general = 9
cdef int piece_marshal = 10
cdef int piece_bomb = 11
cdef int piece_flag = 12
cdef int other_team_offset = 20
cdef int visible_offset = 100

cdef int RED_TEAM_COLOUR = piece_spy
cdef int BLUE_TEAM_COLOUR = RED_TEAM_COLOUR + other_team_offset

cdef tuple ALL_SQUARES = tuple(range(tile_blocked + 1))
cdef tuple ALL_RED_PIECES = tuple(range(piece_spy, tile_blocked))
cdef tuple ALL_BLUE_PIECES = tuple((p + other_team_offset for p in ALL_RED_PIECES))
cdef tuple ALL_PIECES = (*ALL_RED_PIECES, *ALL_BLUE_PIECES)
cdef tuple MOVABLE_PIECES = ALL_RED_PIECES[0:piece_bomb]
cdef list RED_PIECES_TO_PLACE = [piece_spy] + [piece_scout] * 5 + [piece_miner] * 4 + [piece_sergeant] * 2 + \
    [piece_lieutenant] * 2 + [piece_captain] * 3 + [piece_major] * 3 + [piece_colonel] * 2 + \
    [piece_general] + [piece_marshal] + [piece_flag] + [piece_bomb] * 5
cdef list BLUE_PIECES_TO_PLACE = [p + other_team_offset for p in RED_PIECES_TO_PLACE]

cdef tuple DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
cdef np.ndarray START_STATE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 13, 13, 0, 0, 13, 13, 0, 0],
    [0, 0, 13, 13, 0, 0, 13, 13, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)
cdef int HASH_CONST = 0x1f1f1f1f

"""
cdef str START_STATE = (
        (' '.join([str(tile_normal)] * 10) + '\n') * 3 +
        (((str(tile_normal) + ' ') * 2 + (str(tile_blocked) + ' ') * 2) * 2 + ' '.join([str(tile_normal)] * 2) + '\n') * 2 +
        (' '.join([str(tile_normal)] * 10) + '\n') * 3
)
"""


cdef class Board(BaseBoard):
    def __init__(self, state=None):
        self.red_exploded_bombs = []
        self.blue_exploded_bombs = []
        self._red_flag_captured = self._blue_flag_captured = False
        self._red_pieces_to_place = RED_PIECES_TO_PLACE.copy()
        self._blue_pieces_to_place = BLUE_PIECES_TO_PLACE.copy()
        super(Board, self).__init__(
            state or np.copy(START_STATE), ALL_SQUARES, ALL_PIECES,
            Team(RED_TEAM_COLOUR, ALL_RED_PIECES), Team(BLUE_TEAM_COLOUR, ALL_BLUE_PIECES)
        )

    def __getitem__(self, Square key) -> int:
        return self._get_raw_value(self._state[key.y, key.x])

    def __eq__(self, other):
        return (
                super(Board, self).__eq__(other)
                and self.red_exploded_bombs == other.red_exploded_bombs
                and self.blue_exploded_bombs == other.blue_exploded_bombs
                and self._red_flag_captured == other._red_flag_captured
                and self._blue_flag_captured == other._blue_flag_captured
                and self._red_pieces_to_place == other._red_pieces_to_place
                and self._blue_pieces_to_place == other._blue_pieces_to_place
        )

    def __hash__(self):
        return (super(Board, self).__hash__()  * HASH_CONST) ^ ((int(self._blue_flag_captured) + 1) * HASH_CONST) \
               ^ ((int(self._red_flag_captured) + 1) * HASH_CONST) ^ hash(self.red_exploded_bombs) \
               ^ hash(self.blue_exploded_bombs) ^ hash(self._red_pieces_to_place) ^ hash(self._blue_pieces_to_place)

    def __copy__(self):
        cdef Board board = super(Board, self).__copy__()
        board._red_flag_captured = self._red_flag_captured
        board._blue_flag_captured = self._blue_flag_captured
        board.red_exploded_bombs = self.red_exploded_bombs.copy()
        board.blue_exploded_bombs = self.blue_exploded_bombs.copy()
        board._red_pieces_to_place = self._red_pieces_to_place.copy()
        board._blue_pieces_to_place = self._blue_pieces_to_place.copy()
        return board

    @property
    def play_phase(self):
        return not bool(self._red_pieces_to_place) and not bool(self._blue_pieces_to_place)

    cpdef void clear_pieces_to_place(self):
        self._red_pieces_to_place = []
        self._blue_pieces_to_place = []

    cpdef int __base_piece(self, int piece_type):
        return piece_type if piece_type in ALL_RED_PIECES else piece_type - other_team_offset if piece_type in ALL_BLUE_PIECES else 0

    cpdef bint _is_valid(self, Square dest_square, int piece_type):
        if not self._in_bounds(dest_square):  # square must be in bounds
            return False

        cdef int dest_value = self[dest_square]

        return (
            # square's not blocked
            dest_value != tile_blocked
            # pieces are on opposite teams
            and self._team_colour(piece_type) != self._team_colour(dest_value)
        )

    cpdef list legal_moves(self, tuple pieces=(), int piece_type=0):
        cdef Square piece_square, cur_square
        cdef int piece_square_value, team_turn
        cdef tuple move_dir
        cdef list legals = []

        if self.play_phase:
            for piece_square in self._iter_pieces(pieces, piece_type):
                piece_square_value = self[piece_square]
                for move_dir in DIRECTIONS:
                    cur_square = self._relative_square(piece_square, move_dir)

                    while self._is_valid(cur_square, piece_square_value):
                        legals.append((piece_square, cur_square))
                        # get the next square to check in this direction only if the piece is a scout
                        if self.__base_piece(piece_square_value) == piece_scout:
                            cur_square = self._relative_square(cur_square, move_dir)
                        else:
                            break
        else:
            for cur_square in self.get_squares((tile_normal,)):
                team_turn = self.to_play()
                if (
                    (team_turn == RED_TEAM_COLOUR and cur_square.y < 3)
                    or (team_turn == BLUE_TEAM_COLOUR and cur_square.y > 4)
                ):
                    for piece_square_value in set(
                        self._red_pieces_to_place if team_turn == RED_TEAM_COLOUR
                        else self._blue_pieces_to_place
                    ):
                        legals.append((piece_square_value, cur_square))

        return legals

    cpdef bint _has_legals_check(self, Square piece_square):
        cdef tuple move_dir
        cdef Square cur_square
        cdef int piece_square_value = self[piece_square]

        for move_dir in DIRECTIONS:
            cur_square = self._relative_square(piece_square, move_dir)
            if self._is_valid(cur_square, piece_square_value):
                return True
        return False

    cpdef bint has_legal_moves(self, tuple pieces=(), int piece_type=0):
        return not self.play_phase or super(Board, self).has_legal_moves(pieces, piece_type)

    cpdef int get_winner(self):
        if self._red_flag_captured or not self.has_legal_moves(piece_type=RED_TEAM_COLOUR):
            return BLUE_TEAM_COLOUR
        elif self._blue_flag_captured or not self.has_legal_moves(piece_type=BLUE_TEAM_COLOUR):
            return RED_TEAM_COLOUR
        else:
            return 0

    cpdef void move(self, source, dest, bint check_turn=True, bint _check_valid=True, bint _check_win=True):
        cdef int dest_value
        cdef int other_team_colour
        cdef int dest_piece
        cdef int source_piece

        if self.play_phase:
            if _check_valid:
                if not isinstance(source, Square):
                    raise ValueError('Source square must be of type Square in the play phase of the game.')
                if not isinstance(dest, Square):
                    raise ValueError('Destination square must be of type Square in the play phase of the game.')

            dest_value = self[dest]
            other_team_colour = self._team_colour(dest_value)
            dest_piece = self.__base_piece(dest_value)
            source_piece = self.__base_piece(self[source])

            super(Board, self).move(source, dest, check_turn, _check_valid, _check_win)

            # only perform piece checks if the destination square has a piece on it
            if dest_value != tile_normal:
                if dest_piece == piece_flag:
                    # one of the teams has won
                    self._red_flag_captured = other_team_colour == RED_TEAM_COLOUR
                    self._blue_flag_captured = other_team_colour == BLUE_TEAM_COLOUR

                elif (dest_piece == piece_bomb and source_piece != piece_miner) or source_piece == dest_piece:
                    # both pieces are destroyed
                    self[dest] = tile_normal
                    if dest_piece == piece_bomb:
                        self.red_exploded_bombs.append(dest) if other_team_colour == RED_TEAM_COLOUR else self.blue_exploded_bombs.append(dest)
                    return

                elif source_piece < dest_piece and not (source_piece == piece_spy and dest_piece == piece_marshal):
                    # the target piece wins, keep it where it was and make it visible
                    self[dest] = self._get_visible_value(dest_value)

                # in every other case, source piece takes the place of the target piece and becomes visible
                self[dest] = self._get_visible_value(self[dest])

        else:
            if _check_valid:
                if (source, dest) not in self.legal_moves(piece_type=self.to_play()):
                    raise errors.InvalidMoveError(
                        f"Cannot add piece {source} to the square {dest} because it's not a valid move."
                    )
                elif self.__base_piece(source) not in MOVABLE_PIECES:
                    raise errors.InvalidMoveError(
                        f"Cannot add piece {source} to the square {dest} because it's not a valid piece."
                    )
                elif source not in self.teams[super(Board, self).to_play()] or source not in (
                        self._red_pieces_to_place if self.to_play() == RED_TEAM_COLOUR else self._blue_pieces_to_place
                ):
                    raise errors.InvalidMoveError(
                        f"The piece {source} cannot be added to the square {dest} "
                        f"because it's not a valid piece for the player {super(Board, self).to_play()}"
                    )

            self.add_piece(dest, source, replace=False, _check_valid=_check_valid)
            if self.to_play() == RED_TEAM_COLOUR:
                self._red_pieces_to_place.remove(source)
            else:
                self._blue_pieces_to_place.remove(source)
            self.num_turns += 1

    cpdef int to_play(self):
        return RED_TEAM_COLOUR if super(Board, self).to_play() == 0 else BLUE_TEAM_COLOUR

    """
    cpdef np.ndarray get_team_mask(self):
        return self.get_mask(ALL_RED_PIECES if self.to_play() == RED_TEAM_COLOUR else ALL_BLUE_PIECES)

    cpdef np.ndarray get_enemy_mask(self):
        return self.get_mask(ALL_BLUE_PIECES if self.to_play() == RED_TEAM_COLOUR else ALL_RED_PIECES)

    cpdef np.ndarray get_raw_state(self):
        return self._state % visible_offset
    """

    cpdef int _get_raw_value(self, int piece_type):
        return piece_type % visible_offset

    cpdef int _get_visible_value(self, int piece_value):
        return piece_value + visible_offset if piece_value < visible_offset else piece_value


"""
cpdef void unittest():
    # create a new board
    board = Board()
    board.clear_pieces_to_place()

    # check that the board is empty
    assert board.get_mask(ALL_PIECES).all() == False, f"Board is not empty.\n{board}"

    # add test pieces to board
    board.add_piece(Square(0, 0), piece_spy)
    board.add_piece(Square(0, 1), piece_marshal + other_team_offset)

    # check that the board is not empty
    assert board.get_mask(ALL_PIECES).any() == True, f"Board is empty.\n{board}"

    # move the spy to the marshal's square
    board.move(Square(0, 0), Square(0, 1), False)

    # check that the spy is now on the marshal's square
    assert board[Square(0, 1)] == piece_spy, f"Spy is not on marshal's square.\n{board}"

    # check that the spy is now visible
    assert board._state[1, 0] == piece_spy + visible_offset, f"Spy is not visible. {board._state[1, 0]} != {piece_spy + visible_offset}"
"""
