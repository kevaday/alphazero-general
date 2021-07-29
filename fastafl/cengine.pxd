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

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t


cdef class Square:
    cdef public Py_ssize_t x
    cdef public Py_ssize_t y
    cpdef tuple _get_tuple(self)

ctypedef fused RawState_T:
    str
    np.ndarray

cpdef void _raise_invalid_board(RawState_T board_data)

cdef class Board:
    cdef public np.ndarray _state
    cdef public Py_ssize_t width
    cdef public Py_ssize_t height
    cdef public bint king_two_sided_capture
    cdef public int num_turns
    cdef public bint _king_captured
    cdef public bint _king_escaped

    cpdef void _load_str_inner(self, str data)
    cpdef void _load_str(self, str data, bint _skip_error_check=*)
    cpdef Board copy(self)

    cpdef bint _in_bounds(self, Square square)
    cpdef bint _is_valid(self, Square square, bint is_king=*)
    cpdef list __iter_pieces(self, tuple pieces, int piece_type=*)
    cpdef list legal_moves(self, tuple pieces=*, int piece_type=*)
    cpdef bint __has_legals_check(self, Square piece_square)
    cpdef bint has_legal_moves(self, tuple pieces=*, int piece_type=*)

    cpdef bint king_escaped(self)
    cpdef bint __king_captured_lambda(self, Square x)
    cpdef bint king_captured(self)
    cpdef int get_winner(self)
    cpdef bint is_game_over(self)

    cpdef void _check_capture(self, Square moved_piece)
    cpdef tuple __next_check_squares(self, tuple enemy, squares)
    cpdef bint __blocked(self, Square square)
    cpdef tuple __recurse_check(self, Square square, list checked, tuple enemy)
    cpdef void _check_surround(self, Square moved_piece)
    cpdef void move(self, Square source, Square dest, bint check_turn=*, bint _check_valid=*, bint _check_win=*)
    cpdef Board move_(self, Square source, Square dest, bint check_turn=*, bint _check_valid=*, bint _check_win=*)
    cpdef void random_move(self)
    cpdef Board random_move_(self)

    cpdef Square _relative_square(self, Square source, tuple direction)
    cpdef list _surrounding_squares(self, Square source)
    cpdef void _set_square(self, Square square, int new_val)
    cpdef tuple _get_team(self, int piece_type, bint enemy=*)
    cpdef int _team_colour(self, int piece_type)
    cpdef void add_piece(self, Square square, int piece, bint replace=*, _check_valid=*)
    cpdef int remove_piece(self, Square square, bint raise_no_piece=*)
    cpdef np.ndarray get_mask(self, tuple piece_types)
    cpdef list get_squares(self, tuple piece_types)
    cpdef int to_play(self)
