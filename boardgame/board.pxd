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

ctypedef fused RawState_T:
    str
    np.ndarray

cpdef void _raise_invalid_board(RawState_T board_data)

cdef class Square:
    cdef public Py_ssize_t x
    cdef public Py_ssize_t y
    cpdef tuple _get_tuple(self)

cdef class Team:
    cdef public int colour
    cdef public tuple pieces

cdef class BaseBoard:
    cdef public tuple teams
    cdef public bint use_load_whitespace
    cdef public tuple _valid_squares
    cdef public tuple _all_pieces
    cdef public int _empty_square
    cdef public np.ndarray _state
    cdef public Py_ssize_t width
    cdef public Py_ssize_t height
    cdef public int num_turns

    cpdef void _load_str_inner(self, str data)
    cpdef void _load_str(self, str data, bint _skip_error_check=*)
    cpdef BaseBoard copy(self)

    cpdef bint _in_bounds(self, Square square)
    cpdef list _iter_pieces(self, tuple pieces, int piece_type=*)
    cpdef list legal_moves(self, tuple pieces=*, int piece_type=*)
    cpdef bint _has_legals_check(self, Square piece_square)
    cpdef bint has_legal_moves(self, tuple pieces=*, int piece_type=*)

    cpdef int get_winner(self)
    cpdef bint is_game_over(self)

    cpdef void move(self, source, dest, bint check_turn=*, bint _check_valid=*, bint _check_win=*)
    cpdef BaseBoard move_(self, source, dest, bint check_turn=*, bint _check_valid=*, bint _check_win=*)
    cpdef void random_move(self, bint print_move=*)
    cpdef BaseBoard random_move_(self, bint print_move=*)

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
    cpdef bint is_turn(self, Square square)
