# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from boardgame.board cimport BaseBoard, Square


cdef class Board(BaseBoard):
    cdef public list red_exploded_bombs
    cdef public list blue_exploded_bombs
    cdef public bint _red_flag_captured
    cdef public bint _blue_flag_captured
    cdef public list _red_pieces_to_place
    cdef public list _blue_pieces_to_place

    cpdef void clear_pieces_to_place(self)
    cpdef int _base_piece(self, int piece_type)

    cpdef bint _is_valid(self, Square dest_square, int piece_type)
    cpdef list legal_moves(self, tuple pieces=*, int piece_type=*)
    cpdef bint _has_legals_check(self, Square piece_square)
    cpdef bint has_legal_moves(self, tuple pieces=*, int piece_type=*)

    cpdef int get_winner(self)
    cpdef void move(self, source, dest, bint check_turn=*, bint _check_valid=*, bint _check_win=*)
    cpdef int to_play(self)

    cpdef int _get_raw_value(self, int piece_type)
    cpdef int _get_visible_value(self, int piece_value)
