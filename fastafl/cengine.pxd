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
    cdef public bint king_two_sided_capture
    cdef public bint move_over_throne
    cdef public bint king_can_enter_throne
    cdef public bint _king_captured
    cdef public bint _king_escaped

    cpdef bint _is_valid(self, Square square, bint is_king=*)
    cpdef list legal_moves(self, tuple pieces=*, int piece_type=*)
    cpdef bint _has_legals_check(self, Square piece_square)

    cpdef bint king_escaped(self)
    cpdef bint __king_captured_lambda(self, Square x)
    cpdef bint king_captured(self)

    cpdef void _check_capture(self, Square moved_piece)
    cpdef tuple __next_check_squares(self, tuple enemy, squares)
    cpdef bint __blocked(self, Square square)
    cpdef tuple __recurse_check(self, Square square, list checked, tuple enemy)
    cpdef void _check_surround(self, Square moved_piece)
    cpdef void move(self, source, dest, bint check_turn=*, bint _check_valid=*, bint _check_win=*)

    cpdef int to_play(self)
