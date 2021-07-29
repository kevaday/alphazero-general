# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

cimport numpy as np

cdef class GameState:
    cdef public object _board
    cdef public int _player
    cdef public int _turns

    cpdef GameState clone(self)
    @staticmethod
    cdef int action_size()
    @staticmethod
    cdef tuple observation_size()
    @staticmethod
    cdef int num_players()
    cpdef np.ndarray valid_moves(self)
    cpdef int _next_player(self, int player, int turns=*)
    cpdef void _update_turn(self)
    cpdef void play_action(self, int action)
    cpdef np.ndarray win_state(self)
    cpdef float[:, :, :] observation(self)
    cpdef list symmetries(self, float[:] pi)
