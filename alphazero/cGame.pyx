# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from typing import Tuple, List, Any
cimport numpy as np

cdef class GameState:
    def __init__(self, object board):
        self._board = board
        self._player = 0
        self._turns = 0

    def __str__(self) -> str:
        return f'Player:\t{self._player}\n{self._board}\n'

    def __eq__(self, other: 'GameState') -> bool:
        """Compare the current game state to an other"""
        pass

    cpdef GameState clone(self):
        """Return a new clone of the game state, independent of the current one."""
        pass

    @staticmethod
    cdef int action_size():
        """The size of the action space for the game"""
        pass

    @staticmethod
    cdef tuple observation_size():
        """
        Returns:
            observation_size: the shape of observations of the current state,
                             must be in the form channels x width x height.
                             If only one plane is needed for observation, use 1 for channels.
        """
        pass

    cpdef np.ndarray valid_moves(self):
        """Returns a numpy binary array containing zeros for invalid moves and ones for valids."""
        pass

    @staticmethod
    cdef int num_players():
        """
        Returns:
            num_players: the number of total players participating in the game.
        """
        pass

    @property
    def player(self) -> int:
        return self._player

    @property
    def turns(self):
        return self._turns

    cpdef int _next_player(self, int player, int turns=1):
        return (player + turns) % GameState.num_players()

    cpdef void _update_turn(self):
        """Should be called at the end of play_action"""
        self._player = self._next_player(self._player)
        self._turns += 1

    cpdef void play_action(self, int action):
        """Play the action in the current state given by argument action."""
        pass

    cpdef np.ndarray win_state(self):
        """
        Get the win state of the game, a tuple of boolean values
        for each player indicating if they have won, plus one more
        boolean at the end to indicate a draw.
        """
        pass

    cpdef float[:, :, :] observation(self):
        """Get an observation from the game state in the form of a numpy array with the size of self.observation_size"""
        pass

    cpdef list symmetries(self, float[:] pi):
        """
        Args:
            pi: the current policy for the given canonical state

        Returns:
            symmetries: list of state, pi pairs for symmetric samples of
                        the given state and pi (ex: mirror, rotation).
                        This is an optional method as symmetric samples
                        can be disabled for training.
        """
        pass
