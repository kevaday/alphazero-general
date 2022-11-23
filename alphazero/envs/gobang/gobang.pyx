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

import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from alphazero.envs.gobang.GobangLogic import Board
from alphazero.envs.gobang.GobangLogic cimport Board


DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

cdef int NUM_PLAYERS = 2
cdef int BOARD_SIZE = 15
cdef int NUM_IN_ROW = 5
cdef int MAX_MOVES = BOARD_SIZE ** 2

cdef int ACTION_SIZE = BOARD_SIZE ** 2
cdef bint MULTI_PLANE_OBSERVATION = True
cdef int NUM_CHANNELS = 4 if MULTI_PLANE_OBSERVATION else 1
cdef tuple OBSERVATION_SIZE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)


cpdef tuple get_move(int action, int n):
    return action // n, action % n


cpdef int get_action(tuple move, int n):
    return n * move[1] + move[0]


cdef class Game:#(GameState):
    cdef public Board _board
    cdef public int _player
    cdef public int _turns
    
    def __init__(self, _board=None):
        self._board = _board or self._get_board()
        self._player = 0
        self._turns = 0

    @staticmethod
    def _get_board(*args, **kwargs) -> Board:
        return Board(BOARD_SIZE, NUM_IN_ROW, *args, **kwargs)

    def __eq__(self, other: 'Game') -> bool:
        return (
            self._board.pieces == other._board.pieces
            and self._board.n == other._board.n
            and self._board.n_in_row == other._board.n_in_row
            and self._player == other._player
            and self.turns == other.turns
        )

    cpdef Game clone(self):
        cdef Board board = self._get_board(_pieces=np.copy(self._board.pieces))
        cdef Game g = Game(_board=board)
        g._player = self._player
        g._turns = self.turns
        return g
    
    @property
    def player(self) -> int:
        return self._player

    @property
    def turns(self):
        return self._turns
    
    cpdef int _next_player(self, int player, int turns=1):
        return (player + turns) % NUM_PLAYERS

    cpdef void _update_turn(self):
        """Should be called at the end of play_action"""
        self._player = self._next_player(self._player)
        self._turns += 1

    @staticmethod
    def num_players():
        return NUM_PLAYERS

    @staticmethod
    def action_size():
        return ACTION_SIZE

    @staticmethod
    def observation_size():
        return OBSERVATION_SIZE

    @staticmethod
    def max_turns():
        return MAX_MOVES

    @staticmethod
    def has_draw():
        return True

    cpdef np.ndarray valid_moves(self):
        # return a fixed size binary vector
        cdef list valids = [0] * self.action_size()
        cdef tuple move

        for move in self._board.get_legal_moves():
            valids[get_action(move, self._board.n)] = 1

        return np.array(valids, dtype=np.uint8)

    cpdef void play_action(self, int action):
        cdef tuple move = get_move(action, self._board.n)
        self._board.execute_move(move, (1, -1)[self.player])
        self._update_turn()

    cpdef np.ndarray win_state(self):
        cdef list result = [False] * (NUM_PLAYERS + 1)
        cdef bint game_over
        cdef int player
        cdef Py_ssize_t index
        game_over, player = self._board.get_win_state()

        if game_over:
            index = NUM_PLAYERS
            if player == 1:
                index = 0
            elif player == -1:
                index = 1
            result[index] = True

        return np.array(result, dtype=np.uint8)

    cpdef np.ndarray observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            player1 = np.where(pieces == 1, 1, 0)
            player2 = np.where(pieces == -1, 1, 0)
            colour = np.full_like(pieces, self.player)
            turn = np.full_like(pieces, self.turns / self._board.n**2, dtype=np.float32)
            return np.array([player1, player2, colour, turn], dtype=np.float32)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    cpdef list symmetries(self, np.ndarray pi):
        # mirror, rotational

        cdef np.ndarray[np.float32_t, ndim=2] pi_board = np.reshape(pi, (self._board.n, self._board.n))
        cdef np.ndarray[np.int32_t, ndim=2] new_b
        cdef np.ndarray[np.float32_t, ndim=2] new_pi
        cdef list result = []
        cdef Game gs
        cdef Py_ssize_t i
        cdef bint j

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(np.asarray(self._board.pieces), i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)

                gs = self.clone()
                gs._board.pieces = new_b
                result.append((gs, new_pi.ravel()))

        return result


cpdef void display(Game gs, int action=-1):
    cdef np.ndarray[np.int32_t, ndim=2] board = gs._board.pieces
    cdef int n = board.shape[0]
    cdef Py_ssize_t y, x
    cdef int piece
    cdef str prefix = ' '

    if action != -1:
        print(f'Action: {action}, Move: {get_move(action, n)}')

    print(' ' * 4 + '|'.join([str(x) for x in range(n)]))
    print(' ' * 4 + '-' * (n * 2))
    
    for y in range(n):
        if y > 9:
            prefix = ''
        print(prefix + f'{y} |', end='')    # print the row #
        
        for x in range(n):
            piece = board[x, y]    # get the piece to print
            if piece == -1:
                print('b ', end='')
            elif piece == 1:
                print('W ', end='')
            else:
                print('- ', end='')
        print('|')

    print(' ' * 4 + '-' * (n * 2))
